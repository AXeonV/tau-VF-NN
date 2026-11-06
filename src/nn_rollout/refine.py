"""Compare deterministic and diffusion-inspired flow refinement of whole time blocks."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .events import split_events
from .experiment import choose, closest, evaluation_grid, forecast, previous_index, summarize, supervised_loss
from .model import WindowFNO
from ..nn.model import signed_area
from ..nn_world.train import complex_curves, frame_errors
from ..trajectory_data import Trajectories, augment
from ..trajectory_metrics import atomic_json


class BlockRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(11, 32, 1)
        self.blocks = nn.ModuleList(nn.Conv2d(32, 32, (3, 3), dilation=(1, stride))
                                    for stride in (1, 8, 32, 4))
        self.output = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, initial, state, proposal, current, start, queries, noise):
        def over_time(x):
            return x[:, None].expand(-1, queries.shape[1], -1, -1)

        timing = torch.stack([queries/100, (queries-start[:, None])/(1+2*start[:, None]), noise], dim=2)
        x = torch.cat([over_time(initial), over_time(state), proposal, current,
                       timing[..., None].expand(-1, -1, -1, current.shape[-1])], dim=2)
        x = self.input(x.transpose(1, 2))
        for layer in self.blocks:
            width = layer.dilation[1]
            padded = F.pad(F.pad(x, (width, width, 0, 0), mode='circular'), (0, 0, 1, 1), mode='replicate')
            x = x+.5*F.gelu(layer(padded))
        return self.output(x).transpose(1, 2)

    def predict(self, initial, state, proposal, start, queries, steps=0):
        current = proposal
        if steps:
            for index in range(steps, 0, -1):
                level = torch.full_like(queries, index/steps)
                current = current-self(initial, state, proposal, current, start, queries, level)/steps
        else:
            current = current+self(initial, state, proposal, proposal, start, queries, torch.zeros_like(queries))
        flat = current.flatten(0, 1)
        a = signed_area(state)[:, None].expand_as(queries).reshape(-1)
        flat = flat*torch.sqrt(a/torch.maximum(signed_area(flat), .10*a))[:, None, None]
        return torch.where((queries == start[:, None])[:, :, None, None], state[:, None], flat.reshape_as(current))


@torch.no_grad()
def sample(bank, base, batch):
    t = bank.times
    train = bank.splits['train']
    row = train[torch.randint(len(train), (batch,), device=t.device)]
    last = torch.randint(1, len(t), (batch,), device=t.device)
    last[:3*batch//4] = torch.randint(len(t)-19, len(t), (3*batch//4,), device=t.device)
    origin = torch.where(torch.rand(batch, device=t.device) < .35, 0,
                         (last-torch.randint(2, 13, (batch,), device=t.device)).clamp_min(0))
    middle = closest(t, t[origin]+.48*(t[last]-t[origin]))
    before = previous_index(t, origin)
    warmup = choose(t, origin, middle)
    from_origin, from_middle = choose(t, origin, last), choose(t, middle, last)
    use_prediction = torch.rand(batch, device=t.device) < .5
    col = torch.where(use_prediction[:, None], from_middle, from_origin)
    clean = augment(bank.curves[row, 0], bank.curves[row, origin], bank.curves[row, before],
                    *bank.curves[row[:, None], col].unbind(1))
    initial, state, history = clean[:3]
    first = base(initial, state, history, t[before], t[origin], t[warmup])
    state = torch.where(use_prediction[:, None, None], first[:, -1], state)
    history = torch.where(use_prediction[:, None, None], first[:, -2], history)
    start = torch.where(use_prediction, middle, origin)
    before = torch.where(use_prediction, warmup[:, -2], before)
    proposal = base(initial, state, history, t[before], t[start], t[col])
    return initial, state, proposal, t[start], t[col], torch.stack(clean[3:], dim=1)


def fit(bank, base, frames, out, flow, steps):
    torch.manual_seed(20261106)
    began = perf_counter()
    model = BlockRefiner().cuda()
    random = torch.Generator(device='cuda').manual_seed(20261109)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.00005)
    validation = bank.splits['validation']
    target = bank.curves[validation[:, None], frames].cpu().numpy()
    best, history = float('inf'), []
    infer = lambda z, s, p, a, q: model.predict(z, s, p, a, q, steps=4 if flow else 0)
    for step in range(1, steps+1):
        initial, state, proposal, start, queries, expected = sample(bank, base, 8)
        level = torch.zeros_like(queries)
        if flow:
            independent = torch.rand(queries.shape, generator=random, device='cuda')
            mixed = torch.rand((len(queries), 1), generator=random, device='cuda') < .5
            level = torch.where(mixed, independent, independent[:, :1])
            frequency = torch.arange(proposal.shape[-1]//2+1, device='cuda')
            noise = torch.fft.rfft(torch.randn(proposal.shape, generator=random, device='cuda'))
            noise = torch.fft.irfft(noise*torch.exp(-(frequency/32)**4), n=proposal.shape[-1])
            source = proposal+.015*noise/noise.square().mean((2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            current = torch.lerp(expected, source, level[:, :, None, None])
            truth = source-expected
        else:
            current, truth = proposal, expected-proposal
        update = model(initial, state, proposal, current, start, queries, level)
        loss = supervised_loss(update, truth)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step in (1, steps):
            probe = np.arange(0, len(frames), 4)
            direct = forecast(base, bank, validation, frames, 'direct', refiner=infer)
            free = forecast(base, bank, validation, frames, 'free', refiner=infer)
            d = frame_errors(direct[:, probe], target[:, probe])
            r = frame_errors(free[:, probe], target[:, probe])
            score = float(.4*(d.mean()+d[:, -1].mean())/2+.6*(r.mean()+r[:, -1].mean())/2)
            history.append([step, float(loss.detach()), float(np.mean((direct-target)**2)),
                            float(np.mean((free-target)**2)), score])
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(), step=step, flow=flow), out/'model.pt')
            print(out.name, step, 'score', score, 'seconds', perf_counter()-began, flush=True)
    np.save(out/'history.npy', np.asarray(history))
    checkpoint = torch.load(out/'model.pt', weights_only=True, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    t = bank.times[frames].cpu().numpy()
    result = dict(name=out.name, steps=steps, best_step=checkpoint['step'], validation_score=best,
                  inference_iterations=4 if flow else 1)
    for split in ('validation', 'test'):
        rows = bank.splits[split]
        target = bank.curves[rows[:, None], frames].cpu().numpy()
        ids = [bank.ids[i] for i in rows.tolist()]
        events = [split_events(z, t) for z in complex_curves(target)]
        modes = ['direct', 'free', 'teacher']
        result[split] = {}
        export = dict(times=t, ids=np.array(ids), target=target)
        for mode in modes:
            output = forecast(base, bank, rows, frames, mode, refiner=infer)
            result[split][mode] = summarize(output, target, t, ids, events)
            export[mode] = output
        if flow:
            one_step = lambda z, s, p, a, q: model.predict(z, s, p, a, q, steps=1)
            one = forecast(base, bank, rows, frames, 'free', refiner=one_step)
            result[split]['free_one_iteration'] = summarize(one, target, t, ids, events)
        if split == 'test':
            np.savez_compressed(out/'predictions.npz', **export)
    atomic_json(out/'metrics.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/rollout_t100'))
    parser.add_argument('--steps', type=int, default=1200)
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = Trajectories(args.data, 1024, 'cuda', 100.)
    frames = evaluation_grid(bank.times)
    checkpoint = torch.load(args.out/'history/model.pt', weights_only=True, map_location='cuda')
    base = WindowFNO(**checkpoint['config']).cuda().eval()
    base.load_state_dict(checkpoint['model'])
    base.requires_grad_(False)
    for name in ('block_repair', 'block_flow'):
        out = args.out/name
        out.mkdir(exist_ok=True)
        fit(bank, base, frames, out, name == 'block_flow', args.steps)
    atomic_json(args.out/'refinement.json', {name: json.loads((args.out/name/'metrics.json').read_text())
                                             for name in ('block_repair', 'block_flow')})


if __name__ == '__main__':
    main()
