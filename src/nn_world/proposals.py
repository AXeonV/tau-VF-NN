"""Cache frozen full-data and out-of-fold proposals for controlled repair experiments."""
import argparse
from pathlib import Path

import torch

from ..nn.model import CurveFNO
from ..trajectory_data import Trajectories, augment


@torch.no_grad()
def forecast(model, bank, rows):
    frames = len(bank.times)
    row = rows.repeat_interleave(frames)
    col = torch.arange(frames, device=row.device).repeat(len(rows))
    predictions = [model(bank.curves[r, 0], bank.times[c])
                   for r, c in zip(row.split(16), col.split(16))]
    return torch.cat(predictions).reshape(len(rows), frames, 2, -1)


def sample(bank, rows, batch):
    late = torch.nonzero(bank.times >= .30*bank.times[-1]).flatten()
    device = rows.device
    r = rows[torch.randint(len(rows), (batch,), device=device)]
    c = torch.randint(len(bank.times), (batch,), device=device)
    c[:batch//2] = late[torch.randint(len(late), (batch//2,), device=device)]
    return r, c


def shape_loss(residual):
    tangent = (torch.roll(residual, -1, dims=-1)-residual)*(residual.shape[-1]/(2*torch.pi))
    return residual.square().mean()+.005*tangent.square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--baseline', type=Path, default=Path('results/supervised/comparison/t20/nn/model.pt'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/world_t20'))
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.baseline, weights_only=True, map_location='cuda')
    config = checkpoint['config']
    if config['horizon'] != 20 or checkpoint['nodes'] != 1024:
        raise ValueError('Expected the T=20 baseline with 1024 interface nodes')
    bank = Trajectories(args.data, checkpoint['nodes'], 'cuda', 20.)
    train = bank.splits['train']
    if set(checkpoint['training_ids']) != {bank.ids[i] for i in train.tolist()}:
        raise ValueError('The baseline must have been trained on the original training split')
    full_model = CurveFNO(**config).cuda().eval()
    full_model.load_state_dict(checkpoint['model'])
    full = forecast(full_model, bank, torch.arange(len(bank.ids), device='cuda')).cpu()
    torch.save(dict(full=full, ids=bank.ids, times=bank.times.cpu()), args.out/'full.pt')
    # Interleave original IDs so both folds see all eight shape families.
    folds = [train[::2], train[1::2]]
    oof = torch.full_like(full, torch.nan)
    assignments = []
    for index, held in enumerate(folds):
        fit = folds[1-index]
        torch.manual_seed(20261106+index)
        model = CurveFNO(**config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=5e-5)
        for step in range(1, args.steps+1):
            row, col = sample(bank, fit, 8)
            initial, target = augment(bank.curves[row, 0], bank.curves[row, col])
            loss = shape_loss(model(initial, bank.times[col])-target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if step % 1000 == 0 or step == args.steps:
                print(f'fold {index+1} step {step} loss {float(loss.detach()):.6g}', flush=True)
        oof[held.cpu()] = forecast(model.eval(), bank, held).cpu()
        assignments.append(dict(fit=[bank.ids[i] for i in fit.tolist()],
                                predicted=[bank.ids[i] for i in held.tolist()]))
    torch.save(dict(oof=oof, folds=assignments, steps=args.steps), args.out/'oof.pt')


if __name__ == '__main__':
    main()
