"""Shared time selection, curve losses, and pure neural rollout."""
import torch

from .trajectory_data import augment


def closest(times, query):
    right = torch.searchsorted(times, query.contiguous()).clamp_max(len(times)-1)
    left = (right-1).clamp_min(0)
    return torch.where((times[right]-query).abs() < (times[left]-query).abs(), right, left)


def choose(times, first, last, count=4):
    fraction = torch.arange(1, count+1, device=times.device)/count
    query = times[first, None]+fraction*(times[last, None]-times[first, None])
    return closest(times, query)


def evaluation_grid(times):
    if float(times[-1]) != 100.:
        raise ValueError('This comparison uses T=100')
    queries = [.25, 1., 2.4, 5., 8.35, 10., 15., 20., *range(25, 101, 5)]
    return torch.cat([torch.zeros(1, dtype=torch.long, device=times.device),
                      closest(times, times.new_tensor(queries))])


def previous_index(times, first):
    return closest(times, times[first]-.08*(1+times[first]))


def supervised_loss(prediction, target):
    residual = prediction-target
    tangent = (torch.roll(residual, -4, -1)-residual)*(residual.shape[-1]/(8*torch.pi))
    loss = residual.square().mean()+.004*tangent.square().mean()
    return loss


def single_loss(model, bank, grid, step, batch=8):
    """Use contiguous, strictly increasing query sequences on either saved time grid."""
    times = bank.times
    device = times.device
    warmup = 400
    length = 1 if step <= warmup else 2 if step <= 650 else 4 if step <= 950 else 8
    # Both variants see the same time-grid mixture. The full grid includes near-coincident early labels.
    full_grid = step % 4 == 0
    frames = torch.arange(len(times), device=device) if full_grid else grid
    rows = bank.splits['train']
    row = rows[torch.randint(len(rows), (batch,), device=device)]
    start = torch.randint(len(frames)-length, (batch,), device=device)
    late = min(int(torch.searchsorted(times[frames], times.new_tensor(10.))), len(frames)-length-1)
    start[:batch//2] = torch.randint(late, len(frames)-length, (batch//2,), device=device)
    start[-2:] = torch.randint(min(3, len(frames)-length), (2,), device=device)
    sequence = frames[start[:, None]+torch.arange(length+1, device=device)]
    start_index = sequence[:, 0]
    # Match the recent history at deployment, which comes from the preceding output frame.
    before = frames[(start-1).clamp_min(0)]
    initial, state, previous, *labels = augment(
        bank.curves[row, 0], bank.curves[row, start_index], bank.curves[row, before],
        *bank.curves[row[:, None], sequence[:, 1:]].unbind(1))
    first, history, prior = state, previous, before
    loss = []
    for i, label in enumerate(labels):
        current, future = sequence[:, i], sequence[:, i+1:i+2]
        tracked = i >= length-2
        with torch.set_grad_enabled(tracked):
            prediction = model(initial, state, previous, times[before], times[current], times[future])
            if tracked:
                loss.append(supervised_loss(prediction, label[:, None]))
        previous, state, before = state, prediction[:, 0], current
    if length > 1:
        clean = model(initial, first, history, times[prior], times[start_index], times[sequence[:, 1:2]])
        loss.append(supervised_loss(clean, labels[0][:, None]))
    return torch.stack(loss).mean()


def block_loss(model, bank, grid, step, batch=8):
    times = bank.times
    device = times.device
    rows = bank.splits['train']
    row = rows[torch.randint(len(rows), (batch,), device=device)]
    last = torch.randint(1, len(times), (batch,), device=device)
    late = int(torch.searchsorted(times, times.new_tensor(10.)))
    last[:3*batch//4] = torch.randint(late, len(times), (3*batch//4,), device=device)
    start = (last-torch.randint(1, 12, (batch,), device=device)).clamp_min(0)
    wide = closest(times, times[last]*torch.rand(batch, device=device))
    coin = torch.rand(batch, device=device)
    start = torch.where(coin < .45, wide, start)
    start = torch.where(coin > .75, 0, start)
    before = previous_index(times, start)
    clean_queries = choose(times, start, last)
    clean = augment(bank.curves[row, 0], bank.curves[row, start], bank.curves[row, before],
                    *bank.curves[row[:, None], clean_queries].unbind(1))
    clean_prediction = model(*clean[:3], times[before], times[start], times[clean_queries])
    losses = [supervised_loss(clean_prediction, torch.stack(clean[3:], dim=1))]
    if step <= 400:
        return losses[0]
    endpoints = torch.cat([start[:, None], choose(times, start, last, 3)], dim=1)
    windows = [choose(times, a, b) for a, b in zip(endpoints.T, endpoints.T[1:])]
    targets = [bank.curves[row[:, None], query] for query in windows]
    initial, state, previous, *labels = augment(
        bank.curves[row, 0], bank.curves[row, start], bank.curves[row, before],
        *(x for block in targets for x in block.unbind(1)))
    for index, query in enumerate(windows):
        with torch.set_grad_enabled(index > 0):
            prediction = model(initial, state, previous, times[before], times[endpoints[:, index]],
                               times[query])
            if index:
                losses.append(supervised_loss(prediction, torch.stack(labels[4*index:4*index+4], dim=1)))
        previous, state, before = prediction[:, -2], prediction[:, -1], query[:, -2]
    return torch.stack(losses).mean()


@torch.no_grad()
def forecast(model, bank, rows, frames, *, previous_frame=None):
    times = bank.times
    initial = bank.curves[rows, 0]
    before = frames[0] if previous_frame is None else previous_frame
    state = bank.curves[rows, frames[0]]
    previous = bank.curves[rows, before]
    output = [state[:, None]]
    width = model.window_size
    for start, query in zip(frames[::width], frames[1:].split(width)):
        predicted = model(initial, state, previous, times[before].expand(len(rows)),
                          times[start].expand(len(rows)), times[query].expand(len(rows), -1))
        output.append(predicted)
        previous = predicted[:, -2] if len(query) >= 2 else state
        before = query[-2] if len(query) >= 2 else start
        state = predicted[:, -1]
    return torch.cat(output, dim=1)
