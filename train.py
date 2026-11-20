import torch


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_abs += (pred - y).abs().sum().item()
            n += len(y)

    return total_loss / n, total_abs / n