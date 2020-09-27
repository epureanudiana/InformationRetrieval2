import torch
import numpy as np

from torch import nn


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for i, d in enumerate(data_loader):
        optimizer.zero_grad()

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)

        losses.append(loss.item())
        print(i)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return correct_predictions.double() / n_examples, np.mean(losses)