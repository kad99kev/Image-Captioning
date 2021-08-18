import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm


def custom_loss(real, pred, loss_fn):
    mask = torch.logical_not(torch.eq(real, torch.zeros_like(real)))
    loss_ = loss_fn(pred, real)

    mask = mask.type(loss_.dtype)
    loss_ *= mask

    return torch.mean(loss_)


def train_step(encoder, decoder, optimizer, loss_fn, vocab, img_feats, captions):
    encoder.zero_grad()
    decoder.zero_grad()

    loss = 0
    batch_size, seq_len = captions.size()

    hidden = decoder.init_hidden(batch_size)

    dec_input = torch.unsqueeze(torch.tensor(vocab(["<start>"]) * batch_size), 1)

    enc_feats = encoder(img_feats)

    for i in range(1, seq_len):
        preds, hidden, _ = decoder(dec_input, enc_feats, hidden)

        loss += custom_loss(captions[:, i], preds, loss_fn)

        dec_input = torch.unsqueeze(captions[:, i], 1)

    loss.backward()
    loss = loss.item()
    optimizer.step()

    total_loss = loss / seq_len

    return loss, total_loss


def train(dataloader, encoder, decoder, vocab):
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    losses = []

    pbar = tqdm()

    for epoch in range(1, 2):
        print("*" * 10 + f" Epoch {epoch}/{2} " + "*" * 10)
        total_loss = 0

        pbar.reset(total=len(dataloader))
        for i, (img_feats, caps) in enumerate(dataloader):
            batch_loss, t_loss = train_step(
                encoder, decoder, optimizer, ce_loss, vocab, img_feats, caps
            )
            total_loss += t_loss

            if i % 100 == 0:
                average_batch_loss = batch_loss / caps.size(1)
                print(f"Average Batch {i} Loss: {average_batch_loss}")

            pbar.update()

        losses.append(total_loss / len(dataloader))

    print(f"Total Loss: {(total_loss / len(dataloader)):.6f}")
    pbar.refresh()