from json import encoder
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

def _custom_loss(real, pred, loss_fn):
    mask = torch.logical_not(torch.eq(real, torch.zeros_like(real)))
    loss_ = loss_fn(pred, real)

    mask = mask.type(loss_.dtype)
    loss_ *= mask

    return torch.mean(loss_)

class Trainer:
    def __init__(self, encoder, decoder, optimizer, loss_fn, vocab, device=torch.device("cpu")) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer= optimizer
        self.loss_fn = loss_fn
        self.vocab = vocab

        self.device = device

    def train_step(self, img_feats, captions):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        loss = 0
        batch_size, seq_len = captions.size()

        hidden = self.decoder.init_hidden(batch_size)

        dec_input = torch.unsqueeze(torch.tensor(self.vocab(["<start>"]) * batch_size), 1).to(self.device)

        enc_feats = self.encoder(img_feats)

        for i in range(1, seq_len):
            preds, hidden, _ = self.decoder(dec_input, enc_feats, hidden)

            loss += _custom_loss(captions[:, i], preds, self.loss_fn)

            dec_input = torch.unsqueeze(captions[:, i], 1)

        loss.backward()
        loss = loss.item()
        self.optimizer.step()

        total_loss = loss / seq_len
        
        return loss, total_loss

    def train(self, dataloader, epochs):
        losses = []

        pbar = tqdm()

        for epoch in range(1, epochs):
            print("*" * 10 + f" Epoch {epoch}/{epochs} " + "*" * 10)
            total_loss = 0

            pbar.reset(total=len(dataloader))
            for i, (img_feats, captions) in enumerate(dataloader):
                img_feats = img_feats.to(self.device)
                captions = captions.to(self.device)

                batch_loss, t_loss = self.train_step(
                    img_feats, captions
                )
                total_loss += t_loss

                if i % 100 == 0:
                    average_batch_loss = batch_loss / captions.size(1)
                    print(f"Average Batch {i} Loss: {average_batch_loss}")

                pbar.update()
            
            losses.append(total_loss / len(dataloader))
        
        print(f"Total Loss: {(total_loss / len(dataloader)):.6f}")
        pbar.refresh()