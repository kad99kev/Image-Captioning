import numpy as np
import wandb

import torch
from torch.distributions.categorical import Categorical

from .helpers import plot_attention

from tqdm.auto import tqdm


def _custom_loss(real, pred, loss_fn):
    mask = torch.logical_not(torch.eq(real, torch.zeros_like(real)))
    loss_ = loss_fn(pred, real)

    mask = mask.type(loss_.dtype)
    loss_ *= mask

    return torch.mean(loss_)


class Trainer:
    def __init__(
        self,
        encoder,
        decoder,
        optimizer,
        loss_fn,
        vocab,
        max_len,
        embed_dim,
        device=torch.device("cpu"),
        proj_wandb=None,
        sample_image=None,
        sample_features=None,
    ) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.vocab = vocab
        self.max_len = max_len
        self.embed_dim = embed_dim

        if proj_wandb:
            assert (
                sample_image is not None
            ), "You must provide a sample image for evaluation while logging with Weights and Biases!"
            self.sample_image = sample_image

            assert (
                sample_features is not None
            ), "You must provide a sample features for evaluation while logging with Weights and Biases!"
            sample_features = torch.from_numpy(sample_features)
            self.sample_features = torch.unsqueeze(sample_features, 0).to(device)

            self.wandb = wandb.init(project=proj_wandb)

        self.device = device

    def eval_step(self):
        self.encoder.eval()
        self.decoder.eval()

        attention_plot = np.zeros((self.max_len, self.embed_dim))

        hidden = self.decoder.init_hidden(1).to(self.device)

        with torch.no_grad():
            enc_feats = self.encoder(self.sample_features)

            dec_input = torch.unsqueeze(torch.tensor(self.vocab(["<start>"])), 1).to(
                self.device
            )
            result = []

            for i in range(1, self.max_len):
                preds, hidden, attn_weights = self.decoder(dec_input, enc_feats, hidden)

                attention_plot[i] = (
                    attn_weights.view(
                        -1,
                    )
                    .cpu()
                    .numpy()
                )

                cat = Categorical(logits=preds)
                pred_id = cat.sample()
                result.append(self.vocab.lookup_token(pred_id))

                if self.vocab.lookup_token(pred_id) == "<eos>":
                    return result, attention_plot

                dec_input = torch.unsqueeze(pred_id, 1)

            attention_plot = attention_plot[: len(result), :]

        self.encoder.train()
        self.decoder.train()
        return result, attention_plot

    def train_step(self, img_feats, captions):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        loss = 0
        batch_size, seq_len = captions.size()

        hidden = self.decoder.init_hidden(batch_size).to(self.device)

        dec_input = torch.unsqueeze(
            torch.tensor(self.vocab(["<start>"]) * batch_size), 1
        ).to(self.device)

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

        for epoch in range(1, epochs + 1):
            print("*" * 10 + f" Epoch {epoch}/{epochs} " + "*" * 10)
            total_loss = 0

            pbar.reset(total=len(dataloader))
            for i, (img_feats, captions) in enumerate(dataloader):
                img_feats = img_feats.to(self.device)
                captions = captions.to(self.device)

                batch_loss, t_loss = self.train_step(img_feats, captions)
                total_loss += t_loss

                if self.wandb:
                    average_batch_loss = batch_loss / captions.size(1)
                    self.wandb.log({"average_batch_loss": average_batch_loss})

                pbar.update()

            epoch_loss = total_loss / len(dataloader)
            if self.wandb:
                result, attention_plot = self.eval_step()
                pred_sent = " ".join(result)
                fig_ = plot_attention(
                    self.sample_image, result, attention_plot, wandb=True
                )
                self.wandb.log(
                    {
                        "epoch_loss": epoch_loss,
                        "attention": wandb.Image(fig_, caption=pred_sent),
                        "epoch": epoch,
                    }
                )

            losses.append(epoch_loss)
            print(f"Total Loss: {epoch_loss:.6f}")

        pbar.refresh()