from audioop import bias
from turtle import forward
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU())

    def forward(self, inputs) -> torch.Tensor:
        inputs = inputs.permute(
            0, 2, 3, 1
        )  # (batch_size, img_size, img_size, img_encoder_size)

        batch_size = inputs.size(0)
        input_dim = inputs.size(-1)

        inputs_ = inputs.view(
            batch_size, -1, input_dim
        )  # (batch_size, img_size * img_size, img_encoder_size)
        x = self.linear(inputs_)
        return x


class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRUCell(embedding_dim, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, vocab_size)

        self.attn = Attention(hidden_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features, hidden) -> torch.Tensor:
        # Get batch size
        batch_size = features.size(0)

        # Get outputs from attention model
        context_vector, attention_weights = self.attn(features, hidden)

        x = self.embedding(x)


# Reference: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py


class Attention(nn.Module):
    def __init__(self, hidden_size, encoder_size=None, decoder_size=None):
        super().__init__()

        encoder_size = 2 * hidden_size if encoder_size is None else encoder_size
        decoder_size = 2 * hidden_size if decoder_size is None else decoder_size

        self.decoder_attn = nn.Linear(decoder_size, hidden_size, bias=False)
        self.encoder_attn = nn.Linear(encoder_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        enc_out = self.encoder_attn(encoder_out)
        dec_out = self.decoder_attn(decoder_hidden)