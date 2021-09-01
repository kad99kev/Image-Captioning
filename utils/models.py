import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    The CNN Encoder. Passes images features through a fully connected layer.

    Arguments:
        input_dim (int): Number of features from the image encoder.
        embedding_dim (int): Size of the embedding.
    """

    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU())

    def forward(self, inputs):
        inputs = inputs.permute(
            0, 2, 3, 1
        )  # (batch_size, img_size, img_size, img_encoder_size)

        batch_size = inputs.size(0)
        input_dim = inputs.size(-1)

        # inputs_: (batch_size, img_size * img_size, img_encoder_size)
        inputs_ = inputs.view(batch_size, -1, input_dim)

        # After linear layer
        # x: (batch_size, img_size * img_size, embedding_dim)
        x = self.linear(inputs_)
        return x


class RNNDecoder(nn.Module):
    """
    The RNN Decoder. Tries to predict the next word.

    Arguments:
        encoder_dim (int):  Size of the encoder.
        decoder_dim (int):  Size of the decoder.
        attention_dim (int):  Size of the attention layer.
        embedding_dim (int):  Size of the embedding.
        vocab_size (int): Number of words in the vocabulary.
    """

    def __init__(
        self, encoder_dim, decoder_dim, attention_dim, embedding_dim, vocab_size
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(encoder_dim + embedding_dim, decoder_dim, batch_first=True)
        self.fc_1 = nn.Linear(decoder_dim, decoder_dim)
        self.fc_2 = nn.Linear(decoder_dim, vocab_size)

        self.attn = Attention(encoder_dim, decoder_dim, attention_dim)

    def forward(self, decoder_input, encoder_output, hidden):

        # Get outputs from attention model
        # context_vector: (1, encoder_dim)
        # attn_weights: (1, 64, 1)
        context_vector, attention_weights = self.attn(encoder_output, hidden)

        # x: (batch_size, 1, embedding_dim)
        x = self.embedding(decoder_input)

        # x: (batch_size, 1, encoder_dim + embedding_dim)
        x = torch.cat([context_vector.unsqueeze(1), x], dim=-1)

        # output: (batch_size, 1, decoder_dim)
        # state: (batch_size, 1, decoder_dim)
        output, state = self.gru(x)

        # x: (1, 1, decoder_dim)
        x = self.fc_1(output)

        # x: (batch_size * seq_length, decoder_dim)
        x = x.view(-1, x.size(2))

        # x: (batch_size * seq_length, vocab_size)
        x = self.fc_2(x)

        return x, state.view(-1, self.decoder_dim), attention_weights

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.decoder_dim)


class Attention(nn.Module):
    """
    Implements Bahdanau Attention.

    Arguments:
        encoder_dim (int):  Size of the encoder.
        decoder_dim (int):  Size of the decoder.
        attention_dim (int):  Size of the attention layer.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.encoder_attn = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # The 64 here is img_size * img_size i.e. 8 * 8

        # hidden_: (batch_size, 1, encoder_dim)
        hidden_ = decoder_hidden.unsqueeze(1)

        # attn_hidden: (batch_size, 64, attention_dim)
        attn_hidden = torch.tanh(
            self.encoder_attn(encoder_out) + self.decoder_attn(hidden_)
        )

        # score: (1, 64, 1)
        score = self.value(attn_hidden)

        # attn_weights: (1, 64, 1)
        attn_weights = self.softmax(score)

        # context_vector after sum: (1, encoder_dim)
        context_vector = attn_weights * encoder_out
        context_vector = torch.sum(context_vector, 1)

        return context_vector, attn_weights
