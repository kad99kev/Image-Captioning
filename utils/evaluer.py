import numpy as np

import torch
from torch.distributions.categorical import Categorical

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class Evaluer:
    """
    Implements all the necessary evalution functions.

    Arguments:
        encoder (CNNEncoder): The CNN Encoder.
        decoder (RNNDecoder): The RNN Decoder.
        vocab (torchtext.vocab.Vocab): PyTorch Vocab object for the dataset.
        max_len (int): Maximum length of sentence seen in the dataset.
        embed_dim (int): Number of embedding features.
    """

    def __init__(self, encoder, decoder, vocab, max_len, embed_dim):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

        self.vocab = vocab
        self.max_len = max_len
        self.embed_dim = embed_dim

        inception = timm.create_model(
            "inception_v4", pretrained=True, num_classes=0, global_pool=""
        )
        self.inception = inception.eval()

        config = resolve_data_config({}, model=self.inception)
        self.transform = create_transform(**config)

    def evaluate(self, image):
        """
        Performs evaluation.

        Arguments:
            image (PIL Image): Input image.

        Returns:
            Caption and Attention Weights.
        """
        attention_plot = np.zeros((self.max_len, self.embed_dim))

        hidden = self.decoder.init_hidden(1)

        img_ten = torch.unsqueeze(self.transform(image), 0)

        with torch.no_grad():
            features = self.inception(img_ten)

            enc_feats = self.encoder(features)

            dec_input = torch.unsqueeze(torch.tensor(self.vocab(["<start>"])), 1)
            result = []

            for i in range(1, self.max_len):
                preds, hidden, attn_weights = self.decoder(dec_input, enc_feats, hidden)

                attention_plot[i] = attn_weights.view(
                    -1,
                ).numpy()

                cat = Categorical(logits=preds)
                pred_id = cat.sample()
                result.append(self.vocab.lookup_token(pred_id))

                if self.vocab.lookup_token(pred_id) == "<eos>":
                    return result, attention_plot

                dec_input = torch.unsqueeze(pred_id, 1)

            attention_plot = attention_plot[: len(result), :]

        return result, attention_plot
