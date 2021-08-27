import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def show_image(img: np.ndarray, cap=None) -> None:
    """
    Displays an image. (To be used in a notebook environment).

    Parameters:
        img (np.ndarray): Image to be displayed.
        cap: Caption of the image.
    """
    if cap:
        plt.title(cap)
    plt.imshow(img)


def convert_numpy(img: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch image tensor into a `np.ndarray`.

    Parameters:
        img (torch.Tensor): The image to be converted into numpy.

    Returns:
        The image in plt form.
    """
    disp_img = np.squeeze(img.numpy())
    disp_img = (disp_img - np.min(disp_img)) / (np.max(disp_img) - np.min(disp_img))
    disp_img = np.transpose(disp_img, (1, 2, 0))
    return disp_img


def get_tokenizer_vocab(df: pd.DataFrame) -> torchtext.vocab.Vocab:
    """
    Returns a Vocabulary of the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the captions.

    Returns: A Vocab object.

    """

    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    def _build_vocab(df):
        text = df["caption"].str.cat(sep=" ")
        yield tokenizer(text)

    special_tokens = ["<unk>", "<start>", "<eos>", "<pad>"]
    vocab = build_vocab_from_iterator(_build_vocab(df), specials=special_tokens)
    vocab.set_default_index(vocab["<unk>"])

    return tokenizer, vocab


# Reference: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3


def collate_fn_pad(batch):
    """
    Padds batch of variable length
    """

    ## Separate image features with caption indices
    features = [torch.tensor(item[0]) for item in batch]
    idx_captions = [torch.tensor(item[1]) for item in batch]

    ## Convert features from numpy to torch.tensor
    feat_tensor = torch.stack(features, dim=0)

    ## Pad Captions
    padded_captions = torch.nn.utils.rnn.pad_sequence(idx_captions, batch_first=True)

    return feat_tensor, padded_captions


def plot_attention(image, result, attention_plot):
    img_array = np.asarray(image)
    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)

    for i in range(len_result):
        temp_attn = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result / 2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.set_title(result[i])
        img = ax.imshow(img_array)
        ax.imshow(temp_attn, cmap="gray", alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()