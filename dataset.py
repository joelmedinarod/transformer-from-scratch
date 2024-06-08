import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


def causal_mask(size: int) -> torch.Tensor:
    """Avoid attending on previous tokens in sequence"""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):

    def __init__(
        self,
        dataset,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Convert special tokens to number
        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        """Get item from dataset"""
        # Exract pair from original dataset
        src_tgt_pair = self.dataset[index]
        # Extract text in source language
        src_text = src_tgt_pair["translation"][self.src_lang]
        # Extract text in target language
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        # Convert each tag into tokens and then into input IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Fill sentences with padding tokens to reach sequence lenght
        n_enc_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # consider SOS and EOS
        n_dec_padding_tokens = (
            self.seq_len - len(dec_input_tokens) - 1
        )  # consider either EOS for decoder_input or SOS for labels

        if n_enc_padding_tokens < 0 or n_dec_padding_tokens < 0:
            raise ValueError("Sequence is too long")

        # Append special tokens to encoder input: SOS, EOS and PADs
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * n_enc_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )  # shape (seq_len)

        # Append special tokens to decoder input: SOS, EOS and PADs
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * n_dec_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )  # shape (seq_len)

        # Append special tokens to label: EOS and PADs
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * n_dec_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )  # shape (seq_len)

        # for debugging: check that the maximal sequence lenght is reached by all tensors
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Build mask for the encoder: PADs should not be considered
        # by the self-attention mechanism
        # shape (1, 1, seq_len)
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        # Build mask for the decoder: PADs should not be considered
        # by the self-attention mechanism, and neither future tokens
        # (1, seq_len) & (1, seq_len, seq_len) -> (1, seq_len, seq_len)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(
            0
        ).int() & causal_mask(decoder_input.size(0))

        return {
            "enc_input": encoder_input,
            "dec_input": decoder_input,
            "enc_mask": encoder_mask,
            "dec_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
