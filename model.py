"""
Model of a Transformer from Scratch using PyTorch
"""

import math

import torch
from torch import nn


class Embedding(nn.Module):
    """Converts input tokens to vectors of dimension d_model"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward compuation transforming input x into word embedding

        Embedding multiplied by square root of d_model as in paper
        """
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Add positional encoding to input embeddings"""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Args:
        dropout: reduce overfitting by dropping out parameters of network
        seq_len: maximal lenght of input sequence
        d_model: size of vector for a single input token / word
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        # with parameters for positional encoding.
        pos_encoding = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len) that represents
        # the position of a word inside of a sentence.
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Denominator of formula for positional encoding.
        # Calculated in log space for numerical stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply the sin function to even positions of each vector.
        # sin(position * (10000 ** (2i / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply the cos function to odd positions of each vector.
        # cos(position * (10000 ** (2i / d_model))
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add new dimension to positional encoding for considering
        # batches of sequences. Shape will be (1, seq_len, d_model).
        pos_encoding = pos_encoding.unsqueeze(0)

        # Register positional encoding at buffer of the module
        # Postional Encoding will be saved in the files with
        # the state of the model
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to every word in sequence"""
        # The positonal encoding will not be learned.
        # Thus the gradient will not calculated/stored.
        # (batch_size, seq_len, d_model)
        x = x + self.pos_encoding[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Normalize features of a word and the amplify them or reduce them
    using mutiplicative parameter alpha and additive parameter beta.
    """

    def __init__(self, d_model: int, epsilon: float = 10**-6) -> None:
        """
        Args:
        d_model: number of features per word embedding
        epsilon: Guarantee numerical stability and avoid division by 0
        """
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input and perform linear transformation"""
        # Keep the dimesion for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)
        # epsilon is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Feed Forward Block in Encoder and Decoder of Transformer

    It performs the forward computation:
    FFN(x) = max(0, x * W1 + b1) * W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args:
        d_model: dimensionality of input and output
        d_ff: dimensionality of hidden representation
        dropout: avoid overfitting
        """
        super().__init__()
        # weights1 and bias1
        self.linear_layer_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # weights2 and bias2
        self.linear_layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward computation:
        FFN(x) = max(0, x * weights1 + bias1) * weights2 + bias2
        """
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear_layer_2(self.dropout(torch.relu(self.linear_layer_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    Map a query and a set of key-value pairs to an output using multiple heads
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        """
        Args:
        d_model: Dimensionality of embedding vector
        n_heads: Number of heads of Multi-Head Attention Block
        """
        super().__init__()
        self.d_model = d_model
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads

        # Dimensionality of vector analzed by each head
        self.d_k = d_model // n_heads

        # Parameters for Queries Linear Transformation Q' = Wq * Q
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # Parameters for Keys Linear Transformation K' = Wk * K
        self.w_k = nn.Linear(d_model, d_model, bias=False)

        # Parameters for Values Linear Transformation V' = Wv * V
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Linear Transformation for the output of the heads
        # (attention_scores) to word embeddings
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        d_k = query.shape[-1]
        # from the paper: Q * K.T / sqrt(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask values we do not want to get the attention from
        # aasigning them very low attention scores
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax following the formula in the paper
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch_size, n_heads, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Multiply softmax output with value matrix
        attention = attention_scores @ value

        return attention, attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:

        # (batch_size, seq_len, d_model) - > (batch_size, seq_len, d_model)
        query: torch.Tensor = self.w_q(q)

        # (batch_size, seq_len, d_model) - > (batch_size, seq_len, d_model)
        key: torch.Tensor = self.w_k(k)

        # (batch_size, seq_len, d_model) - > (batch_size, seq_len, d_model)
        value: torch.Tensor = self.w_v(v)

        # Each head will see the full sentence but just a subset of the features of each word
        # (batch_size, seq_len, d_model) - > (batch_size, n_heads, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Combine together the attention of all the heads
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # Multiply by Wo matrix: (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        output = self.w_o(x)

        return output


class ResidualConnection(nn.Module):
    """
    Add input of a layer to its output, creating a connection
    in the computational graph, which skips the layer when
    backpropagating loss. As a result, the gradient flows better.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        """
        Args:
        d_model: number of features represented per word embedding
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """res_connection(x) = x + sublayer(x)"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """
        Args:
        d_model: number of features per word embedding
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask) -> torch.Tensor:
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection_2(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        """
        Args:
        d_model: number of features per word embedding
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """
        Args:
        d_model: number of features per word embedding
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)
        self.residual_connection_3 = ResidualConnection(d_model, dropout)

    def forward(
        self, x: torch.Tensor, enc_output: torch.Tensor, src_mask, tgt_mask
    ) -> torch.Tensor:
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connection_2(
            x,
            lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask),
        )
        x = self.residual_connection_3(x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        """
        Args:
        d_model: number of features per word embedding
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(
        self, x: torch.Tensor, enc_output: torch.Tensor, src_mask, tgt_mask
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """Convert output of the decoder back to vocabulary"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)"""
        # Log Softmax for numerical stability instead of Softmax
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: Embedding,
        tgt_embedding: Embedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask) -> torch.Tensor:
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    n_blocks: int = 6,
    n_heads: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Create embedding layers
    src_embedding = Embedding(d_model, src_vocab_size)
    tgt_embedding = Embedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers (one is sufficient)
    src_pos_encoder = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_encoder = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = [
        EncoderBlock(
            d_model=d_model,
            self_attention_block=MultiHeadAttentionBlock(d_model, n_heads, dropout),
            feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout,
        )
        for _ in range(n_blocks)
    ]

    # Create decoder blocks
    decoder_blocks = [
        DecoderBlock(
            d_model=d_model,
            self_attention_block=MultiHeadAttentionBlock(d_model, n_heads, dropout),
            cross_attention_block=MultiHeadAttentionBlock(d_model, n_heads, dropout),
            feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout,
        )
        for _ in range(n_blocks)
    ]

    # Create the encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Create the decoder
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embedding,
        tgt_embedding,
        src_pos_encoder,
        tgt_pos_encoder,
        projection_layer,
    )

    # Initialize the parameters
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer
