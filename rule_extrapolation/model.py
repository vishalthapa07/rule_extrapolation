import math

import torch
from torch import nn as nn
import torch.nn.functional as F


from rule_extrapolation.data import PAD_token


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]  # type: ignore[index]
        )


def get_tgt_mask(size, device) -> torch.Tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.float))
    mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask


def create_pad_mask(
    matrix: torch.Tensor, pad_token: int = PAD_token.item()
) -> torch.Tensor:
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    return torch.as_tensor(matrix == pad_token, device=matrix.device)


class TransformerDecoder(nn.Module):
    """
    Decoder-only Transformer architecture for autoregressive language modeling.
    Uses TransformerEncoder layers with causal masking to implement decoder-only behavior.
    
    Paper Section 3.4 Implementation Details:
    - Implements pre-normalization architecture (norm_first=True)
    - Uses GELU activation (Section 6: "1.1-1.2× faster convergence")
    - Uses sinusoidal positional encodings
    - Xavier uniform initialization for stable training
    
    Paper Section 3.4: "The Transformer used four self-attention layers with eight 
    heads and an embedding dimension of 256"
    """
    # Constructor
    def __init__(
        self,
        num_tokens: int = 6,
        dim_model: int = 256,  # Paper: embedding dimension 256
        num_heads: int = 8,    # Paper: 8 heads
        num_decoder_layers: int = 4,  # Paper: 4 self-attention layers
        dropout_p: float = 0.1,  # Paper: dropout 0.1-0.2
        dim_feedforward: int = 256,
        layer_norm_eps: float = 2e-4,
        relu_rescale: float = 1.0,
    ):
        super().__init__()

        self.dim_model = dim_model
        if relu_rescale <= 0:
            raise ValueError("relu_rescale must be positive")
        self.relu_rescale = nn.Parameter(
            torch.tensor(relu_rescale), requires_grad=False
        )

        # LAYERS
        # Paper Section 3.4: "The positional encodings were sinusoidal"
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)

        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dropout=dropout_p,
            dim_feedforward=dim_feedforward,
            layer_norm_eps=layer_norm_eps,
            activation='gelu',  # Paper Section 6: GELU for "smoother gradients, 1.1-1.2× faster convergence"
            norm_first=True,  # Paper Section 6: Pre-normalization "1.2-1.5× faster convergence"
        )

        self.decoder = nn.TransformerEncoder(layer, num_decoder_layers)

        self.out = nn.Linear(dim_model, num_tokens)
        
        # Paper Section 3.4: "weight initialization was done after Xavier uniform 
        # initialization which guarantees convergence between architectures"
        self._init_weights()
            
        # Ensure all submodules are on the correct device
        if torch.backends.mps.is_available():
            self.to(torch.device("mps"))
    
    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization as per paper Section 3.4.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.decoder(
            src=src,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        out = self.out(transformer_out)
        return out.permute(1, 2, 0)
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self


class LinearLLM(nn.Module):
    def __init__(
        self,
        max_data_length: int = 256,
        num_tokens=6,
        embedding_dim: int = 32,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.max_data_length = max_data_length
        self.num_tokens = num_tokens
        self.device = device
        self.embedding = nn.Embedding(num_tokens, embedding_dim)

        # Weight matrix; +1 because the input has a SOS token at the beginning
        self.weight = torch.nn.Parameter(
            torch.empty(
                (max_data_length + 1, embedding_dim, max_data_length + 1, num_tokens),
                **factory_kwargs
            )
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((max_data_length + 1, num_tokens), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.mask = torch.tril(
            torch.ones(
                max_data_length + 1,
                max_data_length + 1,
                device=device,
                dtype=torch.float,
            )
        )
        self.mask.to(device)

    def reset_parameters(self):
        # Initialize parameters as desired
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, src):
        src = self.embedding(src)
        if src.shape[1] != (self.max_data_length + 1):
            zeros_tensor = torch.zeros(
                src.shape[0],
                self.max_data_length + 1 - src.shape[1],
                src.shape[2],
                device=self.device,
            )
            src = torch.cat((src, zeros_tensor), dim=1)

        out = torch.einsum("bsw,swtv,st->btv", src.float(), self.weight, self.mask)
        if self.bias != None:
            out = out + self.bias[None, :, :]
        return out.permute(0, 2, 1)


class LSTM_LLM(nn.Module):
    """
    LSTM-based Language Model.
    
    Paper Section 3.3 (LSTM Model): "The Long Short-Term Memory (LSTM) network 
    introduces gating mechanisms—input, forget, and output gates—that enable it 
    to store, update, and selectively retain information over long sequences."
    
    Paper Section 3.4: 
    - "RNN and LSTM models shared equivalent hidden sizes for fair comparison"
    - "gradient clipping with a threshold of 1.0 to avoid exploding gradients"
    - Xavier uniform initialization
    """
    # Constructor
    def __init__(
        self,
        num_tokens: int = 6,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_lstm: float = 0.4,
        device=None,
    ):
        super(LSTM_LLM, self).__init__()

        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_lstm if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_lstm)
        self.fc = nn.Linear(hidden_dim, num_tokens)
        
        # Paper Section 3.4: Xavier uniform initialization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization as per paper Section 3.4.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, src):
        src = src.to(self.embedding.weight.device)
        embedded = self.embedding(src)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)

        return out.permute(0, 2, 1)
