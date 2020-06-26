import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.attn import MultiHeadedAttention
from model.position_encoding import PositionalEncoding, RelativePositionEmbedding
from utils import gelu, subsequent_mask


class EncoderDecoder(nn.Module):
    """
        A standard Encoder-Decoder architecture. Base for this and many
        other models.
    """
    def __init__(self, encoder, decoder, code_embed, nl_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.code_embed = code_embed
        self.nl_embed = nl_embed
        self.generator = generator

    def forward(self, inputs):
        if len(inputs) == 6:
            code, relative_par_ids, relative_bro_ids, nl, code_mask, nl_mask = inputs
        else:
            code, nl, code_mask,  nl_mask = inputs
            relative_par_ids = code_mask
            relative_bro_ids = None
        return self.decode(self.encode(code, relative_par_ids, relative_bro_ids),
                           code_mask, nl, nl_mask)

    def encode(self, code, relative_par_ids, relative_bro_ids):
        return self.encoder(self.code_embed(code), relative_par_ids, relative_bro_ids)

    def decode(self, memory, code_mask, nl, nl_mask):
        return self.decoder(self.nl_embed(nl), memory, code_mask, nl_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    def __init__(self, layer, N, relative_pos_emb):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.relative_pos_emb = relative_pos_emb

    def forward(self, code, relative_par_ids=None, relative_bro_ids=None):
        if relative_bro_ids is not None:
            par_k_emb, par_v_emb = self.relative_pos_emb(relative_par_ids, 'parent')
            bro_k_emb, bro_v_emb = self.relative_pos_emb(relative_bro_ids, 'brother')
            par_mask = relative_par_ids != 0
            bro_mask = relative_bro_ids != 0
        else:
            par_k_emb = par_v_emb = bro_k_emb = bro_v_emb = bro_mask = None
            par_mask = relative_par_ids

        for i, layer in enumerate(self.layers):
            code = layer(code, par_mask, bro_mask, par_k_emb, par_v_emb, bro_k_emb, bro_v_emb)
        return self.norm(code)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, code, par_mask, bro_mask=None, par_k_emb=None, par_v_emb=None, bro_k_emb=None, bro_v_emb=None):
        if par_mask is None:
            code_mask = par_mask
            code = self.sublayer[0](code, lambda x: self.self_attn(x, x, x, code_mask))
        else:
            code = self.sublayer[0](code, lambda x: self.self_attn(x, x, x, par_mask, par_k_emb, par_v_emb))
            code = self.sublayer[1](code, lambda x: self.self_attn(x, x, x, bro_mask, bro_k_emb, bro_v_emb))
        return self.sublayer[1](code, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Train(nn.Module):
    def __init__(self, model):
        super(Train, self).__init__()
        self.model = model

    def forward(self, inputs):
        out = self.model.forward(inputs)
        out = self.model.generator(out)
        return out


class GreedyEvaluate(nn.Module):
    def __init__(self, model,  max_nl_len, start_pos):
        super(GreedyEvaluate, self).__init__()
        self.model = model
        self.max_nl_len = max_nl_len
        self.start_pos = start_pos

    def forward(self, inputs):
        if len(inputs) == 6:
            code, relative_par_ids, relative_bro_ids, nl, code_mask, nl_mask = inputs
        else:
            code, nl, code_mask, nl_mask = inputs
            relative_par_ids = None
            relative_bro_ids = None

        batch_size = code.size(0)
        memory = self.model.encode(code, relative_par_ids, relative_bro_ids)
        ys = torch.ones(batch_size, 1).fill_(self.start_pos).type_as(code.data)
        for i in range(self.max_nl_len - 1):
            out = self.model.decode(memory,
                                    code_mask,
                                    Variable(ys),
                                    Variable(subsequent_mask(ys.size(1)).type_as(code.data)))

            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys,
                            next_word.unsqueeze(1).type_as(code.data)], dim=1)
        return ys


def make_model(code_vocab, nl_vocab, N=6,
               d_model=512, d_ff=2048, k=2, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 2 * N,
                RelativePositionEmbedding(d_model // h, k, h, dropout)),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        Embeddings(d_model, code_vocab),
        nn.Sequential(Embeddings(d_model, nl_vocab), c(position)),
        Generator(d_model, nl_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
