import copy
import time

import torch
import torch.nn.functional as F
import math
import torch.nn as nn

import numpy as np


def relative_mul(q, relative):
    """relative position dot product"""
    node_len, dim_per_head = relative.size(2), relative.size(3)
    relative_k = relative.transpose(2, 3).view(-1, dim_per_head, node_len)
    q = q.view(-1, dim_per_head).unsqueeze(1)
    return torch.bmm(q, relative_k).squeeze(1).view(-1, node_len, node_len)


def relative_attn(query, key, value, mask=None, relative_q=None, relative_v=None, dropout=None):
    batch_size, num_heads, length, per_head = query.size()
    query = query.contiguous().view(batch_size * num_heads, length, per_head)
    key = key.contiguous().view(batch_size * num_heads, length, per_head)
    value = value.contiguous().view(batch_size * num_heads, length, per_head)
    mask = mask.repeat(1, num_heads, 1, 1)
    mask = mask.contiguous().view(batch_size * num_heads, length, -1)

    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(per_head)
    if relative_q is not None:
        scores += relative_mul(query, relative_q)
    if mask is not None:
        # 给需要mask的地方设置一个负无穷（因为接下来要输入到softmax层，如果是0还是会有影响）
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算softmax
    p_attn = F.softmax(scores, dim=-1)
    # 添加dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 和V做点积
    context = torch.bmm(p_attn, value)
    if relative_v is not None:
        node_len, dim_per_head = relative_v.size(2), relative_v.size(3)
        att_v = p_attn.view(-1, node_len).unsqueeze(1)
        relative_v = relative_v.view(-1, node_len, dim_per_head)
        context_v = torch.bmm(att_v, relative_v).squeeze(1)
        context_v = context_v.view(-1, node_len, dim_per_head)
        context += context_v
    context = context.contiguous().view(batch_size, num_heads, length, per_head)
    return context, p_attn


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.i = 0

    def forward(self, query, key, value, mask=None, relative_q=None, relative_v=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = attention(query, key, value, mask=mask,
        #                          dropout=self.dropout)
        if relative_q is not None:
            x, self.attn = relative_attn(query, key, value, mask=mask,
                                         relative_q=relative_q, relative_v=relative_v,
                                         dropout=self.dropout)
        else:
            x, self.attn = attention(query, key, value, mask=mask,
                                     dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # if relative_q is not None:
        #     self.attn = self.attn[:, :20, :20]
        #     self.attn = self.attn.contiguous().view(8, -1)
        #     np.savetxt('./visiualize/tree_attn_' + str(time.time()) + '.txt', self.attn.data.numpy())

        # if relative_q is None:
        #     if self.attn.size(-1) == 100 and self.attn.size(-2) == 8:
        #         self.attn = self.attn[:, :, :, :20].squeeze(0)
        #         print(self.attn.size())
        #         self.attn = self.attn.contiguous().view(8, 8*20)
        #         np.savetxt('./visiualize/nl_attn.txt', self.attn.data.numpy())


        return self.linears[-1](x)
