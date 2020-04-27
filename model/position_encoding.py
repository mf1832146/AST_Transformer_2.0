import torch.nn as nn
import torch
from torch.autograd import Variable
import math


class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model, k, num_heads, dropout=0.0):
        """
        生成相对位置信息编码
        :param d_model: 词向量维度
        :param k: 相对位置窗口大小
        :param dropout:
        """
        super(RelativePositionEmbedding, self).__init__()

        self.d_model = d_model
        self.k = k

        self.parent_emb = nn.Embedding(2*k+2, d_model * 2, padding_idx=1)
        self.brother_emb = nn.Embedding(2*k+2, d_model * 2, padding_idx=1)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, relation_type):
        """
        :param inputs: 相对位置矩阵, 即traverse中的relative_parent_ids or
        relative_brother_ids shape [batch_size, max_size, max_size]
        :param relation_type: 'parent' means find relation between parent and child, 'brother' means find relation between brothers
        :return:
        """
        batch_size, max_size = inputs.size(0), inputs.size(1)
        inputs = inputs.unsqueeze(3)
        if relation_type == 'parent':
            position_emb = self.parent_emb(inputs)
        else:
            position_emb = self.brother_emb(inputs)
        position_emb = self.dropout(position_emb)
        position_emb = position_emb.view(batch_size, max_size, max_size, 2, self.d_model)
        k_emb, v_emb = [x.squeeze(3) for x in position_emb.split(1, dim=3)]

        k_emb = k_emb.repeat(1, 1, 1, self.num_heads)
        v_emb = v_emb.repeat(1, 1, 1, self.num_heads)

        k_emb = k_emb.view(-1, max_size, max_size, self.d_model) * math.sqrt(self.d_model)
        v_emb = v_emb.view(-1, max_size, max_size, self.d_model) * math.sqrt(self.d_model)

        return k_emb, v_emb


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

