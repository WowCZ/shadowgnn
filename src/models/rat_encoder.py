# -*- coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.utils

class RATEncoder(nn.Module):
    def __init__(self, args):
        super(RATEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(args.rat_hidden_size, args.rat_head_num, args.rat_ff_dim, args.rat_relation_types,
                          args.dropout) for _ in range(args.rat_layers)])
        '''
        for i in range(len(self.layers)):
            self.layers[i].attention.key_relation_embed = self.layers[0].attention.key_relation_embed
            self.layers[i].attention.value_relation_embed = self.layers[0].attention.value_relation_embed
        '''
    def forward(self, x, rel_matrices, padding_masks):
        enc_repr = x
        for layer in self.layers:
            enc_repr = layer(enc_repr, rel_matrices, padding_masks)
        return enc_repr


class EncoderLayer(nn.Module):
    def __init__(self, enc_hidden_size, head_num, enc_ff_dim, relation_types, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = RelationAwareAttention(enc_hidden_size, head_num, enc_hidden_size, enc_hidden_size,
                                                enc_hidden_size, dropout,
                                                relation_types)

        self.positionwise_FF = nn.Sequential(nn.Linear(enc_hidden_size, enc_ff_dim), nn.ReLU(),
                                             nn.Linear(enc_ff_dim, enc_hidden_size))

        self.SublayerConnect = nn.ModuleList([RisidualConnection(enc_hidden_size, dropout) for i in range(2)])

    def forward(self, x, rel_matrices, padding_masks):
        '''
        :param x: batch * sent_len * hsize
        :param rel_matrices: batch * sent_len * sent_len
        :param padding_masks: batch * sent_len (1:masked)
        :return:
        '''
        attn_out, _ = self.attention(x, x, x, rel_matrices, padding_masks)

        attn_out = self.SublayerConnect[0](x, attn_out)

        ff_out = self.positionwise_FF(attn_out)
        ff_out = self.SublayerConnect[1](attn_out, ff_out)
        if ff_out.device.type == 'cuda':
            torch.cuda.empty_cache()
        return ff_out


class RisidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(RisidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(y))


class RelationAwareAttention(nn.Module):
    def __init__(self, hidden_size, heads_num, q_size, k_size, v_size,dropout, relation_types=0):
        super(RelationAwareAttention, self).__init__()

        assert hidden_size % heads_num == 0
        self.hidden_size = hidden_size // heads_num
        self.heads_num = heads_num
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(q_size, self.hidden_size * self.heads_num,bias=False)
        self.key = nn.Linear(k_size, self.hidden_size * self.heads_num,bias=False)
        self.value = nn.Linear(v_size, self.hidden_size * self.heads_num,bias=False)
        self.dense = nn.Linear(self.hidden_size * self.heads_num, self.hidden_size * self.heads_num)

        self.relation_embed_used = (relation_types > 0)
        if self.relation_embed_used:
            self.key_relation_embed = nn.Embedding(relation_types + 1, self.hidden_size, padding_idx=0)
            self.value_relation_embed =nn.Embedding(relation_types + 1, self.hidden_size, padding_idx=0)

    def forward(self, query, key, value, relation_matrices=None, key_padding_mask=None):
        '''
        :param query: batch * query_len * embed_size
        :param key: batch * key_len * embed_size
        :param value: batch * key_len * embed_size
        :param relation_matrices: batch * query_len * key_len
        :param key_padding_mask: batch * key_len  1 masked, 0 reserved
        :return: context_layer: batch * query_len * hidden_size
        :return: attention_probs: batch * query_len * key_len
        '''
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        # batch * num_heads * query * 1 * hidden
        query_expand = mixed_query_layer.view(mixed_query_layer.shape[0], mixed_query_layer.shape[1], self.heads_num, 1,
                                              self.hidden_size).transpose(1,2)

        # batch * num_heads * query * key * hidden
        key_expand = mixed_key_layer.view(mixed_key_layer.shape[0], mixed_key_layer.shape[1], self.heads_num,
                                          self.hidden_size).transpose(1, 2).unsqueeze(2).expand(
            -1, -1, query_expand.shape[2], -1, -1)

        # batch * num_heads * query * key * hidden
        value_expand = mixed_value_layer.view(mixed_value_layer.shape[0], mixed_value_layer.shape[1], self.heads_num,
                                              self.hidden_size).transpose(1, 2).unsqueeze(2).expand(
            -1, -1, query_expand.shape[2], -1, -1)

        if self.relation_embed_used and relation_matrices is not None:
            k_relation_embedding = self.key_relation_embed(relation_matrices).unsqueeze(1).expand_as(key_expand)
            # batch * num_heads * query * key * hidden
            v_relation_embedding = self.value_relation_embed(relation_matrices).unsqueeze(1).expand_as(value_expand)
            # batch * num_heads * query * key * hidden
            key_expand = key_expand + k_relation_embedding
            value_expand = value_expand + v_relation_embedding

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_expand, key_expand.transpose(-1, -2)).squeeze(-2) / math.sqrt(
            self.hidden_size)
        # batch * num_heads * query * key

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if key_padding_mask is not None:
            attention_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand_as(
                attention_scores).contiguous()  # batch * num_heads * query * key
            attention_scores = attention_scores.masked_fill(attention_mask, -float('inf'))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch * num_heads * query * key
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs.unsqueeze(dim=3), value_expand).squeeze(dim=3)
        # batch * num_heads * query * hidden

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1],
                                           self.heads_num * self.hidden_size)
        # batch * query * num_heads X hidden

        context_layer = self.dense(context_layer)
        return context_layer, attention_probs
