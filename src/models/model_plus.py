#!/usr/bin/env python
# @Time    : 2020-05-10 13:55
# @Author  : Zhi Chen
# @Desc    : model_plus

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : model.py
# @Software: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.dataset import Batch
from src.models import nn_utils
from src.models.basic_model import BasicModel, HeteroRGCN, HeteroRelGCN, HeteroGAT, HAN
from src.models.pointer_net import PointerNet
from src.rule import semQL as define_rule


def get_batch_embedding(tensors, lens, use_cuda):
    batch_size = len(tensors)
    max_len = max(lens)
    hidden_size = tensors[0].size(1)

    batch_embedding = torch.zeros((batch_size, max_len, hidden_size)).float()
    if use_cuda:
        batch_embedding = batch_embedding.cuda()

    for bi, (tensor, l) in enumerate(zip(tensors, lens)):
        batch_embedding[bi, :l] = tensor

    return batch_embedding


class IRNet(BasicModel):

    def __init__(self, args, grammar):
        super(IRNet, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer
        self.use_sentence_features = args.sentence_features
        self.layer_num = 2

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size // 2, bidirectional=True,
                                    batch_first=True)

        self.schema_encoder_lstm = nn.LSTM(args.embed_size, args.embed_size // 2, bidirectional=True,
                                           batch_first=True)

        self.table_node = nn.Parameter(torch.Tensor(1, args.hidden_size))
        self.colum_node = nn.Parameter(torch.Tensor(1, args.hidden_size))

        etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 'fore_t2c', 'fore_c2t', 's2t', 't2s']
        # etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 's2t', 't2s']
        # self.schema_rgcn = []
        # self.shadow_rgcn = []
        # self.schema_att_linear = []
        # self.src_att_linear = []
        # self.schame_shadow_cat_linear = []
        # self.src_shadow_ctx_linear = []
        # self.schema_src_ctx_linear = []
        # for l in range(self.layer_num):
        #     self.schema_rgcn.append(HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size))
        #     self.shadow_rgcn.append(HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size))
        #     self.schema_att_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
        #     self.src_att_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
        #     self.schame_shadow_cat_linear.append(nn.Linear(2 * args.hidden_size, args.hidden_size))
        #     self.src_shadow_ctx_linear.append(nn.Linear(2 * args.hidden_size, args.hidden_size))
        #     self.schema_src_ctx_linear.append(nn.Linear(2 * args.hidden_size, args.hidden_size))
        #
        # if args.cuda:
        #     for l in range(self.layer_num):
        #         self.schema_rgcn[l] = self.schema_rgcn[l].cuda()
        #         self.shadow_rgcn[l] = self.shadow_rgcn[l].cuda()
        #         self.schema_att_linear[l] = self.schema_att_linear[l].cuda()
        #         self.src_att_linear[l] = self.src_att_linear[l].cuda()
        #         self.schame_shadow_cat_linear[l] = self.schame_shadow_cat_linear[l].cuda()
        #         self.src_shadow_ctx_linear[l] = self.src_shadow_ctx_linear[l].cuda()
        #         self.schema_src_ctx_linear[l] = self.schema_src_ctx_linear[l].cuda()
        #
        # self.encoder_again_lstm = nn.LSTM(args.hidden_size, args.hidden_size // 2, bidirectional=True,
        #                                   batch_first=True)

        # self.schema_rgcn = HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size)
        # self.shadow_rgcn = HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size)

        self.schema_rgcn = HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, self.layer_num)
        self.shadow_rgcn = HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, self.layer_num)

        # self.schema_rgcn = HeteroGAT(etypes, args.hidden_size, args.hidden_size, self.layer_num)
        # self.shadow_rgcn = HeteroGAT(etypes, args.hidden_size, args.hidden_size, self.layer_num)

        # self.schema_rgcn = HAN(
        #     meta_paths=[['norm_t2c', 'norm_c2t'], ['prim_t2c', 'prim_c2t'], ['fore_t2c', 'fore_c2t'], ['s2t', 't2s']],
        #     in_size=args.hidden_size,
        #     hidden_size=args.hidden_size,
        #     out_size=args.hidden_size // 2,
        #     num_heads=[4, 4])
        #
        # self.shadow_rgcn = HAN(
        #     meta_paths=[['norm_t2c', 'norm_c2t'], ['prim_t2c', 'prim_c2t'], ['fore_t2c', 'fore_c2t'], ['s2t', 't2s']],
        #     in_size=args.hidden_size,
        #     hidden_size=args.hidden_size,
        #     out_size=args.hidden_size // 2,
        #     num_heads=[4, 4])

        self.schema_att_linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.src_att_linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.schema_layer_norm = nn.LayerNorm(args.hidden_size)
        self.src_layer_norm = nn.LayerNorm(args.hidden_size)
        self.schame_shadow_cat_linear = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.src_shadow_cat_linear = nn.Linear(2 * args.hidden_size, args.hidden_size)

        self.schema_cross_att_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.shadow_cross_att_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.src_cross_att_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.shadow_src_cat_linear = nn.Linear(2 * args.hidden_size, args.hidden_size)

        self.encoder_again_lstm = nn.LSTM(2*args.hidden_size, args.hidden_size // 2, bidirectional=True,
                                          batch_first=True)

        self.schema_link_embed = nn.Embedding(9, args.hidden_size)
        self.schema_link_linear = nn.Linear(args.hidden_size, 1)

        input_dim = args.action_embed_size + \
                    args.att_vec_size + \
                    args.type_embed_size
        # previous action
        # input feeding
        # pre type embedding

        self.lf_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        self.att_sketch_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(4, args.col_embed_size)
        self.sketch_encoder_lstm = nn.LSTM(args.action_embed_size + args.type_embed_size, args.action_embed_size // 2, bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        self.att_project = nn.Linear(args.hidden_size + args.type_embed_size, args.hidden_size)

        self.N_embed = nn.Embedding(len(define_rule.N._init_grammar()), args.action_embed_size)

        # tanh
        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.q_att = nn.Linear(args.hidden_size, args.embed_size)

        self.column_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.table_node.data)
        nn.init.xavier_normal_(self.colum_node.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)

    def one_layer_see_schema(self, src_encoding, column_embedding, table_embedding, shadow_c_emb, shadow_t_emb,
                             schema_graph, layer_id):
        t_len = table_embedding.size(0)
        hetero_schema_input = {
            'table': table_embedding,
            'column': column_embedding
        }
        hetero_schema_output = self.schema_rgcn[layer_id](schema_graph, hetero_schema_input)
        ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
        schema_item_encoding = torch.relu(self.schema_att_linear[layer_id](ori_schema_item_encoding))
        schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)

        att_src_encoding = torch.relu(self.src_att_linear[layer_id](src_encoding))
        src_schema_att_score = torch.bmm(att_src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

        hetero_shadow_input = {
            'table': shadow_t_emb,
            'column': shadow_c_emb
        }
        hetero_shadow_output = self.shadow_rgcn[layer_id](schema_graph, hetero_shadow_input)
        shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)

        schema_item_encoding = torch.cat((ori_schema_item_encoding, shadow_item_encoding), dim=1)
        schema_item_encoding = torch.relu(self.schame_shadow_cat_linear[layer_id](schema_item_encoding))

        src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
        src_shadow_ctx = torch.bmm(src_schema_att_prob, schema_item_encoding.unsqueeze(0)).squeeze(0)
        assert src_shadow_ctx.size(0) == src_encoding.size(0)

        src_shadow_ctx_encoding = torch.cat((src_encoding, src_shadow_ctx), dim=1)
        src_shadow_encoding = torch.relu(self.src_shadow_ctx_linear[layer_id](src_shadow_ctx_encoding))

        schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
        max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
        y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
        if self.args.cuda:
            y_idx = y_idx.cuda()
        max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)

        assert max_schema_arc_att_prob.size(0) == ori_schema_item_encoding.size(0)
        ori_schema_item_encoding = max_schema_arc_att_prob * ori_schema_item_encoding
        shadow_item_encoding = max_schema_arc_att_prob * shadow_item_encoding

        table_encoding = ori_schema_item_encoding[:t_len]
        column_encoding = ori_schema_item_encoding[t_len:]

        table_shadow_encoding = shadow_item_encoding[:t_len]
        column_shadow_encoding = shadow_item_encoding[t_len:]

        max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
        max_schema_arc_att_prob_dict = {}
        max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
        max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

        return src_shadow_encoding, table_encoding, column_encoding, table_shadow_encoding, column_shadow_encoding, max_schema_arc_att_prob_dict

    def one_layer_see_schema1(self, src_encoding, column_embedding, table_embedding, shadow_c_emb, shadow_t_emb,
                             schema_graph, layer_id):
        t_len = table_embedding.size(0)
        hetero_schema_input = {
            'table': table_embedding,
            'column': column_embedding
        }
        hetero_schema_output = self.schema_rgcn[layer_id](schema_graph, hetero_schema_input)
        ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
        schema_item_encoding = torch.relu(self.schema_att_linear[layer_id](ori_schema_item_encoding))
        schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)

        att_src_encoding = torch.relu(self.src_att_linear[layer_id](src_encoding))
        src_schema_att_score = torch.bmm(att_src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

        hetero_shadow_input = {
            'table': shadow_t_emb,
            'column': shadow_c_emb
        }
        hetero_shadow_output = self.shadow_rgcn[layer_id](schema_graph, hetero_shadow_input)
        shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)

        # schema_item_encoding = torch.cat((ori_schema_item_encoding, shadow_item_encoding), dim=1)
        # schema_item_encoding = torch.relu(self.schame_shadow_cat_linear[layer_id](schema_item_encoding))

        src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
        src_shadow_ctx = torch.bmm(src_schema_att_prob, ori_schema_item_encoding.unsqueeze(0)).squeeze(0)
        assert src_shadow_ctx.size(0) == src_encoding.size(0)

        src_shadow_ctx_encoding = torch.cat((src_encoding, src_shadow_ctx), dim=1)
        src_shadow_encoding = torch.relu(self.src_shadow_ctx_linear[layer_id](src_shadow_ctx_encoding))

        schema_src_att_prob = torch.softmax(torch.transpose(src_schema_att_score, 1, 2), dim=2)
        schema_src_ctx = torch.bmm(schema_src_att_prob, src_encoding.unsqueeze(0)).squeeze(0)
        assert schema_src_ctx.size(0) == ori_schema_item_encoding.size(0)

        schema_src_ctx_encoding = torch.cat((ori_schema_item_encoding, schema_src_ctx), dim=1)
        schema_src_encoding = torch.relu(self.schema_src_ctx_linear[layer_id](schema_src_ctx_encoding))

        schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
        max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
        y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
        if self.args.cuda:
            y_idx = y_idx.cuda()
        max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)

        assert max_schema_arc_att_prob.size(0) == shadow_item_encoding.size(0)
        # schema_src_encoding = max_schema_arc_att_prob * schema_src_encoding
        shadow_item_encoding = max_schema_arc_att_prob * shadow_item_encoding

        table_encoding = schema_src_encoding[:t_len]
        column_encoding = schema_src_encoding[t_len:]

        table_shadow_encoding = shadow_item_encoding[:t_len]
        column_shadow_encoding = shadow_item_encoding[t_len:]

        max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
        max_schema_arc_att_prob_dict = {}
        max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
        max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

        return src_shadow_encoding, table_encoding, column_encoding, table_shadow_encoding, column_shadow_encoding, max_schema_arc_att_prob_dict

    def sent_see_schema12(self, batch, src_encodings, table_embeddings, schema_embeddings):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs

        # table_names = batch.table_names
        # table_col_name = batch.table_sents

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_schema_arc_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):

            shadow_t_emb = torch.repeat_interleave(self.table_node, t_len, dim=0)
            shadow_c_emb = torch.repeat_interleave(self.colum_node, c_len, dim=0)

            src_encoding = src_encodings[bi, :ss_len]
            column_encoding = table_embeddings[bi, :c_len]
            table_encoding = schema_embeddings[bi, :t_len]
            schema_graph = schema_graphs[bi]

            for l_id in range(self.layer_num):
                src_encoding, table_encoding, column_encoding, shadow_t_emb, shadow_c_emb, max_schema_arc_att_prob = \
                    self.one_layer_see_schema1(src_encoding, column_encoding, table_encoding, shadow_c_emb, shadow_t_emb,
                            schema_graph, layer_id=l_id)

            src_repre.append(src_encoding)
            table_repre.append(table_encoding)
            column_repre.append(column_encoding)
            batch_max_schema_arc_att_prob.append(max_schema_arc_att_prob)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_schema_arc_att_prob

    def sent_see_schema11(self, batch, src_encodings, table_embeddings, schema_embeddings):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs

        # table_names = batch.table_names
        # table_col_name = batch.table_sents

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_schema_arc_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):
            ori_src_encoding = src_encodings[bi, :ss_len]
            column_embedding = table_embeddings[bi, :c_len]
            table_embedding = schema_embeddings[bi, :t_len]

            hetero_schema_input = {
                'table': table_embedding,
                'column': column_embedding
            }
            hetero_schema_output = self.schema_rgcn(schema_graphs[bi], hetero_schema_input)
            ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
            schema_item_encoding = torch.relu(self.schema_att_linear(ori_schema_item_encoding))

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            # src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            # src_schema_att_score = self.cosine_attention(src_encoding, schema_item_encoding)
            # src_schema_att_score = src_schema_att_score.unsqueeze(dim=0)

            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0),
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0)
            }
            hetero_shadow_output = self.shadow_rgcn(schema_graphs[bi], hetero_shadow_input)
            shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)

            ori_schema_item_encoding = torch.cat((ori_schema_item_encoding, shadow_item_encoding), dim=1)
            ori_schema_item_encoding = torch.relu(self.schame_shadow_cat_linear(ori_schema_item_encoding))

            src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
            src_shadow_ctx = torch.bmm(src_schema_att_prob, ori_schema_item_encoding.unsqueeze(0)).squeeze(0)
            assert src_shadow_ctx.size(0) == ori_src_encoding.size(0)

            src_shadow_ctx_encoding = torch.cat((ori_src_encoding, src_shadow_ctx), dim=1)

            schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
            max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
            y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)

            assert max_schema_arc_att_prob.size(0) == ori_schema_item_encoding.size(0)
            ori_schema_item_encoding = max_schema_arc_att_prob * ori_schema_item_encoding

            table_shadow_encoding = ori_schema_item_encoding[:t_len]
            column_shadow_encoding = ori_schema_item_encoding[t_len:]

            max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
            max_schema_arc_att_prob_dict = {}
            max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
            max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_schema_arc_att_prob.append(max_schema_arc_att_prob_dict)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_schema_arc_att_prob

    def sent_see_schema10(self, batch, src_encodings, table_embeddings, schema_embeddings):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs

        # table_names = batch.table_names
        # table_col_name = batch.table_sents

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_schema_arc_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):
            ori_src_encoding = src_encodings[bi, :ss_len]
            column_embedding = table_embeddings[bi, :c_len]
            table_embedding = schema_embeddings[bi, :t_len]

            hetero_schema_input = {
                'table': table_embedding,
                'column': column_embedding
            }
            hetero_schema_output = self.schema_rgcn(schema_graphs[bi], hetero_schema_input)
            ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
            schema_item_encoding = torch.relu(self.schema_att_linear(ori_schema_item_encoding))
            # schema_item_encoding_v = torch.relu(self.schema_cross_att_v(ori_schema_item_encoding))

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            src_encoding_v = torch.relu(self.src_cross_att_v(ori_src_encoding))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            # src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            # src_schema_att_score = self.cosine_attention(src_encoding, schema_item_encoding)
            # src_schema_att_score = src_schema_att_score.unsqueeze(dim=0)

            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0),
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0)
            }
            hetero_shadow_output = self.shadow_rgcn(schema_graphs[bi], hetero_shadow_input)
            shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)
            shadow_item_encoding_v = torch.relu(self.shadow_cross_att_v(shadow_item_encoding))
            # shadow_item_encoding_v_mean = torch.mean(shadow_item_encoding_v, dim=0).unsqueeze(0)

            # schema_item_encoding = torch.cat((ori_schema_item_encoding, shadow_item_encoding), dim=1)
            # schema_item_encoding = torch.relu(self.schame_shadow_cat_linear(schema_item_encoding))

            src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
            src_shadow_ctx = torch.bmm(src_schema_att_prob, shadow_item_encoding_v.unsqueeze(0)).squeeze(0)
            # src_shadow_encoding_v_mean = torch.repeat_interleave(shadow_item_encoding_v_mean,
            #                                                       src_shadow_ctx.size(0), dim=0)
            # src_shadow_ctx = src_shadow_ctx - src_shadow_encoding_v_mean
            assert src_shadow_ctx.size(0) == ori_src_encoding.size(0)

            src_shadow_ctx_encoding = torch.cat((ori_src_encoding, src_shadow_ctx), dim=1)
            # src_shadow_ctx_encoding = torch.relu(self.src_shadow_cat_linear(src_shadow_ctx_encoding))

            schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
            max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
            y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)

            schema_src_att_prob = torch.transpose(schema_arc_att_prob, 0, 1).unsqueeze(0)
            shadow_src_ctx = torch.bmm(schema_src_att_prob, src_encoding_v.unsqueeze(0)).squeeze(0)

            shadow_src_ctx = torch.cat((shadow_item_encoding, shadow_src_ctx), dim=1)
            shadow_src_ctx = torch.relu(self.shadow_src_cat_linear(shadow_src_ctx))

            assert max_schema_arc_att_prob.size(0) == ori_schema_item_encoding.size(0)
            shadow_src_ctx = max_schema_arc_att_prob * shadow_src_ctx

            table_shadow_encoding = shadow_src_ctx[:t_len]
            column_shadow_encoding = shadow_src_ctx[t_len:]

            max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
            max_schema_arc_att_prob_dict = {}
            max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
            max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_schema_arc_att_prob.append(max_schema_arc_att_prob_dict)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_schema_arc_att_prob

    def sent_see_schema9(self, batch, src_encodings, table_embeddings, schema_embeddings):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs

        # table_names = batch.table_names
        # table_col_name = batch.table_sents

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_schema_arc_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):
            ori_src_encoding = src_encodings[bi, :ss_len]
            column_embedding = table_embeddings[bi, :c_len]
            table_embedding = schema_embeddings[bi, :t_len]

            hetero_schema_input = {
                'table': table_embedding,
                'column': column_embedding
            }
            hetero_schema_output = self.schema_rgcn(schema_graphs[bi], hetero_schema_input)
            ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
            schema_item_encoding = torch.relu(self.schema_att_linear(ori_schema_item_encoding))
            # schema_item_encoding_v = torch.relu(self.schema_cross_att_v(ori_schema_item_encoding))

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            src_encoding_v = torch.relu(self.src_cross_att_v(ori_src_encoding))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
            max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
            y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)
            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0)*max_schema_arc_att_prob[:t_len],
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0)*max_schema_arc_att_prob[t_len:]
            }

            hetero_shadow_output = self.shadow_rgcn(schema_graphs[bi], hetero_shadow_input)
            shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)
            shadow_item_encoding_v = torch.relu(self.shadow_cross_att_v(shadow_item_encoding))

            schema_src_att_prob = torch.transpose(schema_arc_att_prob, 0, 1).unsqueeze(0)
            shadow_src_ctx = torch.bmm(schema_src_att_prob, src_encoding_v.unsqueeze(0)).squeeze(0)

            shadow_src_ctx = torch.cat((shadow_item_encoding, shadow_src_ctx), dim=1)
            shadow_src_ctx = torch.relu(self.shadow_src_cat_linear(shadow_src_ctx))

            # assert max_schema_arc_att_prob.size(0) == ori_schema_item_encoding.size(0)
            # shadow_src_ctx = max_schema_arc_att_prob * shadow_src_ctx

            table_shadow_encoding = shadow_src_ctx[:t_len]
            column_shadow_encoding = shadow_src_ctx[t_len:]

            # src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            # src_schema_att_score = self.cosine_attention(src_encoding, schema_item_encoding)
            # src_schema_att_score = src_schema_att_score.unsqueeze(dim=0)

            # schema_item_encoding = torch.cat((ori_schema_item_encoding, shadow_item_encoding), dim=1)
            # schema_item_encoding = torch.relu(self.schame_shadow_cat_linear(schema_item_encoding))

            src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
            src_shadow_ctx = torch.bmm(src_schema_att_prob, shadow_item_encoding_v.unsqueeze(0)).squeeze(0)
            # src_shadow_encoding_v_mean = torch.repeat_interleave(shadow_item_encoding_v_mean,
            #                                                       src_shadow_ctx.size(0), dim=0)
            # src_shadow_ctx = src_shadow_ctx - src_shadow_encoding_v_mean
            assert src_shadow_ctx.size(0) == ori_src_encoding.size(0)

            src_shadow_ctx_encoding = torch.cat((ori_src_encoding, src_shadow_ctx), dim=1)
            # src_shadow_ctx_encoding = torch.relu(self.src_shadow_cat_linear(src_shadow_ctx_encoding))

            max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
            max_schema_arc_att_prob_dict = {}
            max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
            max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_schema_arc_att_prob.append(max_schema_arc_att_prob_dict)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_schema_arc_att_prob

    def sent_see_schema8(self, batch, src_encodings, table_embeddings, schema_embeddings):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs
        schema_links = batch.schema_links
        src_sents = batch.src_sents

        # table_names = batch.table_names
        # table_col_name = batch.table_sents

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_schema_arc_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):
            ori_src_encoding = src_encodings[bi, :ss_len]
            column_embedding = table_embeddings[bi, :c_len]
            table_embedding = schema_embeddings[bi, :t_len]
            schema_link = schema_links[bi]

            schema_link = torch.tensor(schema_link)
            if torch.cuda.is_available():
                schema_link = schema_link.cuda()

            schema_link_embed = self.schema_link_embed(schema_link)
            schema_link_score = self.schema_link_linear(schema_link_embed)
            # schema_link_score = torch.sigmoid(schema_link_score)
            schema_link_score = schema_link_score.squeeze(-1).unsqueeze(0)

            hetero_schema_input = {
                'table': table_embedding,
                'column': column_embedding
            }
            hetero_schema_output = self.schema_rgcn(schema_graphs[bi], hetero_schema_input)
            ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
            schema_item_encoding = torch.relu(self.schema_att_linear(ori_schema_item_encoding))
            # schema_item_encoding_v = torch.relu(self.schema_cross_att_v(ori_schema_item_encoding))

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            src_encoding_v = torch.relu(self.src_cross_att_v(ori_src_encoding))
            # src_schema_att_score = torch.sigmoid(torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0)))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            # print(src_schema_att_score.shape)
            # print(schema_link_score.shape)

            # assert src_schema_att_score.shape == schema_link_score.shape
            # print(src_schema_att_score.shape)
            # print(schema_link_score.shape)

            if src_schema_att_score.shape != schema_link_score.shape:
                print('#'*100)
                print(src_sents[bi])
                print(src_schema_att_score.shape)
                print(schema_link_score.shape)
                exit(0)
            src_schema_att_score = src_schema_att_score + schema_link_score

            schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
            max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
            y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)
            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0)*max_schema_arc_att_prob[:t_len],
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0)*max_schema_arc_att_prob[t_len:]
            }

            hetero_shadow_output = self.shadow_rgcn(schema_graphs[bi], hetero_shadow_input)
            # hetero_shadow_output = self.schema_rgcn(schema_graphs[bi], hetero_shadow_input)
            shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)
            shadow_item_encoding_v = torch.relu(self.shadow_cross_att_v(shadow_item_encoding))

            schema_src_att_prob = torch.transpose(schema_arc_att_prob, 0, 1).unsqueeze(0)
            shadow_src_ctx = torch.bmm(schema_src_att_prob, src_encoding_v.unsqueeze(0)).squeeze(0)

            shadow_src_ctx = torch.cat((shadow_item_encoding, shadow_src_ctx), dim=1)
            shadow_src_ctx = torch.relu(self.shadow_src_cat_linear(shadow_src_ctx))

            # assert max_schema_arc_att_prob.size(0) == ori_schema_item_encoding.size(0)
            # shadow_src_ctx = max_schema_arc_att_prob * shadow_src_ctx

            table_shadow_encoding = shadow_src_ctx[:t_len]
            column_shadow_encoding = shadow_src_ctx[t_len:]

            # src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            # src_schema_att_score = self.cosine_attention(src_encoding, schema_item_encoding)
            # src_schema_att_score = src_schema_att_score.unsqueeze(dim=0)

            # schema_item_encoding = torch.cat((ori_schema_item_encoding, shadow_item_encoding), dim=1)
            # schema_item_encoding = torch.relu(self.schame_shadow_cat_linear(schema_item_encoding))

            src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
            src_shadow_ctx = torch.bmm(src_schema_att_prob, shadow_item_encoding_v.unsqueeze(0)).squeeze(0)
            # src_shadow_encoding_v_mean = torch.repeat_interleave(shadow_item_encoding_v_mean,
            #                                                       src_shadow_ctx.size(0), dim=0)
            # src_shadow_ctx = src_shadow_ctx - src_shadow_encoding_v_mean
            assert src_shadow_ctx.size(0) == ori_src_encoding.size(0)

            src_shadow_ctx_encoding = torch.cat((ori_src_encoding, src_shadow_ctx), dim=1)
            # src_shadow_ctx_encoding = torch.relu(self.src_shadow_cat_linear(src_shadow_ctx_encoding))

            max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
            max_schema_arc_att_prob_dict = {}
            max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
            max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_schema_arc_att_prob.append(max_schema_arc_att_prob_dict)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_schema_arc_att_prob

    def sent_see_schema7(self, batch, src_encodings, table_embeddings, schema_embeddings):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs
        schema_links = batch.schema_links
        src_sents = batch.src_sents

        # table_names = batch.table_names
        # table_col_name = batch.table_sents

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_schema_arc_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):
            ori_src_encoding = src_encodings[bi, :ss_len]
            column_embedding = table_embeddings[bi, :c_len]
            table_embedding = schema_embeddings[bi, :t_len]
            schema_link = schema_links[bi]

            schema_link = torch.tensor(schema_link)
            if torch.cuda.is_available():
                schema_link = schema_link.cuda()

            schema_link_embed = self.schema_link_embed(schema_link)
            schema_link_score = self.schema_link_linear(schema_link_embed)
            schema_link_score = torch.sigmoid(schema_link_score)
            schema_link_score = schema_link_score.squeeze(-1).unsqueeze(0)

            hetero_schema_input = {
                'table': table_embedding,
                'column': column_embedding
            }
            hetero_schema_output = self.schema_rgcn(schema_graphs[bi], hetero_schema_input)
            ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
            schema_item_encoding = torch.relu(self.schema_att_linear(ori_schema_item_encoding))
            schema_item_encoding_v = torch.relu(self.schema_cross_att_v(ori_schema_item_encoding))

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            src_encoding_v = torch.relu(self.src_cross_att_v(ori_src_encoding))
            # src_schema_att_score = torch.sigmoid(torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0)))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            # print(src_schema_att_score.shape)
            # print(schema_link_score.shape)

            # assert src_schema_att_score.shape == schema_link_score.shape
            # print(src_schema_att_score.shape)
            # print(schema_link_score.shape)

            if src_schema_att_score.shape != schema_link_score.shape:
                print('#'*100)
                print(src_sents[bi])
                print(src_schema_att_score.shape)
                print(schema_link_score.shape)
                exit(0)
            src_schema_att_score = src_schema_att_score * schema_link_score

            schema_arc_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=0)
            max_schema_arc_att_idx = torch.argmax(schema_arc_att_prob, dim=0)
            y_idx = torch.tensor([i for i in range(max_schema_arc_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_schema_arc_att_prob = schema_arc_att_prob[max_schema_arc_att_idx, y_idx].unsqueeze(1)
            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0)*max_schema_arc_att_prob[:t_len],
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0)*max_schema_arc_att_prob[t_len:]
            }

            # hetero_shadow_output = self.shadow_rgcn(schema_graphs[bi], hetero_shadow_input)
            hetero_shadow_output = self.schema_rgcn(schema_graphs[bi], hetero_shadow_input)
            shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)
            shadow_item_encoding_v = torch.relu(self.shadow_cross_att_v(shadow_item_encoding))

            schema_src_att_prob = torch.transpose(schema_arc_att_prob, 0, 1).unsqueeze(0)
            shadow_src_ctx = torch.bmm(schema_src_att_prob, src_encoding_v.unsqueeze(0)).squeeze(0)

            shadow_src_ctx = torch.cat((shadow_item_encoding, shadow_src_ctx), dim=1)
            shadow_src_ctx = torch.relu(self.shadow_src_cat_linear(shadow_src_ctx))

            # assert max_schema_arc_att_prob.size(0) == ori_schema_item_encoding.size(0)
            # shadow_src_ctx = max_schema_arc_att_prob * shadow_src_ctx

            table_shadow_encoding = shadow_src_ctx[:t_len]
            column_shadow_encoding = shadow_src_ctx[t_len:]

            # src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            # src_schema_att_score = self.cosine_attention(src_encoding, schema_item_encoding)
            # src_schema_att_score = src_schema_att_score.unsqueeze(dim=0)

            schema_item_encoding = torch.cat((schema_item_encoding_v, shadow_item_encoding_v), dim=1)
            schema_item_encoding = torch.relu(self.schame_shadow_cat_linear(schema_item_encoding))

            src_schema_att_prob = torch.softmax(src_schema_att_score, dim=2)
            src_shadow_ctx = torch.bmm(src_schema_att_prob, schema_item_encoding.unsqueeze(0)).squeeze(0)
            # src_shadow_encoding_v_mean = torch.repeat_interleave(shadow_item_encoding_v_mean,
            #                                                       src_shadow_ctx.size(0), dim=0)
            # src_shadow_ctx = src_shadow_ctx - src_shadow_encoding_v_mean
            assert src_shadow_ctx.size(0) == ori_src_encoding.size(0)

            src_shadow_ctx_encoding = torch.cat((ori_src_encoding, src_shadow_ctx), dim=1)
            # src_shadow_ctx_encoding = torch.relu(self.src_shadow_cat_linear(src_shadow_ctx_encoding))

            max_schema_arc_att_prob = max_schema_arc_att_prob.squeeze(1)
            max_schema_arc_att_prob_dict = {}
            max_schema_arc_att_prob_dict['table'] = max_schema_arc_att_prob[:t_len]
            max_schema_arc_att_prob_dict['column'] = max_schema_arc_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_schema_arc_att_prob.append(max_schema_arc_att_prob_dict)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_schema_arc_att_prob

    def get_schema_embedding(self, batch_schema_discribs):
        schema_discrib_len = [len(schema_discrib) for schema_discrib in batch_schema_discribs]
        total_schema_discribs = []
        total_schema_discrib_lens = []
        for schema_discrib in batch_schema_discribs:
            for one_discrib in schema_discrib:
                total_schema_discribs.append(one_discrib)
                total_schema_discrib_lens.append(len(one_discrib))

        total_schema_discrib_embedding = self.gen_x_batch(total_schema_discribs)

        _, (last_state, last_cell) = self.schema_encoder(total_schema_discrib_embedding, total_schema_discrib_lens)
        assert last_state.size(0) == len(total_schema_discrib_lens)

        batch_schema_embeddding = []
        flag_id = 0
        for discrib_len in schema_discrib_len:
            batch_schema_embeddding.append(last_state[flag_id:flag_id+discrib_len, :])
            flag_id += discrib_len

        assert len(batch_schema_embeddding) == len(schema_discrib_len)
        batch_schema_embeddding = get_batch_embedding(batch_schema_embeddding, schema_discrib_len, self.args.cuda)
        return batch_schema_embeddding

    def forward(self, examples):
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask

        new_batch_table_sents = []
        for i, table_sents in enumerate(batch.table_sents):
            new_table_sents = []
            for table_sent in table_sents:
                new_table_sents.append(['column'] + table_sent)
            new_batch_table_sents.append(new_table_sents)

        new_batch_table_names = []
        for i, table_names in enumerate(batch.table_names):
            new_table_names = []
            for table_name in table_names:
                new_table_names.append(['table'] + table_name)
            new_batch_table_names.append(new_table_names)

        ori_table_embedding = self.get_schema_embedding(new_batch_table_sents)
        ori_schema_embedding = self.get_schema_embedding(new_batch_table_names)
        ori_src_embedding = self.get_schema_embedding(batch.src_sents)

        src_encodings, _ = self.encode(ori_src_embedding, batch.src_sents_len, None, src_embed=True)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema8(batch, src_encodings,
                                                                                  ori_table_embedding,
                                                                                  ori_schema_embedding)

        src_encodings, (last_state, last_cell) = self.encode_again(src_encodings, batch.src_sents_len)

        src_encodings = self.dropout(src_encodings)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec
        action_probs = [[] for _ in examples]

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        zero_type_embed = Variable(self.new_tensor(args.type_embed_size).zero_())

        sketch_attention_history = list()

        for t in range(batch.max_sketch_num):
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.sketch_decoder_lstm.input_size).zero_(),
                             requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, example in enumerate(examples):

                    if t < len(example.sketch):
                        # get the last action
                        # This is the action embedding
                        action_tm1 = example.sketch[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order]:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        else:
                            print(action_tm1, 'only for sketch')
                            quit()
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, example in enumerate(examples):
                    if t < len(example.sketch):
                        action_tm = example.sketch[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstm,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)
            sketch_attention_history.append(att_t)

            # get the Root possibility
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            for e_id, example in enumerate(examples):
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        # batch_sketch_encoder_input = []
        # batch_sketch_len = batch.sketch_len
        # for t in range(batch.max_sketch_num):
        #     a_t_embeds = []
        #     types = []
        #     for e_id, example in enumerate(examples):
        #         if t < len(example.sketch):
        #             # get the last action
        #             # This is the action embedding
        #             action_t = example.sketch[t]
        #             if type(action_t) in [define_rule.Root1,
        #                                     define_rule.Root,
        #                                     define_rule.Sel,
        #                                     define_rule.Filter,
        #                                     define_rule.Sup,
        #                                     define_rule.N,
        #                                     define_rule.Order]:
        #                 a_t_embed = self.production_embed.weight[self.grammar.prod2id[action_t.production]]
        #             else:
        #                 print(action_t, 'only for sketch')
        #                 quit()
        #                 a_t_embed = zero_action_embed
        #                 pass
        #         else:
        #             a_t_embed = zero_action_embed
        #
        #         a_t_embeds.append(a_t_embed)
        #
        #     a_t_embeds = torch.stack(a_t_embeds)
        #
        #     for e_id, example in enumerate(examples):
        #         if t < len(example.sketch):
        #             action_t = example.sketch[t]
        #             act_type = self.type_embed.weight[self.grammar.type2id[type(action_t)]]
        #         else:
        #             act_type = zero_type_embed
        #         types.append(act_type)
        #
        #     types = torch.stack(types)
        #     batch_sketch_encoder_input.append(torch.cat((a_t_embeds, types), dim=-1).unsqueeze(dim=1))
        #
        # batch_sketch_encoder_input = torch.cat(batch_sketch_encoder_input, dim=1)
        # batch_sketch_encoder_output, _ = self.sketch_encoder(batch_sketch_encoder_input, batch_sketch_len)

        sketch_prob_var = torch.stack(
                [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        # get emb differ
        embedding_differ = self.embedding_cosine(src_embedding=ori_src_embedding, table_embedding=ori_table_embedding,
                                                 table_unk_mask=batch.table_unk_mask)

        schema_differ = self.embedding_cosine(src_embedding=ori_src_embedding, table_embedding=ori_schema_embedding,
                                              table_unk_mask=batch.schema_token_mask)

        tab_ctx = (ori_src_embedding.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (ori_src_embedding.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        table_embedding = table_embedding + tab_ctx

        schema_embedding = schema_embedding + schema_ctx

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        table_embedding = table_embedding + col_type_var

        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]
        max_att_ct_probs = [[] for _ in examples]

        h_tm1 = dec_init_vec

        batch_sketch_flag = [0 for _ in batch.sketch_len]

        for t in range(batch.max_action_num):
            if t == 0:
                # x = self.lf_begin_vec.unsqueeze(0).repeat(len(batch), 1)
                x = Variable(self.new_tensor(len(batch), self.lf_decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order,
                                                ]:

                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            # a_tm1_embed = batch_sketch_encoder_output[e_id, batch_sketch_flag[e_id], :]
                            # batch_sketch_flag[e_id] += 1

                        else:
                            if isinstance(action_tm1, define_rule.C):
                                a_tm1_embed = self.column_rnn_input(table_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.column_rnn_input(schema_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.A):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            else:
                                print(action_tm1, 'not implement')
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                # tgt t-1 action type
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm = example.tgt_actions[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)
            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
            else:
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)

            weights.data.masked_fill_(batch.table_token_mask.bool(), -float('inf'))

            column_attention_weights = F.softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.softmax(table_weights, dim=-1)
            # now get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        table_appear_mask[e_id, action_t.id_c] = 1
                        table_enable[e_id] = action_t.id_c
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)

                        max_att_ct_probs[e_id].append(batch_max_att_p[e_id]['column'][action_t.id_c])
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)

                        max_att_ct_probs[e_id].append(batch_max_att_p[e_id]['table'][action_t.id_c])
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        att_prob_var = torch.stack(
            [torch.stack(att_probs_i, dim=0).log().sum() for att_probs_i in max_att_ct_probs], dim=0)

        return [sketch_prob_var, lf_prob_var, att_prob_var]

    def parse(self, examples, beam_size=5):
        """
        one example a time
        :param examples:
        :param beam_size:
        :return:
        """
        batch = Batch([examples], self.grammar, cuda=self.args.cuda)

        new_batch_table_sents = []
        for i, table_sents in enumerate(batch.table_sents):
            new_table_sents = []
            for table_sent in table_sents:
                new_table_sents.append(['column'] + table_sent)
            new_batch_table_sents.append(new_table_sents)

        new_batch_table_names = []
        for i, table_names in enumerate(batch.table_names):
            new_table_names = []
            for table_name in table_names:
                new_table_names.append(['table'] + table_name)
            new_batch_table_names.append(new_table_names)

        ori_table_embedding = self.get_schema_embedding(new_batch_table_sents)
        ori_schema_embedding = self.get_schema_embedding(new_batch_table_names)
        ori_src_embedding = self.get_schema_embedding(batch.src_sents)

        src_encodings, _ = self.encode(ori_src_embedding, batch.src_sents_len, None, src_embed=True)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema8(batch, src_encodings,
                                                                                  ori_table_embedding,
                                                                                  ori_schema_embedding)

        src_encodings, (last_state, last_cell) = self.encode_again(src_encodings, batch.src_sents_len)

        src_encodings = self.dropout(src_encodings)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=True)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(beams)
            exp_src_enconding = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(hyp_num,
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           1),
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           2))
            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_enconding,
                                             exp_src_encodings_sketch_linear, self.sketch_decoder_lstm,
                                             self.sketch_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                action_class = hyp.get_availableClass()
                if action_class in [define_rule.Root1,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.N,
                                    define_rule.Order]:
                    possible_productions = self.grammar.get_production(action_class)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]
                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    raise RuntimeError('No right action class')

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]
                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']
                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        # now get the sketch result
        completed_beams.sort(key=lambda hyp: -hyp.score)
        if len(completed_beams) == 0:
            return [[], []]

        sketch_actions = completed_beams[0].actions
        # sketch_len = [len(sketch_actions)]
        # batch_sketch_encoder_input = []
        # for t in range(sketch_len[0]):
        #     # This is the action embedding
        #     action_t = sketch_actions[t]
        #     if type(action_t) in [define_rule.Root1,
        #                           define_rule.Root,
        #                           define_rule.Sel,
        #                           define_rule.Filter,
        #                           define_rule.Sup,
        #                           define_rule.N,
        #                           define_rule.Order]:
        #         a_t_embed = self.production_embed.weight[self.grammar.prod2id[action_t.production]]
        #     else:
        #         print(action_t, 'only for sketch')
        #         quit()
        #         pass
        #
        #     a_t_embeds = a_t_embed.unsqueeze(0)
        #
        #     action_t = sketch_actions[t]
        #     act_type = self.type_embed.weight[self.grammar.type2id[type(action_t)]]
        #     types = act_type.unsqueeze(0)
        #
        #     batch_sketch_encoder_input.append(torch.cat((a_t_embeds, types), dim=-1).unsqueeze(dim=1))
        #
        # batch_sketch_encoder_input = torch.cat(batch_sketch_encoder_input, dim=1)
        # batch_sketch_encoder_output, _ = self.sketch_encoder(batch_sketch_encoder_input, sketch_len)
        # batch_sketch_encoder_output = batch_sketch_encoder_output.squeeze(0)

        padding_sketch = self.padding_sketch(sketch_actions)

        # get emb differ
        embedding_differ = self.embedding_cosine(src_embedding=ori_src_embedding, table_embedding=ori_table_embedding,
                                                 table_unk_mask=batch.table_unk_mask)

        schema_differ = self.embedding_cosine(src_embedding=ori_src_embedding, table_embedding=ori_schema_embedding,
                                              table_unk_mask=batch.schema_token_mask)

        tab_ctx = (ori_src_embedding.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (ori_src_embedding.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        table_embedding = table_embedding + tab_ctx

        schema_embedding = schema_embedding + schema_ctx

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        table_embedding = table_embedding + col_type_var

        batch_table_dict = batch.col_table_dict

        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(beams)

            # expand value
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(hyp_num,
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         1),
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         2))
            exp_table_embedding = table_embedding.expand(hyp_num, table_embedding.size(1),
                                                         table_embedding.size(2))

            exp_schema_embedding = schema_embedding.expand(hyp_num, schema_embedding.size(1),
                                                           schema_embedding.size(2))

            table_appear_mask = batch.table_appear_mask
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        # a_tm1_embed = batch_sketch_encoder_output[hyp.sketch_step, :]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        a_tm1_embed = self.column_rnn_input(table_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.T):
                        a_tm1_embed = self.column_rnn_input(schema_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                             self.lf_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            table_appear_mask_val = torch.from_numpy(table_appear_mask)

            if self.args.cuda: table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
                # weights = weights + self.col_attention_out(exp_embedding_differ).squeeze()
            else:
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)
            # weights.data.masked_fill_(exp_col_pred_mask, -float('inf'))

            column_selection_log_prob = F.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)
            # table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))

            table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.log_softmax(table_weights, dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.A:
                    possible_productions = self.grammar.get_production(define_rule.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': define_rule.A, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_selection_log_prob[hyp_id, col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': define_rule.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                                  'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']
                if action_type_str == define_rule.C:
                    col_id = hyp_meta_entry['col_id']
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = hyp_meta_entry['t_id']
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]

                beams = new_beams
                t += 1
            else:
                break

        completed_beams.sort(key=lambda hyp: -hyp.score)

        return [completed_beams, sketch_actions]

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        # tanh
        att_t = F.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        # tanh
        h_0 = F.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

