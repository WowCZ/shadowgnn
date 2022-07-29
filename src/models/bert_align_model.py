#!/usr/bin/env python
# @Time    : 2020-05-23 17:00
# @Author  : Zhi Chen
# @Desc    : bert_model_plus

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
from src.models.basic_model import BasicModel, HeteroRGCN, HeteroRelGCN, HeteroGAT, HAN, GAT
from src.models.pointer_net import PointerNet
from src.rule import semQL as define_rule
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

from transformers import BertModel, BertTokenizer, AlbertTokenizer, AlbertModel, ElectraModel, ElectraTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-model/')
# bert_tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2/')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncase/')
# bert_tokenizer = ElectraTokenizer.from_pretrained('electra_large_discriminator/')


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


def merge_pad_idx(sequences, pad_idx):
    '''
    merge from batch * sent_len to batch * max_len
    '''
    lengths = [len(seq) for seq in sequences]
    max_len = 1 if max(lengths) == 0 else max(lengths)
    padded_seqs = torch.ones(len(sequences), max_len).long() * pad_idx
    att_mask = torch.zeros(len(sequences), max_len).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = torch.tensor(seq[:end])
        att_mask[i, :end] = 1
    padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
    return padded_seqs, lengths, att_mask


def merge_bert_data(sequences, segm=True):
    batch_size = len(sequences)

    bert_tokens = []
    segment_ids = []
    ori_seq_lens = []
    x_axis = []
    y_axis = []
    bert_distill_flag = []
    for bi, seq in enumerate(sequences):
        bert_token = []
        segment_id = []
        segment_flag = False
        seq_split = seq.split(' ')
        ori_seq_lens.append(len(seq_split))
        x = []
        y = []
        distill_flag = []
        for ti, token in enumerate(seq_split):
            if token != '[SEP]' and token != '[CLS]':
                distill_flag.append(ti)
            cur_bert_tokens = bert_tokenizer.tokenize(token)
            start_y = len(bert_token)
            bert_token = bert_token + cur_bert_tokens
            if segment_flag:
                segment_id = segment_id + [1] * len(cur_bert_tokens)
            else:
                segment_id = segment_id + [0] * len(cur_bert_tokens)
            end_y = len(bert_token)
            x = x + [ti] * len(cur_bert_tokens)
            y = y + [yi for yi in range(start_y, end_y)]

            if token == '[SEP]' and segm:
                segment_flag = True

        assert len(x) == len(y)
        x_axis.append(x)
        y_axis.append(y)
        bert_distill_flag.append(distill_flag)

        bert_tokens.append(bert_tokenizer.convert_tokens_to_ids(bert_token))
        segment_ids.append(segment_id)
        assert len(bert_tokens[-1]) == len(bert_token)

    bert_token_ids, bert_token_lens, attention_mask = merge_pad_idx(bert_tokens, bert_tokenizer.pad_token_id)
    segment_ids, _, _ = merge_pad_idx(segment_ids, bert_tokenizer.pad_token_id)

    max_ori_seq_len = max(ori_seq_lens)
    max_bert_seq_len = bert_token_ids.size(1)

    bert_to_ori_matrix = torch.zeros((batch_size, max_ori_seq_len, max_bert_seq_len)).float()
    for bi, (x, y) in enumerate(zip(x_axis, y_axis)):
        bert_to_ori_matrix[bi, x, y] = 1.0

    return bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens


def get_stc_embedding(bert_encodes, bert_distill_flag, batch_stc_lens, args_cuda=True):
    batch_size = len(bert_distill_flag)
    src_embeddings = []
    table_embeddings = []
    column_embeddings = []
    batch_s_len, batch_t_len, batch_c_len = [], [], []
    for bi in range(batch_size):
        bert_embedding = bert_encodes[bi, bert_distill_flag[bi]]
        s_len, t_len, c_len = batch_stc_lens[bi]
        src_embeddings.append(bert_embedding[: s_len])
        table_embeddings.append(bert_embedding[s_len: s_len + t_len])
        column_embeddings.append(bert_embedding[s_len + t_len: s_len + t_len + c_len])
        batch_s_len.append(s_len)
        batch_t_len.append(t_len)
        batch_c_len.append(c_len)

    src_embeddings = get_batch_embedding(src_embeddings, batch_s_len, args_cuda)
    table_embeddings = get_batch_embedding(table_embeddings, batch_t_len, args_cuda)
    column_embeddings = get_batch_embedding(column_embeddings, batch_c_len, args_cuda)

    return src_embeddings, table_embeddings, column_embeddings


def get_bert_style_input(src_sents, table_names, table_sents):
    batch_bert_seqs = []
    batch_stc_lens = []

    def get_str_from_list(l_sent):
        l_str = []
        for s in l_sent:
            # l_str.append('-'.join(s))
            l_str.append(''.join(s))
        return ' '.join(l_str), len(l_sent)

    for (src, table, column) in zip(src_sents, table_names, table_sents):
        src, s_len = get_str_from_list(src)
        table, t_len = get_str_from_list(table)
        column, c_len = get_str_from_list(column)
        bert_seq = '[CLS] ' + src + ' [SEP] ' + table + ' ' + column + ' [SEP]'

        batch_bert_seqs.append(bert_seq)
        batch_stc_lens.append((s_len, t_len, c_len))

    return batch_bert_seqs, batch_stc_lens


class IRNet(BasicModel):

    def __init__(self, args, grammar):
        super(IRNet, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer
        self.use_sentence_features = args.sentence_features
        self.layer_num = args.layer_num

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        # self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size // 2, bidirectional=True,
        #                             batch_first=True)
        #
        # self.schema_encoder_lstm = nn.LSTM(args.embed_size, args.embed_size // 2, bidirectional=True,
        #                                    batch_first=True)

        self.table_node = nn.Parameter(torch.Tensor(1, args.hidden_size))
        self.colum_node = nn.Parameter(torch.Tensor(1, args.hidden_size))

        etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 'fore_t2c', 'fore_c2t', 's2t', 't2s']
        # schema_enc_etypes = ['p2c', 'c2p']

        # self.schema_rgcn = HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size)
        # self.shadow_rgcn = HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size)

        self.schema_rgcn = HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, self.layer_num)
        self.shadow_rgcn = HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, self.layer_num)

        # self.schema_rgcn = HeteroGAT(etypes, args.hidden_size, args.hidden_size, self.layer_num)
        # self.shadow_rgcn = HeteroGAT(etypes, args.hidden_size, args.hidden_size, self.layer_num)

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased-model/')

        # self.bert_encoder = AlbertModel.from_pretrained('albert-xxlarge-v2/')
        # self.bert_encode_linear = nn.Linear(4096, args.hidden_size)

        # self.bert_encoder = ElectraModel.from_pretrained('electra_large_discriminator/')

        # self.bert_encoder = BertModel.from_pretrained('bert-large-uncase/')
        # self.bert_encode_linear = nn.Linear(1024, args.hidden_size)

        # self.bert_s_linear = nn.Linear(1024, args.hidden_size)
        # self.bert_t_linear = nn.Linear(1024, args.hidden_size)
        # self.bert_c_linear = nn.Linear(1024, args.hidden_size)

        self.parse_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 4, args.cuda)

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

        self.encoder_again_lstm = nn.LSTM(2 * args.hidden_size, args.hidden_size // 2, bidirectional=True,
                                          batch_first=True)

        self.schema_link_embed = nn.Embedding(9, args.hidden_size)
        self.parse_node_embed = nn.Embedding(75, args.hidden_size)

        # self.schema_link_linear = nn.Linear(args.hidden_size, 1)
        self.schema_link_linear = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )
        self.schema_link_drop = nn.Dropout(args.sldp)

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

        self.sketch_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 2, args.cuda)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        # tanh
        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.q_att = nn.Linear(args.hidden_size, args.embed_size)

        self.column_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.align_table_linear = nn.Linear(args.hidden_size, 1)
        self.align_column_linear = nn.Linear(args.hidden_size, 1)
        self.align_loss = nn.BCEWithLogitsLoss()

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.table_node.data)
        nn.init.xavier_normal_(self.colum_node.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)

    def sent_see_schema8(self, batch, src_encodings, table_embeddings, schema_embeddings, schema_link_drop=False):
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs
        schema_links = batch.schema_links

        assert len(src_sents_len) == len(table_len) == len(table_col_len)
        src_repre = []
        table_repre = []
        column_repre = []
        batch_max_src_schema_att_prob = []

        for bi, (ss_len, t_len, c_len) in enumerate(zip(src_sents_len, table_len, table_col_len)):
            ori_src_encoding = src_encodings[bi, :ss_len]
            column_embedding = table_embeddings[bi, :c_len]
            table_embedding = schema_embeddings[bi, :t_len]
            schema_link = schema_links[bi]

            schema_link = torch.tensor(schema_link)
            if torch.cuda.is_available():
                schema_link = schema_link.cuda()

            schema_link_embed = self.schema_link_embed(schema_link)
            if schema_link_drop:
                schema_link_embed = self.schema_link_drop(schema_link_embed)
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
            schema_item_encoding = self.schema_layer_norm(schema_item_encoding)

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            src_encoding = self.src_layer_norm(src_encoding)
            src_encoding_v = torch.relu(self.src_cross_att_v(ori_src_encoding))
            # src_schema_att_score = torch.sigmoid(torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0)))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            assert src_schema_att_score.shape == schema_link_score.shape
            src_schema_att_score = src_schema_att_score * schema_link_score
            src_schema_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=1)
            schema_src_att_prob = torch.transpose(src_schema_att_score.squeeze(0), 0, 1)
            schema_src_att_prob = torch.softmax(schema_src_att_prob, dim=1)

            # max_schema_src_att_idx = torch.argmax(src_schema_att_prob, dim=0)
            max_schema_src_att_idx = torch.argmax(schema_src_att_prob, dim=1)
            y_idx = torch.tensor([i for i in range(max_schema_src_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_src_schema_att_prob = src_schema_att_prob[max_schema_src_att_idx, y_idx].unsqueeze(1)
            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0)*max_src_schema_att_prob[:t_len],
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0)*max_src_schema_att_prob[t_len:]
            }

            hetero_shadow_output = self.shadow_rgcn(schema_graphs[bi], hetero_shadow_input)
            # hetero_shadow_output = self.schema_rgcn(schema_graphs[bi], hetero_shadow_input)
            shadow_item_encoding = torch.cat((hetero_shadow_output['table'], hetero_shadow_output['column']), dim=0)
            shadow_item_encoding_v = torch.relu(self.shadow_cross_att_v(shadow_item_encoding))

            shadow_src_ctx = torch.bmm(schema_src_att_prob.unsqueeze(0), src_encoding_v.unsqueeze(0)).squeeze(0)

            shadow_src_ctx = torch.cat((shadow_item_encoding, shadow_src_ctx), dim=1)
            shadow_src_ctx = torch.relu(self.shadow_src_cat_linear(shadow_src_ctx))

            table_shadow_encoding = shadow_src_ctx[:t_len]
            column_shadow_encoding = shadow_src_ctx[t_len:]

            src_shadow_ctx = torch.bmm(src_schema_att_prob.unsqueeze(0), shadow_item_encoding_v.unsqueeze(0)).squeeze(0)
            assert src_shadow_ctx.size(0) == ori_src_encoding.size(0)

            src_shadow_ctx_encoding = torch.cat((ori_src_encoding, src_shadow_ctx), dim=1)
            # src_shadow_ctx_encoding = torch.relu(self.src_shadow_cat_linear(src_shadow_ctx_encoding))

            max_src_schema_att_prob = max_src_schema_att_prob.squeeze(1)
            max_src_schema_att_prob_dict = {}
            max_src_schema_att_prob_dict['table'] = max_src_schema_att_prob[:t_len]
            max_src_schema_att_prob_dict['column'] = max_src_schema_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_src_schema_att_prob.append(max_src_schema_att_prob_dict)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_src_schema_att_prob

    def _get_align_loss(self, batch_align_score, schema_lens, ground_truth_onehots):
        align_loss = []
        for bi, schema_len in enumerate(schema_lens):
            onehot_vector = torch.tensor(ground_truth_onehots[bi]).unsqueeze(0)
            if self.args.cuda:
                onehot_vector = onehot_vector.cuda().float()
            align_score = batch_align_score[bi, :schema_len].squeeze(-1).unsqueeze(0)
            align_loss.append(self.align_loss(align_score, onehot_vector))
        align_loss = torch.stack(align_loss)
        return align_loss

    def forward(self, examples):
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        batch_bert_seqs, batch_stc_lens = \
            get_bert_style_input(batch.src_sents, batch.table_names, batch.table_sents)

        bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens = \
            merge_bert_data(batch_bert_seqs)

        if self.args.cuda:
            bert_token_ids = bert_token_ids.cuda()
            segment_ids = segment_ids.cuda()
            attention_mask = attention_mask.cuda()
            bert_to_ori_matrix = bert_to_ori_matrix.cuda()

        if bert_to_ori_matrix.size(2) > 511:
            print(batch_bert_seqs)
            print('idx greater than 512 !!!')
            exit(0)

        bert_src_schema_outputs = self.bert_encoder(bert_token_ids, attention_mask=attention_mask,
                                                    token_type_ids=segment_ids)
        # src_schema_outputs = self.bert_encode_linear(bert_src_schema_outputs[0])
        src_schema_outputs = bert_src_schema_outputs[0]
        src_schema_outputs = bert_to_ori_matrix.bmm(src_schema_outputs)
        # src_schema_outputs = self.bert_encode_linear(src_schema_outputs)

        ori_src_embedding, ori_schema_embedding, ori_table_embedding = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens)

        # src_encodings, _ = self.encode(ori_src_embedding, batch.src_sents_len, None, src_embed=True)
        src_encodings = ori_src_embedding
        # src_encodings = self.parse_encode(batch, ori_src_embedding)
        # src_encodings = get_batch_embedding(src_encodings, batch.src_sents_len, self.args.cuda)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema8(batch, src_encodings,
                                                                                  ori_table_embedding,
                                                                                  ori_schema_embedding, True)

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

        batch_t_lens = [stc[1] for stc in batch_stc_lens]
        batch_c_lens = [stc[2] for stc in batch_stc_lens]
        # table_align_score = self.align_table_linear(schema_embedding)
        # column_align_score = self.align_table_linear(table_embedding)
        table_align_score = torch.sigmoid(self.align_table_linear(schema_embedding))
        column_align_score = torch.sigmoid(self.align_table_linear(table_embedding))
        # print('#'*100)
        # print(column_align_score)
        # print(batch.align_column_one_hots)
        table_align_loss = self._get_align_loss(table_align_score, batch_t_lens, batch.align_table_one_hots)
        column_align_loss = self._get_align_loss(column_align_score, batch_c_lens, batch.align_column_one_hots)
        align_loss = table_align_loss + column_align_loss

        return align_loss

    def align_parse(self, examples):
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        batch_bert_seqs, batch_stc_lens = \
            get_bert_style_input(batch.src_sents, batch.table_names, batch.table_sents)

        bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens = \
            merge_bert_data(batch_bert_seqs)

        if self.args.cuda:
            bert_token_ids = bert_token_ids.cuda()
            segment_ids = segment_ids.cuda()
            attention_mask = attention_mask.cuda()
            bert_to_ori_matrix = bert_to_ori_matrix.cuda()

        if bert_to_ori_matrix.size(2) > 511:
            print(batch_bert_seqs)
            print('idx greater than 512 !!!')
            exit(0)

        bert_src_schema_outputs = self.bert_encoder(bert_token_ids, attention_mask=attention_mask,
                                                    token_type_ids=segment_ids)
        # src_schema_outputs = self.bert_encode_linear(bert_src_schema_outputs[0])
        src_schema_outputs = bert_src_schema_outputs[0]
        src_schema_outputs = bert_to_ori_matrix.bmm(src_schema_outputs)
        # src_schema_outputs = self.bert_encode_linear(src_schema_outputs)

        ori_src_embedding, ori_schema_embedding, ori_table_embedding = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens)

        # src_encodings, _ = self.encode(ori_src_embedding, batch.src_sents_len, None, src_embed=True)
        src_encodings = ori_src_embedding
        # src_encodings = self.parse_encode(batch, ori_src_embedding)
        # src_encodings = get_batch_embedding(src_encodings, batch.src_sents_len, self.args.cuda)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema8(batch, src_encodings,
                                                                                                  ori_table_embedding,
                                                                                                  ori_schema_embedding,
                                                                                                  True)


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

        batch_t_lens = [stc[1] for stc in batch_stc_lens]
        batch_c_lens = [stc[2] for stc in batch_stc_lens]
        # table_align_score = self.align_table_linear(schema_embedding).squeeze(-1)
        # column_align_score = self.align_table_linear(table_embedding).squeeze(-1)
        table_align_score = torch.sigmoid(self.align_table_linear(schema_embedding)).squeeze(-1)
        column_align_score = torch.sigmoid(self.align_table_linear(table_embedding)).squeeze(-1)

        table_align_scores = []
        column_align_scores = []
        table_ground_truth_ids = []
        column_ground_truth_ids = []
        for bi, (t_len, c_len) in enumerate(zip(batch_t_lens, batch_c_lens)):
            t_s = table_align_score[bi, :t_len].cpu().detach().numpy()
            table_align_scores.append(t_s)

            c_s = column_align_score[bi, :c_len].cpu().detach().numpy()
            column_align_scores.append(c_s)

            table_ground_truth_ids.append(np.array(batch.align_table_one_hots[bi]))
            column_ground_truth_ids.append(np.array(batch.align_column_one_hots[bi]))

        return table_align_scores, table_ground_truth_ids, column_align_scores, column_ground_truth_ids

