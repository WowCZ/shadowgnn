#!/usr/bin/env python
# @Time    : 2020-06-02 16:19
# @Author  : Zhi Chen
# @Desc    : bert_layer_wise

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import dgl
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.dataset import Batch
from src.models import nn_utils
from src.models.basic_model import BasicModel, HeteroRGCN, HeteroRelGCN, GATLayer, MultiHeadGATLayer, GAT, SublayerConnection, PositionwiseFeedForward
from src.models.pointer_net import PointerNet
from src.rule import semQL as define_rule
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1
from src.models.rat_encoder import RATEncoder

from transformers import BertModel, BertTokenizer, AlbertTokenizer, AlbertModel, ElectraModel, ElectraTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-model/')
# bert_tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2/')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncase/')
# bert_tokenizer = ElectraTokenizer.from_pretrained('electra_large_discriminator/')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking/')


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
            l_str.append('-'.join(s))
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

        self.table_node = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.colum_node = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))

        etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 'fore_t2c', 'fore_c2t', 'fore_c2c', 's2t', 't2s']
        schema_enc_etypes = ['p2c', 'c2p']
        dependency_etypes = ['advmod', 'amod', 'nsubj', 'case', 'nmod', 'cop', 'obl', 'punct', 'compound', 'conj', 'cc', 'dep', 'det', 'nmod:poss', 'obj', 'acl:relcl', 'nsubj:pass', 'aux:pass', 'acl', 'obl:npmod', 'nummod', 'expl', 'aux', 'discourse', 'fixed', 'xcomp', 'mark', 'ccomp', 'parataxis', 'cc:preconj', 'iobj', 'appos', 'advcl', 'obl:tmod', 'det:predet', 'csubj', 'compound:prt', 'csubj:pass', 'goeswith']
        # 9,19,43
        self.schema_link_embed = nn.Embedding(9, args.hidden_size)
        self.colset_type_embed = nn.Embedding(5, args.hidden_size)
        self.parse_node_embed = nn.Embedding(75, args.hidden_size)
        # self.schema_link_linear = nn.Linear(args.hidden_size, 1)
        self.tab_link_linear = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )
        self.col_link_linear = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )
        self.schema_link_drop = nn.Dropout(args.sldp)
        self.schema_layer_norm = nn.LayerNorm(args.hidden_size)
        self.src_layer_norm = nn.LayerNorm(args.hidden_size)

        self.schema_rgcn = []
        self.shadow_rgcn = []
        self.src_transformer_encode = []
        self.src_att_k_linear = []
        self.tab_link_q_linear = []
        self.col_link_q_linear = []
        self.tab_att_q_linear = []
        self.tab_att_v_linear = []
        self.shadow_tab_att_v_linear = []

        self.col_att_q_linear = []
        self.col_att_v_linear = []
        self.shadow_col_att_v_linear = []

        self.src_att_shadow_v_linear = []
        self.src_att_schema_v_linear = []

        self.src_shadow_ctx_score = []
        self.shadow_src_ctx_score = []
        self.schema_src_ctx_score = []

        self.tab_src_multiattention = []
        self.col_src_multiattention = []
        self.schema_src_attn_heads = 8
        self.src_trans_heads = 8


        for l in range(self.layer_num):
            self.schema_rgcn.append(HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, args.ave_layer))

            self.shadow_rgcn.append(HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, args.ave_layer))

            self.tab_src_multiattention.append(
                nn.MultiheadAttention(args.hidden_size, num_heads=self.schema_src_attn_heads, dropout=args.dropout))
            self.col_src_multiattention.append(
                nn.MultiheadAttention(args.hidden_size, num_heads=self.schema_src_attn_heads, dropout=args.dropout))

            # self.schema_rgcn.append(GATLayer(args.hidden_size, args.hidden_size // 8, etypes,
            #     8, activation=F.relu, self_loop=False, dropout=args.dropout))
            #
            # self.shadow_rgcn.append(GATLayer(args.hidden_size, args.hidden_size // 8, etypes,
            #     8, activation=F.relu, self_loop=False, dropout=args.dropout))

            self.src_transformer_encode.append(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=self.src_trans_heads))
            # self.src_transformer_encode.append(MultiHeadGATLayer(args.hidden_size, args.hidden_size // 4, 4))

            self.src_att_k_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            # self.schema_link_q_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.tab_link_q_linear.append(nn.Linear(args.hidden_size, 1))
            self.col_link_q_linear.append(nn.Linear(args.hidden_size, 1))

            self.tab_att_q_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.tab_att_v_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.shadow_tab_att_v_linear.append(nn.Linear(args.hidden_size, args.hidden_size))

            self.col_att_q_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.col_att_v_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.shadow_col_att_v_linear.append(nn.Linear(args.hidden_size, args.hidden_size))

            self.src_att_shadow_v_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
            self.src_att_schema_v_linear.append(nn.Linear(args.hidden_size, args.hidden_size))

            self.src_shadow_ctx_score.append(nn.Linear(args.hidden_size, 1))
            self.shadow_src_ctx_score.append(nn.Linear(args.hidden_size, 1))
            self.schema_src_ctx_score.append(nn.Linear(args.hidden_size, 1))

        if args.cuda:
            for l in range(self.layer_num):
                self.schema_rgcn[l] = self.schema_rgcn[l].cuda()
                self.shadow_rgcn[l] = self.shadow_rgcn[l].cuda()
                self.src_transformer_encode[l] = self.src_transformer_encode[l].cuda()
                self.src_att_k_linear[l] = self.src_att_k_linear[l].cuda()
                self.tab_link_q_linear[l] = self.tab_link_q_linear[l].cuda()
                self.col_link_q_linear[l] = self.col_link_q_linear[l].cuda()
                self.tab_att_q_linear[l] = self.tab_att_q_linear[l].cuda()
                self.tab_att_v_linear[l] = self.tab_att_v_linear[l].cuda()
                self.shadow_tab_att_v_linear[l] = self.shadow_tab_att_v_linear[l].cuda()

                self.col_att_q_linear[l] = self.col_att_q_linear[l].cuda()
                self.col_att_v_linear[l] = self.col_att_v_linear[l].cuda()
                self.shadow_col_att_v_linear[l] = self.shadow_col_att_v_linear[l].cuda()
                self.src_att_shadow_v_linear[l] = self.src_att_shadow_v_linear[l].cuda()
                self.src_att_schema_v_linear[l] = self.src_att_schema_v_linear[l].cuda()

                self.src_shadow_ctx_score[l] = self.src_shadow_ctx_score[l].cuda()
                self.shadow_src_ctx_score[l] = self.shadow_src_ctx_score[l].cuda()
                self.schema_src_ctx_score[l] = self.schema_src_ctx_score[l].cuda()

                self.tab_src_multiattention[l] = self.tab_src_multiattention[l].cuda()
                self.col_src_multiattention[l] = self.col_src_multiattention[l].cuda()

        self.align_table_linear = nn.Linear(args.hidden_size, 1)
        self.align_column_linear = nn.Linear(args.hidden_size, 1)

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased-model/')
        self.bert_encode_linear = nn.Linear(768, args.hidden_size)

        # self.bert_encoder = AlbertModel.from_pretrained('albert-xxlarge-v2/')
        # self.bert_encode_linear = nn.Linear(4096, args.hidden_size)

        # self.bert_encoder = BertModel.from_pretrained('bert-large-uncase/')
        # self.bert_encode_linear = nn.Linear(1024, args.hidden_size)

        # self.bert_encoder = ElectraModel.from_pretrained('electra_large_discriminator/')
        # self.bert_encode_linear = nn.Linear(1024, args.hidden_size)

        # self.bert_encoder = BertModel.from_pretrained('bert-large-uncased-whole-word-masking/')
        # self.bert_encode_linear = nn.Linear(1024, args.hidden_size)

        self.rat_encoder = RATEncoder(args)

        self.encoder_again_lstm = nn.LSTM(args.hidden_size, args.hidden_size // 2, bidirectional=True,
                                          batch_first=True)

        self.parse_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 4, args.cuda)

        input_dim = args.action_embed_size + \
                    args.att_vec_size + \
                    args.type_embed_size

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

        # self.sketch_graph_enc = HeteroRelGCN(schema_enc_etypes, args.action_embed_size, args.action_embed_size, self.layer_num)
        self.sketch_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 2, args.cuda)

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

        self.align_loss = nn.BCEWithLogitsLoss()

        self.relu = nn.ReLU(inplace=False)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.table_node.data)
        nn.init.xavier_normal_(self.colum_node.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)

    def one_layer_see_schema(self, src_embedding, column_embedding, table_embedding, shadow_c_emb, shadow_t_emb,
                             schema_graph, schema_linking_q, src_schema_mask, src_pad_mask, t_lens, c_lens, layer_id):
        max_t_len = table_embedding.size(1)
        max_c_len = column_embedding.size(1)
        max_src_len = src_embedding.size(1)

        assert schema_linking_q.size(1) == max_src_len and schema_linking_q.size(2) == max_t_len + max_c_len

        tab_linking_attn = schema_linking_q[:, :, :max_t_len]
        col_linking_attn = schema_linking_q[:, :, max_t_len:]
        tab_linking_attn = torch.cat([tab_linking_attn.transpose(1, 2) for _ in range(self.schema_src_attn_heads)], dim=0)
        col_linking_attn = torch.cat([col_linking_attn.transpose(1, 2) for _ in range(self.schema_src_attn_heads)], dim=0)

        src_tab_mask = src_schema_mask[:, :, :max_t_len]
        src_col_mask = src_schema_mask[:, :, max_t_len:]
        tab_src_mask = torch.transpose(src_tab_mask, 1, 2)
        col_src_mask = torch.transpose(src_col_mask, 1, 2)

        src_encoding_k = torch.relu(self.src_att_k_linear[layer_id](src_embedding))
        src_encoding_shadow_v = torch.relu(self.src_att_shadow_v_linear[layer_id](src_embedding))
        src_encoding_schema_v = torch.relu(self.src_att_schema_v_linear[layer_id](src_embedding))

        tab_item_encoding_q = torch.relu(self.tab_att_q_linear[layer_id](table_embedding))
        tab_item_encoding_v = torch.relu(self.tab_att_v_linear[layer_id](table_embedding))
        col_item_encoding_q = torch.relu(self.col_att_q_linear[layer_id](column_embedding))
        col_item_encoding_v = torch.relu(self.col_att_v_linear[layer_id](column_embedding))
        schema_item_encoding_v = torch.cat((tab_item_encoding_v, col_item_encoding_v), dim=1)

        # print(table_embedding.shape)
        # print(src_embedding.shape)
        # print(tab_linking_attn.shape)
        # exit(0)

        tab_src_ctx, tab_src_weights = self.tab_src_multiattention[layer_id](tab_item_encoding_q.transpose(0, 1),
                                                                             src_encoding_k.transpose(0, 1),
                                                                             src_encoding_schema_v.transpose(0, 1),
                                                                             key_padding_mask=src_pad_mask,
                                                                             attn_mask=tab_linking_attn)
        tab_src_ctx = tab_src_ctx.transpose(0, 1)
        tab_src_weights = tab_src_weights.masked_fill(tab_src_mask.bool(), -np.inf)

        col_src_ctx, col_src_weights = self.col_src_multiattention[layer_id](col_item_encoding_q.transpose(0, 1),
                                                                             src_encoding_k.transpose(0, 1),
                                                                             src_encoding_schema_v.transpose(0, 1),
                                                                             key_padding_mask=src_pad_mask,
                                                                             attn_mask=col_linking_attn)
        col_src_ctx = col_src_ctx.transpose(0, 1)
        col_src_weights = col_src_weights.masked_fill(col_src_mask.bool(), -np.inf)

        schema_src_ctx = torch.cat((tab_src_ctx, col_src_ctx), dim=1)
        schema_src_ctx_score = torch.sigmoid(self.schema_src_ctx_score[layer_id](schema_src_ctx))
        schema_src_ctx = (1 - schema_src_ctx_score) * schema_item_encoding_v + schema_src_ctx_score * schema_src_ctx

        src_tab_att_prob = torch.softmax(tab_src_weights.transpose(1, 2), dim=2)
        src_col_att_prob = torch.softmax(col_src_weights.transpose(1, 2), dim=2)
        # print(src_tab_att_prob)
        # print(src_col_att_prob)
        src_schema_att_prob = torch.cat((src_tab_att_prob, src_col_att_prob), dim=2)
        src_schema_att_prob = src_schema_att_prob.masked_fill(src_schema_mask, 0)
        max_src_schema_att_prob, _ = torch.max(src_schema_att_prob.transpose(1, 2), dim=-1)
        max_src_schema_att_prob = max_src_schema_att_prob.unsqueeze(-1)

        shadow_t_emb_weight = shadow_t_emb * max_src_schema_att_prob[:, :max_t_len]
        shadow_c_emb_weight = shadow_c_emb * max_src_schema_att_prob[:, max_t_len:]

        schema_src_attn_weights = torch.cat((tab_src_weights, col_src_weights), dim=1)
        schema_src_att_prob = torch.softmax(schema_src_attn_weights, dim=2)
        schema_src_att_prob = schema_src_att_prob.masked_fill(src_schema_mask.transpose(1, 2), 0)

        shadow_tab_item_encoding_v = torch.relu(self.shadow_tab_att_v_linear[layer_id](shadow_t_emb_weight))
        shadow_col_item_encoding_v = torch.relu(self.shadow_col_att_v_linear[layer_id](shadow_c_emb_weight))
        shadow_item_encoding_v = torch.cat((shadow_tab_item_encoding_v, shadow_col_item_encoding_v), dim=1)

        shadow_src_ctx = torch.bmm(schema_src_att_prob, src_encoding_shadow_v)
        shadow_src_ctx_score = torch.sigmoid(self.shadow_src_ctx_score[layer_id](shadow_src_ctx))
        shadow_src_ctx = (1 - shadow_src_ctx_score) * shadow_item_encoding_v + shadow_src_ctx_score * shadow_src_ctx

        src_shadow_ctx = torch.bmm(src_schema_att_prob, shadow_item_encoding_v)
        src_shadow_ctx_score = torch.sigmoid(self.src_shadow_ctx_score[layer_id](src_shadow_ctx))
        src_shadow_ctx = (1 - src_shadow_ctx_score) * src_encoding_shadow_v + src_shadow_ctx_score * src_shadow_ctx

        # def get_dgl_style(batch_t_len, batch_c_len, batch_schema_input):
        #     batch_t_x = []
        #     batch_t_y = []
        #     batch_c_x = []
        #     batch_c_y = []
        #     for bi, (t_len, c_len) in enumerate(batch_t_len, batch_c_len):
        #         for ti in range(t_len):
        #             batch_t_x.append(bi)
        #             batch_t_y.append(ti)
        #
        #         for ti in range(c_len):
        #             batch_c_x.append(bi)
        #             batch_c_y.append(ti+t_len)
        #
        #     dgl_tab_input = batch_schema_input[batch_t_x, batch_t_y]
        #     dgl_col_input = batch_schema_input[batch_c_x, batch_c_y]
        #
        #     return dgl_tab_input, dgl_col_input

        batch_t_x = []
        batch_t_y = []
        batch_c_x = []
        batch_c_y = []
        for bi, (t_len, c_len) in enumerate(zip(t_lens, c_lens)):
            for ti in range(t_len):
                batch_t_x.append(bi)
                batch_t_y.append(ti)

            for ti in range(c_len):
                batch_c_x.append(bi)
                batch_c_y.append(ti + t_len)

        dgl_tab_shadow_input = shadow_src_ctx[batch_t_x, batch_t_y]
        dgl_col_shadow_input = shadow_src_ctx[batch_c_x, batch_c_y]

        dgl_tab_schema_input = schema_src_ctx[batch_t_x, batch_t_y]
        dgl_col_schema_input = schema_src_ctx[batch_c_x, batch_c_y]

        hetero_schema_input = {
            'table': dgl_tab_schema_input,
            'column': dgl_col_schema_input
        }

        hetero_schema_output = self.schema_rgcn[layer_id](schema_graph, hetero_schema_input)

        src_shadow_ctx = self.src_transformer_encode[layer_id](src_shadow_ctx.transpose(0, 1), src_key_padding_mask=src_pad_mask)
        src_shadow_ctx = src_shadow_ctx.transpose(0, 1)
        # res
        src_shadow_ctx = src_shadow_ctx + src_embedding

        hetero_shadow_input = {
            'table': dgl_tab_shadow_input,
            'column': dgl_col_shadow_input
        }

        hetero_shadow_output = self.shadow_rgcn[layer_id](schema_graph, hetero_shadow_input)

        shadow_encoding = torch.zeros_like(shadow_src_ctx).float()
        schema_encoding = torch.zeros_like(schema_src_ctx).float()

        if self.args.cuda:
            shadow_encoding = shadow_encoding.cuda()
            schema_encoding = schema_encoding.cuda()

        shadow_encoding[batch_t_x, batch_t_y] = hetero_shadow_output['table']
        shadow_encoding[batch_c_x, batch_c_y] = hetero_shadow_output['column']

        schema_encoding[batch_t_x, batch_t_y] = hetero_schema_output['table']
        schema_encoding[batch_c_x, batch_c_y] = hetero_schema_output['column']

        # res
        table_shadow_encoding = shadow_encoding[:, :max_t_len] + shadow_t_emb
        column_shadow_encoding = shadow_encoding[:, max_t_len:] + shadow_c_emb
        table_schema_encoding = schema_encoding[:, :max_t_len] + table_embedding
        column_schema_encoding = schema_encoding[:, max_t_len:] + column_embedding

        max_src_schema_att_prob = max_src_schema_att_prob.squeeze(-1)
        max_src_schema_att_prob_dict = {}
        max_src_schema_att_prob_dict['table'] = max_src_schema_att_prob[:max_t_len]
        max_src_schema_att_prob_dict['column'] = max_src_schema_att_prob[max_t_len:]

        return src_shadow_ctx, table_schema_encoding, column_schema_encoding, table_shadow_encoding, \
               column_shadow_encoding, max_src_schema_att_prob_dict, src_schema_att_prob

    def sent_see_schema(self, batch, src_encodings, table_embeddings, schema_embeddings):
        batch_size = len(batch.src_sents_len)
        src_sents_len = batch.src_sents_len
        table_len = batch.table_len
        table_col_len = batch.col_num
        schema_graphs = batch.schema_graphs
        schema_links = batch.schema_links
        relative_matirxs = batch.relative_matrixs

        # get emb differ
        embedding_differ = self.embedding_cosine(src_embedding=src_encodings, table_embedding=table_embeddings,
                                                 table_unk_mask=batch.table_unk_mask)

        schema_differ = self.embedding_cosine(src_embedding=src_encodings, table_embedding=schema_embeddings,
                                              table_unk_mask=batch.schema_token_mask)

        tab_ctx = (src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        init_table_embedding = tab_ctx

        init_schema_embedding = schema_ctx

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        init_table_embedding = init_table_embedding + col_type_var

        assert len(src_sents_len) == len(table_len) == len(table_col_len)

        max_t_len = max(table_len)
        max_c_len = max(table_col_len)
        max_ss_len = max(src_sents_len)

        shadow_t_emb = torch.repeat_interleave(torch.repeat_interleave(self.table_node, max_t_len, dim=1),
                                               batch_size, dim=0)
        shadow_t_emb = shadow_t_emb + init_schema_embedding
        shadow_c_emb = torch.repeat_interleave(torch.repeat_interleave(self.colum_node, max_c_len, dim=1),
                                               batch_size, dim=0)
        shadow_c_emb = shadow_c_emb + init_table_embedding

        schema_link_weight = torch.ones((batch_size, max_ss_len, max_t_len+max_c_len)).float()
        src_schema_mask = torch.ones_like(schema_link_weight).byte()
        src_pad_mask = torch.ones((batch_size, max_ss_len)).byte()
        rat_pad_mask = torch.ones((batch_size, max_ss_len+max_t_len+max_c_len)).byte()
        relative_pad_matirxs = torch.zeros(
            (batch_size, max_ss_len + max_t_len + max_c_len, max_ss_len + max_t_len + max_c_len)).long()
        if self.args.cuda:
            schema_link_weight = schema_link_weight.cuda()
            src_schema_mask = src_schema_mask.cuda()
            src_pad_mask = src_pad_mask.cuda()
            rat_pad_mask = rat_pad_mask.cuda()
            relative_pad_matirxs = relative_pad_matirxs.cuda()

        colset_text_types = []
        for bi, t_len in enumerate(table_len):
            c_len = table_col_len[bi]
            ss_len = src_sents_len[bi]
            relative_matirx = relative_matirxs[bi]
            src_pad_mask[bi, :src_sents_len[bi]] = 0
            colset_type_id = batch.colset_text_types[bi]
            colset_type_id = torch.tensor(colset_type_id)
            schema_link = schema_links[bi]
            schema_link = torch.tensor(schema_link)
            if self.args.cuda:
                colset_type_id = colset_type_id.cuda()
                schema_link = schema_link.cuda()
            c_type_emb = self.colset_type_embed(colset_type_id)
            schema_link_embed = self.schema_link_embed(schema_link)
            tab_link_embed = schema_link_embed[:, :t_len]
            col_link_embed = schema_link_embed[:, t_len:]

            tab_link_weight = self.tab_link_linear(tab_link_embed).squeeze(-1)
            col_link_weight = self.col_link_linear(col_link_embed).squeeze(-1)

            schema_link_weight[bi, :ss_len, :t_len] = tab_link_weight
            src_schema_mask[bi, :ss_len, :t_len] = 0
            schema_link_weight[bi, :ss_len, max_t_len:max_t_len+c_len] = col_link_weight
            src_schema_mask[bi, :ss_len, max_t_len:max_t_len+c_len] = 0

            rat_pad_mask[bi, :ss_len] = 0
            rat_pad_mask[bi, max_ss_len: max_ss_len+c_len] = 0
            rat_pad_mask[bi, max_ss_len + max_c_len: max_ss_len + max_c_len + table_len[bi]] = 0

            # Q -> Q C T
            relative_pad_matirxs[bi, :ss_len, :ss_len] = relative_matirx[:ss_len, :ss_len]
            relative_pad_matirxs[bi, :ss_len, max_ss_len :max_ss_len + c_len] = relative_matirx[:ss_len,
                                                                               ss_len:ss_len + c_len]
            relative_pad_matirxs[bi, :ss_len, max_ss_len+max_c_len:max_ss_len+max_c_len+t_len] = \
                relative_matirx[:ss_len, ss_len + c_len:]

            # C -> Q C T
            relative_pad_matirxs[bi, max_ss_len: max_ss_len + c_len, :ss_len] = relative_matirx[ss_len:ss_len + c_len,
                                                                                :ss_len]
            relative_pad_matirxs[bi, max_ss_len: max_ss_len + c_len, max_ss_len:max_ss_len + c_len] = \
                relative_matirx[ss_len:ss_len + c_len, ss_len:ss_len + c_len]
            relative_pad_matirxs[bi, max_ss_len: max_ss_len + c_len,
            max_ss_len + max_c_len:max_ss_len + max_c_len + t_len] = \
                relative_matirx[ss_len:ss_len + c_len, ss_len + c_len:]

            # T -> Q C T
            relative_pad_matirxs[bi, max_ss_len + max_c_len: max_ss_len + max_c_len + t_len, :ss_len] = relative_matirx[
                                                                                                        ss_len + c_len:,
                                                                                                        :ss_len]
            relative_pad_matirxs[bi, max_ss_len + max_c_len: max_ss_len + max_c_len + t_len,
            max_ss_len:max_ss_len + c_len] = \
                relative_matirx[ss_len + c_len:, ss_len:ss_len + c_len]
            relative_pad_matirxs[bi, max_ss_len + max_c_len: max_ss_len + max_c_len + t_len,
            max_ss_len + max_c_len:max_ss_len + max_c_len + t_len] = \
                relative_matirx[ss_len + c_len:, ss_len + c_len:]

            colset_text_types.append(c_type_emb)

        c_type_emb = get_batch_embedding(colset_text_types, table_col_len, self.args.cuda)
        shadow_c_emb = shadow_c_emb + c_type_emb

        schema_graph = dgl.batch_hetero(schema_graphs)
        if self.args.cuda:
            schema_graph = schema_graph.to(torch.device('cuda:0'))

        table_encoding = schema_embeddings
        column_encoding = table_embeddings

        src_schema_mask = src_schema_mask.bool()
        src_pad_mask = src_pad_mask.bool()
        rat_pad_mask = rat_pad_mask.bool()

        for l_id in range(self.layer_num):
            src_encodings, table_encoding, column_encoding, shadow_t_emb, shadow_c_emb, max_schema_arc_att_prob, src_schema_att_prob = \
                self.one_layer_see_schema(src_encodings, column_encoding, table_encoding, shadow_c_emb, shadow_t_emb,
                             schema_graph, schema_link_weight, src_schema_mask, src_pad_mask, table_len, table_col_len, l_id)

        stc_rat_input = torch.cat((src_encodings, shadow_c_emb, shadow_t_emb), dim=1)
        stc_encoding = self.rat_encoder(stc_rat_input, relative_pad_matirxs, rat_pad_mask)
        # stc_encoding = stc_rat_input

        src_encoding = stc_encoding[:, :max_ss_len]
        table_encoding = stc_encoding[:, max_ss_len + max_c_len:]
        column_encoding = stc_encoding[:, max_ss_len:max_ss_len + max_c_len]

        return src_encoding, table_encoding, column_encoding, max_schema_arc_att_prob

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
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask

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
        src_schema_outputs = bert_src_schema_outputs[0]
        src_schema_outputs = bert_to_ori_matrix.bmm(src_schema_outputs)
        src_schema_outputs = self.bert_encode_linear(src_schema_outputs)

        ori_src_embedding, ori_schema_embedding, ori_table_embedding = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema(batch, ori_src_embedding,
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

        sketch_prob_var = torch.stack(
                [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        batch_t_lens = [stc[1] for stc in batch_stc_lens]
        batch_c_lens = [stc[2] for stc in batch_stc_lens]
        # table_align_score = self.align_table_linear(schema_embedding)
        # column_align_score = self.align_table_linear(table_embedding)
        table_align_score = torch.sigmoid(self.align_table_linear(schema_embedding))
        column_align_score = torch.sigmoid(self.align_table_linear(table_embedding))
        table_align_loss = self._get_align_loss(table_align_score, batch_t_lens, batch.align_table_one_hots)
        column_align_loss = self._get_align_loss(column_align_score, batch_c_lens, batch.align_column_one_hots)
        align_loss = table_align_loss + column_align_loss

        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]

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
                            # a_tm1_embed = batch_sketch_encoder_output[e_id][batch_sketch_flag[e_id], :]
                            batch_sketch_flag[e_id] += 1
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
                        # print('C:', act_prob_t_i)

                        # max_att_ct_probs[e_id].append(batch_max_att_p[e_id]['column'][action_t.id_c])
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                        # print('#'*100)
                        # print(example.src_sent)
                        # print(example.table_names)
                        # print(example.table_col_name)
                        # print('e_id:', e_id)
                        # print(action_t.id_c)
                        # print('T:', act_prob_t_i)
                        # print(table_dict[e_id])

                        # max_att_ct_probs[e_id].append(batch_max_att_p[e_id]['table'][action_t.id_c])
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                        # print('A:', act_prob_t_i)
                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        att_prob_var = 0

        return [sketch_prob_var, lf_prob_var, att_prob_var, align_loss]

    def forward_ssl(self, examples):
        batch_size = len(examples)
        # print(examples[0].src_sent)
        # print(examples[1].src_sent)
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask

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
        src_schema_outputs = bert_src_schema_outputs[0]
        src_schema_outputs = bert_to_ori_matrix.bmm(src_schema_outputs)
        src_schema_outputs = self.bert_encode_linear(src_schema_outputs)

        ori_src_embedding, ori_schema_embedding, ori_table_embedding = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema(batch,
                                                                                                 ori_src_embedding,
                                                                                                 ori_table_embedding,
                                                                                                 ori_schema_embedding)

        ave_schema_embedding = torch.mean(schema_embedding, dim=0)
        ave_schema_embedding = torch.repeat_interleave(ave_schema_embedding.unsqueeze(0), batch_size, dim=0)

        # print(schema_embedding.size())
        # print(ave_schema_embedding.size())

        assert schema_embedding.size() == ave_schema_embedding.size()
        schema_similarity = F.cosine_similarity(schema_embedding, ave_schema_embedding)
        schema_similarity_loss = -torch.sum(schema_similarity, dim=1)

        ave_table_embedding = torch.mean(table_embedding, dim=0)
        ave_table_embedding = torch.repeat_interleave(ave_table_embedding.unsqueeze(0), batch_size, dim=0)

        # print(table_embedding.size())
        # print(ave_table_embedding.size())

        assert table_embedding.size() == ave_table_embedding.size()
        table_similarity = F.cosine_similarity(table_embedding, ave_table_embedding)
        table_similarity_loss = -torch.sum(table_similarity, dim=1)

        similarity_loss = schema_similarity_loss + table_similarity_loss

        # print(similarity_loss.size())
        # exit(0)

        return similarity_loss


    def parse(self, examples, beam_size=5):
        """
        one example a time
        :param examples:
        :param beam_size:
        :return:
        """
        batch = Batch([examples], self.grammar, cuda=self.args.cuda)

        batch_bert_seqs, batch_stc_lens = \
            get_bert_style_input(batch.src_sents, batch.table_names, batch.table_sents)

        bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens = \
            merge_bert_data(batch_bert_seqs)

        if self.args.cuda:
            bert_token_ids = bert_token_ids.cuda()
            segment_ids = segment_ids.cuda()
            attention_mask = attention_mask.cuda()
            bert_to_ori_matrix = bert_to_ori_matrix.cuda()

        bert_src_schema_outputs = self.bert_encoder(bert_token_ids, attention_mask=attention_mask,
                                                    token_type_ids=segment_ids)
        src_schema_outputs = bert_src_schema_outputs[0]
        src_schema_outputs = bert_to_ori_matrix.bmm(src_schema_outputs)
        src_schema_outputs = self.bert_encode_linear(src_schema_outputs)

        ori_src_embedding, ori_schema_embedding, ori_table_embedding = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens)

        src_encodings, schema_embedding, table_embedding, batch_max_att_p = self.sent_see_schema(batch, ori_src_embedding,
                                                                                  ori_table_embedding,
                                                                                  ori_schema_embedding)

        src_encodings, (last_state, last_cell) = self.encode_again(src_encodings, batch.src_sents_len)

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

        padding_sketch = self.padding_sketch(sketch_actions)

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
            else:
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)

            column_selection_log_prob = F.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

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

