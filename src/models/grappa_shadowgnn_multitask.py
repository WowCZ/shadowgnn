#!/usr/bin/env python
# @Time    : 2020-06-02 16:19
# @Author  : Zhi Chen
# @Desc    : bert_layer_wise

# -*- coding: utf-8 -*-
import numpy as np
import torch
import dgl
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.dataset import Batch
from src.models import nn_utils
from src.models.basic_model import BasicModel, HeteroRelGCN
from src.models.pointer_net import PointerNet
from src.rule import semQLPro as define_rule
from src.models.rat_encoder import RATEncoder, RelationAwareAttention

from preprocess.utils import wordnet_lemmatizer
from pattern.en import lemma

from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaModel

bert_tokenizer = AutoTokenizer.from_pretrained("grappa_large_jnt")
sql_hardness_list = ["easy", "medium", "hard", "extra"]


def Lemmanize(x):
    y = [lemma(wordnet_lemmatizer.lemmatize(x_item.lower())) for x_item in x.split('_')]
    return ' '.join(y)


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


def merge_roberta_data(sequences, segm=True):
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
            if token != '<s>' and token != '</s>':
                distill_flag.append(ti)
            cur_bert_tokens = bert_tokenizer.tokenize(token.replace("-", " "))
            start_y = len(bert_token)
            bert_token = bert_token + cur_bert_tokens
            if segment_flag:
                segment_id = segment_id + [1] * len(cur_bert_tokens)
            else:
                segment_id = segment_id + [0] * len(cur_bert_tokens)
            end_y = len(bert_token)
            x = x + [ti] * len(cur_bert_tokens)
            y = y + [yi for yi in range(start_y, end_y)]

            if token == '</s>' and segm:
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

    sum_to_ori_matrix = torch.repeat_interleave(bert_to_ori_matrix.sum(dim=2).unsqueeze(2), bert_to_ori_matrix.size(2),
                                                dim=2) + 1e-7

    bert_to_ori_matrix = bert_to_ori_matrix / sum_to_ori_matrix

    return bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens


def get_stc_embedding(bert_encodes, bert_distill_flag, batch_stc_lens, batch_tab_col_match, args_cuda=True):
    batch_size = len(bert_distill_flag)
    src_embeddings = []
    column_embeddings = []
    batch_s_len, batch_t_len, batch_c_len = [], [], []
    x_axis, y_axis = [], []
    max_tab_len = 0
    max_col_len = 0
    for bi in range(batch_size):
        bert_embedding = bert_encodes[bi, bert_distill_flag[bi]]
        s_len, c_len = batch_stc_lens[bi]
        src_embeddings.append(bert_embedding[: s_len])
        column_embeddings.append(bert_embedding[s_len: s_len + c_len])
        batch_s_len.append(s_len)
        batch_c_len.append(c_len)

        tab_col_match = batch_tab_col_match[bi]
        col_len = len(tab_col_match)
        tab_len = max(tab_col_match) + 1

        if col_len > max_col_len:
            max_col_len = col_len

        if tab_len > max_tab_len:
            max_tab_len = tab_len

        x, y = [], []
        for col_id, t_id in enumerate(tab_col_match):
            if t_id != -1:
                x.append(t_id)
                y.append(col_id)

        x_axis.append(x)
        y_axis.append(y)

    col_to_tab_matrix = torch.zeros((batch_size, max_tab_len, max_col_len)).float()
    for bi, (x, y) in enumerate(zip(x_axis, y_axis)):
        col_to_tab_matrix[bi, x, y] = 1.0

    sum_to_ori_matrix = torch.repeat_interleave(col_to_tab_matrix.sum(dim=2).unsqueeze(2), col_to_tab_matrix.size(2),
                                                dim=2) + 1e-7

    col_to_tab_matrix = col_to_tab_matrix / sum_to_ori_matrix

    src_embeddings = get_batch_embedding(src_embeddings, batch_s_len, args_cuda)
    column_embeddings = get_batch_embedding(column_embeddings, batch_c_len, args_cuda)

    if args_cuda:
        col_to_tab_matrix = col_to_tab_matrix.cuda()

    table_embeddings = col_to_tab_matrix.bmm(column_embeddings)

    return src_embeddings, table_embeddings, column_embeddings


def get_roberta_style_input(src_sents, table_names, table_sents, tab_col_matches):
    batch_roberta_seqs = []
    batch_stc_lens = []

    def get_str_from_list(l_sent):
        l_str = []
        for s in l_sent:
            l_str.append('-'.join(s))
        return ' '.join(l_str), len(l_sent)

    new_table_sents = []
    for table_name, table_sent, tab_col_match in zip(table_names, table_sents, tab_col_matches):
        assert len(table_sent) == len(tab_col_match)
        new_table_sent = []
        new_table_sent.append(table_sent[0])

        for col, tab_id in zip(table_sent, tab_col_match):
            if tab_id != -1:
                if col[0] == 'id' or col[0] == 'name':
                    new_table_sent.append(table_name[tab_id] + col)
                else:
                    new_table_sent.append(col)

        new_table_sents.append(new_table_sent)

    for (src, column) in zip(src_sents, new_table_sents):
        src, s_len = get_str_from_list(src)
        column, c_len = get_str_from_list(column)
        column = column.replace(' ', ' </s> ')
        roberta_seq = '<s> ' + src + ' </s> ' + column + ' </s>'

        batch_roberta_seqs.append(roberta_seq)
        batch_stc_lens.append((s_len, c_len))

    return batch_roberta_seqs, batch_stc_lens


class Shadowgnn(BasicModel):

    def __init__(self, args, grammar, vocab):
        super(Shadowgnn, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer
        self.use_sentence_features = args.sentence_features
        self.layer_num = args.layer_num
        self.vocab_len = len(vocab)
        self.vocab = {}
        for k, v in vocab.items():
            self.vocab[v] = k

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.table_node = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.colum_node = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))

        etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 'fore_c2c', 'fore_invc2c', 's2t', 't2s']

        # 9,19,43
        self.schema_link_embed = nn.Embedding(9, args.hidden_size)
        self.colset_type_embed = nn.Embedding(5, args.hidden_size)
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

            self.src_transformer_encode.append(
                nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=self.src_trans_heads))

            self.src_att_k_linear.append(nn.Linear(args.hidden_size, args.hidden_size))
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

        self.bert_encoder = RobertaModel.from_pretrained("grappa_large_jnt", return_dict=True)
        self.bert_encode_linear = nn.Linear(1024, args.hidden_size)

        self.encoder_again_lstm = nn.LSTM(args.hidden_size, args.hidden_size // 2, bidirectional=True,
                                          batch_first=True)

        # self.parse_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 4, args.cuda)

        self.rat_encoder = RATEncoder(args)

        self.col_memory_align = RelationAwareAttention(args.hidden_size, 1, args.hidden_size,
                                                       args.hidden_size, args.hidden_size, 0,
                                                       relation_types=args.rat_relation_types)

        self.table_memory_align = RelationAwareAttention(args.hidden_size, 1, args.hidden_size,
                                                       args.hidden_size, args.hidden_size, 0,
                                                       relation_types=args.rat_relation_types)

        self.table_memory_align.key_relation_embed = self.col_memory_align.key_relation_embed
        self.table_memory_align.value_relation_embed = self.col_memory_align.value_relation_embed

        self.final_select_col_wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.final_select_col_wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.final_select_tab_wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.final_select_tab_wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.rephrase_select_col_wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.rephrase_select_col_wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.rephrase_select_tab_wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.rephrase_select_tab_wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        input_dim = args.action_embed_size + \
                    args.hidden_size + \
                    args.type_embed_size

        self.step_input_dim = args.action_embed_size + \
                              args.hidden_size + \
                              args.type_embed_size

        self.lf_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.rephrase_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)
        self.to_dec_att = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=8)
        self.enc_dec = nn.Linear(args.hidden_size, args.hidden_size)

        self.att_sketch_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.att_rephrase_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)
        self.rephrase_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(4, args.col_embed_size)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        # tanh
        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)


        # rephrase
        self.vocab_embed = nn.Embedding(self.vocab_len, args.action_embed_size)
        self.rephrase_type_embed = nn.Embedding(3, args.type_embed_size)
        self.vocab_readout_b = nn.Parameter(torch.FloatTensor(self.vocab_len).zero_())

        self.rephrase_query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.vocab_readout = lambda q: F.linear(self.read_out_act(self.rephrase_query_vec_to_action_embed(q)),
                                                self.vocab_embed.weight, self.vocab_readout_b)

        self.rephrase_type_classification = nn.Linear(args.hidden_size, 3)

        self.column_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.column_rephrase_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.table_rephrase_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.vocab_embed.weight.data)
        nn.init.xavier_normal_(self.rephrase_type_embed.weight.data)
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
        tab_linking_attn = torch.cat([tab_linking_attn.transpose(1, 2) for _ in range(self.schema_src_attn_heads)],
                                     dim=0)
        col_linking_attn = torch.cat([col_linking_attn.transpose(1, 2) for _ in range(self.schema_src_attn_heads)],
                                     dim=0)

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

        src_shadow_ctx = self.src_transformer_encode[layer_id](src_shadow_ctx.transpose(0, 1),
                                                               src_key_padding_mask=src_pad_mask)
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

        assert len(src_sents_len) == len(table_len) == len(table_col_len)

        max_t_len = max(table_len)
        max_c_len = max(table_col_len)
        max_ss_len = max(src_sents_len)

        shadow_t_emb = torch.repeat_interleave(torch.repeat_interleave(self.table_node, max_t_len, dim=1),
                                               batch_size, dim=0)
        shadow_c_emb = torch.repeat_interleave(torch.repeat_interleave(self.colum_node, max_c_len, dim=1),
                                               batch_size, dim=0)

        schema_link_weight = torch.ones((batch_size, max_ss_len, max_t_len + max_c_len)).float()
        src_schema_mask = torch.ones_like(schema_link_weight).byte()
        src_pad_mask = torch.ones((batch_size, max_ss_len)).byte()
        rat_pad_mask = torch.ones((batch_size, max_ss_len + max_t_len + max_c_len)).byte()
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
            colset_type_id = batch.col_text_types[bi]
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
            schema_link_weight[bi, :ss_len, max_t_len:max_t_len + c_len] = col_link_weight
            src_schema_mask[bi, :ss_len, max_t_len:max_t_len + c_len] = 0

            rat_pad_mask[bi, :ss_len] = 0
            rat_pad_mask[bi, max_ss_len: max_ss_len + c_len] = 0
            rat_pad_mask[bi, max_ss_len + max_c_len: max_ss_len + max_c_len + table_len[bi]] = 0

            # Q -> Q C T
            relative_pad_matirxs[bi, :ss_len, :ss_len] = relative_matirx[:ss_len, :ss_len]
            relative_pad_matirxs[bi, :ss_len, max_ss_len:max_ss_len + c_len] = relative_matirx[:ss_len,
                                                                               ss_len:ss_len + c_len]
            relative_pad_matirxs[bi, :ss_len, max_ss_len + max_c_len:max_ss_len + max_c_len + t_len] = \
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

        def _get_initial_shadow_emb(src_encodings, table_embeddings, schema_embeddings, shadow_t_emb, shadow_c_emb):
            # get emb differ
            embedding_differ = self.embedding_cosine(src_embedding=src_encodings, table_embedding=table_embeddings,
                                                     table_unk_mask=batch.table_unk_mask)

            schema_differ = self.embedding_cosine(src_embedding=src_encodings, table_embedding=schema_embeddings,
                                                  table_unk_mask=batch.schema_token_mask)

            tab_ctx = (src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
            schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

            shadow_t_emb = shadow_t_emb + schema_ctx
            shadow_c_emb = shadow_c_emb + tab_ctx

            return shadow_t_emb, shadow_c_emb

        if self.args.enc_type == 'shadowgnn':
            shadow_t_emb, shadow_c_emb = _get_initial_shadow_emb(src_encodings, table_embeddings, schema_embeddings,
                                                                 shadow_t_emb, shadow_c_emb)
            for l_id in range(self.layer_num):
                src_encodings, table_encoding, column_encoding, shadow_t_emb, shadow_c_emb, max_schema_arc_att_prob, src_schema_att_prob = \
                    self.one_layer_see_schema(src_encodings, column_encoding, table_encoding, shadow_c_emb, shadow_t_emb,
                                              schema_graph, schema_link_weight, src_schema_mask, src_pad_mask, table_len,
                                              table_col_len, l_id)

            table_encoding = shadow_t_emb
            column_encoding = shadow_c_emb

        elif self.args.enc_type == 'rat':
            stc_rat_input = torch.cat((src_encodings, column_encoding, table_encoding), dim=1)
            stc_encoding = self.rat_encoder(stc_rat_input, relative_pad_matirxs, rat_pad_mask)

            src_encodings = stc_encoding[:, :max_ss_len]
            table_encoding = stc_encoding[:, max_ss_len + max_c_len:]
            column_encoding = stc_encoding[:, max_ss_len:max_ss_len + max_c_len]

        elif self.args.enc_type == 'shadowgnn_rat':
            shadow_t_emb, shadow_c_emb = _get_initial_shadow_emb(src_encodings, table_embeddings, schema_embeddings,
                                                                 shadow_t_emb, shadow_c_emb)
            for l_id in range(self.layer_num):
                src_encodings, table_encoding, column_encoding, shadow_t_emb, shadow_c_emb, max_schema_arc_att_prob, src_schema_att_prob = \
                    self.one_layer_see_schema(src_encodings, column_encoding, table_encoding, shadow_c_emb,
                                              shadow_t_emb,
                                              schema_graph, schema_link_weight, src_schema_mask, src_pad_mask,
                                              table_len,
                                              table_col_len, l_id)

            stc_rat_input = torch.cat((src_encodings, shadow_c_emb, shadow_t_emb), dim=1)
            stc_encoding = self.rat_encoder(stc_rat_input, relative_pad_matirxs, rat_pad_mask)

            src_encodings = stc_encoding[:, :max_ss_len]
            table_encoding = stc_encoding[:, max_ss_len + max_c_len:]
            column_encoding = stc_encoding[:, max_ss_len:max_ss_len + max_c_len]

        elif self.args.enc_type == 'rat_shadowgnn':
            stc_rat_input = torch.cat((src_encodings, column_encoding, table_encoding), dim=1)
            stc_encoding = self.rat_encoder(stc_rat_input, relative_pad_matirxs, rat_pad_mask)

            src_encodings = stc_encoding[:, :max_ss_len]
            table_encoding = stc_encoding[:, max_ss_len + max_c_len:]
            column_encoding = stc_encoding[:, max_ss_len:max_ss_len + max_c_len]

            shadow_t_emb, shadow_c_emb = _get_initial_shadow_emb(src_encodings, column_encoding, table_encoding,
                                                                 shadow_t_emb, shadow_c_emb)

            for l_id in range(self.layer_num):
                src_encodings, table_encoding, column_encoding, shadow_t_emb, shadow_c_emb, max_schema_arc_att_prob, src_schema_att_prob = \
                    self.one_layer_see_schema(src_encodings, column_encoding, table_encoding, shadow_c_emb,
                                              shadow_t_emb,
                                              schema_graph, schema_link_weight, src_schema_mask, src_pad_mask,
                                              table_len,
                                              table_col_len, l_id)

            table_encoding = shadow_t_emb
            column_encoding = shadow_c_emb

        else:
            print('encoder type error!')
            exit(0)

        col_align_matr = self.col_memory_align(stc_encoding, column_encoding, column_encoding,
                                               relative_pad_matirxs[:, :, max_ss_len:max_ss_len + max_c_len],
                                               key_padding_mask=rat_pad_mask[:, max_ss_len:max_ss_len + max_c_len])[1].squeeze(1)
        table_align_matr = self.table_memory_align(stc_encoding, table_encoding, table_encoding,
                                                   relative_pad_matirxs[:, :, max_ss_len + max_c_len:],
                                                   key_padding_mask=rat_pad_mask[:, max_ss_len + max_c_len:])[1].squeeze(1)

        return src_encodings, table_encoding, column_encoding, table_align_matr, col_align_matr, rat_pad_mask

    def shadowgnn_encoder(self, examples):
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        batch_bert_seqs, batch_stc_lens = \
            get_roberta_style_input(batch.src_sents, batch.table_names, batch.table_sents, batch.tab_col_matches)

        bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens = \
            merge_roberta_data(batch_bert_seqs)

        if self.args.cuda:
            bert_token_ids = bert_token_ids.cuda()
            segment_ids = segment_ids.cuda()
            attention_mask = attention_mask.cuda()
            bert_to_ori_matrix = bert_to_ori_matrix.cuda()

        if bert_to_ori_matrix.size(2) > 511:
            print(batch_bert_seqs)
            print('idx greater than 512 !!!')
            exit(0)

        segment_ids = torch.zeros_like(segment_ids)
        if self.args.cuda:
            segment_ids = segment_ids.cuda()

        bert_src_schema_outputs = self.bert_encoder(bert_token_ids, attention_mask=attention_mask,
                                                    token_type_ids=segment_ids)
        src_schema_outputs = bert_src_schema_outputs['last_hidden_state']

        src_schema_outputs = bert_to_ori_matrix.bmm(src_schema_outputs)
        src_schema_outputs = self.bert_encode_linear(src_schema_outputs)

        ori_src_embedding, ori_schema_embedding, ori_table_embedding = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens, batch.tab_col_matches)

        src_encodings, schema_embedding, table_embedding, table_align_matr, col_align_matr, rat_pad_mask = \
            self.sent_see_schema(batch, ori_src_embedding, ori_table_embedding, ori_schema_embedding)

        src_encodings, (last_state, last_cell) = self.encode_again(src_encodings, batch.src_sents_len)

        src_encodings = self.dropout(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)

        return batch, src_encodings, schema_embedding, table_embedding, table_align_matr, col_align_matr, \
               rat_pad_mask, dec_init_vec

    def text2sql_forward(self, examples):
        args = self.args

        batch, src_encodings, schema_embedding, table_embedding, table_align_matr, col_align_matr, rat_pad_mask, dec_init_vec \
            = self.shadowgnn_encoder(examples)

        enc_output = torch.cat((src_encodings, table_embedding, schema_embedding), dim=1)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)
        h_tm1 = dec_init_vec
        sketch_action_probs = [[] for _ in examples]

        zero_action_embed = torch.zeros(args.action_embed_size, dtype=torch.float)
        zero_type_embed = torch.zeros(args.type_embed_size, dtype=torch.float)
        if self.args.cuda:
            zero_action_embed = zero_action_embed.cuda()
            zero_type_embed = zero_type_embed.cuda()

        sketch_attention_history = list()

        for t in range(batch.max_sketch_num):
            if t == 0:
                x = torch.zeros((len(batch), self.step_input_dim), dtype=torch.float)
                if self.args.cuda:
                    x = x.cuda()
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
                                                define_rule.Order,
                                                define_rule.Group,
                                                define_rule.From,
                                                ]:
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
                    sketch_action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_prob_var = torch.stack(
                [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in sketch_action_probs], dim=0)

        action_probs = [[] for _ in examples]

        h_tm1 = dec_init_vec

        batch_sketch_flag = [0 for _ in batch.sketch_len]
        for t in range(batch.max_action_num):
            if t == 0:
                x = torch.zeros((len(batch), self.step_input_dim), dtype=torch.float)
                if self.args.cuda:
                    x = x.cuda()
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        # action_tm1 = example.tgt_actions[parent_match[e_id][t]]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Order,
                                                define_rule.Group,
                                                define_rule.From,
                                                ]:

                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            # a_tm1_embed = batch_sketch_encoder_output[e_id][batch_sketch_flag[e_id], :]
                            batch_sketch_flag[e_id] += 1
                        else:
                            if isinstance(action_tm1, define_rule.C):
                                a_tm1_embed = self.column_rnn_input(table_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.table_rnn_input(schema_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.A) or isinstance(action_tm1, define_rule.V) or isinstance(action_tm1, define_rule.C1):
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
                        # action_tm = example.tgt_actions[parent_match[e_id][t]]
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

            column_weights = self.schema_select(att_t.unsqueeze(1), enc_output, col_align_matr, self.final_select_col_wq,
                                                self.final_select_col_wk, rat_pad_mask)

            table_weights = self.schema_select(att_t.unsqueeze(1), enc_output, table_align_matr, self.final_select_tab_wq,
                                               self.final_select_tab_wk, rat_pad_mask)

            # now get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        act_prob_t_i = column_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)

                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)

                    elif isinstance(action_t, define_rule.A) or isinstance(action_t, define_rule.V) or isinstance(action_t, define_rule.C1):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        pass

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        att_prob_var = 0

        fine_tune_prob_var = []
        for (sketch_prob, lf_prob) in zip(sketch_action_probs, action_probs):
            stack_prob = torch.stack(sketch_prob + lf_prob, dim=0)
            fine_tune_var = (1 - stack_prob)**args.finetune_lambda * stack_prob.log()
            fine_tune_var = fine_tune_var.mean()
            fine_tune_prob_var.append(fine_tune_var)

        fine_tune_prob_var = torch.stack(fine_tune_prob_var, dim=0)

        sql_hardnesses = []
        for hard in batch.sql_hardnesses:
            sql_hardnesses.append(sql_hardness_list.index(hard))

        return [lf_prob_var, att_prob_var, sketch_prob_var, fine_tune_prob_var, sql_hardnesses]

    def text2sql_parse(self, examples, beam_size=5):
        examples = [examples]

        batch, src_encodings, schema_embedding, table_embedding, table_align_matr, col_align_matr, rat_pad_mask, dec_init_vec \
            = self.shadowgnn_encoder(examples)

        enc_output = torch.cat((src_encodings, table_embedding, schema_embedding), dim=1)
        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)
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
                    x = torch.zeros((1, self.step_input_dim), dtype=torch.float)
                    if self.args.cuda:
                        x = x.cuda()
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Order,
                                            define_rule.Group,
                                            define_rule.From,
                                            ]:
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
                                    define_rule.Order,
                                    define_rule.Group,
                                    define_rule.From,
                                    ]:
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
                    print(action_class)
                    raise RuntimeError('No right action class')

            if not new_hyp_meta:
                break

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

                # if new_hyp.is_valid is False:
                #     continue

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
            print(beams[0].actions)
            print('shadow_test1: sketch beam empty !')
            # exit(0)
            return [[], []]

        sketch_actions = completed_beams[0].actions
        padding_sketch = self.padding_sketch(sketch_actions)

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

            exp_enc_output = enc_output.expand(hyp_num, enc_output.size(1), enc_output.size(2))
            exp_col_align_matr = col_align_matr.expand(hyp_num, col_align_matr.size(1), col_align_matr.size(2))
            exp_table_align_matr = table_align_matr.expand(hyp_num, table_align_matr.size(1), table_align_matr.size(2))
            exp_rat_pad_mask = rat_pad_mask.expand(hyp_num, rat_pad_mask.size(1))

            table_appear_mask = batch.table_appear_mask
            table_col_dicts = batch.table_col_dicts
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_col_match_mask = np.ones_like(table_appear_mask, dtype=np.uint8)
            table_col_masks = [[] for _ in range(len(beams))]
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        table_appear_mask[e_id][act.id_c] = 1

                    if type(act) == define_rule.T:
                        table_col_masks[e_id].extend(table_col_dicts[0][act.id_c])

                table_col_match_mask[e_id, list(set(table_col_masks[e_id]))] = 0

            if t == 0:
                with torch.no_grad():
                    x = torch.zeros((1, self.step_input_dim), dtype=torch.float)
                    if self.args.cuda:
                        x = x.cuda()
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Order,
                                            define_rule.Group,
                                            define_rule.From,
                                            ]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        # a_tm1_embed = batch_sketch_encoder_output[hyp.sketch_step, :]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        a_tm1_embed = self.column_rnn_input(table_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.T):
                        a_tm1_embed = self.table_rnn_input(schema_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A) or isinstance(action_tm1, define_rule.V) or isinstance(
                            action_tm1, define_rule.C1):
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

            column_weights = self.schema_select(att_t.unsqueeze(1), exp_enc_output, exp_col_align_matr,
                                                self.final_select_col_wq, self.final_select_col_wk, exp_rat_pad_mask)

            table_weights = self.schema_select(att_t.unsqueeze(1), exp_enc_output, exp_table_align_matr,
                                               self.final_select_tab_wq, self.final_select_tab_wk, exp_rat_pad_mask)

            column_weights = column_weights.log()
            table_weights = table_weights.log()

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                j = 0
                for i, act in enumerate(hyp.actions):
                    if type(act) == define_rule.V:
                        if act.id_c == 0 and t-i > 2:
                            j += 2

                if type(padding_sketch[t + j]) == define_rule.A or type(
                        padding_sketch[t + j]) == define_rule.V or type(padding_sketch[t + j]) == define_rule.C1:
                    possible_productions = self.grammar.get_production(type(padding_sketch[t+j]))
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': type(padding_sketch[t+j]), 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t+j]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_weights[hyp_id, col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t+j]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': define_rule.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t+j].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {'action_type': type(padding_sketch[t+j]), 'prod_id': prod_id,
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

    def rephrase_forward(self, examples):
        args = self.args
        # now should implement the examples
        batch, src_encodings, schema_embedding, table_embedding, table_align_matr, col_align_matr, rat_pad_mask, dec_init_vec \
            = self.shadowgnn_encoder(examples)

        utterance_encodings_lf_linear = self.att_rephrase_linear(src_encodings)
        enc_output = torch.cat((src_encodings, table_embedding, schema_embedding), dim=1)

        zero_action_embed = torch.zeros(args.action_embed_size, dtype=torch.float)
        zero_type_embed = torch.zeros(args.type_embed_size, dtype=torch.float)
        if self.args.cuda:
            zero_action_embed = zero_action_embed.cuda()
            zero_type_embed = zero_type_embed.cuda()

        h_tm1 = dec_init_vec

        schema_flags = [0 for _ in examples]
        action_probs = [[] for _ in examples]
        class_probs = [[] for _ in examples]

        for t in range(batch.max_rephrase_len):
            if t == 0:
                x = torch.zeros((len(batch), self.step_input_dim), dtype=torch.float)
                if self.args.cuda:
                    x = x.cuda()
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if t < len(example.rephrase_sentence):
                        action_tm1 = example.rephrase_sentence[t - 1]
                        if action_tm1 > 1:
                            # print('batch {}: '.format(e_id), self.vocab[action_tm1])
                            a_tm1_embed = self.vocab_embed.weight[action_tm1]
                            pre_type = self.rephrase_type_embed.weight[2]
                        else:
                            if action_tm1 == 1:
                                column_idx = example.rephrase_schema[schema_flags[e_id]]
                                a_tm1_embed = self.column_rephrase_input(table_embedding[e_id, column_idx])
                                pre_type = self.rephrase_type_embed.weight[1]
                                schema_flags[e_id] += 1
                                # print('batch {}: '.format(e_id), '<COLUMN>')
                            elif action_tm1 == 0:
                                table_idx = example.rephrase_schema[schema_flags[e_id]]
                                a_tm1_embed = self.table_rephrase_input(schema_embedding[e_id, table_idx])
                                pre_type = self.rephrase_type_embed.weight[0]
                                schema_flags[e_id] += 1
                                # print('batch {}: '.format(e_id), '<TABLE>')
                    else:
                        a_tm1_embed = zero_action_embed
                        pre_type = zero_type_embed
                    a_tm1_embeds.append(a_tm1_embed)
                    pre_types.append(pre_type)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                pre_types = torch.stack(pre_types)

                x = torch.cat([a_tm1_embeds, att_tm1, pre_types], dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.rephrase_decoder_lstm,
                                                 self.rephrase_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            rephrase_type_prob = F.softmax(self.rephrase_type_classification(att_t), dim=-1)

            apply_rule_prob = F.softmax(self.vocab_readout(att_t), dim=-1)

            column_weights = self.schema_select(att_t.unsqueeze(1), enc_output, col_align_matr, self.rephrase_select_col_wq,
                                                self.rephrase_select_col_wk, rat_pad_mask)

            table_weights = self.schema_select(att_t.unsqueeze(1), enc_output, table_align_matr, self.rephrase_select_tab_wq,
                                               self.rephrase_select_tab_wk, rat_pad_mask)

            # now get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.rephrase_sentence):
                    action_t = example.rephrase_sentence[t]
                    if action_t == 1:
                        column_idx = example.rephrase_schema[schema_flags[e_id]]
                        act_prob_t_i = column_weights[e_id, column_idx]
                        action_probs[e_id].append(act_prob_t_i)
                        # schema_flags[e_id] += 1

                        class_prob_t_i = rephrase_type_prob[e_id, 1]
                        class_probs[e_id].append(class_prob_t_i)

                    elif action_t == 0:
                        table_idx = example.rephrase_schema[schema_flags[e_id]]
                        act_prob_t_i = table_weights[e_id, table_idx]
                        action_probs[e_id].append(act_prob_t_i)
                        # schema_flags[e_id] += 1

                        class_prob_t_i = rephrase_type_prob[e_id, 0]
                        class_probs[e_id].append(class_prob_t_i)

                    elif action_t >= 2:
                        act_prob_t_i = apply_rule_prob[e_id, action_t]
                        action_probs[e_id].append(act_prob_t_i)

                        class_prob_t_i = rephrase_type_prob[e_id, 2]
                        class_probs[e_id].append(class_prob_t_i)
                    else:
                        pass

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        act_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        class_prob_var = torch.stack(
            [torch.stack(class_probs_i, dim=0).log().sum() for class_probs_i in class_probs], dim=0)

        return [act_prob_var, class_prob_var]

    def rephrase_parse(self, examples, beam_size=5):
        args = self.args

        batch, src_encodings, schema_embedding, table_embedding, table_align_matr, col_align_matr, rat_pad_mask, dec_init_vec \
            = self.shadowgnn_encoder(examples)

        enc_output = torch.cat((src_encodings, table_embedding, schema_embedding), dim=1)

        utterance_encodings_lf_linear = self.att_rephrase_linear(src_encodings)

        zero_action_embed = torch.zeros(args.action_embed_size, dtype=torch.float)
        zero_type_embed = torch.zeros(args.type_embed_size, dtype=torch.float)
        if self.args.cuda:
            zero_action_embed = zero_action_embed.cuda()
            zero_type_embed = zero_type_embed.cuda()

        stop_flags = [False for _ in examples]
        rephrase_type = [[] for _ in examples]
        rephrase_sentence = [[] for _ in examples]
        h_tm1 = dec_init_vec

        for t in range(self.args.decode_max_time_step):
            if t == 0:
                x = torch.zeros((len(batch), self.step_input_dim), dtype=torch.float)
                if self.args.cuda:
                    x = x.cuda()
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if not stop_flags[e_id]:
                        action_tm1 = rephrase_type[e_id][-1]
                        action_tm1_id = rephrase_sentence[e_id][-1]

                        if action_tm1 > 1:
                            a_tm1_embed = self.vocab_embed.weight[action_tm1_id]
                            pre_type = self.rephrase_type_embed.weight[2]
                        else:
                            if action_tm1 == 1:
                                a_tm1_embed = self.column_rephrase_input(table_embedding[e_id, action_tm1_id])
                                pre_type = self.rephrase_type_embed.weight[1]
                            elif action_tm1 == 0:
                                a_tm1_embed = self.table_rephrase_input(schema_embedding[e_id, action_tm1_id])
                                pre_type = self.rephrase_type_embed.weight[0]
                    else:
                        a_tm1_embed = zero_action_embed
                        pre_type = zero_type_embed
                    a_tm1_embeds.append(a_tm1_embed)
                    pre_types.append(pre_type)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                pre_types = torch.stack(pre_types)

                x = torch.cat([a_tm1_embeds, att_tm1, pre_types], dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.rephrase_decoder_lstm,
                                                 self.rephrase_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            rephrase_type_prob = F.softmax(self.rephrase_type_classification(att_t), dim=-1)

            rephrase_type_idx = torch.argmax(rephrase_type_prob, dim=1)

            apply_rule_prob = F.softmax(self.vocab_readout(att_t), dim=-1)
            apply_rule_idx = torch.argmax(apply_rule_prob, dim=1)

            column_weights = self.schema_select(att_t.unsqueeze(1), enc_output, col_align_matr,
                                                self.rephrase_select_col_wq,
                                                self.rephrase_select_col_wk, rat_pad_mask)
            column_idx = torch.argmax(column_weights, dim=1)

            table_weights = self.schema_select(att_t.unsqueeze(1), enc_output, table_align_matr,
                                               self.rephrase_select_tab_wq,
                                               self.rephrase_select_tab_wk, rat_pad_mask)
            table_idx = torch.argmax(table_weights, dim=1)

            for e_id, example in enumerate(examples):
                if not stop_flags[e_id]:
                    action_type = rephrase_type_idx[e_id].data.cpu()
                    if action_type == 2:
                        act_prob_t_i = apply_rule_idx[e_id].data.cpu()
                        rephrase_sentence[e_id].append(act_prob_t_i)
                        rephrase_type[e_id].append(2)
                        if act_prob_t_i == 2:
                            stop_flags[e_id] = True

                    elif action_type == 1:
                        act_prob_t_i = column_idx[e_id].data.cpu()
                        rephrase_sentence[e_id].append(act_prob_t_i)
                        rephrase_type[e_id].append(1)

                    elif action_type == 0:
                        act_prob_t_i = table_idx[e_id].data.cpu()
                        rephrase_sentence[e_id].append(act_prob_t_i)
                        rephrase_type[e_id].append(0)
                    else:
                        pass

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        rephrase_sentence_tokens = [[] for _ in examples]
        chosen_schemas = [[] for _ in examples]
        for e_id, (sentence, type) in enumerate(zip(rephrase_sentence, rephrase_type)):
            table_names = examples[e_id].table_names
            col_names = examples[e_id].cols

            assert len(sentence) == len(type)

            for s, t in zip(sentence, type):
                if t == 0:
                    rephrase_sentence_tokens[e_id].append(Lemmanize(' '.join(table_names[s])))
                    chosen_schemas[e_id].append(' '.join(table_names[s]))
                elif t == 1:
                    rephrase_sentence_tokens[e_id].append(Lemmanize(col_names[s]))
                    chosen_schemas[e_id].append(col_names[s])
                else:
                    rephrase_sentence_tokens[e_id].append(self.vocab[int(s)])

        return rephrase_sentence_tokens, chosen_schemas

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):

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

    def schema_select(self, hidden_state, enc_output, align_matr, Wq, Wk, mask=None):
        q = Wq(hidden_state)  # batch * 1 * hidden
        k = Wk(enc_output)  # batch * src_len * hidden
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])  # batch*1*src_len
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(dim=1), -float('inf'))
        scores = F.softmax(scores, -1)
        weight = torch.matmul(scores, align_matr)
        return weight.squeeze(1)