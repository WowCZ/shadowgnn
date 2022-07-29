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
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.dataset import Batch
from src.models import nn_utils
from src.models.basic_model import BasicModel, HeteroRGCN, HeteroRelGCN, RelGraphConvLayer, MultiHeadGATLayer, HeteroGAT, SublayerConnection, PositionwiseFeedForward
from src.models.pointer_net import PointerNet
from src.rule import semQLPro as define_rule
from src.utils import get_parent_match

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


class Shadowgnn(BasicModel):

    def __init__(self, args, grammar):
        super(Shadowgnn, self).__init__()
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

        # etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 'fore_c2c', 'fore_invc2c', 's2t', 't2s']
        etypes = ['norm_t2c', 'norm_c2t', 'prim_t2c', 'prim_c2t', 'fore_c2c', 'fore_invc2c', 's2t', 't2s', 'fore_t2c',
                  'fore_c2t']

        # self.schema_rgcn = HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size)
        # self.shadow_rgcn = HeteroRGCN(etypes, args.hidden_size, args.hidden_size, args.hidden_size)

        self.schema_rgcn = HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, self.layer_num)
        self.shadow_rgcn = HeteroRelGCN(etypes, args.hidden_size, args.hidden_size, self.layer_num)

        # self.schema_rgcn = HeteroGAT(etypes, args.hidden_size, args.hidden_size, self.layer_num)
        # self.shadow_rgcn = HeteroGAT(etypes, args.hidden_size, args.hidden_size, self.layer_num)

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased-model/')
        self.bert_encode_linear = nn.Linear(768, args.hidden_size)

        # self.bert_encoder = AlbertModel.from_pretrained('albert-xxlarge-v2/')
        # self.bert_encode_linear = nn.Linear(4096, args.hidden_size)

        # self.bert_encoder = ElectraModel.from_pretrained('electra_large_discriminator/')

        # self.bert_encoder = BertModel.from_pretrained('bert-large-uncase/')
        # self.bert_encode_linear = nn.Linear(1024, args.hidden_size)

        # self.bert_s_linear = nn.Linear(1024, args.hidden_size)
        # self.bert_t_linear = nn.Linear(1024, args.hidden_size)
        # self.bert_c_linear = nn.Linear(1024, args.hidden_size)

        # self.parse_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 4, args.cuda)

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

        self.encoder_again_lstm = nn.LSTM(args.hidden_size, args.hidden_size // 2, bidirectional=True,
                                          batch_first=True)

        self.schema_link_embed = nn.Embedding(9, args.hidden_size)
        # self.colset_type_embed = nn.Embedding(5, args.hidden_size)
        # self.parse_node_embed = nn.Embedding(75, args.hidden_size)

        # self.schema_link_linear = nn.Linear(args.hidden_size, 1)
        self.schema_link_linear = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )
        self.schema_link_drop = nn.Dropout(args.sldp)

        # input_dim = args.action_embed_size + \
        #             args.att_vec_size + \
        #             args.type_embed_size
        input_dim = args.action_embed_size + \
                    args.hidden_size + \
                    args.type_embed_size
        self.step_input_dim = args.action_embed_size + \
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

        self.prob_att = nn.Linear(args.hidden_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(4, args.col_embed_size)

        # self.sketch_graph_enc = GAT(args.hidden_size, args.hidden_size // 4, args.hidden_size, 4, 2, args.cuda)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)

        # tanh
        self.production_readout = nn.Sequential(
            nn.Linear(args.hidden_size, args.action_embed_size),
            nn.Tanh(),
            nn.Linear(args.action_embed_size, len(self.grammar.prod2id)))

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

    def sent_see_schema(self, batch, src_encodings, table_embeddings, schema_embeddings, schema_link_drop=False):
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
        batch_src_schema_att_prob = []

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
            schema_link_score = torch.ones_like(schema_link_score)
            if self.args.cuda:
                schema_link_score = schema_link_score.cuda()

            hetero_schema_input = {
                'table': table_embedding,
                'column': column_embedding
            }
            hetero_schema_output = self.schema_rgcn(schema_graphs[bi], hetero_schema_input)
            ori_schema_item_encoding = torch.cat((hetero_schema_output['table'], hetero_schema_output['column']), dim=0)
            schema_item_encoding = torch.relu(self.schema_att_linear(ori_schema_item_encoding))
            # schema_item_encoding = self.schema_layer_norm(schema_item_encoding)

            schema_item_encoding = torch.transpose(schema_item_encoding, 0, 1)
            src_encoding = torch.relu(self.src_att_linear(ori_src_encoding))
            # src_encoding = self.src_layer_norm(src_encoding)
            src_encoding_v = torch.relu(self.src_cross_att_v(ori_src_encoding))
            # src_schema_att_score = torch.sigmoid(torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0)))
            src_schema_att_score = torch.bmm(src_encoding.unsqueeze(0), schema_item_encoding.unsqueeze(0))

            assert src_schema_att_score.shape == schema_link_score.shape
            src_schema_att_score = src_schema_att_score * schema_link_score
            # src_schema_att_prob = torch.softmax(src_schema_att_score.squeeze(0), dim=1)

            src_tab_att_prob = torch.softmax(src_schema_att_score.squeeze(0)[:, :t_len], dim=1)
            src_col_att_prob = torch.softmax(src_schema_att_score.squeeze(0)[:, t_len:], dim=1)
            src_schema_att_prob = torch.cat((src_tab_att_prob, src_col_att_prob), dim=1)

            schema_src_att_prob = torch.transpose(src_schema_att_score.squeeze(0), 0, 1)
            schema_src_att_prob = torch.softmax(schema_src_att_prob, dim=1)

            max_schema_src_att_idx = torch.argmax(src_schema_att_prob, dim=0)
            # max_schema_src_att_idx = torch.argmax(schema_src_att_prob, dim=1)
            y_idx = torch.tensor([i for i in range(max_schema_src_att_idx.size(0))])
            if self.args.cuda:
                y_idx = y_idx.cuda()
            max_src_schema_att_prob = src_schema_att_prob[max_schema_src_att_idx, y_idx].unsqueeze(1)
            hetero_shadow_input = {
                'table': torch.repeat_interleave(self.table_node, t_len, dim=0) * max_src_schema_att_prob[:t_len],
                'column': torch.repeat_interleave(self.colum_node, c_len, dim=0) * max_src_schema_att_prob[t_len:]
            }
            # colset_type_id = batch.colset_text_types[bi]
            # colset_type_id = torch.tensor(colset_type_id)
            # if self.args.cuda:
            #     colset_type_id = colset_type_id.cuda()
            # c_type_emb = self.colset_type_embed(colset_type_id)
            # hetero_shadow_input['column'] = hetero_shadow_input['column'] + c_type_emb

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
            src_shadow_ctx_encoding = torch.relu(self.src_shadow_cat_linear(src_shadow_ctx_encoding))

            max_src_schema_att_prob = max_src_schema_att_prob.squeeze(1)
            max_src_schema_att_prob_dict = {}
            max_src_schema_att_prob_dict['table'] = max_src_schema_att_prob[:t_len]
            max_src_schema_att_prob_dict['column'] = max_src_schema_att_prob[t_len:]

            src_repre.append(src_shadow_ctx_encoding)
            table_repre.append(table_shadow_encoding)
            column_repre.append(column_shadow_encoding)
            batch_max_src_schema_att_prob.append(max_src_schema_att_prob_dict)
            batch_src_schema_att_prob.append(src_schema_att_prob)

        src_encoding = get_batch_embedding(src_repre, src_sents_len, self.args.cuda)
        table_encoding = get_batch_embedding(table_repre, table_len, self.args.cuda)
        column_encoding = get_batch_embedding(column_repre, table_col_len, self.args.cuda)

        return src_encoding, table_encoding, column_encoding, batch_max_src_schema_att_prob, batch_src_schema_att_prob

    def forward(self, examples):
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)
        table_appear_mask = batch.table_appear_mask
        table_col_match_mask = np.ones_like(table_appear_mask, dtype=np.uint8)
        table_col_dicts = batch.table_col_dicts

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

        src_encodings, schema_embedding, table_embedding, batch_max_att_p, batch_src_schema_att_prob\
            = self.sent_see_schema(batch, ori_src_embedding, ori_table_embedding, ori_schema_embedding)

        _, (last_state, last_cell) = self.encode_again(src_encodings, batch.src_sents_len)

        src_encodings = self.dropout(src_encodings)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec
        action_probs = [[] for _ in examples]

        zero_action_embed = torch.zeros(args.action_embed_size, dtype=torch.float)
        zero_type_embed = torch.zeros(args.type_embed_size, dtype=torch.float)
        if self.args.cuda:
            zero_action_embed = zero_action_embed.cuda()
            zero_type_embed = zero_type_embed.cuda()

        sketch_attention_history = list()
        sketch_parent_match = [get_parent_match(example.sketch) for example in examples]

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
                        # action_tm1 = example.sketch[sketch_parent_match[e_id][t]]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.A,
                                                define_rule.V,
                                                define_rule.C1,
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
                        # action_tm = example.sketch[sketch_parent_match[e_id][t]]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstm,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask)
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

        action_probs = [[] for _ in examples]
        max_att_ct_probs = [[] for _ in examples]
        parent_match = [get_parent_match(example.tgt_actions) for example in examples]

        h_tm1 = dec_init_vec

        batch_sketch_flag = [0 for _ in batch.sketch_len]
        table_col_masks = [[] for _ in batch.sketch_len]

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
                                                define_rule.A,
                                                define_rule.V,
                                                define_rule.C1,
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

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask)

            def get_schema_weight(src_schema_att, decoder_att, t_lens, src_lens):
                column_attention_weights = []
                table_weights = []
                batch_size = len(src_schema_att)
                assert len(src_schema_att) == len(t_lens)
                for bi in range(batch_size):
                    ss_att = src_schema_att[bi]
                    assert ss_att.size(0) == src_lens[bi]
                    ss_tab_att = ss_att[:, :t_lens[bi]]  # src*tab
                    ss_col_att = ss_att[:, t_lens[bi]:]  # src*col
                    dec_att = decoder_att[bi, :src_lens[bi]].unsqueeze(0)  # 1*src

                    column_attention_weights.append(torch.matmul(dec_att, ss_col_att).squeeze(dim=0))
                    table_weights.append(torch.matmul(dec_att, ss_tab_att).squeeze(dim=0))
                return column_attention_weights, table_weights

            column_attention_weights, table_weights = get_schema_weight(batch_src_schema_att_prob, aw, batch.table_len,
                                                                        batch.src_sents_len)

            # now get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        table_appear_mask[e_id, action_t.id_c] = 1
                        act_prob_t_i = column_attention_weights[e_id][action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                        # print(example.tgt_actions)
                        # print('C weight:', weights[e_id])
                        # print('C att:', column_attention_weights[e_id])

                        max_att_ct_probs[e_id].append(batch_max_att_p[e_id]['column'][action_t.id_c])
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id][action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                        table_col_masks[e_id].extend(table_col_dicts[e_id][action_t.id_c])
                        table_col_match_mask[e_id, list(set(table_col_masks[e_id]))] = 0

                        max_att_ct_probs[e_id].append(batch_max_att_p[e_id]['table'][action_t.id_c])
                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        att_prob_var = torch.stack(
            [torch.stack(att_probs_i, dim=0).log().sum() for att_probs_i in max_att_ct_probs], dim=0)

        return [lf_prob_var, att_prob_var, sketch_prob_var, 0]

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

        src_encodings, schema_embedding, table_embedding, batch_max_att_p, batch_src_schema_att_prob \
            = self.sent_see_schema(batch, ori_src_embedding, ori_table_embedding, ori_schema_embedding)

        _, (last_state, last_cell) = self.encode_again(src_encodings, batch.src_sents_len)

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
                                            define_rule.A,
                                            define_rule.V,
                                            define_rule.C1,
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
                    # action_tm = hyp.actions[get_parent_match(hyp.actions)[-1]]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, exp_src_enconding,
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
                                    define_rule.A,
                                    define_rule.V,
                                    define_rule.C1,
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
            table_appear_mask = batch.table_appear_mask
            table_col_dicts = batch.table_col_dicts
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_col_match_mask = np.ones_like(table_appear_mask, dtype=np.uint8)
            table_col_masks = [[] for _ in range(len(beams))]
            hyp_t_lens = []
            hyp_src_lens = []
            exp_src_schema_att_prob = []
            for e_id, hyp in enumerate(beams):
                hyp_t_lens.append(batch.table_len[0])
                hyp_src_lens.append(batch.src_sents_len[0])
                exp_src_schema_att_prob.append(batch_src_schema_att_prob[0])

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
                    # action_tm1 = hyp.actions[get_parent_match(hyp.actions)[-1]]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.A,
                                            define_rule.V,
                                            define_rule.C1,
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
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    # action_tm = hyp.actions[get_parent_match(hyp.actions)[-1]]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, exp_src_encodings,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                             self.lf_att_vec_linear,
                                             src_token_mask=None)

            def get_schema_weight(src_schema_att, decoder_att, t_lens, src_lens):
                column_attention_weights = []
                table_weights = []
                batch_size = len(src_schema_att)
                assert len(src_schema_att) == len(t_lens)
                for bi in range(batch_size):
                    ss_att = src_schema_att[bi]
                    assert ss_att.size(0) == src_lens[bi]
                    ss_tab_att = ss_att[:, :t_lens[bi]]  # src*tab
                    ss_col_att = ss_att[:, t_lens[bi]:]  # src*col
                    dec_att = decoder_att[bi, :src_lens[bi]].unsqueeze(0)  # 1*src

                    column_attention_weights.append(torch.matmul(dec_att, ss_col_att).squeeze(dim=0))
                    table_weights.append(torch.matmul(dec_att, ss_tab_att).squeeze(dim=0))
                    table_weights[-1] = torch.log(table_weights[-1])
                    column_attention_weights[-1] = torch.log(column_attention_weights[-1])
                return column_attention_weights, table_weights

            column_selection_log_prob, table_weights = get_schema_weight(exp_src_schema_att_prob, aw, hyp_t_lens,
                                                                         hyp_src_lens)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_selection_log_prob[hyp_id][col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id][t_id]
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

        completed_beams.sort(key=lambda hyp: -hyp.score)

        return [completed_beams, sketch_actions]

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None):
        # h_t: (batch_size, hidden_size)
        att_t, alpha_t = nn_utils.dot_prod_attention(h_tm1[0],
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)
        x = torch.cat([x, att_t], 1)
        h_t, cell_t = decoder(x, h_tm1)

        # # tanh
        # att_t = F.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        # att_t = self.dropout(att_t)

        return (h_t, cell_t), h_t, alpha_t

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        # tanh
        h_0 = F.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

