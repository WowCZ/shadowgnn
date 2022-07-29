import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


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
    tc_embeddings = []
    batch_s_len, batch_tc_len = [], []
    for bi in range(batch_size):
        bert_embedding = bert_encodes[bi, bert_distill_flag[bi]]
        s_len, t_len, c_len = batch_stc_lens[bi]
        src_embeddings.append(bert_embedding[: s_len])
        tc_embeddings.append(bert_embedding[s_len: s_len + t_len + c_len])
        batch_s_len.append(s_len)
        batch_tc_len.append(t_len + c_len)

    src_embeddings = get_batch_embedding(src_embeddings, batch_s_len, args_cuda)
    tc_embeddings = get_batch_embedding(tc_embeddings, batch_tc_len, args_cuda)

    return src_embeddings, tc_embeddings


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


class ValueGenerator(nn.Module):
    def __init__(self, use_cuda, hidden_size, sketch_layer_num, sketch_enc_head_num, sketch_vocab_size):
        super(ValueGenerator, self).__init__()
        self.use_cuda = use_cuda
        # Query and schema encoder
        self.bert_encoder = BertModel.from_pretrained('bert-base-chinese')

        # Sketch encoder
        self.sketch_emb = nn.Embedding(num_embeddings=sketch_vocab_size, embedding_dim=hidden_size)
        transE = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=sketch_enc_head_num)
        self.sketch_encoder = nn.TransformerEncoder(transE, num_layers=sketch_layer_num)

        self.start_point_linear = nn.Linear(2*hidden_size, 1)
        self.end_point_linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, batch):
        ### aggr:
        # batch = {
        #   src_sents: [[['你'], ['好'], ['表', '1']], [['列'], ['出'], ['表', '2']]],
        #   table_names: [[['表', '1']], [['表', '2']]],
        #   table_sents: [[['人', '名'], ['列', '2']], [['地', '名'], ['列', '3']]],
        #   tc_appear_schema_idxs: [sketch中包含schema里出现的table和column在schema中的标号],
        #   sketch_ids: [sketch在sketch vocabulary中对应的idx，用来喂给transformer],
        #   tc_appear_sketch_idxs: [sketch中包含schema里出现的table和column在sketch中的标号],
        #   truth_start_ids: [value开始的位置],
        #   truth_end_ids: [value结束的位置],
        #   value_appear_ids: [sketch中包含的value在schema中的标号]
        # }
        #
        batch_size = len(batch.src_sents)
        # Encode query and schema
        batch_bert_seqs, batch_stc_lens = \
            get_bert_style_input(batch.src_sents, batch.table_names, batch.table_sents)

        bert_token_ids, segment_ids, attention_mask, bert_to_ori_matrix, bert_distill_flag, ori_seq_lens = \
            merge_bert_data(batch_bert_seqs)

        if self.use_cuda:
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

        # ori_src_embedding: (batch_size * max_len_query * hidden_size)
        ori_src_embedding, ori_tc_embeddings = \
            get_stc_embedding(src_schema_outputs, bert_distill_flag, batch_stc_lens)

        # Encode sketch
        tc_appear_enc_vecs = []
        tc_appear_schema_idxs = batch.tc_appear_schema_idxs
        for bi in range(batch_size):
            tc_appear_enc_vecs.append(ori_tc_embeddings[bi, tc_appear_schema_idxs[bi]])

        sketch_padded_seqs, sketch_lengths, sketch_att_mask = merge_pad_idx(batch.sketch_ids, pad_idx=0)
        if self.use_cuda:
            sketch_padded_seqs = sketch_padded_seqs.cuda()
            sketch_att_mask = sketch_att_mask.cuda()

        tc_appear_sketch_idxs = batch.tc_appear_sketch_idxs
        for bi in range(batch_size):
            sketch_padded_seqs[bi, tc_appear_sketch_idxs] = tc_appear_enc_vecs[bi]

        sketch_enc_outputs = self.sketch_encoder(sketch_padded_seqs, mask=sketch_att_mask)

        truth_start_ids = batch.truth_start_ids
        truth_end_ids = batch.truth_end_ids
        start_id_probs = []
        end_id_probs = []
        for bi in range(batch_size):
            truth_start_id = truth_start_ids[bi]
            truth_end_id = truth_end_ids[bi]
            src_len = batch_stc_lens[bi][0]
            value_appear_id = batch.value_appear_ids[bi]
            src_bert_enc_vec = ori_src_embedding[bi, :src_len].unsqueeze(0)

            value_vec = sketch_enc_outputs[bi, value_appear_id].unsqueeze(1)
            value_len = value_vec.size(0)

            src_bert_enc_vec = torch.repeat_interleave(src_bert_enc_vec, value_len, dim=0)
            value_vec = torch.repeat_interleave(value_vec, src_len, dim=1)
            src_value_cat_vec = torch.cat((src_bert_enc_vec, value_vec), dim=2)

            start_id_score = self.start_point_linear(src_value_cat_vec)
            end_id_score = self.end_point_linear(src_value_cat_vec)
            start_id_prob = torch.softmax(start_id_score)
            end_id_prob = torch.softmax(end_id_score)

            value_id = [i for i in range(value_len)]
            start_id_probs.append(start_id_prob[value_id, truth_start_id])
            end_id_probs.append(end_id_prob[value_id, truth_end_id])

        start_prob_var = torch.stack(
            [torch.stack(probs_i, dim=0).log().sum() for probs_i in start_id_probs], dim=0)

        end_prob_var = torch.stack(
            [torch.stack(probs_i, dim=0).log().sum() for probs_i in end_id_probs], dim=0)

        return start_prob_var, end_prob_var

