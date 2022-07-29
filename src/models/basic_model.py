# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/26
# @Author  : Jiaqi&Zecheng
# @File    : basic_model.py
# @Software: PyCharm
"""

import numpy as np
import os
import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import dgl
import dgl.function as fn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import dgl.nn.pytorch as dglnn
from dgl.data.utils import save_graphs
from src.rule import lf
from src.rule.semQLPro import Sel, Order, Root, Filter, A, C, T, Root1, From, C1, V

sketch_cls = [Root1, Root, Sel, Filter, Order, A, C1, V]
sketch_name = ['root1', 'root', 'sel', 'filter', 'order', 'agg', 'C1', 'V']

from src.rule import semQLPro as define_rule


class SchemaLinkingTypes():
    # ======== Schema Linking ========
    # None-Match
    NONE = 0
    # Exact-Match
    Q_T_E = 1  # Query-Table-Exact
    Q_C_E = 2
    # Partial-Match
    Q_T_P = 3
    Q_C_P = 4
    # ======== Schema Content Linking ========
    Q_T_C_E = 5  # Query-Table-Content-Exact
    Q_T_C_P = 6  # Query-Table-Content-Partial
    Q_C_C_E = 7  # Query-Column-Content-Exact
    Q_C_C_P = 8  # Query-Column-Content-Partial

    @classmethod
    def type_num(cls):
        return 9


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes, ntypes=['table', 'column']):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })
        # self.weight_h = nn.ModuleDict({
        #     name: nn.Linear(in_size, out_size) for name in ntypes
        # })
        self.weight_h = nn.Linear(in_size, out_size)

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            # leaky_relu
            Wh = F.leaky_relu(self.weight[etype](feat_dict[srctype]))
            # Wh = self.weight[etype](feat_dict[srctype])

            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'mean')
        # return the updated node feature dictionary
        # return {ntype: F.leaky_relu(self.weight_h[ntype](G.nodes[ntype].data['h'])) + feat_dict[ntype] for ntype in
        #         G.ntypes}
        return {ntype: F.leaky_relu(self.weight_h(G.nodes[ntype].data['h'])) + feat_dict[ntype] for ntype in
                G.ntypes}
        # return {ntype: feat_dict[ntype] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, g_etypes, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # # Use trainable node embeddings as featureless inputs.
        # embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
        #               for ntype in G.ntypes}
        # for key, embed in embed_dict.items():
        #     nn.init.xavier_uniform_(embed)
        # self.embed = nn.ParameterDict(embed_dict)

        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, g_etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, g_etypes)

    def forward(self, G, node_embed):
        h_dict = self.layer1(G, node_embed)
        # h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class HeteroRelGCN(nn.Module):
    def __init__(self,
                 g_etypes,
                 h_dim, out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(HeteroRelGCN, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = g_etypes
        self.num_bases = len(self.rel_names)

        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()

        # h2h
        for i in range(self.num_hidden_layers-1):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=F.relu,
            self_loop=self.use_self_loop))

    def forward(self, g, h, blocks=None):
        if blocks is None:
            # full graph training
            blocks = [g] * len(self.layers)

        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


class GATLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_heads,
                 *,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(GATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel: dglnn.conv.GATConv((in_feat, in_feat), out_feat, num_heads)
                for rel in rel_names
            })
        # self.conv = dglnn.HeteroGraphConv({
        #     rel: dglnn.conv.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
        #     for rel in rel_names
        # })

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        g = g.local_var()
        # for stype, etype, dtype in g.canonical_etypes:
        #     rel_graph = g[stype, etype, dtype]
        #     print(rel_graph.number_of_edges())
        #     print(inputs[stype].shape)

        hs = self.conv(g, (inputs, inputs))

        def _apply(ntype, h):
            # print(h.shape)
            h = h.view(h.size(0), -1)
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class HeteroGAT(nn.Module):
    def __init__(self,
                 g_etypes,
                 h_dim, out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(HeteroGAT, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = g_etypes
        self.num_heads = 4

        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()

        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(GATLayer(
                self.h_dim, self.h_dim // self.num_heads, self.rel_names,
                self.num_heads, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(GATLayer(
            self.h_dim, self.out_dim // self.num_heads, self.rel_names,
            self.num_heads, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, g, h, blocks=None):
        if blocks is None:
            # full graph training
            blocks = [g] * len(self.layers)

        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)

        # for name, h_item in h.items():
        #     print(name, h_item.shape)
        # print('Hetero GAT work !!!')
        # exit(0)
        return h


class GATLayerwise(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayerwise, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayerwise(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers, cuda):
        super(GAT, self).__init__()
        self.layers = []
        for nl in range(num_layers - 1):
            self.layers.append(MultiHeadGATLayer(in_dim, hidden_dim, num_heads))

        if cuda:
            self.layers = [layer.cuda() for layer in self.layers]
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.last_layer = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
            h = F.elu(h)
        h = self.last_layer(g, h)
        return h


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout=None):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(dglnn.GATConv(in_size, out_size, layer_num_heads, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout=None):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


class BasicModel(nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()
        pass

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(2).expand(
            table_embedding.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ).bool(), 0)

        return embedding_differ

    def cosine_attention(self, src_embedding, key_embedding):
        embedding_differ = []
        for i in range(key_embedding.size(0)):
            one_key_embedding = key_embedding[i, :]
            one_key_embedding = torch.repeat_interleave(one_key_embedding.unsqueeze(0), src_embedding.size(0), dim=0)

            topk_val = F.cosine_similarity(one_key_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)

        return embedding_differ

    def encode(self, src_sents_var, src_sents_len, q_onehot_project=None, src_embed=False):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        if src_embed:
            src_token_embed = src_sents_var
        else:
            src_token_embed = self.gen_x_batch(src_sents_var)

        if q_onehot_project is not None:
            src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def parse_encode(self, batch, src_encodings):
        batch_size = src_encodings.size(0)
        src_sents_len = batch.src_sents_len
        parse_dfs_labels = batch.parse_dfs_labels
        parse_token_ids = batch.parse_token_ids
        parse_graphs = batch.parse_graphs
        batch_src_encoding = []

        for bi in range(batch_size):
            parse_dfs_label = parse_dfs_labels[bi]
            parse_token_id = parse_token_ids[bi]
            parse_graph = parse_graphs[bi]

            parse_dfs_label = torch.tensor(parse_dfs_label)
            if self.args.cuda:
                parse_dfs_label = parse_dfs_label.cuda()
            parse_graph_input = self.parse_node_embed(parse_dfs_label)
            parse_graph_input[parse_token_id] = src_encodings[bi, :src_sents_len[bi]]

            parse_graph_output = self.parse_graph_enc(parse_graph, parse_graph_input)
            src_encoding = parse_graph_output[parse_token_id]

            batch_src_encoding.append(src_encoding)

        return batch_src_encoding


    def encode_again(self, src_token_embed, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """

        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_again_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def sketch_encoder(self, input, input_lens):
        input_lens = np.array(input_lens)
        sort_idx = np.argsort(-input_lens)
        input_lengths = input_lens[sort_idx]
        sort_input_seq = input[sort_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(sort_input_seq, input_lengths, batch_first=True)
        outputs, (last_state, last_cell) = self.sketch_encoder_lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        invert_sort_idx = np.argsort(sort_idx)
        outputs = outputs[invert_sort_idx]
        last_state = last_state[invert_sort_idx]
        last_cell = last_cell[invert_sort_idx]
        return outputs, (last_state, last_cell)

    def sketch_graph_encoder(self, active_rule_label, sketch_repre):
        sketch_tree_root = lf.build_tree_for_encoder(active_rule_label)
        sketch_graph_input = {
            'root1': [],
            'root': [],
            'sel': [],
            'filter': [],
            'sup': [],
            'N': [],
            'order': []
        }

        recover_flag = {
            'root1': [],
            'root': [],
            'sel': [],
            'filter': [],
            'sup': [],
            'N': [],
            'order': []
        }
        sketch_graph = dict()
        sketch_read_len = 0

        def get_sketch_tree_input(sketch_tree_root, sketch_repre, sketch_graph_input, sketch_graph, recover_flag, sketch_read_len):
            parent_name = sketch_name[sketch_cls.index(type(sketch_tree_root))]
            parent_id = len(sketch_graph_input[parent_name])

            sketch_graph_input[parent_name].append(sketch_repre[sketch_read_len])
            recover_flag[parent_name].append(sketch_read_len)

            while len(sketch_tree_root.children) > 0:
                child = sketch_tree_root.children.pop(0)

                if type(child) not in sketch_cls:
                    continue

                sketch_read_len += 1
                child_name = sketch_name[sketch_cls.index(type(child))]
                child_id = len(sketch_graph_input[child_name])
                if (parent_name, 'p2c', child_name) not in sketch_graph:
                    sketch_graph[(parent_name, 'p2c', child_name)] = []
                    sketch_graph[(child_name, 'c2p', parent_name)] = []

                sketch_graph[(parent_name, 'p2c', child_name)].append((parent_id, child_id))
                sketch_graph[(child_name, 'c2p', parent_name)].append((child_id, parent_id))

                sketch_graph_input, sketch_graph, recover_flag, sketch_read_len = get_sketch_tree_input(child,
                                                                                    sketch_repre, sketch_graph_input,
                                                                                    sketch_graph, recover_flag,
                                                                                    sketch_read_len)

            return sketch_graph_input, sketch_graph, recover_flag, sketch_read_len

        sketch_graph_input, sketch_graph, recover_flag, sketch_read_len = get_sketch_tree_input(sketch_tree_root,
                                                                                    sketch_repre, sketch_graph_input,
                                                                                    sketch_graph, recover_flag,
                                                                                    sketch_read_len)

        sketch_graph_input_dict = dict()
        for k, v in sketch_graph_input.items():
            if len(v) > 0:
                sketch_graph_input_dict[k] = torch.cat(v, dim=0)

        ori_sketch_graph = copy.deepcopy(sketch_graph)

        sketch_graph = dgl.heterograph(sketch_graph)

        sketch_graph_output_dict = self.sketch_graph_enc(sketch_graph, sketch_graph_input_dict)

        sketch_graph_output = []
        flat_recover_flag = []
        for k in sketch_graph_output_dict.keys():
            sketch_graph_output.append(sketch_graph_output_dict[k])
            flat_recover_flag = flat_recover_flag + recover_flag[k]
        try:
            sketch_graph_output = torch.cat(sketch_graph_output, dim=0)
        except:
            print(sketch_graph_output_dict)
            print(ori_sketch_graph)
            exit(0)

        reorder_recover_flag = []
        for i in range(len(flat_recover_flag)):
            reorder_recover_flag.append(flat_recover_flag.index(i))

        sketch_graph_output = sketch_graph_output[reorder_recover_flag]

        return sketch_graph_output

    def sketch_graph_without_r_encoder(self, active_rule_label, sketch_repre):
        sketch_tree_root = lf.build_tree_for_encoder(active_rule_label)
        sketch_graph = []
        sketch_repre = torch.cat(sketch_repre, dim=0)
        graph_node_id_flag = 0

        # sketch_rule = [x for x in active_rule_label if
        #                not isinstance(x, define_rule.C) and not isinstance(x, define_rule.T) and not isinstance(x,
        #                                                                                                         define_rule.A)]
        # print([str(x) for x in sketch_rule])

        def get_sketch_tree_input(sketch_tree_root, sketch_graph, graph_node_id_flag):
            parent_id = graph_node_id_flag

            while len(sketch_tree_root.children) > 0:
                child = sketch_tree_root.children.pop(0)

                if type(child) not in sketch_cls:
                    continue

                graph_node_id_flag += 1
                child_id = graph_node_id_flag
                sketch_graph.append((parent_id, child_id))

                sketch_graph, graph_node_id_flag = get_sketch_tree_input(child, sketch_graph, graph_node_id_flag)

            return sketch_graph, graph_node_id_flag

        sketch_graph, graph_node_id_flag = get_sketch_tree_input(sketch_tree_root, sketch_graph, graph_node_id_flag)
        sketch_graph = dgl.DGLGraph(sketch_graph)

        sketch_graph_output = self.sketch_graph_enc(sketch_graph, sketch_repre)

        return sketch_graph_output

    def schema_encoder(self, input, input_lens):
        input_lens = np.array(input_lens)
        sort_idx = np.argsort(-input_lens)
        input_lengths = input_lens[sort_idx]
        sort_input_seq = input[sort_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(sort_input_seq, input_lengths, batch_first=True)
        outputs, (last_state, last_cell) = self.schema_encoder_lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        invert_sort_idx = np.argsort(sort_idx)
        outputs = outputs[invert_sort_idx]
        last_state = last_state[invert_sort_idx]
        last_cell = last_cell[invert_sort_idx]
        return outputs, (last_state, last_cell)

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    # def padding_sketch(self, sketch):
    #     padding_result = []
    #     for action in sketch:
    #         padding_result.append(action)
    #         if type(action) == define_rule.C1:
    #             padding_result.append(define_rule.C(0))
    #         elif type(action) == define_rule.Group:
    #             padding_result.append(define_rule.C(0))
    #             if action.id_c in [1, 3]:
    #                 padding_result.append(define_rule.C(0))
    #         elif type(action) == define_rule.From:
    #             for _ in range(action.id_c):
    #                 padding_result.append(define_rule.T(0))
    #
    #     return padding_result

    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.Sel:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.V(1))
                    padding_result.append(define_rule.C1(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.C1(0))
                    padding_result.append(define_rule.C(0))
            elif type(action) == define_rule.Filter:
                if action.id_c >= 2:
                    padding_result.append(define_rule.V(1))
                    padding_result.append(define_rule.C1(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.C1(0))
                    padding_result.append(define_rule.C(0))
            elif type(action) == define_rule.Order:
                padding_result.append(define_rule.C1(0))
                padding_result.append(define_rule.C(0))
                if action.id_c in [2, 3, 6, 7]:
                    padding_result.append(define_rule.C1(0))
                    padding_result.append(define_rule.C(0))
            elif type(action) == define_rule.Group:
                padding_result.append(define_rule.C(0))
                if action.id_c in [1, 3]:
                    padding_result.append(define_rule.C(0))
            elif type(action) == define_rule.From:
                for _ in range(action.id_c):
                    padding_result.append(define_rule.T(0))

        return padding_result

    def gen_x_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        is_list = False
        if type(q[0][0]) == list:
            is_list = True
        for i, one_q in enumerate(q):
            if not is_list:
                q_val = list(
                    map(lambda x: self.word_emb.get(x, np.zeros(self.args.col_embed_size, dtype=np.float32)), one_q))
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, self.word_emb['unk']))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))

            val_embs.append(q_val)
            val_len[i] = len(q_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.args.col_embed_size), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        return val_inp

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
