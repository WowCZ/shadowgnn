# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import copy

import src.rule.semQLPro as define_rule
from src.models import nn_utils


class Example:
    """

    """
    def __init__(self, src_sent, tgt_actions=None, vis_seq=None, tab_cols=None, col_num=None, sql=None,
                 one_hot_type=None, col_hot_type=None, schema_len=None, tab_ids=None,
                 table_names=None, table_len=None, col_table_dict=None, cols=None,
                 table_col_name=None, table_col_len=None,
                  col_pred=None, schema_graph=None, col_set_dict=None, tokenized_src_sent=None,
                 reprocess_col_type=None, reprocess_schema_graph=None, tgt_action_sent=None, schema_link=None,
                 align_table_one_hot=None, align_column_one_hot=None, dependency_graph=None, dependency_tree=None,
                 parse_graph=None, parse_token_id=None, parse_dfs_label=None, colset_text_type=None, col_text_type=None,
                 table_col_dict=None, question_table_match=None, question_col_match=None, table_col=None,
                 col_table=None, foreign_keys=None, primary_keys=None, relative_matrix=None, foreign_table_keys=None,
                 tab_col_match=None, sql_hardness=None, rephrase_sentence_idx=None, rephrase_schema_idx=None,
                 rephrase_result=None, rephrase_schema_items=None
        ):

        self.src_sent = src_sent
        self.tokenized_src_sent = tokenized_src_sent
        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.sql = sql
        self.one_hot_type = one_hot_type
        self.col_hot_type = col_hot_type
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.col_pred = col_pred
        self.tgt_actions = tgt_actions
        self.truth_actions = copy.deepcopy(tgt_actions)
        self.schema_graph = schema_graph
        self.col_set_dict = col_set_dict
        self.reprocess_col_type = reprocess_col_type
        self.reprocess_schema_graph = reprocess_schema_graph
        self.schema_link = schema_link
        self.tgt_action_sent = tgt_action_sent
        self.align_table_one_hot = align_table_one_hot
        self.align_column_one_hot = align_column_one_hot
        self.dependency_graph = dependency_graph
        self.dependency_tree = dependency_tree
        self.parse_graph = parse_graph
        self.parse_token_id = parse_token_id
        self.parse_dfs_label = parse_dfs_label
        self.colset_text_type = colset_text_type
        self.col_text_type = col_text_type
        self.table_col_dict = table_col_dict
        self.question_table_match = question_table_match
        self.question_col_match = question_col_match
        self.table_col = table_col
        self.col_table = col_table
        self.foreign_keys = foreign_keys
        self.primary_keys = primary_keys
        self.foreign_table_keys = foreign_table_keys
        self.relative_matrix = relative_matrix
        self.tab_col_match = tab_col_match
        self.sql_hardness = sql_hardness
        self.rephrase_sentence = rephrase_sentence_idx
        self.rephrase_schema = rephrase_schema_idx
        self.rephrase_result = rephrase_result
        self.rephrase_schema_items = rephrase_schema_items

        self.sketch = list()
        if self.truth_actions:
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A) or isinstance(ta, define_rule.V) or isinstance(ta, define_rule.C1):
                    continue
                self.sketch.append(ta)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    def __init__(self, examples, grammar, cuda=False):
        self.examples = examples

        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
        self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.max_rephrase_len = max(len(e.rephrase_sentence) for e in self.examples)

        self.sketch_len = [len(e.sketch) for e in self.examples]

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.tokenized_src_sents = [e.tokenized_src_sent for e in self.examples]
        self.tokenized_src_sents_len = [len(e.tokenized_src_sent) for e in examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]

        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]

        self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]
        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]
        self.schema_graphs = [e.schema_graph for e in examples]
        self.col_set_dicts = [e.col_set_dict for e in examples]
        self.reprocess_col_types = [e.reprocess_col_type for e in examples]
        self.reprocess_schema_graphs = [e.reprocess_schema_graph for e in examples]
        self.schema_links = [e.schema_link for e in examples]
        self.align_table_one_hots = [e.align_table_one_hot for e in examples]
        self.align_column_one_hots = [e.align_column_one_hot for e in examples]
        self.dependency_graphs = [e.dependency_graph for e in examples]
        self.dependency_trees = [e.dependency_tree for e in examples]
        self.parse_graphs = [e.parse_graph for e in examples]
        self.parse_token_ids = [e.parse_token_id for e in examples]
        self.parse_dfs_labels = [e.parse_dfs_label for e in examples]
        self.colset_text_types = [e.colset_text_type for e in examples]
        self.col_text_types = [e.col_text_type for e in examples]
        self.table_col_dicts = [e.table_col_dict for e in examples]
        self.stc_lens = [ss_len + t_len + c_len for ss_len, t_len, c_len in
                         zip(self.src_sents_len, self.table_len, self.col_num)]
        self.relative_matrixs = [e.relative_matrix for e in examples]
        self.tab_col_matches = [e.tab_col_match for e in examples]
        self.sql_hardnesses = [e.sql_hardness for e in examples]
        self.rephrase_sentences = [e.rephrase_sentence for e in examples]
        self.rephrase_schema = [e.rephrase_schema for e in examples]

        if cuda:
            self.relative_matrixs = [mat.cuda() for mat in self.relative_matrixs]

        self.grammar = grammar
        self.cuda = cuda

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    @cached_property
    def pred_col_mask(self):
        return nn_utils.pred_col_mask(self.col_pred, self.col_num)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)

    @cached_property
    def src_total_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.stc_lens,
                                                    cuda=self.cuda)


