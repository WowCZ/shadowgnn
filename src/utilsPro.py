# -*- coding: utf-8 -*-

import json
import time

import copy
import numpy as np
import random
import os
import torch
import dgl
import pickle
from nltk.stem import WordNetLemmatizer

from src.dataset import Example
from src.rule import lf
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1, From, Group

import requests
from nltk.corpus import wordnet as wn

wordnet_lemmatizer = WordNetLemmatizer()

COLSET_TYPE = ['others', 'text', 'boolean', 'time', 'number']

def load_word_emb(file_name, use_small=False):
    print('Loading word embedding from %s' % file_name, use_small)
    ret = {}
    with open(file_name, encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
    return ret


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x


def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ti in range(len(table_col_dict)):
        result.append(table_col_dict[ti])
    return result


def get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def get_reprocess_col_table_dict(tab_cols, tab_ids):
    table_dict = {}
    for cor_id, cor_val in enumerate(tab_cols):
        table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [cor_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def get_col_set_dict(tab_cols, sql):
    col_dict = {}
    inv_col_dict = {}
    for cor_id, cor_val in enumerate(tab_cols):
        for c_id, c_v in enumerate(sql['col_set']):
            if c_v == cor_val:
                col_dict[cor_id] = c_id
                if c_id not in inv_col_dict:
                    inv_col_dict[c_id] = []
                inv_col_dict[c_id].append(cor_id)
    return col_dict, inv_col_dict


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):
    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            # try:
            #     col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
            #     question_arg[count_q] = ['column'] + question_arg[count_q]
            # except:
            #     print(col_set_iter, question_arg[count_q])
            #     raise RuntimeError("not in col set")
            if question_arg[count_q] in col_set_iter:
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]
            else:
                continue
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    # try:
                    #     col_set_type[sql['col_set'].index(col_probase)][2] = 5
                    #     question_arg[count_q] = ['value'] + question_arg[count_q]
                    # except:
                    #     print(sql['col_set'], col_probase)
                    #     raise RuntimeError('not in col')
                    if col_probase in sql['col_set']:
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                        one_hot_type[count_q][5] = 1
                    else:
                        continue
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    col_set_type[sql['col_set'].index(col_probase)][3] += 1


def get_graph(table, col_set_dict, col_table_dict):
    table_column = table['column_names']
    prim_keys = table['primary_keys']
    fore_keys = table['foreign_keys']
    schema_graph = {}

    # edges between primary nodes and table nodes
    prim_tc_edges = []
    prim_ct_edges = []
    for i, k in enumerate(prim_keys):
        prim_t_id = table_column[k][0]
        prim_c_id = col_set_dict[k]
        prim_tc_edges.append((prim_t_id, prim_c_id))
        prim_ct_edges.append((prim_c_id, prim_t_id))

    # edges between foreign nodes and table nodes
    fore_tc_edges = []
    fore_ct_edges = []
    for fp in fore_keys:
        fore_t_id = table_column[fp[0]][0]
        fore_c_id = col_set_dict[fp[0]]
        if (fore_t_id, fore_c_id) not in prim_tc_edges:
            fore_tc_edges.append((fore_t_id, fore_c_id))
            fore_ct_edges.append((fore_c_id, fore_t_id))

        # fore_t_id = table_column[fp[1]][0]
        # fore_c_id = col_set_dict[fp[1]]
        # if (fore_t_id, fore_c_id) not in prim_tc_edges:
        #     fore_tc_edges.append((fore_t_id, fore_c_id))
        #     fore_ct_edges.append((fore_c_id, fore_t_id))

    # edges between norm column nodes except primary nodes and table nodes
    norm_tc_edges = []
    norm_ct_edges = []
    st_nodes = []
    ts_nodes = []
    for col, t_set in col_table_dict.items():
        if col == 0:
            for t in t_set:
                st_nodes.append((col, t))
                ts_nodes.append((t, col))
        else:
            for t in t_set:
                if (col, t) not in fore_ct_edges and (col, t) not in prim_ct_edges:
                    norm_ct_edges.append((col, t))
                    norm_tc_edges.append((t, col))

    schema_graph[('table', 'norm_t2c', 'column')] = norm_tc_edges
    schema_graph[('column', 'norm_c2t', 'table')] = norm_ct_edges

    schema_graph[('table', 'fore_t2c', 'column')] = fore_tc_edges
    schema_graph[('column', 'fore_c2t', 'table')] = fore_ct_edges

    schema_graph[('table', 'prim_t2c', 'column')] = prim_tc_edges
    schema_graph[('column', 'prim_c2t', 'table')] = prim_ct_edges

    schema_graph[('column', 's2t', 'table')] = st_nodes
    schema_graph[('table', 't2s', 'column')] = ts_nodes

    schema_graph = dgl.heterograph(schema_graph)

    return schema_graph


def get_reprocess_graph(table):
    table_id = [i for i, _ in enumerate(table['table_names'])]
    table_column = table['column_names']
    prim_keys = table['primary_keys']
    fore_keys = table['foreign_keys']
    schema_graph = {}

    # edges between primary nodes and table nodes
    prim_tc_edges = []
    prim_ct_edges = []
    for i, k in enumerate(prim_keys):
        prim_tc_edges.append((i, k))
        prim_ct_edges.append((k, i))

    # edges between foreign nodes and table nodes
    fore_cc_edges = []
    fore_invcc_edges = []
    for fp in fore_keys:
        fore_cc_edges.append((fp[0], fp[1]))
        fore_invcc_edges.append((fp[1], fp[0]))

    # edges between norm column nodes except primary nodes and table nodes
    norm_tc_edges = []
    norm_ct_edges = []
    st_nodes = []
    ts_nodes = []
    for col, tc_pair in enumerate(table_column):
        if col == 0:
            for t_id in table_id:
                st_nodes.append((col, t_id))
                ts_nodes.append((t_id, col))
        else:
            if (col, tc_pair[0]) not in prim_ct_edges:
                norm_ct_edges.append((col, tc_pair[0]))
                norm_tc_edges.append((tc_pair[0], col))

    schema_graph[('table', 'norm_t2c', 'column')] = norm_tc_edges
    schema_graph[('column', 'norm_c2t', 'table')] = norm_ct_edges

    schema_graph[('table', 'prim_t2c', 'column')] = prim_tc_edges
    schema_graph[('column', 'prim_c2t', 'table')] = prim_ct_edges

    schema_graph[('column', 'fore_c2c', 'column')] = fore_cc_edges
    schema_graph[('column', 'fore_invc2c', 'column')] = fore_invcc_edges

    schema_graph[('column', 's2t', 'table')] = st_nodes
    schema_graph[('table', 't2s', 'column')] = ts_nodes

    schema_graph = dgl.heterograph(schema_graph)

    return schema_graph


def process(sql, table):
    process_dict = {}

    origin_sql = sql['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]

    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    schema_link = copy.deepcopy(sql['schema_linking_matrix'])
    one_hot_type = np.zeros((len(question_arg_type), 6))
    col_text_type = table['column_types']

    col_set_type = np.zeros((len(col_set_iter), 4))

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['schema_link'] = schema_link
    process_dict['col_text_type'] = col_text_type

    return process_dict


def process_random(sql, table):
    process_dict = {}

    sql = copy.deepcopy(sql)
    table = copy.deepcopy(table)

    def lose_weight_for_database(table, sql):
        rule_label = sql['rule_label'].strip().split(' ')
        gold_col_id_set = [0]
        gold_tab_id_set = []
        for label_item in rule_label:
            if label_item.find('C(') == 0:
                col_id_s = label_item.find('(') + 1
                col_id_e = label_item.find(')')
                col_id = int(label_item[col_id_s:col_id_e])
                if col_id not in gold_col_id_set:
                    gold_col_id_set.append(col_id)
            elif label_item.find('T(') == 0:
                tab_id_s = label_item.find('(') + 1
                tab_id_e = label_item.find(')')
                try:
                    tab_id = int(label_item[tab_id_s:tab_id_e])
                except:
                    print(tab_id_s)
                    print(tab_id_e)
                    print(label_item[tab_id_s:tab_id_e])
                    print(rule_label)
                if tab_id not in gold_tab_id_set:
                    gold_tab_id_set.append(tab_id)
            else:
                pass

        tab_cols = [col[1] for col in table['column_names']]
        tab_ids = [col[0] for col in table['column_names']]
        # small_col_iter = [' '.join([wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]) for x in tab_cols]
        # small_col_set_iter = [' '.join([wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]) for x in sql['col_set']]
        # table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]
        col_iter = copy.deepcopy(tab_cols)
        col_set_iter = copy.deepcopy(sql['col_set'])
        table_names = copy.deepcopy(table['table_names'])

        gold_col_name_set = [col_set_iter[col_id] for col_id in gold_col_id_set]
        gold_tab_idx_with_column = [tab_ids[col_id] for col_id, col_name in enumerate(col_iter) if
                                    col_name in gold_col_name_set and col_id != 0]
        gold_tab_id_set = list(set(gold_tab_id_set + gold_tab_idx_with_column))

        # remove table
        removable_table_idx = [i for i in range(len(table['table_names'])) if i not in gold_tab_id_set]
        if len(removable_table_idx) > 0:
            # removed_tab_cnt = random.randint(1, len(removable_table_idx))
            # random.shuffle(removable_table_idx)
            # removed_tab_idxs = removable_table_idx[:removed_tab_cnt]
            removed_tab_idxs = removable_table_idx
            removed_col_idxs_with_tab = [col_id for col_id, tab_id in enumerate(tab_ids) if
                                         tab_id in removed_tab_idxs]
        else:
            removed_tab_idxs = []
            removed_col_idxs_with_tab = []

        keep_tab_prim_idx = [pk for pk in table['primary_keys'] if tab_ids[pk] not in removed_tab_idxs]
        keep_tab_prim_name = [col_iter[col_id] for col_id in keep_tab_prim_idx]

        # remove column
        removable_column_names = [col_set_iter[i] for i in range(len(col_set_iter)) if i not in gold_col_id_set]
        removable_column_names = [col_name for col_name in removable_column_names if col_name not in keep_tab_prim_name]
        removable_column_idx = [col_id for col_id, col_name in enumerate(col_iter) if col_name in removable_column_names]

        if len(removable_column_idx) > 0:
            # removed_col_cnt = random.randint(1, len(removable_column_idx))
            # random.shuffle(removable_column_idx)
            # removed_col_idxs = removable_column_idx[:removed_col_cnt]
            removed_col_idxs = removable_column_idx
        else:
            removed_col_idxs = []

        total_removed_col_idxs = list(set(removed_col_idxs_with_tab + removed_col_idxs))
        keep_table_idx_with_column = [col[0] for col_id, col in enumerate(table['column_names'][1:]) if
                                      col_id+1 not in total_removed_col_idxs]
        keep_table_idx_with_column = keep_table_idx_with_column + gold_tab_id_set
        keep_table_idx_with_column = list(set(keep_table_idx_with_column))

        keep_table_with_column_dict = {}
        for col_id, tb_id in enumerate(tab_ids):
            if tb_id in keep_table_idx_with_column:
                if tb_id not in keep_table_with_column_dict:
                    keep_table_with_column_dict[tb_id] = {'keep': [], 'remove':[]}
                if col_id not in total_removed_col_idxs:
                    keep_table_with_column_dict[tb_id]['keep'].append(col_id)
                else:
                    keep_table_with_column_dict[tb_id]['remove'].append(col_id)

        for tb_id, tb_col in keep_table_with_column_dict.items():
            if len(tb_col['keep']) == 0:
                # recovered_col = random.randint(1, len(tb_col['remove']))
                recovered_col = 1
                keep_table_with_column_dict[tb_id]['keep'] = tb_col['remove'][:recovered_col]
                keep_table_with_column_dict[tb_id]['remove'] = tb_col['remove'][recovered_col:]

        keep_column_idx = [0]
        for tb_id, tb_col in keep_table_with_column_dict.items():
            keep_column_idx = keep_column_idx + tb_col['keep']

        keep_table_names = [col_name for col_id, col_name in enumerate(table_names) if
                            col_id in keep_table_idx_with_column]
        keep_table_idxs = keep_table_idx_with_column

        # removed_tab_idxs = [tab_id for tab_id in range(len(table_names)) if tab_id not in keep_table_idxs]
        keep_tab_prim_idx = [pk for pk in table['primary_keys'] if tab_ids[pk] in keep_table_idxs]

        # update table
        col_ori_new_pairs = []
        for col_id, col_name in enumerate(col_iter):
            if col_id in keep_column_idx:
                col_ori_new_pair = [col_id, len(col_ori_new_pairs)]
                col_ori_new_pairs.append(col_ori_new_pair)

        col_ori_new_dict = dict(col_ori_new_pairs)

        keep_column_names = [col_name for col_id, col_name in enumerate(col_iter) if
                             col_id in keep_column_idx]

        keep_column_set_idx = list(set([col_set_iter.index(col_name) for col_name in keep_column_names]))

        table['column_names'] = [[keep_table_names.index(table_names[col_name[0]]), col_name[1]]
        for col_id, col_name in enumerate(table['column_names'][1:]) if col_id+1 in keep_column_idx]
        table['column_names'] = [[-1, '*']] + table['column_names']

        table['column_names_original'] = [col_name for col_id, col_name in enumerate(table['column_names_original']) if
                                 col_id in keep_column_idx]

        assert len(table['column_names']) == len(table['column_names_original'])
        table['column_names_original'] = [[col_name[0], col_name_ori[1]] for col_name, col_name_ori in
                                          zip(table['column_names'], table['column_names_original'])]

        table['column_types'] = [col_name for col_id, col_name in enumerate(table['column_types']) if
                                          col_id in keep_column_idx]

        table['primary_keys'] = [col_ori_new_dict[col_id] for col_id in keep_tab_prim_idx]
        table['table_names'] = [col_name for col_id, col_name in enumerate(table['table_names']) if
                                 col_id in keep_table_idxs]
        table['table_names_original'] = [col_name for col_id, col_name in enumerate(table['table_names_original']) if
                                col_id in keep_table_idxs]

        foreigh_keys = []
        for foreigh_key in table['foreign_keys']:
            if foreigh_key[0] not in keep_column_idx or foreigh_key[1] not in keep_column_idx:
                continue
            else:
                foreigh_keys.append([col_ori_new_dict[foreigh_key[0]], col_ori_new_dict[foreigh_key[1]]])

        table['foreign_keys'] = foreigh_keys

        # update sql
        assert len(sql['col_set']) == len(col_set_iter)
        sql['col_set'] = [col_list for (col_list, col_name) in zip(sql['col_set'], col_set_iter) if
                          col_name in keep_column_names]
        col_set_names = [col_name for col_name in col_set_iter if
                          col_name in keep_column_names]

        new_rule_label = []
        for label_item in rule_label:
            if label_item.find('C(') == 0:
                col_id_s = label_item.find('(') + 1
                col_id_e = label_item.find(')')
                col_id = int(label_item[col_id_s:col_id_e])
                col_name = col_set_iter[col_id]
                new_col_id = col_set_names.index(col_name)
                new_rule_label.append('C({})'.format(str(new_col_id)))

            elif label_item.find('T(') == 0:
                tab_id_s = label_item.find('(') + 1
                tab_id_e = label_item.find(')')
                tab_id = int(label_item[tab_id_s:tab_id_e])
                tab_name = table_names[tab_id]
                new_tab_id = keep_table_names.index(tab_name)
                new_rule_label.append('T({})'.format(str(new_tab_id)))
            else:
                new_rule_label.append(label_item)

        sql['rule_label'] = ' '.join(new_rule_label)
        schema_link_matrix = sql['schema_linking_matrix']
        schema_link_matrix = np.array(schema_link_matrix)
        keep_schema_link_idx = keep_table_idxs + [col_id + len(table_names) for col_id in keep_column_set_idx]
        schema_link_matrix = list(schema_link_matrix[:, keep_schema_link_idx])
        sql['schema_linking_matrix'] = schema_link_matrix

        sql['names'] = [col_name for col_id, col_name in enumerate(sql['names']) if
                                 col_id in keep_column_idx]
        sql['col_table'] = [col_name[0] for col_name in table['column_names']]

        sql['table_names'] = copy.deepcopy(table['table_names'])
        sql['foreign_keys'] = copy.deepcopy(table['foreign_keys'])
        sql['primary_keys'] = copy.deepcopy(table['primary_keys'])
        sql['column_types'] = copy.deepcopy(table['column_types'])

        return table, sql

    table, sql = lose_weight_for_database(table, sql)

    origin_sql = sql['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]
    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    schema_link = copy.deepcopy(sql['schema_linking_matrix'])

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]

    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 6))
    col_set_type = np.zeros((len(col_set_iter), 4))
    col_text_type = table['column_types']

    assert len(schema_link[0]) == len(col_set_iter) + len(table_names)

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['schema_link'] = schema_link
    process_dict['col_text_type'] = col_text_type

    return process_dict, table, sql


def process_by_conceptnet(sql, table, is_train):
    process_dict = {}

    sql = copy.deepcopy(sql)
    table = copy.deepcopy(table)

    if is_train:
        conceptnet_path = 'conceptnet_score.json'
    else:
        conceptnet_path = 'dev_conceptnet_score.json'

    with open(os.path.join('save/', conceptnet_path), 'r') as inf:
        conceptnet_value_dict = json.load(inf)

    def lose_weight_for_database(table, sql, conceptnet_value_dict):
        rule_label = sql['rule_label'].strip().split(' ')
        gold_col_id_set = [0]
        gold_tab_id_set = []
        for label_item in rule_label:
            if label_item.find('C(') == 0:
                col_id_s = label_item.find('(') + 1
                col_id_e = label_item.find(')')
                col_id = int(label_item[col_id_s:col_id_e])
                if col_id not in gold_col_id_set:
                    gold_col_id_set.append(col_id)
            elif label_item.find('T(') == 0:
                tab_id_s = label_item.find('(') + 1
                tab_id_e = label_item.find(')')
                try:
                    tab_id = int(label_item[tab_id_s:tab_id_e])
                except:
                    print(tab_id_s)
                    print(tab_id_e)
                    print(label_item[tab_id_s:tab_id_e])
                    print(rule_label)
                if tab_id not in gold_tab_id_set:
                    gold_tab_id_set.append(tab_id)
            else:
                pass

        tab_cols = [col[1] for col in table['column_names']]
        tab_ids = [col[0] for col in table['column_names']]
        col_iter = copy.deepcopy(tab_cols)
        col_set_iter = copy.deepcopy(sql['col_set'])
        table_names = copy.deepcopy(table['table_names'])

        def infer_from_conceptnet(sql, table, conceptnet_value_dict):
            small_col_set_iter = ['_'.join([wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")]) for x in
                                  sql['col_set']]
            small_table_names = ['_'.join([wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')]) for x in
                                 table['table_names']]
            schema_tokens = small_table_names + small_col_set_iter

            question_arg = copy.deepcopy(sql['question_arg'])
            question_arg = ['_'.join(token) for token in question_arg]

            schema_len = len(schema_tokens)
            query_len = len(question_arg)
            schema_link_with_conceptnet = np.zeros((query_len, schema_len))

            schema_link = copy.deepcopy(sql['schema_linking_matrix'])
            schema_link = np.array(schema_link)
            schema_link_score = np.sum(schema_link, axis=0)
            query_link_score = np.sum(schema_link, axis=1)

            schema_link_related_id = np.where(schema_link_score > 0)[0]
            query_link_related_id = np.where(query_link_score > 0)[0]
            no_related_tokens = ['what', 'is', 'of', 'the', '', ' ', ',', '?', '!', 'how', 'which', 'but', 'are', '\'',
                                 'of', 'were', 'was',
                                 'a', 'an', 'and', '.', 'that', 'or', 'for', 'either', 'with', 'do', 'dont', 'we',
                                 'you', 'me',
                                 'at', 'in', 'from', 'return', 'find', 'list', 'give', 'to', 'all', 'their', 'them',
                                 'many',
                                 'much', 'number', 'total', 'have', 'has', 'ha', 'into', 'by'
                                 ]

            def get_wordnet_score(q_token, s_token):
                q_token_syns = wn.synsets(q_token)
                s_token_syns = wn.synsets(s_token)

                qs_smi_score = []
                for q_syn in q_token_syns:
                    for s_syn in s_token_syns:
                        value = q_syn.path_similarity(s_syn)
                        if value:
                            qs_smi_score.append(value)

                if len(qs_smi_score) == 0:
                    return 0

                return max(qs_smi_score)

            for s_id, s_token in enumerate(schema_tokens):
                if s_id not in schema_link_related_id and s_token != '*':
                    qs_value_list = []
                    for q_id, q_token in enumerate(question_arg):
                        if q_token not in no_related_tokens and q_id not in query_link_related_id:
                            qs_token = q_token + '-' + s_token
                            if qs_token not in conceptnet_value_dict:
                                # start_time = time.time()
                                # concept_query = 'http://api.conceptnet.io//relatedness?node1=/c/en/{}&node2=/c/en/{}'.format(
                                #     q_token,
                                #     s_token)
                                # # try:
                                # #     print(q_token, s_token)
                                # #     response = requests.get(concept_query).json()
                                # #     related_value = response['value']
                                # #     print(related_value)
                                # #
                                # #     if related_value > 0:
                                # #         schema_link_with_conceptnet[q_id][s_id] = 1
                                # #     request_cnt += 1
                                # # except:
                                # #     # print('cannt match with {} and {}'.format(q_token, s_token))
                                # #     pass
                                # response = requests.get(concept_query)
                                # try:
                                #     related_value = json.loads(response.text)['value']
                                # except:
                                #     with open(os.path.join('save/', 'dev_conceptnet_score.json'), 'w') as outf:
                                #         json.dump(conceptnet_value_dict, outf)
                                #     print(response.text)
                                #     exit(0)
                                #
                                # conceptnet_value_dict[qs_token] = related_value
                                # end_time = time.time()
                                # visit_time = end_time - start_time
                                # if visit_time < 1.2:
                                #     time.sleep(1.2 - visit_time)
                                # # related_value = get_wordnet_score(q_token, s_token)
                                related_value = 0
                            else:
                                related_value = conceptnet_value_dict[qs_token]

                            qs_value_list.append(related_value)

                            # if related_value > 0.2:
                            #     schema_link_with_conceptnet[q_id][s_id] = 1

                    if qs_value_list:
                        if max(qs_value_list) > 0.2:
                            schema_link_with_conceptnet[np.argmax(np.array(qs_value_list))][s_id] = 1

            return schema_link_with_conceptnet, schema_len, conceptnet_value_dict

        def get_reserved_schema(sql, table, conceptnet_value_dict):
            tab_len = len(table['table_names'])
            col_len = len(sql['col_set'])
            schema_link = copy.deepcopy(sql['schema_linking_matrix'])
            schema_link = np.array(schema_link)
            schema_link_with_conceptnet, schema_len, conceptnet_value_dict \
                = infer_from_conceptnet(sql, table, conceptnet_value_dict)
            schema_link = schema_link + schema_link_with_conceptnet

            assert schema_link.shape[1] == tab_len + col_len
            schema_link_score = np.sum(schema_link, axis=0)
            # if len(np.where(schema_link_score > 0)[0]) == 0:
            #     schema_link_with_conceptnet = infer_from_conceptnet(sql, table)
            #     schema_link = schema_link_with_conceptnet
            #     schema_link_score = np.sum(schema_link, axis=0)

            table_link_score = schema_link_score[:tab_len]
            col_link_score = schema_link_score[tab_len:]

            table_link_related_id = np.where(table_link_score > 0)[0]
            table_link_related_id = list(table_link_related_id)

            col_link_related_id = np.where(col_link_score > 0)[0]
            col_link_related_id = [0] + list(col_link_related_id)

            tab_cols = [col[1] for col in table['column_names']]
            col_set_dict, set_col_dict = get_col_set_dict(tab_cols, sql)

            # find sub-graph from related columns and tables
            # step 1: add related tables with related columns
            for col_set_id in col_link_related_id[1:]:
                col_ids = set_col_dict[col_set_id]
                for col_id in col_ids:
                    tab_id = table['column_names'][col_id][0]

                    if tab_id not in table_link_related_id:
                        table_link_related_id.append(tab_id)

            # step 2: add primary columns of the related tables
            for pri_id in table['primary_keys']:
                pri_col_id = col_set_dict[pri_id]
                tab_pri_id = table['column_names'][pri_id][0]
                if tab_pri_id in table_link_related_id and pri_col_id not in col_link_related_id:
                    col_link_related_id.append(pri_col_id)

            # step 3: add foreign columns between two related tables
            for f_pair in table['foreign_keys']:
                start_tab_id = table['column_names'][f_pair[0]][0]
                end_tab_id = table['column_names'][f_pair[1]][0]

                start_col_id = col_set_dict[f_pair[0]]
                end_col_id = col_set_dict[f_pair[1]]

                if start_tab_id in table_link_related_id and end_tab_id in table_link_related_id:
                    if start_col_id not in col_link_related_id:
                        col_link_related_id.append(start_col_id)

                    if end_col_id not in col_link_related_id:
                        col_link_related_id.append(end_col_id)

            return table_link_related_id, col_link_related_id, schema_len, conceptnet_value_dict, col_set_dict, set_col_dict

        table_link_related_id, col_link_related_id, schema_len, conceptnet_value_dict, col_set_dict, set_col_dict \
            = get_reserved_schema(sql, table, conceptnet_value_dict)

        # keep_table_idxs = list(set(table_link_related_id))
        # keep_column_set_idx = list(set(col_link_related_id))

        # keep_table_idxs = list(set(keep_table_idxs + gold_tab_id_set))
        # keep_column_set_idx = list(set(keep_column_set_idx + gold_col_id_set))

        keep_table_idxs = gold_tab_id_set
        keep_column_set_idx = gold_col_id_set

        keep_tab_prim_idx = [pk for pk in table['primary_keys'] if tab_ids[pk] in keep_table_idxs]
        for tpi in keep_tab_prim_idx:
            if col_set_dict[tpi] not in keep_column_set_idx:
                keep_column_set_idx.append(col_set_dict[tpi])

        # keep_table_idxs = list(set(keep_table_idxs + keep_table_idx_with_column))

        # print('#'*100)
        # print(keep_tab_prim_idx)
        # print(keep_column_idx)

        keep_column_idx = []
        for set_id in keep_column_set_idx:
            keep_column_idx.extend(set_col_dict[set_id])

        keep_table_names = [tab_name for tab_id, tab_name in enumerate(table_names) if
                            tab_id in keep_table_idxs]

        remain_col_index = [col_id + 1 for col_id, col_name in enumerate(table['column_names'][1:]) if
                            col_id + 1 in keep_column_idx and table_names[col_name[0]] in keep_table_names]

        keep_table_idx_with_column = [col[0] for col_id, col in enumerate(table['column_names'][1:]) if
                                      col_id + 1 in remain_col_index]

        for ti in keep_table_idx_with_column:
            if ti not in keep_table_idxs:
                keep_table_idxs.append(ti)

        # update table
        col_ori_new_pairs = []
        for col_id, col_name in enumerate(col_iter):
            if col_id in [0] + remain_col_index:
                col_ori_new_pair = [col_id, len(col_ori_new_pairs)]
                col_ori_new_pairs.append(col_ori_new_pair)

        col_ori_new_dict = dict(col_ori_new_pairs)

        keep_column_names = [col_name for col_id, col_name in enumerate(col_iter) if
                             col_id in [0] + remain_col_index]

        keep_table_idxs = list(set([col_name[0] for col_id, col_name in enumerate(table['column_names'][1:]) if
                               col_id + 1 in remain_col_index]))

        keep_table_names = [tab_name for tab_id, tab_name in enumerate(table_names) if
                            tab_id in keep_table_idxs]

        # keep_column_set_idx = list(set([col_set_iter.index(col_name) for col_name in keep_column_names]))

        table['column_names'] = [[keep_table_names.index(table_names[col_name[0]]), col_name[1]]
            for col_id, col_name in enumerate(table['column_names'][1:]) if col_id+1 in remain_col_index]
        table['column_names'] = [[-1, '*']] + table['column_names']

        table['column_names_original'] = [col_name for col_id, col_name in enumerate(table['column_names_original'][1:])
                                          if col_id + 1 in remain_col_index]

        table['column_names_original'] = [[-1, '*']] + table['column_names_original']

        assert len(table['column_names']) == len(table['column_names_original'])
        table['column_names_original'] = [[col_name[0], col_name_ori[1]] for col_name, col_name_ori in
                                          zip(table['column_names'], table['column_names_original'])]

        remain_col_index = [0] + remain_col_index
        table['column_types'] = [col_type for col_id, col_type in enumerate(table['column_types']) if
                                          col_id in remain_col_index]

        table['primary_keys'] = [col_ori_new_dict[col_id] for col_id in keep_tab_prim_idx]

        for pk in table['primary_keys']:
            if pk not in [k for k in range(len(table['column_names']))]:
                print(table['primary_keys'])
                print(len(table['column_names']))
                exit(0)
        table['table_names'] = [col_name for col_id, col_name in enumerate(table['table_names']) if
                                 col_id in keep_table_idxs]
        table['table_names_original'] = [col_name for col_id, col_name in enumerate(table['table_names_original']) if
                                col_id in keep_table_idxs]

        foreigh_keys = []
        for foreigh_key in table['foreign_keys']:
            if foreigh_key[0] not in remain_col_index or foreigh_key[1] not in remain_col_index:
                continue
            else:
                foreigh_keys.append([col_ori_new_dict[foreigh_key[0]], col_ori_new_dict[foreigh_key[1]]])

        table['foreign_keys'] = foreigh_keys

        # update sql
        assert len(sql['col_set']) == len(col_set_iter)
        sql['col_set'] = [col_list for (col_list, col_name) in zip(sql['col_set'], col_set_iter) if
                          col_name in keep_column_names]
        col_set_names = [col_name for col_name in col_set_iter if
                          col_name in keep_column_names]

        new_rule_label = []
        for label_item in rule_label:
            if label_item.find('C(') == 0:
                col_id_s = label_item.find('(') + 1
                col_id_e = label_item.find(')')
                col_id = int(label_item[col_id_s:col_id_e])
                col_name = col_set_iter[col_id]
                if col_name in col_set_names:
                    new_col_id = col_set_names.index(col_name)
                else:
                    new_col_id = 0
                new_rule_label.append('C({})'.format(str(new_col_id)))

            elif label_item.find('T(') == 0:
                tab_id_s = label_item.find('(') + 1
                tab_id_e = label_item.find(')')
                tab_id = int(label_item[tab_id_s:tab_id_e])
                tab_name = table_names[tab_id]
                if tab_name in keep_table_names:
                    new_tab_id = keep_table_names.index(tab_name)
                else:
                    new_tab_id = 0
                new_rule_label.append('T({})'.format(str(new_tab_id)))
            else:
                new_rule_label.append(label_item)

        sql['rule_label'] = ' '.join(new_rule_label)
        schema_link_matrix = sql['schema_linking_matrix']
        schema_link_matrix = np.array(schema_link_matrix)
        keep_schema_link_idx = keep_table_idxs + [col_id + len(table_names) for col_id in keep_column_set_idx]
        schema_link_matrix = list(schema_link_matrix[:, keep_schema_link_idx])
        sql['schema_linking_matrix'] = schema_link_matrix

        sql['names'] = [col_name for col_id, col_name in enumerate(sql['names']) if
                                 col_id in remain_col_index]
        sql['col_table'] = [col_name[0] for col_name in table['column_names']]

        sql['table_names'] = copy.deepcopy(table['table_names'])
        sql['foreign_keys'] = copy.deepcopy(table['foreign_keys'])
        sql['primary_keys'] = copy.deepcopy(table['primary_keys'])
        sql['column_types'] = copy.deepcopy(table['column_types'])

        return table, sql

    table, sql = lose_weight_for_database(table, sql, conceptnet_value_dict)

    origin_sql = sql['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]
    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    schema_link = copy.deepcopy(sql['schema_linking_matrix'])

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]

    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 6))
    col_set_type = np.zeros((len(col_set_iter), 4))
    col_text_type = table['column_types']

    assert len(schema_link[0]) == len(col_set_iter) + len(table_names)

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['schema_link'] = schema_link
    process_dict['col_text_type'] = col_text_type

    return process_dict, table, sql


def is_valid(rule_label, col_table_dict, sql):
    try:
        lf.build_tree(copy.copy(rule_label))
    except:
        print(rule_label)

    flag = False
    for r_id, rule in enumerate(rule_label):
        if type(rule) == C:
            try:
                assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
            except:
                flag = True
                print(sql['question'])
    return flag is False


def get_col_type(col_set_type, set_col_dict, col_set_dict):
    col_len = len(col_set_dict)
    col_type = np.zeros((col_len, 4))
    for set_id, col_ids in set_col_dict.items():
        for col_id in col_ids:
            col_type[col_id] = col_set_type[set_id]

    return col_type


def get_gold_schema_node(rule_label, tab_len, col_len):
    one_hot_tab = np.zeros(tab_len)
    one_hot_col = np.zeros(col_len)

    rule_label = rule_label.strip().split(' ')
    gold_col_id_set = []
    gold_tab_id_set = []
    for label_item in rule_label:
        if label_item.find('C(') == 0:
            col_id_s = label_item.find('(') + 1
            col_id_e = label_item.find(')')
            col_id = int(label_item[col_id_s:col_id_e])
            if col_id not in gold_col_id_set:
                gold_col_id_set.append(col_id)
        elif label_item.find('T(') == 0:
            tab_id_s = label_item.find('(') + 1
            tab_id_e = label_item.find(')')
            tab_id = int(label_item[tab_id_s:tab_id_e])
            if tab_id not in gold_tab_id_set:
                gold_tab_id_set.append(tab_id)
        else:
            pass

    one_hot_tab[gold_tab_id_set] = 1
    one_hot_col[gold_col_id_set] = 1
    return one_hot_tab, one_hot_col


def get_dependency_graph(dependency_tree):
    graph = []
    root_flag = 0
    node_id_set = [0]
    for node in dependency_tree:
        if node[0] == 'ROOT':
            root_flag = max(node_id_set)
            continue

        if root_flag+node[1] not in node_id_set:
            node_id_set.append(root_flag+node[1])
        if root_flag+node[2] not in node_id_set:
            node_id_set.append(root_flag+node[2])

        graph.append((root_flag+node[2]-1, root_flag+node[1]-1))

    dependency_graph = dgl.DGLGraph(graph)

    return dependency_graph


def get_parse_graph(parse_tree_edegs, parse_token_ids):

    for i in range(len(parse_token_ids)-1):
        parse_tree_edegs.append((parse_token_ids[i], parse_token_ids[i+1]))

    parse_graph = dgl.DGLGraph(parse_tree_edegs)

    return parse_graph


def to_batch_seq(sql_data, table_data, idxes, st, ed, epoch,
                 is_train=True):
    """

    :return:
    """
    examples = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        table = table_data[sql['db_id']]

        process_dict = process(sql, table)
        # if is_train and epoch < 20:
        #     process_dict, table, sql = process_random(sql, table)
        # else:
        #     process_dict = process(sql, table)
        # process_dict, table, sql = process_random(sql, table)

        # process_dict, table, sql = process_by_conceptnet(sql, table, is_train)

        for c_id, col_ in enumerate(process_dict['col_set_iter']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in col_:
                    process_dict['col_set_type'][c_id][0] += 1

        schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                       process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'], sql)

        col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)
        try:
            table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])
        except:
            print('#'*100)
            print(process_dict['tab_ids'])
            print(process_dict['col_iter'])
            print(table['column_names'])
            exit(0)

        col_set_dict, set_col_dict = get_col_set_dict(process_dict['tab_cols'], sql)

        colset_text_type = [COLSET_TYPE.index(process_dict['col_text_type'][set_col_dict[c_id][0]]) for c_id in
                            range(len(process_dict['col_set_iter']))]

        schema_graph = get_graph(table, col_set_dict, col_table_dict)
        schema_link = process_dict['schema_link']

        # dependency_graph = get_dependency_graph(sql['dependency_tree'])
        dependency_graph = None
        # parse_graph = get_parse_graph(sql['parse_tree_edegs'], sql['parse_token_ids'])
        parse_graph = None

        process_dict['col_set_iter'][0] = ['count', 'number', 'many']
        process_dict['col_iter'][0] = ['count', 'number', 'many']

        rule_label = None
        # if 'rule_label' in sql:
        #     try:
        #         rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]
        #     except:
        #         continue
        #     if is_valid(rule_label, col_table_dict=col_table_dict, sql=sql) is False:
        #         continue
        rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]

        one_hot_tab, one_hot_col = \
            get_gold_schema_node(sql['rule_label'], len(process_dict['table_names']), len(process_dict['col_set_iter']))

        example = Example(
            src_sent=process_dict['question_arg'],
            col_num=len(process_dict['col_set_iter']),
            vis_seq=(sql['question'], process_dict['col_set_iter'], sql['query']),
            tab_cols=process_dict['col_set_iter'],
            sql=sql['query'],
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_set_type'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            col_table_dict=col_table_dict,
            cols=process_dict['tab_cols'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict['col_set_type'],
            schema_graph=schema_graph,
            col_set_dict=col_set_dict,
            tgt_actions=rule_label,
            tgt_action_sent=copy.deepcopy(sql['rule_label']),
            schema_link=schema_link,
            align_table_one_hot=one_hot_tab,
            align_column_one_hot=one_hot_col,
            dependency_graph=dependency_graph,
            dependency_tree=None,
            parse_graph=parse_graph,
            parse_token_id=None,
            parse_dfs_label=None,
            colset_text_type=colset_text_type
        )
        example.sql_json = copy.deepcopy(sql)
        examples.append(example)

    if is_train:
        examples.sort(key=lambda e: -len(e.src_sent))
        return examples
    else:
        return examples


def epoch_train(model, optimizer, scheduler, batch_size, sql_data, table_data,
                args, epoch=0, loss_epoch_threshold=20, sketch_loss_coefficient=0.2):
    model.train()

    new_sql_data = []
    error_rule_label = 0
    for sql in sql_data:
        rule_label = sql['rule_label'].strip()
        if rule_label.find('{') >= 0:
            error_rule_label += 1
            continue
        else:
            new_sql_data.append(sql)

    sql_data = new_sql_data

    # shuffe
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    batch_cnt = 0
    # for s_data in sql_data:
    #     col_set = s_data['col_set']
    #     col_set = [' '.join([wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')]) for x in col_set]
    #     if len(col_set) != len(list(set(col_set))):
    #         print(col_set)
    # print('sql scan over !!!')
    # exit(0)

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch)

        if len(examples) == 0:
            st = ed
            batch_cnt += 1
            continue

        score = model.forward(examples)
        loss_sketch = -score[0]
        loss_lf = -score[1]
        # loss_att = -score[2]
        loss_align = score[3]

        loss_sketch = torch.mean(loss_sketch)
        loss_lf = torch.mean(loss_lf)
        # loss_att = torch.mean(loss_att)
        loss_align = torch.mean(loss_align)

        if epoch > loss_epoch_threshold:
            # loss = loss_lf + sketch_loss_coefficient * loss_sketch + loss_att
            loss = loss_lf + sketch_loss_coefficient * loss_sketch
            # loss = sketch_loss_coefficient * loss_sketch
        else:
            # loss = loss_lf + loss_sketch + loss_att
            loss = loss_lf + loss_sketch
            # loss = loss_sketch

        if torch.isnan(loss) or loss == np.inf:
            optimizer.zero_grad()
        else:
            loss.backward()

        # if loss == np.inf:
        #     print(loss_lf)
        #     print(loss_sketch)
        #     for example in examples:
        #         print([str(act) for act in example.tgt_actions])
        #     exit(0)

        loss = loss / args.acc_batch

        if ((batch_cnt + 1) % args.acc_batch) == 0:
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        if not torch.isnan(loss) and loss != np.inf:
            cum_loss += loss.data.cpu().numpy() * (ed - st)
        st = ed
        batch_cnt += 1

    if args.clip_grad > 0.:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()

    return cum_loss / len(sql_data)


def epoch_align_train(model, optimizer, batch_size, sql_data, table_data,
                args, epoch=0):
    model.train()
    # shuffe
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    batch_cnt = 0

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch)
        # examples = to_reprocess_batch_seq(sql_data, table_data, perm, st, ed)
        # optimizer.zero_grad()
        if len(examples) == 0:
            st = ed
            batch_cnt += 1
            continue

        loss_align = model.forward(examples)

        loss = torch.mean(loss_align)

        if torch.isnan(loss):
            optimizer.zero_grad()
        else:
            loss.backward()

        # loss = loss / args.acc_batch

        if ((batch_cnt + 1) % args.acc_batch) == 0:
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        if not torch.isnan(loss):
            cum_loss += loss.data.cpu().numpy() * (ed - st)
        st = ed
        batch_cnt += 1

    if args.clip_grad > 0.:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()

    return cum_loss / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, beam_size=3):
    model.eval()

    new_sql_data = []
    for sql in sql_data:
        rule_label = sql['rule_label'].strip()
        if rule_label.find('{') >= 0:
            continue
        else:
            new_sql_data.append(sql)

    sql_data = new_sql_data

    perm = list(range(len(sql_data)))
    st = 0

    json_datas = []
    sketch_correct, rule_label_correct, total = 0, 0, 0
    error = 0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch=0,
                                is_train=False)
        if len(examples) == 0:
            st = ed
            continue
        # examples = to_reprocess_batch_seq(sql_data, table_data, perm, st, ed,
        #                         is_train=False)
        for example in examples:
            results_all = model.parse(example, beam_size=beam_size)
            results = results_all[0]
            try:
                pred = " ".join([str(x) for x in results[0].actions])
            except Exception as e:
                # print('Epoch Acc: ', e)
                # print(results)
                # print(results_all)
                pred = 'Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)'

            simple_json = example.sql_json['pre_sql']

            simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
            simple_json['model_result'] = pred

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if truth_sketch == simple_json['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == simple_json['model_result']:
                rule_label_correct += 1
            total += 1

            json_datas.append(simple_json)
        st = ed
    return json_datas, float(sketch_correct) / float(total), float(rule_label_correct) / float(total)


def epoch_gold_acc(batch_size, sql_data, table_data):
    new_sql_data = []
    for sql in sql_data:
        rule_label = sql['rule_label'].strip()
        if rule_label.find('{') >= 0:
            continue
        else:
            new_sql_data.append(sql)

    sql_data = new_sql_data

    perm = list(range(len(sql_data)))
    st = 0

    json_datas = []
    sketch_correct, rule_label_correct, total = 0, 0, 0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch=0,
                                is_train=False)
        if len(examples) == 0:
            st = ed
            continue

        for example in examples:
            simple_json = example.sql_json['pre_sql']

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            simple_json['sketch_result'] = truth_sketch
            simple_json['model_result'] = truth_rule_label

            if truth_sketch == simple_json['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == simple_json['model_result']:
                rule_label_correct += 1
            total += 1

            json_datas.append(simple_json)
        st = ed
    return json_datas, float(sketch_correct) / float(total), float(rule_label_correct) / float(total)


def epoch_acc_with_spider_script(val_sql_data, val_table_data, data_path, print_log=False, dump_result=False):
    from eval_script.evaluation import build_foreign_key_map_from_json, evaluate_sqls
    from eval_script.semQL2sqlPro import transform
    kmaps = build_foreign_key_map_from_json(os.path.join(data_path, 'tables.json'))
    quess = []
    golds = []
    preds = []
    dbs = []
    semsqls = []
    presems = []
    for i, val in enumerate(val_sql_data):
        presems.append(val['model_result'])
        semsqls.append(val['rule_label'])
        quess.append(val['question'])
        golds.append(val['query'])
        # try:
        #     result = transform(val, val_table_data[val['db_id']], origin=val['model_result'])[0]
        # except:
        #     result = transform(val, val_table_data[val['db_id']], origin='Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)')[0]
        result = transform(val, val_table_data[val['db_id']], origin=val['model_result'])[0]

        preds.append(result)
        dbs.append(val['db_id'])
    acc = evaluate_sqls(presems, semsqls, quess, golds, preds, dbs, os.path.join(data_path, 'database'), 'match', kmaps, print_log=print_log, dump_result=dump_result)
    return acc


def remove_irrelevant_t(rule_label):
    rule_label_split = rule_label.split(' ')
    skip_flag = False
    rt_rule_label = []
    for label in rule_label_split:
        if skip_flag:
            skip_flag = False
            continue
        if label.find('C(') == 0:
            if label != 'C(0)':
                skip_flag = True
        rt_rule_label.append(label)

    return ' '.join(rt_rule_label)


def eval_acc(preds, sqls):
    sketch_correct, best_correct = 0, 0
    for i, (pred, sql) in enumerate(zip(preds, sqls)):
        # reprocess
        # if remove_irrelevant_t(pred['model_result']) == remove_irrelevant_t(sql['rule_label_reprocess']):
        # if pred['model_result'] == sql['rule_label_reprocess']:
        if pred['model_result'] == sql['rule_label']:
            best_correct += 1
    print(best_correct / len(preds))
    return best_correct / len(preds)


def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    table_data_new = {table['db_id']: table for table in table_data}

    if use_small:
        return sql_data[:], table_data_new
    else:
        return sql_data, table_data_new


# def precess_dataset(dataset_dir):
#     print("Producing for datasets...")
#
#     TRAIN_PATH = os.path.join(dataset_dir, "train.json")
#     DEV_PATH = os.path.join(dataset_dir, "dev.json")
#
#     PRO_TRAIN_PATH = os.path.join("spider/train_reprocess.json")
#     PRO_DEV_PATH = os.path.join("spider/dev_reprocess.json")
#
#     OUT_TRAIN_PATH = os.path.join(dataset_dir, "train_reprocess.json")
#     OUT_DEV_PATH = os.path.join(dataset_dir, "dev_reprocess.json")
#
#     def process(TRAIN_PATH, PRO_TRAIN_PATH, OUT_TRAIN_PATH):
#         train_sql_data = []
#         with open(TRAIN_PATH) as inf:
#             data = json.load(inf)
#             train_sql_data += data
#
#         pro_train_sql_data = []
#         with open(PRO_TRAIN_PATH) as inf:
#             data = json.load(inf)
#             pro_train_sql_data += data
#
#         assert len(pro_train_sql_data) == len(train_sql_data)
#         for (data, prodata) in zip(train_sql_data, pro_train_sql_data):
#             if data['origin_question_toks'] == prodata['origin_question_toks']:
#                 data['rule_label_reprocess'] = prodata['rule_label_reprocess']
#             else:
#                 print('NO MATCH!!!')
#                 print(data['origin_question_toks'])
#                 print(prodata['origin_question_toks'])
#                 exit(0)
#
#         with open(OUT_TRAIN_PATH, 'w') as outf:
#             json.dump(train_sql_data, outf)
#
#     process(TRAIN_PATH, PRO_TRAIN_PATH, OUT_TRAIN_PATH)
#     process(DEV_PATH, PRO_DEV_PATH, OUT_DEV_PATH)


def precess_dataset(dataset_dir):
    print("Producing for datasets...")

    TRAIN_PATH = os.path.join(dataset_dir, "train.json")
    DEV_PATH = os.path.join(dataset_dir, "dev.json")
    TABLE_PATH = os.path.join(dataset_dir, "tables.json")

    with open(TABLE_PATH) as inf:
        print("Loading data from %s" % TABLE_PATH)
        table_data = json.load(inf)

    table_data_new = {table['db_id']: table for table in table_data}

    OUT_TRAIN_PATH = os.path.join(dataset_dir, "train_reprocess.json")
    OUT_DEV_PATH = os.path.join(dataset_dir, "dev_reprocess.json")

    def process(TRAIN_PATH, OUT_TRAIN_PATH, table_data):
        train_sql_data = []
        with open(TRAIN_PATH) as inf:
            data = json.load(inf)
            train_sql_data += data

        unvalid = 0
        for data in train_sql_data:
            col_set = data['col_set']
            rule_label = data['rule_label']
            db_id = data['db_id']

            ori_rule_label = []
            rule_label_split = rule_label.split(' ')
            for i, rl_item in enumerate(rule_label_split):
                if rl_item == 'C(0)':
                    ori_rule_label.append(rl_item)
                    continue

                if rl_item.find('C(') == 0:
                    col_id_s = rl_item.find('(') + 1
                    col_id_e = rl_item.find(')')
                    col_id = int(rl_item[col_id_s:col_id_e])
                    assert rule_label_split[i + 1].find('T(') == 0
                    tab_id_s = rule_label_split[i + 1].find('(') + 1
                    tab_id_e = rule_label_split[i + 1].find(')')
                    table_id = int(rule_label_split[i + 1][tab_id_s:tab_id_e])
                    col_token = col_set[col_id]

                    for j, tab_col_pair in enumerate(table_data[db_id]['column_names']):
                        if col_token == tab_col_pair[1] and table_id == tab_col_pair[0]:
                            ori_rule_label.append('C({})'.format(str(j)))

                else:
                    ori_rule_label.append(rl_item)

            if len(rule_label_split) != len(ori_rule_label):
                print(rule_label_split)
                print(ori_rule_label)
                unvalid += 1
                for i, rl_item in enumerate(rule_label_split):
                    if rl_item.find('C(') == 0:
                        col_id_s = rl_item.find('(') + 1
                        col_id_e = rl_item.find(')')
                        col_id = int(rl_item[col_id_s:col_id_e])
                #         print(col_id)
                #         print(col_set[col_id])
                # print(table_data[db_id]['column_names'])
                # print(data['query'])
                # exit(0)
            else:
                data['rule_label_reprocess'] = ' '.join(ori_rule_label)
        print('wrong count: ', unvalid)
        with open(OUT_TRAIN_PATH, 'w') as outf:
            json.dump(train_sql_data, outf)

    process(TRAIN_PATH, OUT_TRAIN_PATH, table_data_new)
    process(DEV_PATH, OUT_DEV_PATH, table_data_new)


def load_dataset(dataset_dir, use_small=False):
    print("Loading from datasets...")

    # TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TABLE_PATH = os.path.join("save/regular_schemas.json")
    # TABLE_PATH = os.path.join("save/regular_schema_dbcontent.json")
    # reprocess
    # TRAIN_PATH = os.path.join(dataset_dir, "train.json")
    # DEV_PATH = os.path.join(dataset_dir, "dev.json")
    # TRAIN_PATH = os.path.join(dataset_dir, "train_reprocess.json")
    # DEV_PATH = os.path.join(dataset_dir, "dev_reprocess.json")
    # TRAIN_PATH = os.path.join(dataset_dir, "train_correct_linking_matrix_parse_v3_semql_v4.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train_link_test1.json")
    # DEV_PATH = os.path.join(dataset_dir, "dev_linking_matrix_parse_v3_semql_v4.json")
    # TRAIN_PATH = os.path.join(dataset_dir, "train_linking_matrix_v4.json")
    DEV_PATH = os.path.join(dataset_dir, "dev_link_test1.json")
    with open(TABLE_PATH) as inf:
        print("Loading data from %s" % TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)

    return train_sql_data, train_table_data, val_sql_data, val_table_data


def load_pointed_dataset(database_dir, dataset_dir, use_small=False):
    TABLE_PATH = database_dir
    DATA_PATH = dataset_dir

    with open(TABLE_PATH) as inf:
        print("Loading data from %s" % TABLE_PATH)
        table_data = json.load(inf)
    sql_data, table_data = load_data_new(DATA_PATH, table_data, use_small=use_small)

    return sql_data, table_data

def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))


def init_log_checkpoint_path(args):
    save_path = args.save
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dir_name = save_path + str(time_str)
    save_path = os.path.join(os.path.curdir, 'saved_model', dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path
