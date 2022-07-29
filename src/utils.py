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
from nltk.translate.bleu_score import sentence_bleu

from src.dataset import Example
from src.rule import lf
from src.rule.semQLPro import Sel, Order, Root, Filter, A, C, T, Root1, From, Group, C1, V

from src.models.rat_utils import get_relation_matrices
from preprocess.schema_linking import generate_matrix
from eval_script.evaluation import return_sql_type
from data_statistic.rephrase_generation import generate_rephrase_label
from data_statistic.rephrase_generation import Lemmanize

wordnet_lemmatizer = WordNetLemmatizer()
COLSET_TYPE = ['others', 'text', 'boolean', 'time', 'number']


def get_parent_match(rule_label):
    import src.rule.semQLPro as semQLPro
    from src.rule.semQLPro import Root1, Root, Group, From, C, C1, V, Sel, Filter, A, T, Order

    type_parent_match = {
        semQLPro.C: [semQLPro.C1, semQLPro.Group],
        semQLPro.C1: [semQLPro.V, semQLPro.Order],
        semQLPro.V: [semQLPro.A, semQLPro.Filter],
        semQLPro.A: [semQLPro.Sel],
        semQLPro.T: [semQLPro.From],
        semQLPro.Sel: [semQLPro.Root],
        semQLPro.From: [semQLPro.Root],
        semQLPro.Order: [semQLPro.Root],
        semQLPro.Group: [semQLPro.Root],
        semQLPro.Filter: [semQLPro.Filter, semQLPro.Root, semQLPro.Group],
        semQLPro.Root: [semQLPro.Root1, semQLPro.Filter, semQLPro.From],
    }

    rule_label.reverse()
    rl_len = len(rule_label) - 1
    parent_match = []
    for i, c_act in enumerate(rule_label[:rl_len]):
        for j, p_act in enumerate(rule_label[i + 1:]):
            if type(p_act) in type_parent_match[type(c_act)]:
                parent_match.append(rl_len - (i + 1 + j))
                break
    parent_match.append(-1)

    parent_match.reverse()
    rule_label.reverse()
    return parent_match


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


def get_table_colDict(tab_ids):
    table_col_dict = {}
    for ci, ti in enumerate(tab_ids):
        if ti != -1:
            table_col_dict[ti] = table_col_dict.get(ti, [0]) + [ci]

    col_table_dict = {}
    for key_item, value_item in table_col_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]

    return table_col_dict, col_table_dict


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


# *********** test **********
def schema_linking(question_arg, question_arg_type, one_hot_type, col_type, col_iter, sql):
    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            # question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            if question_arg[count_q] in col_iter:
                for i, col in enumerate(col_iter):
                    if col == question_arg[count_q]:
                        col_type[i][1] = 5
                # question_arg[count_q] = ['column'] + question_arg[count_q]
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
            # question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    if col_probase in col_iter:
                        for i, col in enumerate(col_iter):
                            if ' '.join(col) == col_probase:
                                col_type[i][2] = 5

                        # question_arg[count_q] = ['value'] + question_arg[count_q]
                        one_hot_type[count_q][5] = 1
                    else:
                        continue
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue

                    if col_probase in col_iter:
                        for i, col in enumerate(col_iter):
                            if ' '.join(col) == col_probase:
                                col_type[i][3] += 1

# *********** test **********
# def get_graph(table):
#     table_id = [i for i, _ in enumerate(table['table_names'])]
#     table_column = table['column_names']
#     prim_keys = table['primary_keys']
#     fore_keys = table['foreign_keys']
#     schema_graph = {}
#
#     # edges between primary nodes and table nodes
#     prim_tc_edges = []
#     prim_ct_edges = []
#     for i, k in enumerate(prim_keys):
#         prim_tc_edges.append((i, k))
#         prim_ct_edges.append((k, i))
#
#     # edges between foreign nodes and table nodes
#     fore_cc_edges = []
#     fore_invcc_edges = []
#     for fp in fore_keys:
#         fore_cc_edges.append((fp[0], fp[1]))
#         fore_invcc_edges.append((fp[1], fp[0]))
#
#     # edges between norm column nodes except primary nodes and table nodes
#     norm_tc_edges = []
#     norm_ct_edges = []
#     st_nodes = []
#     ts_nodes = []
#     for col, tc_pair in enumerate(table_column):
#         if col == 0:
#             for t_id in table_id:
#                 st_nodes.append((col, t_id))
#                 ts_nodes.append((t_id, col))
#         else:
#             if (col, tc_pair[0]) not in prim_ct_edges:
#                 norm_ct_edges.append((col, tc_pair[0]))
#                 norm_tc_edges.append((tc_pair[0], col))
#
#     schema_graph[('table', 'norm_t2c', 'column')] = norm_tc_edges
#     schema_graph[('column', 'norm_c2t', 'table')] = norm_ct_edges
#
#     schema_graph[('table', 'prim_t2c', 'column')] = prim_tc_edges
#     schema_graph[('column', 'prim_c2t', 'table')] = prim_ct_edges
#
#     schema_graph[('column', 'fore_c2c', 'column')] = fore_cc_edges
#     schema_graph[('column', 'fore_invc2c', 'column')] = fore_invcc_edges
#
#     schema_graph[('column', 's2t', 'table')] = st_nodes
#     schema_graph[('table', 't2s', 'column')] = ts_nodes
#
#     schema_graph = dgl.heterograph(schema_graph)
#
#     return schema_graph


def get_graph(table):
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
    fore_tc_edges = []
    fore_ct_edges = []
    for fp in fore_keys:
        # fore_t_id0 = table_column[fp[0]][0]
        # fore_t_id1 = table_column[fp[1]][0]
        # if (fore_t_id0, fp[0]) not in prim_tc_edges:
        #     fore_tc_edges.append((fore_t_id0, fp[0]))
        #     fore_ct_edges.append((fp[0], fore_t_id0))
        #
        # if (fore_t_id1, fp[1]) not in prim_tc_edges:
        #     fore_tc_edges.append((fore_t_id1, fp[1]))
        #     fore_ct_edges.append((fp[1], fore_t_id1))

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

    # schema_graph[('table', 'fore_t2c', 'column')] = fore_tc_edges
    # schema_graph[('column', 'fore_c2t', 'table')] = fore_ct_edges

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

    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    schema_link = copy.deepcopy(sql['schema_linking_matrix'])
    one_hot_type = np.zeros((len(question_arg_type), 6))
    col_text_type = table['column_types']

    col_type = np.zeros((len(col_iter), 4))

    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_text_type'] = col_text_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['names'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['schema_link'] = schema_link
    process_dict['col_type'] = col_type
    process_dict['question_col_match'] = copy.deepcopy(sql['question_col_match'])
    process_dict['question_table_match'] = copy.deepcopy(sql['question_table_match'])
    process_dict['rephrase_sentence_idx'] = copy.deepcopy(sql['rephrase_sentence_idx'])
    process_dict['rephrase_schema_idx'] = copy.deepcopy(sql['rephrase_schema_idx'])
    process_dict['rephrase_result'] = copy.deepcopy(sql['rephrase_result'])
    process_dict['schema_items'] = copy.deepcopy(sql['schema_items'])

    return process_dict


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


def feed_vocab(vocab, template_sentence):
    sentence_idx = []
    for token in template_sentence:
        if token not in vocab:
            vocab[token] = len(vocab)

        sentence_idx.append(vocab[token])

    return vocab, sentence_idx


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

        for c_id, col_ in enumerate(process_dict['col_iter']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in col_:
                    process_dict['col_type'][c_id][0] += 1

        # schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
        #                process_dict['one_hot_type'], process_dict['col_type'], process_dict['col_iter'], sql)

        table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])
        table_col_dict, col_table_dict = get_table_colDict(process_dict['tab_ids'])
        # print(process_dict['tab_ids'])
        # print(table_col_dict)
        # print(col_table_dict)
        # exit(0)

        primary_keys = copy.deepcopy(table['primary_keys'])
        foreign_keys = copy.deepcopy(table['foreign_keys'])
        foreign_table_keys = [[process_dict['tab_ids'][f_key[0]], process_dict['tab_ids'][f_key[1]]] for f_key in
                              table['foreign_keys']]

        col_text_type = [COLSET_TYPE.index(process_dict['col_text_type'][c_id]) for c_id in
                            range(len(process_dict['col_iter']))]

        schema_graph = get_graph(table)
        # schema_link = process_dict['schema_link']

        process_dict['col_iter'][0] = ['count', 'all']
        rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]

        one_hot_tab, one_hot_col = \
            get_gold_schema_node(sql['rule_label'], len(process_dict['table_names']), len(process_dict['col_iter']))

        question_table_match = copy.deepcopy(process_dict['question_table_match'])
        question_col_match = copy.deepcopy(process_dict['question_col_match'])

        rephrase_sentence_idx = copy.deepcopy(process_dict['rephrase_sentence_idx'])
        rephrase_schema_idx = copy.deepcopy(process_dict['rephrase_schema_idx'])

        rephrase_result = copy.deepcopy(sql['rephrase_result'])
        rephrase_schema_items = copy.deepcopy(sql['schema_items'])

        def filter_match(question_table_match, question_col_match, col_table_dict):
            new_qc_match = {}
            matched_table = []
            qt_match = []
            for value, qt in question_table_match.items():
                new_qc_match[value] = []
                qt_match = qt_match + qt

            for _, t_id in qt_match:
                matched_table.append(t_id)

            for value, qc in question_col_match.items():
                for q_id, c_id in qc:
                    t_id = col_table_dict[c_id][0]
                    if t_id in matched_table:
                        new_qc_match[value].append([q_id, c_id])

            # question_table_match['content_exact'] = []
            # question_table_match['content_partial'] = []

            return question_table_match, new_qc_match

        question_table_match, question_col_match = filter_match(question_table_match, question_col_match,
                                                                col_table_dict)

        process_dict['question_table_match'] = question_table_match
        process_dict['question_col_match'] = question_col_match

        schema_link = generate_matrix(process_dict)

        sql_hardness = return_sql_type(sql['query'], sql['db_id'])

        example = Example(
            src_sent=process_dict['question_arg'],
            col_num=len(process_dict['col_iter']),
            vis_seq=(sql['question'], process_dict['col_iter'], sql['query']),
            tab_cols=process_dict['col_iter'],
            sql=sql['query'],
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_type'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            cols=process_dict['tab_cols'],
            tab_col_match=process_dict['tab_ids'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict['question_arg'],
            schema_graph=schema_graph,
            tgt_actions=rule_label,
            tgt_action_sent=copy.deepcopy(sql['rule_label']),
            schema_link=schema_link,
            align_table_one_hot=one_hot_tab,
            align_column_one_hot=one_hot_col,
            col_text_type=col_text_type,
            table_col_dict=table_col_dict,
            question_table_match=question_table_match,
            question_col_match=question_col_match,
            table_col=table_col_dict,
            col_table=col_table_dict,
            foreign_keys=foreign_keys,
            primary_keys=primary_keys,
            foreign_table_keys=foreign_table_keys,
            sql_hardness=sql_hardness,
            rephrase_sentence_idx=rephrase_sentence_idx,
            rephrase_schema_idx=rephrase_schema_idx,
            rephrase_result=rephrase_result,
            rephrase_schema_items=rephrase_schema_items
        )
        relative_matrix = get_relation_matrices(example)
        example.relative_matrix = copy.deepcopy(relative_matrix)
        example.sql_json = copy.deepcopy(sql)
        examples.append(example)

    if is_train:
        # examples.sort(key=lambda e: -(len(e.src_sent)+e.col_num+e.table_len))
        examples.sort(key=lambda e: -len(e.src_sent))
        return examples
    else:
        return examples


# TODO: UNCOMPLETED
def get_schema_idx(schema_items, rs_type, tab_names, col_names, col_tab_type):
    schema_type = [t for t in rs_type if t < 2]

    assert len(schema_type) == len(schema_items)

    # print('#' * 100)
    # print(schema_type)
    # print(schema_items)
    # print(tab_names)
    # print(col_names)

    schema_idx = []
    renew_idx = []
    for e, (type, item) in enumerate(zip(schema_type, schema_items)):
        item_idx = []
        if type == 0:
            for i, tab in enumerate(tab_names):
                if tab == item:
                    item_idx.append(i)
        else:
            for i, col in enumerate(col_names):
                if col == item:
                    item_idx.append(i)

        if len(item_idx) == 0:
            continue
        schema_idx.append(item_idx)
        renew_idx.append(e)

    schema_type = [t for e, t in enumerate(schema_type) if e in renew_idx]
    # print(schema_idx)
    # print(len(schema_idx))

    single_schema_idx = []
    for i, (type, item_idx) in enumerate(zip(schema_type, schema_idx)):
        if len(item_idx) == 1:
            single_schema_idx.append(item_idx[0])

        else:
            cur_tab = -1

            all_previous_tabs = []
            for j in range(len(schema_idx)):
                if schema_type[j] == 0:
                    cur_tab = schema_idx[j][0]
                    all_previous_tabs.append(cur_tab)
                else:
                    if len(schema_idx[j]) == 1:
                        all_previous_tabs.append(col_tab_type[schema_idx[j][0]])

            all_previous_tabs = set(all_previous_tabs)
            if len(all_previous_tabs) == 1:
                cur_tab = list(all_previous_tabs)[0]

            for j in range(i, len(schema_idx)):
                if schema_type[j] == 0:
                    cur_tab = schema_idx[j][0]
                    break

            for i_idx in item_idx:
                if cur_tab >= 0 and col_tab_type[i_idx] == cur_tab:
                    single_schema_idx.append(i_idx)
                    break

            if len(single_schema_idx) != i + 1:
                single_schema_idx.append(item_idx[0])

    # print(col_tab_type)
    # print(single_schema_idx)
    # print(len(single_schema_idx))

    assert len(single_schema_idx) == len(schema_idx)

    return single_schema_idx


def feed_rephrase(sql, vocab, table):
    rephrase_sentence, schema_items, rephrase_sentence_type, rephrase_template = generate_rephrase_label(
        sql['rule_label'],
        table[sql['db_id']])
    vocab, sentence_idx = feed_vocab(vocab, rephrase_template)

    tab_names = table[sql['db_id']]['table_names']
    col_names = [col[1] for col in table[sql['db_id']]['column_names']]
    col_tab_type = [col[0] for col in table[sql['db_id']]['column_names']]

    schema_idx = get_schema_idx(schema_items, rephrase_sentence_type, tab_names, col_names, col_tab_type)

    return vocab, sentence_idx, schema_idx, rephrase_sentence, schema_items


def epoch_train(model, optimizer, scheduler, batch_size, sql_data, table_data,
                args, epoch=0, loss_epoch_threshold=20, sketch_loss_coefficient=0.2,
                fine_tune_alpha=np.array([0.25, 0.25, 0.25, 0.25])):
    model.train()

    # shuffle
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    batch_cnt = 0

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch)

        if len(examples) == 0:
            st = ed
            batch_cnt += 1
            continue

        score = model.forward(examples)
        if len(score) == 5:
            if args.train_type == 'CE':
                loss_sketch = -score[2]
                loss_lf = -score[0]

                loss_sketch = torch.mean(loss_sketch)
                loss_lf = torch.mean(loss_lf)
                if epoch > loss_epoch_threshold:
                    loss = loss_lf + sketch_loss_coefficient * loss_sketch
                else:
                    loss = loss_lf + loss_sketch
            elif args.train_type == 'FT':
                loss_finetune = -score[3]
                sql_hardness_id = score[4]
                batch_fine_tune_alpha = fine_tune_alpha[sql_hardness_id]

                if args.cuda:
                    batch_fine_tune_alpha = torch.from_numpy(batch_fine_tune_alpha).cuda()

                loss = batch_fine_tune_alpha * loss_finetune
                loss = loss.mean()

        elif len(score) == 2:
            loss = -torch.mean(score[0])
        else:
            print('loss wrong!')
            exit(0)

        loss = loss / args.acc_batch

        if torch.isnan(loss) or loss == np.inf:
            optimizer.zero_grad()
        else:
            loss.backward()

        if ((batch_cnt + 1) % args.acc_batch) == 0:
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        cum_loss += loss.data.cpu().numpy() * (ed - st)
        st = ed
        batch_cnt += 1

    if args.clip_grad > 0.:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()

    return cum_loss / len(sql_data)


def epoch_rephrase_train(model, optimizer, scheduler, batch_size, sql_data, table_data,
                args, epoch=0, loss_epoch_threshold=20, sketch_loss_coefficient=0.2,
                fine_tune_alpha=np.array([0.25, 0.25, 0.25, 0.25])):
    model.train()

    # shuffle
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    batch_cnt = 0

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch)
        if len(examples) == 0:
            st = ed
            batch_cnt += 1
            continue

        # schema_flags = [0 for _ in examples]
        # for i, example in enumerate(examples):
        #     for action_tm1 in example.rephrase_sentence:
        #         if action_tm1 == 1 or action_tm1 == 0:
        #             schema_flags[i] += 1
        #
        # for i, example in enumerate(examples):
        #     if len(example.rephrase_schema) != schema_flags[i]:
        #         print('#'*100)
        #         print(example.rephrase_result)
        #         print(example.rephrase_schema_items)
        #         count += 1
        score = model.forward(examples)
        loss_act = -score[0]
        loss_class = -score[1]

        loss_act = torch.mean(loss_act)
        loss_class = torch.mean(loss_class)

        if epoch > loss_epoch_threshold:
            loss = loss_act + sketch_loss_coefficient * loss_class
        else:
            loss = loss_act + loss_class

        loss = loss / args.acc_batch

        if torch.isnan(loss) or loss == np.inf:
            optimizer.zero_grad()
        else:
            loss.backward()

        if ((batch_cnt + 1) % args.acc_batch) == 0:
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        cum_loss += loss.data.cpu().numpy() * (ed - st)
        st = ed
        batch_cnt += 1

    if args.clip_grad > 0.:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()

    return cum_loss / len(sql_data)


def epoch_multitask_train(model, optimizer, scheduler, batch_size, sql_data, table_data,
                args, epoch=0, loss_epoch_threshold=20, sketch_loss_coefficient=0.2,
                fine_tune_alpha=np.array([0.25, 0.25, 0.25, 0.25])):
    model.train()

    # shuffle
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    batch_cnt = 0

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch)
        if len(examples) == 0:
            st = ed
            batch_cnt += 1
            continue

        # rephrase
        score = model.rephrase_forward(examples)
        loss_act = -score[0]
        loss_class = -score[1]

        loss_act = torch.mean(loss_act)
        loss_class = torch.mean(loss_class)

        if epoch > loss_epoch_threshold:
            loss_rephrase = loss_act + loss_class
        else:
            loss_rephrase = loss_act + loss_class

        # text2sql
        score = model.text2sql_forward(examples)
        if args.train_type == 'CE':
            loss_sketch = -score[2]
            loss_lf = -score[0]

            loss_sketch = torch.mean(loss_sketch)
            loss_lf = torch.mean(loss_lf)
            if epoch > loss_epoch_threshold:
                loss_text2sql = loss_lf + sketch_loss_coefficient * loss_sketch
            else:
                loss_text2sql = loss_lf + loss_sketch
        elif args.train_type == 'FT':
            loss_finetune = -score[3]
            sql_hardness_id = score[4]
            batch_fine_tune_alpha = fine_tune_alpha[sql_hardness_id]

            if args.cuda:
                batch_fine_tune_alpha = torch.from_numpy(batch_fine_tune_alpha).cuda()

            loss_text2sql = batch_fine_tune_alpha * loss_finetune
            loss_text2sql = loss_text2sql.mean()

        loss = loss_rephrase + loss_text2sql

        loss = loss / args.acc_batch

        if torch.isnan(loss) or loss == np.inf:
            optimizer.zero_grad()
        else:
            loss.backward()

        if ((batch_cnt + 1) % args.acc_batch) == 0:
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

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

        for example in examples:
            results_all = model.parse(example, beam_size=beam_size)

            results = results_all[0]
            if len(results) == 0:
                pred = 'Root1(3) Root(7) From(1) T(1) Sel(0) A(3) V(0) C1(0) C(0)'
                sketch = 'Root1(3) Root(7) From(1) Sel(0) A(3) V(0) C1(0)'
            else:
                pred = " ".join([str(x) for x in results[0].actions])
                sketch = " ".join(str(x) for x in results_all[1])

            simple_json = example.sql_json['pre_sql']

            simple_json['sketch_result'] = sketch
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


def epoch_rephrase_acc(model, batch_size, sql_data, table_data, beam_size=3):
    model.eval()

    perm = list(range(len(sql_data)))
    st = 0

    schema_item_correct, rephrase_exact_correct, total, bleu = 0, 0, 0, 0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch=0,
                                is_train=False)
        if len(examples) == 0:
            st = ed
            continue

        pred_rephrases, chosen_schemas = model.parse(examples, beam_size=beam_size)

        for e_id, example in enumerate(examples):

            pred_rephrase = pred_rephrases[e_id]
            chosen_schema = [Lemmanize(item) for item in chosen_schemas[e_id]]

            pred_rephrase = ' '.join(pred_rephrase[:-1])
            pred_rephrase = pred_rephrase.replace('<VALUE>', 'value')
            truth_rephrase = example.rephrase_result
            truth_rephrase = ' '.join(truth_rephrase.split(' ')[:-1])
            truth_schema_items = [Lemmanize(item) for item in example.rephrase_schema_items]

            if len(truth_schema_items) == len(chosen_schema):
                schema_match = True
                for item in truth_schema_items:
                    if item not in chosen_schema:
                        schema_match = False
                        break

                for item in chosen_schema:
                    if item not in truth_schema_items:
                        schema_match = False
                        break

                if schema_match:
                    schema_item_correct += 1

            if truth_rephrase == pred_rephrase:
                rephrase_exact_correct += 1

            bleu += sentence_bleu([pred_rephrase.split(' ')], truth_rephrase.split(' '))

            total += 1

        st = ed
    return float(schema_item_correct) / float(total), float(rephrase_exact_correct) / float(total), bleu / float(total)


def epoch_multitask_acc(model, batch_size, sql_data, table_data, beam_size=3):
    model.eval()

    perm = list(range(len(sql_data)))
    st = 0

    json_datas = []
    sketch_correct, rule_label_correct, schema_item_correct, rephrase_exact_correct, total, bleu = 0, 0, 0, 0, 0, 0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch=0,
                                is_train=False)
        if len(examples) == 0:
            st = ed
            continue

        # rephrase
        pred_rephrases, chosen_schemas = model.rephrase_parse(examples, beam_size=beam_size)

        for e_id, example in enumerate(examples):
            pred_rephrase = pred_rephrases[e_id]
            chosen_schema = [Lemmanize(item) for item in chosen_schemas[e_id]]

            pred_rephrase = ' '.join(pred_rephrase[:-1])
            pred_rephrase = pred_rephrase.replace('<VALUE>', 'value')
            truth_rephrase = example.rephrase_result
            truth_rephrase = ' '.join(truth_rephrase.split(' ')[:-1])
            truth_schema_items = [Lemmanize(item) for item in example.rephrase_schema_items]

            if len(truth_schema_items) == len(chosen_schema):
                schema_match = True
                for item in truth_schema_items:
                    if item not in chosen_schema:
                        schema_match = False
                        break

                for item in chosen_schema:
                    if item not in truth_schema_items:
                        schema_match = False
                        break

                if schema_match:
                    schema_item_correct += 1

            if truth_rephrase == pred_rephrase:
                rephrase_exact_correct += 1

            bleu += sentence_bleu([pred_rephrase.split(' ')], truth_rephrase.split(' '))

            # text2sql
            results_all = model.text2sql_parse(example, beam_size=beam_size)

            results = results_all[0]
            if len(results) == 0:
                pred = 'Root1(3) Root(7) From(1) T(1) Sel(0) A(3) V(0) C1(0) C(0)'
                sketch = 'Root1(3) Root(7) From(1) Sel(0) A(3) V(0) C1(0)'
            else:
                pred = " ".join([str(x) for x in results[0].actions])
                sketch = " ".join(str(x) for x in results_all[1])

            simple_json = example.sql_json['pre_sql']

            simple_json['sketch_result'] = sketch
            simple_json['model_result'] = pred

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if truth_sketch == simple_json['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == simple_json['model_result']:
                rule_label_correct += 1

            json_datas.append(simple_json)

            total += 1

        st = ed
    return json_datas, float(schema_item_correct) / float(total), float(rephrase_exact_correct) / float(total), \
           bleu / float(total)


def dump_rephrase_predict(model, batch_size, sql_data, table_data, output_file, beam_size=3):
    model.eval()

    perm = list(range(len(sql_data)))
    st = 0
    new_sql_data = []

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        # reprocess
        examples = to_batch_seq(sql_data, table_data, perm, st, ed, epoch=0,
                                is_train=False)
        if len(examples) == 0:
            st = ed
            continue

        pred_rephrases, chosen_schemas = model.parse(examples, beam_size=beam_size)

        for e_id, example in enumerate(examples):
            new_data = {}
            data = sql_data[perm[st+e_id]]
            pred_rephrase = pred_rephrases[e_id]

            pred_rephrase = ' '.join(pred_rephrase[:-1])
            pred_rephrase = pred_rephrase.replace('<VALUE>', 'value')
            truth_rephrase = example.rephrase_result
            truth_rephrase = ' '.join(truth_rephrase.split(' ')[:-1])

            new_data['db_id'] = data['db_id']
            new_data['query'] = data['query']
            new_data['query_toks'] = data['query_toks']
            new_data['query_toks_no_value'] = data['query_toks_no_value']
            new_data['sql'] = data['sql']

            new_data['question'] = pred_rephrase
            new_data['gold_question'] = truth_rephrase
            new_data['question_toks'] = pred_rephrase.split(' ')

            new_sql_data.append(new_data)

        st = ed

    with open(output_file, 'w') as f:
        json.dump(new_sql_data, f, indent=4)

    print('Success dumping to: ', output_file)


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

        try:
            result = transform(val, val_table_data[val['db_id']], origin=val['model_result'])[0]
        except:
            print('cannot be transformed', val['model_result'])
            result = transform(val, val_table_data[val['db_id']],
                               origin='Root1(3) Root(7) Sel(0) A(3) V(0) C1(0) C(0) From(1) T(0)')[0]

        preds.append(result)
        dbs.append(val['db_id'])
    acc = evaluate_sqls(presems, semsqls, quess, golds, preds, dbs, 'database', 'match', kmaps, print_log=print_log, dump_result=dump_result)
    return acc


def eval_acc(preds, sqls):
    sketch_correct, best_correct = 0, 0
    for i, (pred, sql) in enumerate(zip(preds, sqls)):
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


def load_dataset(dataset_dir, vocabulary={}, use_small=False):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join("spider/tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train_link_test6.json")
    # DEV_PATH = os.path.join(dataset_dir, "rewrite_sparc_cong_noacc_dev_link.json")
    # DEV_PATH = os.path.join(dataset_dir, "predict_link_rephrase.json")
    DEV_PATH = os.path.join(dataset_dir, "dev_link.json")

    # TABLE_PATH = os.path.join("spider/tables.json")
    # TRAIN_PATH = os.path.join(dataset_dir, "train_link_rephrase.json")
    # DEV_PATH = os.path.join(dataset_dir, "dev_link_rephrase.json")

    with open(TABLE_PATH) as inf:
        print("Loading data from %s" % TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)

    new_sql_data = []
    error_rule_label = 0
    rm_label = 0
    for sql in train_sql_data:
        if sql['db_id'] == 'baseball_1':
            rm_label += 1
            continue
        rule_label = sql['rule_label'].strip()
        if rule_label.find('{') >= 0:
            error_rule_label += 1
            continue
        else:
            vocabulary, sentence_idx, schema_idx, rephrase_result, schema_items = feed_rephrase(sql, vocabulary, train_table_data)
            if '*' in schema_items or '' in schema_items:
                continue
            sql['rephrase_result'] = rephrase_result
            sql['schema_items'] = schema_items
            sql['rephrase_sentence_idx'] = sentence_idx
            sql['rephrase_schema_idx'] = schema_idx
            new_sql_data.append(sql)

    train_sql_data = new_sql_data

    new_val_sql_data = []
    error_rule_label = 0
    rm_label = 0
    for sql in val_sql_data:
        if sql['db_id'] == 'baseball_1':
            rm_label += 1
            continue
        rule_label = sql['rule_label'].strip()
        if rule_label.find('{') >= 0:
            error_rule_label += 1
            continue
        else:
            vocabulary, sentence_idx, schema_idx, rephrase_result, schema_items = feed_rephrase(sql, vocabulary, val_table_data)
            if '*' in schema_items or '' in schema_items:
                continue

            sql['rephrase_result'] = rephrase_result
            sql['schema_items'] = schema_items
            sql['rephrase_sentence_idx'] = sentence_idx
            sql['rephrase_schema_idx'] = schema_idx
            new_val_sql_data.append(sql)

    val_sql_data = new_val_sql_data

    return train_sql_data, train_table_data, val_sql_data, val_table_data, vocabulary


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
