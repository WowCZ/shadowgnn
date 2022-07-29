# -*- coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.rule.semQLPro as semQLPro
from src.rule.semQLPro import Sel, Order, Root, Filter, A, C, T, Root1, From, Group, C1, V

def get_parent_match(rule_label):
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


def get_all_valQ_path(rule_label, parent_match):
    valQ_path = []
    for node_id, (apply_rule, p_id) in enumerate(zip(rule_label, parent_match)):
        if type(apply_rule) == V:
            parent_rule = rule_label[p_id]
            if type(parent_rule) == Filter:
                parent_rule_id = parent_rule.id_c

                if parent_rule_id >= 2 and parent_rule_id <= 10:
                    col_info = []
                    cur_v_id = apply_rule.id_c
                    if cur_v_id > 0:
                        col_info_len = 4
                    else:
                        col_info_len = 2
                    for i in range(col_info_len):
                        col_info.append(str(rule_label[node_id+i+1]))

                    p_p_id = p_id
                    v_path_info = [str(rule_label[node_id])]
                    while p_p_id != -1:
                        v_path_info.append(str(rule_label[p_p_id]))
                        p_p_id = parent_match[p_p_id]

                    v_path_info.reverse()

                    valQ_path.append(v_path_info + col_info + ['column_value'])

        elif type(apply_rule) == Order:
            cur_node_id = apply_rule.id_c
            if cur_node_id > 3:
                col_info = []
                if cur_node_id > 5:
                    col_info_len = 4
                else:
                    col_info_len = 2

                for i in range(col_info_len):
                    col_info.append(str(rule_label[node_id + i + 1]))

                p_p_id = p_id
                v_path_info = [str(rule_label[node_id])]
                while p_p_id != -1:
                    v_path_info.append(str(rule_label[p_p_id]))
                    p_p_id = parent_match[p_p_id]

                v_path_info.reverse()

                valQ_path.append(v_path_info + col_info + ['limit_value'])

    return valQ_path


if __name__ == '__main__':
    import json
    from tqdm import tqdm
    with open('data/train_link_test6.json', 'r', encoding='utf8') as f1:
        sql_datas = json.load(f1)

    dump_datas = {}
    for data in tqdm(sql_datas):
        question_arg = ' '.join(data['origin_question_toks'])
        rule_label = data['rule_label']
        rule_label = [eval(rule) for rule in rule_label.split(' ')]

        parent_match = get_parent_match(rule_label)
        valQ_path = get_all_valQ_path(rule_label, parent_match)

        dump_datas[question_arg] = valQ_path

    with open('data/train_value_Q.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(dump_datas))
