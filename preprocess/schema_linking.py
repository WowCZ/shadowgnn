# -*- coding: utf-8 -*-

import os, sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.utils import check_match
from preprocess.utils import wordnet_lemmatizer
from preprocess.relation_types import SchemaLinkingTypes
from pattern.en import lemma


def generate_matrix(sample):
    num_toks = len(sample['question_arg'])
    num_table, num_column = len(sample['table_names']), len(sample['names'])
    matrix = [[SchemaLinkingTypes.NONE for _ in range(num_table + num_column)] for _ in range(num_toks)]
    for match in sample['question_table_match']['exact']:
        matrix[match[0]][match[1]] = SchemaLinkingTypes.Q_T_E
    for match in sample['question_table_match']['partial']:
        matrix[match[0]][match[1]] = SchemaLinkingTypes.Q_T_P
    for match in sample['question_table_match']['content_exact']:
        matrix[match[0]][match[1]] = SchemaLinkingTypes.Q_T_C_E
    for match in sample['question_table_match']['content_partial']:
        matrix[match[0]][match[1]] = SchemaLinkingTypes.Q_T_C_P
    for match in sample['question_col_match']['exact']:
        matrix[match[0]][match[1] + num_table] = SchemaLinkingTypes.Q_C_E
    for match in sample['question_col_match']['partial']:
        matrix[match[0]][match[1] + num_table] = SchemaLinkingTypes.Q_C_P
    for match in sample['question_col_match']['content_exact']:
        matrix[match[0]][match[1] + num_table] = SchemaLinkingTypes.Q_C_C_E
    for match in sample['question_col_match']['content_partial']:
        matrix[match[0]][match[1] + num_table] = SchemaLinkingTypes.Q_C_C_P
    return matrix


def Lemmanize(x):
    y = [lemma(wordnet_lemmatizer.lemmatize(x_item.lower())) for x_item in x]
    return ' '.join(y)


def process_samples(samples, table):
    """

    :param samples:
    :param args:
    :return:
    """

    from tqdm import tqdm
    for entry in tqdm(samples, total=len(samples)):
        # question_toks = [' '.join(x) for x in entry['question_arg']]
        # table_names = entry['table_names']
        # header_toks = entry['names']
        question_toks = [Lemmanize(x) for x in entry['question_arg']]
        table_names = [Lemmanize(x.split(' ')) for x in entry['table_names']]
        header_toks = [Lemmanize(x.split(' ')) for x in entry['names']]

        question_col_match = check_match(question_toks, header_toks)
        question_table_match = check_match(question_toks, table_names)

        question_col_content_exact_match = []
        question_col_content_partial_match = []
        question_table_content_exact_match = []
        question_table_content_partial_match = []
        for c_id, column in enumerate(table[entry['db_id']]['column_names_original']):
            if c_id == 0:
                continue
            # col_set_id = entry['col_set'].index(entry['names'][c_id])
            table_id = column[0]
            try:
                contents = [str(x).lower() for x in column[2]]
            except:
                contents = []
            question_content_match = check_match(question_toks, contents)
            for content_exact_m in question_content_match[0]:
                if [content_exact_m[0], c_id] not in question_col_content_exact_match:
                    question_col_content_exact_match.append([content_exact_m[0], c_id])
                if [content_exact_m[0], table_id] not in question_table_content_exact_match:
                    question_table_content_exact_match.append([content_exact_m[0], table_id])
            for content_partial_m in question_content_match[1]:
                if [content_partial_m[0], c_id] not in question_col_content_partial_match:
                    question_col_content_partial_match.append([content_partial_m[0], c_id])
                if [content_partial_m[0], table_id] not in question_table_content_partial_match:
                    question_table_content_partial_match.append([content_partial_m[0], table_id])

        entry['question_col_match'] = {'exact': question_col_match[0], 'partial': question_col_match[1],
                                       'content_exact': question_col_content_exact_match, 'content_partial': question_col_content_partial_match}
        entry['question_table_match'] = {'exact': question_table_match[0], 'partial': question_table_match[1],
                                       'content_exact': question_table_content_exact_match, 'content_partial': question_table_content_partial_match}

        entry['schema_linking_matrix'] = generate_matrix(entry)

    return samples


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()

    # loading samples
    with open(args.table_path, 'r', encoding='utf8') as f:
        table = json.load(f)
    with open(args.data_path, 'r', encoding='utf8') as f:
        raw_samples = json.load(f)

    table_new = {tab['db_id']: tab for tab in table}

    # process samples
    processed_samples = process_samples(raw_samples, table_new)

    with open(args.output, 'w') as f:
        json.dump(processed_samples, f, indent=4)

    # from tqdm import tqdm
    #
    # max_len = 0
    # total_len = []
    # longer_than_rl = []
    # for entry in tqdm(raw_samples, total=len(raw_samples)):
    #     rule_label = entry['rule_label']
    #     rl_len = len(rule_label.split(' '))
    #     total_len.append(rl_len)
    #     if rl_len > 60:
    #         longer_than_rl.append(rl_len)
    #     if rl_len > max_len:
    #         max_len = rl_len
    #
    # print(max_len)
    # print(sum(total_len) / len(total_len))
    # print(len(longer_than_rl))
    # print(longer_than_rl)

    # import copy
    # import src.rule.semQLPro as semQLPro
    # from src.rule.semQLPro import Root1, Root, Group, From, C, C1, V, Sel, Filter, A, T, Order
    #
    # rule_label = 'Root1(3) Root(6) From(1) T(1) Sel(2) A(0) V(0) C1(0) C(9) A(0) V(0) C1(0) C(10) A(0) V(0) C1(0) C(13) Order(0) C1(0) C(13)'
    # rule_label = [eval(l) for l in rule_label.split(' ')]
    #
    # type_parent_match = {
    #     semQLPro.C: [semQLPro.C1, semQLPro.Group],
    #     semQLPro.C1: [semQLPro.V, semQLPro.Order],
    #     semQLPro.V: [semQLPro.A, semQLPro.Filter],
    #     semQLPro.A: [semQLPro.Sel],
    #     semQLPro.T: [semQLPro.From],
    #     semQLPro.Sel: [semQLPro.Root],
    #     semQLPro.From: [semQLPro.Root],
    #     semQLPro.Order: [semQLPro.Root],
    #     semQLPro.Filter: [semQLPro.Filter, semQLPro.Root, semQLPro.Group],
    #     semQLPro.Root: [semQLPro.Root1, semQLPro.Filter, semQLPro.From],
    # }
    #
    # rule_label.reverse()
    # rl_len = len(rule_label) - 1
    # parent_match = []
    # for i, c_act in enumerate(rule_label[:rl_len]):
    #     for j, p_act in enumerate(rule_label[i+1:]):
    #         if type(p_act) in type_parent_match[type(c_act)]:
    #             parent_match.append(rl_len - (i+1+j))
    #             break
    # parent_match.append(-1)
    #
    # parent_match.reverse()
    # rule_label.reverse()
    # print(parent_match)
    # print(' '.join([str(act) for act in rule_label]))

    # import src.rule.semQLPro as semQLPro
    # from src.rule.semQLPro import Root1, Root, Group, From, C, C1, V, Sel, Filter, A, T, Order
    # def get_availableClass(actions):
    #     """
    #     return the available action class
    #     :return:
    #     """
    #
    #     # TODO: it could be update by speed
    #     # return the available class using rule
    #     # FIXME: now should change for these 11: "Filter 1 ROOT",
    #     def check_type(lists):
    #         for s in lists:
    #             if type(s) == int:
    #                 return False
    #         return True
    #
    #     stack = [semQLPro.Root1]
    #     for action in actions:
    #         # if type(action) == semQLPro.Root1 and action.id_c == 0 and not self.is_sketch:
    #         #     infer_action = action.get_next_action(is_sketch=self.is_sketch)
    #         #     print(infer_action)
    #         #     exit(0)
    #         infer_action = action.get_next_action()
    #         infer_action.reverse()
    #         if stack[-1] is type(action):
    #             stack.pop()
    #             # check if the are non-terminal
    #             # if check_type(infer_action):
    #             stack.extend(infer_action)
    #         else:
    #             print(str(action))
    #             print([str(action) for action in actions])
    #             print(str(stack[-1]))
    #             print([str(action) for action in stack])
    #             raise RuntimeError("Not the right action")
    #
    #     result = stack[-1] if len(stack) > 0 else None
    #
    #     return result
    #
    #
    # rule_label = 'Root1(3) Root(6) From(1) T(1) Sel(2) A(0) V(0) C1(0) C(9) A(0) V(0) C1(0) C(10) A(0) V(0) C1(0) C(13) Order(0) C1(0) C(13)'
    # rule_label = [eval(l) for l in rule_label.split(' ')]
    #
    # result = get_availableClass(rule_label)
    #
    # if not result:
    #     print('compele')
    # else:
    #     print('error')

