import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import random
from src.rule import semQLPro
from src import args as arg
from src import utils
from src.rule.semQLPro import Sel, Order, Root, Filter, A, C, T, Root1, From, Group, C1, V
from data_statistic.template_utils import transform


def sample_template(sample_data):

    coarse_rule_label_dict = {}

    for data in sample_data:
        rule_label = [eval(x) for x in data['rule_label'].strip().split(' ')]
        coarse_rule_label = [x for x in rule_label if (type(x) is not semQLPro.C and type(x) is not semQLPro.T) or (
                    type(x) is semQLPro.C and x.id_c == 0)]

        # coarse_rule_label_type = [str(type(x)) for x in coarse_rule_label]
        coarse_rule_label_type = [str(x) for x in coarse_rule_label]
        coarse_rule_label = ' '.join(coarse_rule_label_type)
        # data['coarse_rule_label'] = coarse_rule_label
        if coarse_rule_label not in coarse_rule_label_dict:
            coarse_rule_label_dict[coarse_rule_label] = []
        coarse_rule_label_dict[coarse_rule_label].append(data)

    return coarse_rule_label_dict


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)

    sql_data, table_data, val_sql_data, \
    val_table_data = utils.load_dataset(args.dataset, use_small=args.toy)

    train_coarse_rule_label_dict = sample_template(sql_data)

    dev_coarse_rule_label_dict = sample_template(val_sql_data)
    coarse_len = []

    j = 0
    for i, (k, v) in enumerate(train_coarse_rule_label_dict.items()):

        # if k in dev_coarse_rule_label_dict:
        #     print('#' * 50, ' Template {}'.format(j), '#' * 50)
        #     print(len(v), len(dev_coarse_rule_label_dict[k]))
        #     j += 1
        # # else:
        # #     print(len(v))

        coarse_len.append(len(v))

    assert sum(coarse_len) == len(sql_data)

    print('Total template number in train set: ', len(train_coarse_rule_label_dict))

    print('Total template number in dev set: ', len(dev_coarse_rule_label_dict))

    extra_dk = []
    for dk in dev_coarse_rule_label_dict.keys():
        if dk not in train_coarse_rule_label_dict.keys():
            extra_dk.append(dk)

    print('Extra SQL in dev set: ', len(extra_dk))

    extra_v_len = 0
    for k, v in dev_coarse_rule_label_dict.items():
        if k not in train_coarse_rule_label_dict:
            extra_v_len += len(v)

    print('Extra SQL question in dev set({}): '.format(len(val_sql_data)), extra_v_len)

    template_envidence = {}

    for i, (k, v) in enumerate(train_coarse_rule_label_dict.items()):

        if k not in template_envidence:
            template_envidence[k] = []

        for d in v:
            query_toks_no_value = ' '.join(d['query_toks_no_value'])
            template_envidence[k].append([query_toks_no_value, d['db_id'], d['rule_label'], d['question']])

    # with open(os.path.join('save/', 'template_envidence.json'), 'w') as outf:
    #     json.dump(template_envidence, outf)

    annotated_template = open(os.path.join('save/', 'annotated_template.txt'), 'w')

    for k, v in template_envidence.items():
        annotated_template.write('RL: ' + k + '\n')
        sql_template, tc_cols = transform(v[0][2], table_data[v[0][1]], v[0][2])
        annotated_template.write('SQL: ' + sql_template + '\n')
        N = len(v)
        n = min(3, N)
        sample_n = random.sample(range(0, N), n)
        quess = []
        sample_tc_cols = []
        for j in sample_n:
            ques = v[j][3].encode('ascii', 'ignore').decode('ascii')
            query = v[j][0].encode('ascii', 'ignore').decode('ascii')
            quess.append(ques + ' -> ' + query)

            sql_template, tc_cols = transform(v[j][2], table_data[v[j][1]], v[j][2])
            sample_tc_cols.append(tc_cols)

        for h, eg in enumerate(quess):
            annotated_template.write('EG{}: '.format(h) + str(eg) + '\n')
            annotated_template.write('TC{}: '.format(h) + str(sample_tc_cols[h]) + '\n')

        annotated_template.write('TP: ' + 'TBA' + '\n')
        annotated_template.write('\n')

    annotated_template.close()