import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import random
import copy
import tqdm
from src.rule import semQLPro
from src import args as arg
from src import utils
from src.rule.semQLPro import Sel, Order, Root, Filter, A, C, T, Root1, From, Group, C1, V
from data_statistic.template_utils import transform
from data_statistic.generation_rule import generate_sentence_template

from preprocess.utils import wordnet_lemmatizer
from pattern.en import lemma
from bert_score import score


def Lemmanize(x):
    y = [lemma(wordnet_lemmatizer.lemmatize(x_item.lower())) for x_item in x.split('_')]
    return ' '.join(y)


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

        query_toks_no_value = ' '.join(data['query_toks_no_value'])
        coarse_rule_label_dict[coarse_rule_label].append(
            [query_toks_no_value, data['db_id'], data['rule_label'], data['question']])

    return coarse_rule_label_dict


def rephare_rule(template, tc_cols):
    schema_items = []
    template_split = template.split(' ')

    rephrase_sentence = []
    rephrase_sentence_type = []
    rephrase_template = []
    col_count = 0
    jump = 0
    tab_name = ''
    for i, word in enumerate(template_split):
        # word = word.lstrip('(').rstrip(')')
        # if word != '':
        #     continue

        if jump > 0:
            jump -= 1
            continue

        if word == '<COLUMN>':
            cur_column = tc_cols[col_count].split('.')
            tab_name = cur_column[0].replace('_', ' ')
            col_name = cur_column[1].replace('_', ' ')
            if col_name.find(tab_name) >= 0:
                col_contain_tab = True
            else:
                col_contain_tab = False

            if i+2 < len(template_split) and template_split[i+2] == '<TABLE>' and col_contain_tab:
                jump = 2

            col_count += 1

            rephrase_sentence.append(Lemmanize(col_name))
            rephrase_sentence_type.append(1)
            rephrase_template.append('<COLUMN>')

            schema_items.append(col_name)

        elif word == '<TABLE>':
            if col_count < len(tc_cols) and tc_cols[col_count].find('*') >= 0:
                tab_name = tc_cols[col_count].split('.')[0]
                col_count += 1

            rephrase_sentence.append(Lemmanize(tab_name))
            rephrase_sentence_type.append(0)
            rephrase_template.append('<TABLE>')

            schema_items.append(tab_name)

        elif word == '<VALUE>':
            rephrase_sentence.append('value')
            rephrase_sentence_type.append(2)
            rephrase_template.append('<VALUE>')

        else:
            rephrase_sentence.append(word)
            rephrase_sentence_type.append(2)
            rephrase_template.append(word)

    return ' '.join(rephrase_sentence), schema_items, rephrase_sentence_type, rephrase_template


def generate_rephrase_label(rule_label, schema):
    sql_template, tc_cols = transform(rule_label, schema, rule_label)
    template = generate_sentence_template(sql_template)
    rephrase_sentence, schema_items, rephrase_sentence_type, rephrase_template = rephare_rule(template, tc_cols)

    return rephrase_sentence, schema_items, rephrase_sentence_type, rephrase_template


def dump_rephrase_data(sql_data, table_data, output_file):
    new_sql_data = []
    for data in sql_data:
        new_data = {}
        sql_template, tc_cols = transform(data['rule_label'], table_data[data['db_id']], data['rule_label'])
        template = generate_sentence_template(sql_template)
        rephrase_sentence, _, _, _ = rephare_rule(template, tc_cols)

        new_data['db_id'] = data['db_id']
        new_data['query'] = data['query']
        new_data['query_toks'] = data['query_toks']
        new_data['query_toks_no_value'] = data['query_toks_no_value']
        new_data['sql'] = data['sql']

        new_data['question'] = rephrase_sentence
        new_data['question_toks'] = rephrase_sentence.split(' ')

        new_sql_data.append(new_data)

    with open(output_file, 'w') as f:
        json.dump(new_sql_data, f, indent=4)

    print('Success dumping to: ', output_file)

if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)

    sql_data, table_data, val_sql_data, \
    val_table_data = utils.load_dataset(args.dataset, use_small=args.toy)

    dump_rephrase_data(sql_data, table_data, 'data/train_rephrase.json')
    dump_rephrase_data(val_sql_data, table_data, 'data/dev_rephrase.json')

    # train_template_envidence = sample_template(sql_data)
    #
    # dev_template_envidence = sample_template(val_sql_data)

    # rephrase_score_f = open(os.path.join('example/', 'rephrase_score.txt'), 'w')
    # query_score_f = open(os.path.join('example/', 'query_score.txt'), 'w')
    #
    # question_f = open(os.path.join('example/', 'question.txt'), 'w')
    # rephrase_f = open(os.path.join('example/', 'rephrase.txt'), 'w')
    # query_f = open(os.path.join('example/', 'query.txt'), 'w')

    #
    # Q_F1_scores = []
    # R_F1_scores = []
    # for k, v in tqdm.tqdm(train_template_envidence.items()):
    #     sql_template, tc_cols = transform(v[0][2], table_data[v[0][1]], v[0][2])
    #     template = generate_sentence_template(sql_template)
    #
    #     questions = []
    #     queries = []
    #     rephrase_sentences = []
    #
    #     for j in range(len(v)):
    #         ques = v[j][3].encode('ascii', 'ignore').decode('ascii')
    #         query = v[j][0].encode('ascii', 'ignore').decode('ascii')
    #
    #         _, tc_cols = transform(v[j][2], table_data[v[j][1]], v[j][2])
    #
    #         rephrase_sentence = rephare_rule(template, tc_cols)
    #
    #         questions.append(ques)
    #         queries.append(query)
    #         rephrase_sentences.append(rephrase_sentence)
    #         question_f.write(ques + '\n')
    #         rephrase_f.write(rephrase_sentence + '\n')
    #         query_f.write(query + '\n')
    #
    #     # Q_P, Q_R, Q_F1 = score(
    #     #     questions, queries,
    #     #     rescale_with_baseline=True, lang="en"
    #     # )
    #     #
    #     # R_P, R_R, R_F1 = score(
    #     #     questions, rephrase_sentences,
    #     #     rescale_with_baseline=True, lang="en"
    #     # )
    #     #
    #     # Q_F1_avg = Q_F1.mean().item()
    #     # R_F1_avg = R_F1.mean().item()
    #     #
    #     # rephrase_score_f.write(str(R_F1_avg) + ' ')
    #     # query_score_f.write(str(Q_F1_avg) + ' ')
    #     #
    #     # Q_F1_scores.append(Q_F1_avg)
    #     # R_F1_scores.append(R_F1_avg)
    #
    # # rephrase_score_f.flush()
    # # query_score_f.flush()
    # question_f.flush()
    # rephrase_f.flush()
    # query_f.flush()
    #
    # # rephrase_score_f.close()
    # # query_score_f.close()
    # question_f.close()
    # rephrase_f.close()
    # query_f.close()
    #
    # # print('AVG of Q_F1_scores: ', sum(Q_F1_scores) / len(Q_F1_scores))
    # # print('AVG of R_F1_scores: ', sum(R_F1_scores) / len(R_F1_scores))



    # # rephrase_score_f = open(os.path.join('example/', 'rephrase_score.txt'), 'w')
    # query_score_f = open(os.path.join('example/', 'query_score.txt'), 'w')
    #
    # with open("example/question.txt") as f:
    #     questions = [line.strip() for line in f]
    #
    # with open("example/query.txt") as f:
    #     queries = [line.strip() for line in f]
    #
    # with open("example/rephrase.txt") as f:
    #     rephrase_sentences = [line.strip() for line in f]
    #
    # Q_P, Q_R, Q_F1 = score(
    #     questions, queries,
    #     rescale_with_baseline=True, lang="en"
    # )
    #
    # for s in Q_F1.tolist():
    #     query_score_f.write(str(round(s, 4)) + ' ')
    #
    # query_score_f.flush()
    # query_score_f.close()