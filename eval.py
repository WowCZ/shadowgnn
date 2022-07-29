# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/27
# @Author  : Jiaqi&Zecheng
# @File    : eval.py
# @Software: PyCharm
"""


import torch
from src import args as arg
from src import utils
from src.models.bert_shadowgnn_rat_baseline import Shadowgnn
from src.rule import semQLPro


def evaluate(args):
    """
    :param args:
    :return:
    """

    grammar = semQLPro.Grammar()
    sql_data, table_data, val_sql_data,\
    val_table_data= utils.load_dataset(args.dataset, use_small=args.toy)

    model = Shadowgnn(args, grammar)

    if args.cuda: model.cuda()

    # print('load pretrained model from %s'% (args.load_model))
    # pretrained_model = torch.load(args.load_model,
    #                                  map_location=lambda storage, loc: storage)
    # import copy
    # pretrained_modeled = copy.deepcopy(pretrained_model)
    # for k in pretrained_model.keys():
    #     if k not in model.state_dict().keys():
    #         del pretrained_modeled[k]
    #
    # model.load_state_dict(pretrained_modeled)

    # model.word_emb = utils.load_word_emb(args.glove_embed_path)
    model.word_emb = None

    # json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
    #                        beam_size=args.beam_size)
    json_datas, sketch_acc, acc = utils.epoch_gold_acc(args.batch_size, sql_data, table_data)
    spider_acc = utils.epoch_acc_with_spider_script(json_datas, table_data, args.dataset,
                                                    print_log=args.toy, dump_result=True)
    print('Sketch Acc: %f, Acc: %f' % (sketch_acc, acc))
    # import json
    # with open('./predict_lf.json', 'w') as f:
    #     json.dump(json_datas, f)

if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    print(args)
    evaluate(args)
