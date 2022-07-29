# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import traceback

import os
import torch
import torch.optim as optim
import tqdm
import copy
import re
import numpy as np

from src import args as arg
from src import utils
# from src.models.bert_shadowgnn_rat_baseline import Shadowgnn
# from src.models.bert_shadowgnn_rat0 import Shadowgnn
from src.models.grappa_shadowgnn_rat2 import Shadowgnn
# from src.models.grappa_rat_baseline import Shadowgnn

from src.rule import semQLPro
from src.BertAdam import *
from transformers import get_linear_schedule_with_warmup, AdamW


def _get_bert_optimizer(model, lr_rate, optim_func):
    bert_params_id = list(map(id, model.bert_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params_id, model.parameters())
    param_group = [
        {'params': model.bert_encoder.parameters(), 'lr': 0.1 * lr_rate},
        {'params': base_params, 'lr': lr_rate}
    ]

    optimizer = optim_func(lr=lr_rate, params=param_group)

    return optimizer


def _get_rat_optimizer(model, lr_rate=7.44e-4, optim_func=None):
    bert_params_id = list(map(id, model.bert_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params_id, model.parameters())
    param_group = [
        {'params': model.bert_encoder.parameters(), 'lr': 3e-6},
        {'params': base_params, 'lr': lr_rate}
    ]

    optimizer = AdamW(params=param_group)

    return optimizer


def _get_bert_decay_optimizer(model, lr_rate, optim_func):
    bert_params_id = list(map(id, model.bert_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params_id, model.parameters())

    param_groups = [
        {'params': base_params, 'lr': lr_rate}
    ]

    layerwise_lr_decay = 0.9

    pretrain_config_layer = model.bert_encoder.config.num_hidden_layers
    for n, p in model.bert_encoder.named_parameters():
        param_group = {}
        param_group["params"] = p
        if 'embeddings' in n:
            depth = 0
        elif 'encoder.layer' in n:
            depth = int(re.search(r"encoder.layer.(\d+)", n).group(1)) + 1
        else:
            depth = pretrain_config_layer

        param_group["lr"] = lr_rate * layerwise_lr_decay ** (pretrain_config_layer - depth)

        param_groups.append(param_group)

    optimizer = optim_func(lr=lr_rate, params=param_groups)

    return optimizer


def _get_bertadam_optimizer(args, model, train_cnt, lr_scheduler='warmup_linear'):
    bert_named_params = list(filter(lambda p: p[1].requires_grad and 'bert' in p[0], model.named_parameters()))
    other_params = list(
        filter(lambda p: id(p[1]) not in list(map(id, [i[1] for i in bert_named_params]))
                         and p[1].requires_grad, model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_named_params if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': 1e-5},
        {'params': [p for n, p in bert_named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 1e-5},
        {'params': [p for n, p in other_params]}
    ]
    num_train_optimization_steps = (train_cnt // args.batch_size) * args.epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.05,
                         t_total=num_train_optimization_steps, schedule=lr_scheduler, e=1e-9,
                         max_grad_norm=args.clip_grad)

    return optimizer


def train(args):
    """
    :param args:
    :return:
    """

    vocabulary = {
        '<TABLE>': 0,
        '<COLUMN>': 1,
        '<EOS>': 2
    }

    grammar = semQLPro.Grammar()
    sql_data, table_data, val_sql_data,\
    val_table_data, vocabulary = utils.load_dataset(args.dataset, vocabulary=vocabulary, use_small=args.toy)

    model = Shadowgnn(args, grammar)

    if args.cuda: model.cuda()

    # now get the optimizer
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)
    # optimizer_cls = AdamW
    if args.bert:
        # optimizer = _get_bert_decay_optimizer(model, args.lr, optimizer_cls)
        optimizer = _get_bert_optimizer(model, args.lr, optimizer_cls)
        # optimizer = _get_rat_optimizer(model, args.lr, optimizer_cls)
        # optimizer = _get_bertadam_optimizer(args, model, len(table_data))
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    print('Enable Learning Rate Scheduler: ', args.lr_scheduler)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21, 41], gamma=args.lr_scheduler_gammar)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=800, num_training_steps=40000)
    else:
        scheduler = None

    print('Loss epoch threshold: %d' % args.loss_epoch_threshold)
    print('Sketch loss coefficient: %f' % args.sketch_loss_coefficient)

    if args.load_model:
        print('load pretrained model from %s'% (args.load_model))
        pretrained_model = torch.load(args.load_model,
                                         map_location=lambda storage, loc: storage)
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in model.state_dict().keys():
                del pretrained_modeled[k]

        model.load_state_dict(pretrained_modeled)

    eval_data_distribution = np.array([250, 440, 174, 170])
    test_data_distribution = np.array([470, 857, 461, 357])

    eval_data_distribution = eval_data_distribution / np.sum(eval_data_distribution)
    test_data_distribution = test_data_distribution / np.sum(test_data_distribution)

    # # TODO: remove
    fine_tune_alpha = np.array([0.25, 0.25, 0.25, 0.25])
    if args.train_type == 'FT':
        model.load_state_dict(torch.load('eval_model/best_model723.model'))
        json_datas, train_acc, _ = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
                                          beam_size=args.beam_size)
        spider_acc = utils.epoch_acc_with_spider_script(json_datas, val_table_data, args.dataset,
                                                        print_log=args.toy, dump_result=True)
        # hardness_eval_acc = spider_acc[:-1]
        # hardness_eval_acc = np.array(hardness_eval_acc)
        # fine_tune_alpha = 1 - hardness_eval_acc
        fine_tune_alpha = eval_data_distribution
        fine_tune_alpha = fine_tune_alpha / np.sum(fine_tune_alpha)

    if args.bert:
        model.word_emb = None
    else:
        model.word_emb = utils.load_word_emb(args.glove_embed_path, args.toy)
    # model.word_emb = None
    # begin train

    model_save_path = utils.init_log_checkpoint_path(args)
    utils.save_args(args, os.path.join(model_save_path, 'config.json'))
    best_dev_acc = .0

    try:
        with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
            for epoch in tqdm.tqdm(range(args.epoch)):
                if args.lr_scheduler:
                    scheduler.step()

                # model.load_state_dict(torch.load('eval_model/best_model723.model'))
                # json_datas, train_acc, _ = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
                #                                   beam_size=args.beam_size)
                # spider_acc = utils.epoch_acc_with_spider_script(json_datas, val_table_data, args.dataset,
                #                                                 print_log=args.toy, dump_result=True)
                # print('SUC !')
                # exit(0)

                # tqdm.tqdm.write('fine_tune_alpha: ' + str(fine_tune_alpha.tolist()))
                epoch_begin = time.time()
                loss = utils.epoch_train(model, optimizer, scheduler, args.batch_size, sql_data, table_data, args,
                                   loss_epoch_threshold=args.loss_epoch_threshold,
                                   sketch_loss_coefficient=args.sketch_loss_coefficient,
                                   fine_tune_alpha=fine_tune_alpha)
                epoch_end = time.time()
                json_datas, dev_sketch_acc, dev_acc = utils.epoch_acc(model, args.batch_size, val_sql_data,
                                                                      val_table_data, beam_size=args.beam_size)

                spider_acc = utils.epoch_acc_with_spider_script(json_datas, val_table_data, args.dataset,
                                                                print_log=args.toy, dump_result=True)
                # hardness_eval_acc = spider_acc[:-1]
                # hardness_eval_acc = np.array(hardness_eval_acc)
                # fine_tune_alpha = 1 - hardness_eval_acc
                fine_tune_alpha = eval_data_distribution
                fine_tune_alpha = fine_tune_alpha / np.sum(fine_tune_alpha)

                if spider_acc[-1] >= best_dev_acc:
                    utils.save_checkpoint(model, os.path.join(model_save_path, 'best_model.model'))
                    best_dev_acc = spider_acc[-1]

                # utils.save_checkpoint(model, os.path.join(model_save_path, '{%s}_{%s}.model') % (epoch, spider_acc))

                log_str = 'Epoch: %d, Loss: %f, dev sketch Acc: %f, dev Acc: %f, time: %f\n' % (
                    epoch + 1, loss, dev_sketch_acc, spider_acc[-1], epoch_end - epoch_begin)
                tqdm.tqdm.write(log_str)
                epoch_fd.write(log_str)
                epoch_fd.flush()

    except Exception as e:
        # Save model
        utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
        print(e)
        tb = traceback.format_exc()
        print(tb)


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    print(args)
    train(args)
