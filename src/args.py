# -*- coding: utf-8 -*-

import random
import argparse
import torch
import numpy as np

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--evaluating', action='store_true', help='is evaling')
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--lr_scheduler', action='store_true', help='use learning rate scheduler')
    arg_parser.add_argument('--lr_scheduler_gammar', default=0.5, type=float, help='decay rate of learning rate scheduler')
    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')
    arg_parser.add_argument('--loss_epoch_threshold', default=20, type=int, help='loss epoch threshold')
    arg_parser.add_argument('--sketch_loss_coefficient', default=0.2, type=float, help='sketch loss coefficient')
    arg_parser.add_argument('--sentence_features', action='store_true', help='use sentence features')
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='rnn',
                            help='model name')
    arg_parser.add_argument('--enc_type', choices=['shadowgnn', 'rat', 'shadowgnn_rat', 'rat_shadowgnn'], default='shadowgnn',
                            help='encoder type')

    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')

    arg_parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    arg_parser.add_argument('--glove_embed_path', default="glove.42B.300d.txt", type=str)
    arg_parser.add_argument('--bert', action='store_true', help='use bert')

    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    arg_parser.add_argument('--acc_batch', default=4, type=int, help='accumulation size')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='size of word embeddings')
    arg_parser.add_argument('--col_embed_size', default=768, type=int, help='size of word embeddings')

    arg_parser.add_argument('--action_embed_size', default=768, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=768, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=768, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=768, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    arg_parser.add_argument('--sldp', default=0.2, type=float, help='schema linking dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')
    arg_parser.add_argument('--layer_num', default=2, type=int, help='layer size')
    arg_parser.add_argument('--ave_layer', default=2, type=int, help='average layer size')

    # rat config
    arg_parser.add_argument('--rat_hidden_size', default=768, type=int, help='rat_hidden_size')
    arg_parser.add_argument('--rat_head_num', default=8, type=int, help='rat_head_num')
    arg_parser.add_argument('--rat_ff_dim', default=768, type=int, help='rat_ff_dim')
    arg_parser.add_argument('--rat_relation_types', default=39, type=int, help='rat_relation_types')
    arg_parser.add_argument('--rat_layers', default=4, type=int, help='rat_layers')

    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')


    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')

    arg_parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')


    arg_parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    arg_parser.add_argument('--toy', action='store_true',
                            help='If set, use small data; used for fast debugging.')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    arg_parser.add_argument('--finetune_lambda', default=2, type=int, help='finetune hyperparameter')
    arg_parser.add_argument('--train_type', choices=['CE', 'FT'], default='CE', help='cross entropy or fine tune')

    arg_parser.add_argument('--dataset', default="./data", type=str)

    arg_parser.add_argument('--epoch', default=50, type=int, help='Maximum Epoch')
    arg_parser.add_argument('--save', default='./', type=str,
                            help="Path to save the checkpoint and logs of epoch")

    return arg_parser

def init_config(arg_parser):
    args = arg_parser.parse_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))
    random.seed(int(args.seed))
    return args

