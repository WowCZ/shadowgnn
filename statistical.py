# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-

from src import args as arg
from src import utils
from src.rule import semQLPro


def statistical(args):
    """
    :param args:
    :return:
    """
    sql_data, table_data, val_sql_data,\
    val_table_data= utils.load_dataset(args.dataset, use_small=args.toy)

    train_dbs = {}
    for data in sql_data:
        if data['db_id'] not in train_dbs:
            train_dbs[data['db_id']] = {
                'labels_cnt': 0,
                'columns_cnt': 0,
                'tabels_cnt': 0
            }
            train_dbs[data['db_id']]['columns_cnt'] = len(data['names'])
            train_dbs[data['db_id']]['tabels_cnt'] = len(data['table_names'])
        train_dbs[data['db_id']]['labels_cnt'] += 1

    print(len(train_dbs))

    labels_count = []
    columns_count = []
    tables_count = []
    for k, v in train_dbs.items():
        labels_count.append(v['labels_cnt'])
        columns_count.append(v['columns_cnt'])
        tables_count.append(v['tabels_cnt'])

    print(sum(labels_count) / len(labels_count))
    print(sum(columns_count) / len(columns_count))
    print(sum(tables_count) / len(tables_count))


if __name__ == '__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    # print(args)
    statistical(args)
