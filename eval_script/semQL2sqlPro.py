# -*- coding: utf-8 -*-

import argparse
import traceback
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rule.graph import Graph
from src.rule.semQLPro import Sel, Order, Root, Filter, A, C, T, Root1, From, Group, C1, V
from preprocess.utils import load_dataSets
from eval_script.evaluation import evaluate_sqls, build_foreign_key_map_from_json


def split_logical_form(lf):
    indexs = [i + 1 for i, letter in enumerate(lf) if letter == ')']
    indexs.insert(0, 0)
    components = list()
    for i in range(1, len(indexs)):
        components.append(lf[indexs[i - 1]:indexs[i]].strip())
    return components


def pop_front(array):
    if len(array) == 0:
        return 'None'
    return array.pop(0)


def is_end(components, transformed_sql, is_root_processed, is_where):
    end = False
    c = pop_front(components)
    c_instance = eval(c)

    if isinstance(c_instance, Root) and is_root_processed:
        # intersect, union, except
        end = True
    elif isinstance(c_instance, Filter):
        if is_where:
            if 'where' not in transformed_sql:
                end = True
            else:
                num_conjunction = 0
                for f in transformed_sql['where']:
                    if isinstance(f, str) and (f == 'and' or f == 'or'):
                        num_conjunction += 1
                current_filters = len(transformed_sql['where'])
                valid_filters = current_filters - num_conjunction
                if valid_filters >= num_conjunction + 1:
                    end = True
        else:
            if 'having' not in transformed_sql:
                end = True

    elif isinstance(c_instance, Group):
        if 'group' not in transformed_sql:
            end = True
        elif len(transformed_sql['group']) == 0:
            end = False
        else:
            end = True
    elif isinstance(c_instance, Order):
        if 'order' not in transformed_sql:
            end = True
        elif len(transformed_sql['order']) == 0:
            end = False
        else:
            end = True

    elif isinstance(c_instance, Sel):
        if len(transformed_sql['select']) == 0:
            end = False
        else:
            end = True

    elif isinstance(c_instance, From):
        if len(transformed_sql['from']) == 0:
            end = False
        else:
            end = True

    components.insert(0, c)
    return end


def _transform(components, transformed_sql, ori_col_names, table_names, col_table):
    processed_root = False

    def _get_filter_clause(components, c_instance, ori_col_names, table_names, col_table):

        op = c_instance.production.split()[1]
        if op == 'and' or op == 'or':
            filter_clause = op
        else:
            val = eval(pop_front(components))
            if len(val.production.split()) == 2:
                val_op = 'none'
            else:
                val_op = val.production.split()[1]

            if len(c_instance.production.split()) == 3:
                if val_op == 'none':
                    c1_left = eval(pop_front(components))
                    c1_left_op = c1_left.production.split()[1]
                    c1_left_c = eval(pop_front(components))
                    c1_left_c_name = ori_col_names[c1_left_c.id_c]
                    c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                    filter_clause = [op, val_op, [c1_left_op, c1_left_t_name, c1_left_c_name], None, None]
                else:
                    c1_left = eval(pop_front(components))
                    c1_left_op = c1_left.production.split()[1]
                    c1_left_c = eval(pop_front(components))
                    c1_left_c_name = ori_col_names[c1_left_c.id_c]
                    c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                    c1_right = eval(pop_front(components))
                    c1_right_op = c1_right.production.split()[1]
                    c1_right_c = eval(pop_front(components))
                    c1_right_c_name = ori_col_names[c1_right_c.id_c]
                    c1_right_t_name = table_names[col_table[c1_right_c.id_c]]
                    filter_clause = \
                        [op, val_op, [c1_left_op, c1_left_t_name, c1_left_c_name],
                         [c1_right_op, c1_right_t_name, c1_right_c_name], None]
            else:
                # Subquery
                new_dict = dict()
                new_dict['sql'] = transformed_sql['sql']
                if val_op == 'none':
                    c1_left = eval(pop_front(components))
                    c1_left_op = c1_left.production.split()[1]
                    c1_left_c = eval(pop_front(components))
                    c1_left_c_name = ori_col_names[c1_left_c.id_c]
                    c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                    filter_clause = [op, val_op, [c1_left_op, c1_left_t_name, c1_left_c_name], None,
                                     _transform(components, new_dict, ori_col_names, table_names, col_table)]
                else:
                    c1_left = eval(pop_front(components))
                    c1_left_op = c1_left.production.split()[1]
                    c1_left_c = eval(pop_front(components))
                    c1_left_c_name = ori_col_names[c1_left_c.id_c]
                    c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                    c1_right = eval(pop_front(components))
                    c1_right_op = c1_right.production.split()[1]
                    c1_right_c = eval(pop_front(components))
                    c1_right_c_name = ori_col_names[c1_right_c.id_c]
                    c1_right_t_name = table_names[col_table[c1_right_c.id_c]]
                    filter_clause = \
                        [op, val_op, [c1_left_op, c1_left_t_name, c1_left_c_name],
                         [c1_right_op, c1_right_t_name, c1_right_c_name],
                         _transform(components, new_dict, ori_col_names, table_names, col_table)]
        return components, filter_clause

    is_where_flag = False
    is_having_flag = False
    while len(components) > 0:
        if is_end(components, transformed_sql, processed_root, is_where_flag):
            break
        c = pop_front(components)
        c_instance = eval(c)
        if isinstance(c_instance, Root):
            processed_root = True
            transformed_sql['from'] = list()
            transformed_sql['select'] = list()
            if c_instance.id_c == 0:
                transformed_sql['where'] = list()
                transformed_sql['group'] = list()
                transformed_sql['having'] = list()
                transformed_sql['order'] = list()
                is_where_flag = True
            elif c_instance.id_c == 1:
                transformed_sql['group'] = list()
                transformed_sql['having'] = list()
                transformed_sql['order'] = list()
            elif c_instance.id_c == 2:
                transformed_sql['where'] = list()
                transformed_sql['order'] = list()
                is_where_flag = True
            elif c_instance.id_c == 3:
                transformed_sql['where'] = list()
                transformed_sql['group'] = list()
                transformed_sql['having'] = list()
                is_where_flag = True
            elif c_instance.id_c == 4:
                transformed_sql['where'] = list()
                is_where_flag = True
            elif c_instance.id_c == 5:
                transformed_sql['group'] = list()
                transformed_sql['having'] = list()
            elif c_instance.id_c == 6:
                transformed_sql['order'] = list()

        elif isinstance(c_instance, From):
            if c_instance.id_c > 0:
                for i in range(c_instance.id_c):
                    _table = pop_front(components)
                    table = eval(_table)
                    assert isinstance(table, T)
                    transformed_sql['from'].append(table_names[table.id_c])
            else:
                new_dict = dict()
                new_dict['sql'] = transformed_sql['sql']
                transformed_sql['from'].append(_transform(components, new_dict, ori_col_names, table_names, col_table))

        elif isinstance(c_instance, Sel):
            for i in range(c_instance.id_c + 1):
                agg = eval(pop_front(components))
                agg_op = agg.production.split()[1]
                val = eval(pop_front(components))
                if len(val.production.split()) == 2:
                    val_op = 'none'
                else:
                    val_op = val.production.split()[1]

                if val_op == 'none':
                    c1_left = eval(pop_front(components))
                    c1_left_op = c1_left.production.split()[1]
                    c1_left_c = eval(pop_front(components))
                    c1_left_c_name = ori_col_names[c1_left_c.id_c]
                    c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                    transformed_sql['select'].append([agg_op, val_op, [c1_left_op, c1_left_t_name, c1_left_c_name], None])
                else:
                    c1_left = eval(pop_front(components))
                    c1_left_op = c1_left.production.split()[1]
                    c1_left_c = eval(pop_front(components))
                    c1_left_c_name = ori_col_names[c1_left_c.id_c]
                    c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                    c1_right = eval(pop_front(components))
                    c1_right_op = c1_right.production.split()[1]
                    c1_right_c = eval(pop_front(components))
                    c1_right_c_name = ori_col_names[c1_right_c.id_c]
                    c1_right_t_name = table_names[col_table[c1_right_c.id_c]]
                    transformed_sql['select'].append(
                        [agg_op, val_op, [c1_left_op, c1_left_t_name, c1_left_c_name],
                         [c1_right_op, c1_right_t_name, c1_right_c_name]])

        elif isinstance(c_instance, Filter):
            components, filter_clause = _get_filter_clause(components, c_instance, ori_col_names, table_names, col_table)
            if is_where_flag:
                transformed_sql['where'].append(filter_clause)
            elif not is_where_flag and is_having_flag:
                transformed_sql['having'].append(filter_clause)

        elif isinstance(c_instance, Group):
            group_id = c_instance.id_c

            if group_id == 3:
                first_c = eval(pop_front(components))
                second_c = eval(pop_front(components))
                first_c_name = ori_col_names[first_c.id_c]
                first_t_name = table_names[col_table[first_c.id_c]]
                second_c_name = ori_col_names[second_c.id_c]
                second_t_name = table_names[col_table[second_c.id_c]]
                first_group_component = [first_t_name, first_c_name]
                second_group_component = [second_t_name, second_c_name]
                is_where_flag = False
                is_having_flag = True

            elif group_id == 2:
                first_c = eval(pop_front(components))
                first_c_name = ori_col_names[first_c.id_c]
                first_t_name = table_names[col_table[first_c.id_c]]
                first_group_component = [first_t_name, first_c_name]
                second_group_component = None
                is_where_flag = False
                is_having_flag = True

            elif group_id == 1:
                first_c = eval(pop_front(components))
                second_c = eval(pop_front(components))
                first_c_name = ori_col_names[first_c.id_c]
                first_t_name = table_names[col_table[first_c.id_c]]
                second_c_name = ori_col_names[second_c.id_c]
                second_t_name = table_names[col_table[second_c.id_c]]
                first_group_component = [first_t_name, first_c_name]
                second_group_component = [second_t_name, second_c_name]

            elif group_id == 0:
                first_c = eval(pop_front(components))
                first_c_name = ori_col_names[first_c.id_c]
                first_t_name = table_names[col_table[first_c.id_c]]
                first_group_component = [first_t_name, first_c_name]
                second_group_component = None

            else:
                raise RuntimeError('semQL2sqlPro group transform FIL')

            transformed_sql['group'].extend([first_group_component, second_group_component])

        elif isinstance(c_instance, Order):
            order_op = c_instance.production.split()[1]
            transformed_sql['order'].append(order_op)

            c1_left_component = None
            c1_right_component =None
            is_limit = False
            order_id = c_instance.id_c
            if order_id in [0, 1, 4, 5]:
                c1_left = eval(pop_front(components))
                c1_left_op = c1_left.production.split()[1]
                c1_left_c = eval(pop_front(components))
                c1_left_c_name = ori_col_names[c1_left_c.id_c]
                c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                c1_left_component = [c1_left_op, c1_left_t_name, c1_left_c_name]
                c1_right_component = None
            elif order_id in [2, 3, 6, 7]:
                c1_left = eval(pop_front(components))
                c1_left_op = c1_left.production.split()[1]
                c1_left_c = eval(pop_front(components))
                c1_left_c_name = ori_col_names[c1_left_c.id_c]
                c1_left_t_name = table_names[col_table[c1_left_c.id_c]]
                c1_left_component = [c1_left_op, c1_left_t_name, c1_left_c_name]
                c1_right = eval(pop_front(components))
                c1_right_op = c1_right.production.split()[1]
                c1_right_c = eval(pop_front(components))
                c1_right_c_name = ori_col_names[c1_right_c.id_c]
                c1_right_t_name = table_names[col_table[c1_right_c.id_c]]
                c1_right_component = [c1_right_op, c1_right_t_name, c1_right_c_name]

            if order_id in [0, 1, 2, 3]:
                is_limit = False
            elif order_id in [4, 5, 6, 7]:
                is_limit = True

            transformed_sql['order'].extend([c1_left_component, c1_right_component, is_limit])

    return transformed_sql


def transform(query, schema, origin=None):
    preprocess_schema(schema)
    if origin is None:
        # lf = query['model_result_replace']
        # lf = query['model_result']
        lf = query['rule_label']
    else:
        lf = origin
    # lf = ' '.join([str(x) for x in query['rule_label']])

    ori_col_names = [x[1] for x in schema['column_names_original']]
    # table_names = query['table_names']
    table_names = schema['table_names_original']
    current_table = schema

    current_table['schema_content_clean'] = [x[1] for x in current_table['column_names']]
    current_table['schema_content'] = [x[1] for x in current_table['column_names_original']]

    components = split_logical_form(lf)
    # print(components)

    transformed_sql = dict()
    transformed_sql['sql'] = query
    c = pop_front(components)
    c_instance = eval(c)
    assert isinstance(c_instance, Root1)
    if c_instance.id_c == 0:
        transformed_sql['intersect'] = dict()
        transformed_sql['intersect']['sql'] = query

        _transform(components, transformed_sql, ori_col_names, table_names, schema['col_table'])
        _transform(components, transformed_sql['intersect'], ori_col_names, table_names, schema['col_table'])
    elif c_instance.id_c == 1:
        transformed_sql['union'] = dict()
        transformed_sql['union']['sql'] = query
        _transform(components, transformed_sql, ori_col_names, table_names, schema['col_table'])
        _transform(components, transformed_sql['union'], ori_col_names, table_names, schema['col_table'])
    elif c_instance.id_c == 2:
        transformed_sql['except'] = dict()
        transformed_sql['except']['sql'] = query
        _transform(components, transformed_sql, ori_col_names, table_names, schema['col_table'])
        _transform(components, transformed_sql['except'], ori_col_names, table_names, schema['col_table'])
    else:
        # print(col_set, table_names)
        _transform(components, transformed_sql, ori_col_names, table_names, schema['col_table'])

    # print(transformed_sql)
    if len(transformed_sql['from']) == 0:
        print(lf)
        print(query['question'])
        exit(0)
    parse_result = to_str(transformed_sql, 1, schema)

    parse_result = parse_result.replace('\t', '')
    return [parse_result]


def col_to_str(component, table_names, table_alias, N=1):
    exp_ops = ['count', 'max', 'min', 'avg', 'sum']
    [agg_op, val_op, left_component, right_component] = component
    [c1_left_op, c1_left_t_name, c1_left_c_name] = left_component
    c1_left_c_name = c1_left_c_name.replace(' ', '_')
    if c1_left_c_name != '*' and c1_left_t_name not in table_names:
        table_names[c1_left_t_name] = ['T' + str(len(table_names) + N)]
        table_alias.append(table_names[c1_left_t_name][-1])

    if c1_left_c_name != '*':
        left_table_alias = table_names[c1_left_t_name][-1]
        left_col_repre = '%s.%s' % (left_table_alias, c1_left_c_name)
    else:
        left_col_repre = '*'

    if c1_left_op != 'none':
        left_agg_col_repre = '%s(%s)' % (c1_left_op, left_col_repre)
    else:
        left_agg_col_repre = left_col_repre

    if right_component is not None:
        assert val_op != 'none'
        [c1_right_op, c1_right_t_name, c1_right_c_name] = right_component
        c1_right_c_name = c1_right_c_name.replace(' ', '_')
        if c1_right_c_name != '*' and c1_right_t_name not in table_names:
            table_names[c1_right_t_name] = ['T' + str(len(table_names) + N)]
            table_alias.append(table_names[c1_right_t_name][-1])

        if c1_right_c_name != '*':
            right_table_alias = table_names[c1_right_t_name][-1]
            right_col_repre = '%s.%s' % (right_table_alias, c1_right_c_name)
        else:
            right_col_repre = '*'

        if c1_right_op != 'none':
            right_agg_col_repre = '%s(%s)' % (c1_right_op, right_col_repre)
        else:
            right_agg_col_repre = right_col_repre

        col_repre = '%s %s %s' % (left_agg_col_repre, val_op, right_agg_col_repre)

        if agg_op != 'none' and agg_op in exp_ops:
            return '%s(%s)' % (agg_op, col_repre), table_names, table_alias
        else:
            return col_repre, table_names, table_alias
    else:
        if agg_op != 'none' and agg_op in exp_ops:
            return '%s(%s)' % (agg_op, left_agg_col_repre), table_names, table_alias
        else:
            return left_agg_col_repre, table_names, table_alias


def infer_from_clause(table_names, table_alias, from_tables, N):
    join_clause = list()

    for from_tab in from_tables:
        if from_tab not in table_names:
            table_names[from_tab] = []
            table_names[from_tab].append('T' + str(len(table_alias) + N))
            table_alias.append(table_names[from_tab][-1])

    for tab_name, tab_ali_list in table_names.items():
        for tab_ali in tab_ali_list:
            join_clause.append((tab_name, tab_ali))

    join_clause = ' JOIN '.join(['%s AS %s' % (jc[0], jc[1]) for jc in join_clause])
    return 'FROM ' + join_clause


def build_graph(schema):
    relations = list()
    foreign_keys = schema['foreign_keys']
    for (fkey, pkey) in foreign_keys:
        fkey_table = schema['table_names_original'][schema['column_names'][fkey][0]]
        pkey_table = schema['table_names_original'][schema['column_names'][pkey][0]]
        relations.append((fkey_table, pkey_table))
        relations.append((pkey_table, fkey_table))
    return Graph(relations)


def preprocess_schema(schema):
    # print table
    schema['schema_content'] = [col[1] for col in schema['column_names']]
    schema['col_table'] = [col[0] for col in schema['column_names']]
    graph = build_graph(schema)
    schema['graph'] = graph


def to_str(sql_json, N_T, schema, pre_table_names=None, slot_filling='1'):
    select_clause = list()
    table_names = dict()
    table_alias = []
    current_table = schema
    # if 'select' not in sql_json:
    #     print(sql_json.keys())
    #     exit(0)

    for sel_component in sql_json['select']:
        col_clause, table_names, table_alias = col_to_str(sel_component, table_names, table_alias, N_T)
        select_clause.append(col_clause)
    select_clause = 'SELECT ' + ', '.join(select_clause).strip()

    order_clause = ''
    direction_map = {"des": 'DESC', 'asc': 'ASC'}

    if 'order' in sql_json:
        direction, first_c1_component, second_c1_component, is_limit = sql_json['order']
        first_c1_repre, table_names, table_alias = col_to_str(['none', 'none', first_c1_component, None], table_names, table_alias)

        if second_c1_component is not None:
            second_c1_repre, table_names, table_alias = col_to_str(['none', 'none', second_c1_component, None], table_names, table_alias)
            if is_limit:
                order_clause = ('ORDER BY %s, %s %s LIMIT %s' % (
                first_c1_repre, second_c1_repre, direction_map[direction], slot_filling)).strip()
            else:
                order_clause = ('ORDER BY %s, %s %s' % (
                first_c1_repre, second_c1_repre, direction_map[direction])).strip()
        else:
            if is_limit:
                order_clause = ('ORDER BY %s %s LIMIT %s' % (
                first_c1_repre, direction_map[direction], slot_filling)).strip()
            else:
                order_clause = ('ORDER BY %s %s' % (
                first_c1_repre, direction_map[direction])).strip()

    where_clause = ''
    having_clause = ''

    def get_filter_clause(table_names, table_alias, filter_sql, slot_filling, N_T, prefix):
        conjunctions = list()
        filters = list()
        for f in filter_sql:
            if isinstance(f, str):
                conjunctions.append(f)
            else:
                [op, val_op, left_component, right_component, sub_query] = f
                subject, table_names, table_alias = col_to_str([op, val_op, left_component, right_component], table_names, table_alias, N_T)

                if sub_query is None:
                    where_value = '%s' % slot_filling
                    if op == 'between':
                        where_value = '%s AND %s' % (slot_filling, slot_filling)
                    filters.append('%s %s %s' % (subject, op, where_value))
                else:
                    filters.append(
                        '%s %s %s' % (subject, op, '(' + to_str(sub_query, len(table_alias) + N_T, schema) + ')'))

                if len(conjunctions):
                    filters.append(conjunctions.pop())

        if len(filters) > 0:
            filter_clause = prefix + ' ' + ' '.join(filters).strip()
            filter_clause = filter_clause.replace('not_in', 'NOT IN')
            filter_clause = filter_clause.replace('not_like', 'NOT LIKE')
        else:
            filter_clause = ''

        return filter_clause, table_names, table_alias

    has_group_by = False

    if 'where' in sql_json:
        where_clause, table_names, table_alias = get_filter_clause(table_names, table_alias, sql_json['where'], slot_filling, N_T, prefix='WHERE')

    if 'having' in sql_json:
        having_clause, table_names, table_alias = get_filter_clause(table_names, table_alias, sql_json['having'], slot_filling, N_T, prefix='HAVING')
        has_group_by = True

    # if 'group' in sql_json and len(sql_json['group']) > 0:
    #     groupby_clause = "GROUP BY"
    #     first_component, second_component = sql_json['group']
    #     # for (tab, col) in sql_json['group']:
    #     col_clause, table_names, table_alias = col_to_str(
    #         ['none', 'none', ['none', first_component[0], first_component[1]], None], table_names, table_alias, N_T)
    #     groupby_clause = groupby_clause + ' ' + col_clause
    #     if second_component is not None:
    #         col_clause, table_names, table_alias = col_to_str(
    #             ['none', 'none', ['none', second_component[0], second_component[1]], None], table_names, table_alias, N_T)
    #         groupby_clause = groupby_clause + ', ' + col_clause
    # else:
    #     groupby_clause = ''

    for agg in ['count(', 'avg(', 'min(', 'max(', 'sum(']:
        if (len(sql_json['select']) > 1 and agg in select_clause) or agg in order_clause:
            has_group_by = True
            break

    # groupby_clause = ''
    # if has_group_by:
    #     if len(table_names) == 1:
    #         # check none agg
    #         is_agg_flag = False
    #         for sel_component in sql_json['select']:
    #
    #             if agg == 'none':
    #                 col_clause, table_names, table_alias = col_to_str(sel_component, table_names, table_alias, N_T)
    #                 groupby_clause = 'GROUP BY ' + col_clause
    #             else:
    #                 is_agg_flag = True
    #
    #         if is_agg_flag is False and len(groupby_clause) > 5:
    #             groupby_clause = "GROUP BY"
    #             for sel_component in sql_json['select']:
    #                 col_clause, table_names, table_alias = col_to_str(sel_component, table_names, table_alias, N_T)
    #                 groupby_clause = groupby_clause + ' ' + col_clause
    #
    #         if len(groupby_clause) < 5:
    #             if 'count(*)' in select_clause:
    #                 current_table = schema
    #                 for primary in current_table['primary_keys']:
    #                     if current_table['table_names'][current_table['col_table'][primary]] in table_names:
    #                         col_clause, table_names, table_alias = col_to_str(
    #                             ['none', 'none', ['none', current_table['table_names'][current_table['col_table'][primary]],
    #                             current_table['schema_content'][primary]], None], table_names,
    #                             table_alias, N_T)
    #
    #                         groupby_clause = 'GROUP BY ' + col_clause
    #     else:
    #         # if only one select
    #         if len(sql_json['select']) == 1:
    #             tab = sql_json['select'][0][2][1]
    #             non_lists = [tab]
    #             fix_flag = False
    #             # add tab from other part
    #             for key, value in table_names.items():
    #                 if key not in non_lists:
    #                     non_lists.append(key)
    #
    #             a = non_lists[0]
    #             b = None
    #             for non in non_lists:
    #                 if a != non:
    #                     b = non
    #             if b:
    #                 for pair in current_table['foreign_keys']:
    #                     t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
    #                     t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
    #                     if t1 in [a, b] and t2 in [a, b]:
    #                         if pre_table_names and t1 not in pre_table_names:
    #                             assert t2 in pre_table_names
    #                             t1 = t2
    #                         col_clause, table_names, table_alias = col_to_str(
    #                             ['none', 'none', ['none', t1, current_table['schema_content'][pair[0]]], None], table_names,
    #                             table_alias, N_T)
    #
    #                         groupby_clause = 'GROUP BY ' + col_clause
    #                         fix_flag = True
    #                         break
    #
    #             if fix_flag is False:
    #                 col_clause, table_names, table_alias = col_to_str(sql_json['select'][0], table_names, table_alias, N_T)
    #                 groupby_clause = 'GROUP BY ' + col_clause
    #
    #         else:
    #             if 'group' in sql_json and len(sql_json['group']) > 0:
    #                 groupby_clause = "GROUP BY"
    #                 first_component, second_component = sql_json['group']
    #                 # for (tab, col) in sql_json['group']:
    #                 col_clause, table_names, table_alias = col_to_str(
    #                     ['none', 'none', ['none', first_component[0], first_component[1]], None], table_names,
    #                     table_alias, N_T)
    #                 groupby_clause = groupby_clause + ' ' + col_clause
    #                 if second_component is not None:
    #                     col_clause, table_names, table_alias = col_to_str(
    #                         ['none', 'none', ['none', second_component[0], second_component[1]], None], table_names,
    #                         table_alias, N_T)
    #                     groupby_clause = groupby_clause + ', ' + col_clause
    #             else:
    #                 groupby_clause = ''
    #
    # else:
    if 'group' in sql_json and len(sql_json['group']) > 0:
        groupby_clause = "GROUP BY"
        first_component, second_component = sql_json['group']
        # for (tab, col) in sql_json['group']:
        col_clause, table_names, table_alias = col_to_str(
            ['none', 'none', ['none', first_component[0], first_component[1]], None], table_names, table_alias, N_T)
        groupby_clause = groupby_clause + ' ' + col_clause
        if second_component is not None:
            col_clause, table_names, table_alias = col_to_str(
                ['none', 'none', ['none', second_component[0], second_component[1]], None], table_names,
                table_alias, N_T)
            groupby_clause = groupby_clause + ', ' + col_clause
    else:
        groupby_clause = ''

    intersect_clause = ''
    if 'intersect' in sql_json:
        sql_json['intersect']['sql'] = sql_json['sql']
        # print(sql_json['sql']['query'])
        intersect_clause = 'INTERSECT ' + to_str(sql_json['intersect'], len(table_alias) + N_T, schema, table_names)
    union_clause = ''
    if 'union' in sql_json:
        sql_json['union']['sql'] = sql_json['sql']
        union_clause = 'UNION ' + to_str(sql_json['union'], len(table_alias) + N_T, schema, table_names)
    except_clause = ''
    if 'except' in sql_json:
        sql_json['except']['sql'] = sql_json['sql']
        except_clause = 'EXCEPT ' + to_str(sql_json['except'], len(table_alias) + N_T, schema, table_names)

    # print(current_table['table_names_original'])

    # table_names_replace = {}
    # for a, b in zip(current_table['table_names_original'], current_table['table_names']):
    #     table_names_replace[b] = a
    # new_table_names = {}
    # for key, value in table_names.items():
    #     if key is None:
    #         continue
    #     new_table_names[table_names_replace[key]] = value
    # from_clause = infer_from_clause(new_table_names, schema, all_columns).strip()

    if type(sql_json['from'][0]) == dict:
        sql_json['from'][0]['sql'] = sql_json['sql']
        from_clause = 'FROM (' + to_str(sql_json['from'][0], len(table_alias) + N_T, schema, table_names) + ')'
    else:
        from_clause = infer_from_clause(table_names, table_alias, sql_json['from'], len(table_alias) + N_T)

    sql_components = [select_clause, from_clause, where_clause, groupby_clause, having_clause, order_clause,
         intersect_clause, union_clause, except_clause]
    sql_components = [clause for clause in sql_components if clause != '']
    sql = ' '.join(sql_components)

    return sql


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset path', required=True)
    arg_parser.add_argument('--table_path', type=str, help='predicted logical form', required=True)
    args = arg_parser.parse_args()

    # loading dataSets
    datas, schemas = load_dataSets(args)
    # alter_not_in(datas, schemas=schemas)
    # alter_inter(datas)
    # alter_column0(datas)

    presems = []
    semsqls = []
    quess = []
    gold_sqls = []
    pred_sqls = []
    db_list = []
    db_dir = []
    etype = 'match'
    kmaps = build_foreign_key_map_from_json(args.table_path)

    for i, d in enumerate(datas):
        # print('#' * 100)
        # print(d['query'])
        # print(d['rule_label'])
        result = transform(d, schemas[d['db_id']])[0]
        presems.append(d['rule_label'])
        semsqls.append(d['rule_label'])
        quess.append(d['question'])
        gold_sqls.append(str(d['query']))
        # print(type(gold_sqls[0]))
        # print(gold_sqls[0])
        # exit(0)
        pred_sqls.append(result)
        db_list.append(d['db_id'])

        # if quess[-1] == 'What is the total number of routes for each country and airline in that country?':
        #     print('#' * 100)
        #     print(d['sql'])
        #     print(d['query'])
        #     print(d['rule_label'])
        #     print('#'*100)
        #     print(result)
        #     exit(0)

    print('eval ...')
    evaluate_sqls(presems, semsqls, quess, gold_sqls, pred_sqls, db_list, 'spider/database', etype, kmaps, True)