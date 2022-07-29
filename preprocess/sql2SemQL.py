# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import copy
from utils import load_dataSets
from src.rule.semQL import Root1, Root, N, A, C, T, Sel, Sup, Filter, Order, From, Group


class Parser:
    def __init__(self):
        self.copy_selec = None
        self.sel_result = []
        self.colSet = set()

    def _init_rule(self):
        self.copy_selec = None
        self.colSet = set()

    def _parse_root(self, sql):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] == None:
            use_sup = False

        if sql['sql']['orderBy'] == []:
            use_ord = False
        elif sql['sql']['limit'] != None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] != [] or \
                        sql['sql']['having'] != []:
            use_fil = True

        if use_fil and use_sup:
            return [Root(0)], ['FILTER', 'SUP', 'GROUP', 'FROM', 'SEL']
        elif use_fil and use_ord:
            return [Root(1)], ['ORDER', 'FILTER', 'GROUP', 'FROM', 'SEL']
        elif use_sup:
            return [Root(2)], ['SUP', 'GROUP', 'FROM', 'SEL']
        elif use_fil:
            return [Root(3)], ['FILTER', 'GROUP', 'FROM', 'SEL']
        elif use_ord:
            return [Root(4)], ['ORDER', 'GROUP', 'FROM', 'SEL']
        else:
            return [Root(5)], ['GROUP', 'FROM', 'SEL']

    def _parser_column0(self, sql, select):
        """
        Find table of column '*'
        :return: T(table_id)
        """
        if len(sql['sql']['from']['table_units']) == 1:
            return T(sql['sql']['from']['table_units'][0][1])
        else:
            table_list = []
            for tmp_t in sql['sql']['from']['table_units']:
                if type(tmp_t[1]) == int:
                    table_list.append(tmp_t[1])
            table_set, other_set = set(table_list), set()
            for sel_p in select:
                if sel_p[1][1][1] != 0:
                    other_set.add(sql['col_table'][sel_p[1][1][1]])

            if len(sql['sql']['where']) == 1:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
            elif len(sql['sql']['where']) == 3:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
            elif len(sql['sql']['where']) == 5:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][4][2][1][1]])
            table_set = table_set - other_set
            if len(table_set) == 1:
                return T(list(table_set)[0])
            elif len(table_set) == 0 and sql['sql']['groupBy'] != []:
                return T(sql['col_table'][sql['sql']['groupBy'][0][1]])
            else:
                question = sql['question']
                self.sel_result.append(question)
                print('column * table error')
                return T(sql['sql']['from']['table_units'][0][1])

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        result.append(Sel(0))
        ori_sel_len = len(select) - 1
        result.append(N(len(select) - 1))
        sel_tabs = []

        for sel in select:
            result.append(A(sel[0]))
            self.colSet.add(sql['col_set'].index(sql['names'][sel[1][1][1]]))
            result.append(C(sql['col_set'].index(sql['names'][sel[1][1][1]])))
            # now check for the situation with *
            if sel[1][1][1] == 0:
                result.append(self._parser_column0(sql, select))
            else:
                result.append(T(sql['col_table'][sel[1][1][1]]))
                sel_tabs.append(sql['col_table'][sel[1][1][1]])
            if not self.copy_selec:
                self.copy_selec = [copy.deepcopy(result[-2]), copy.deepcopy(result[-1])]

        return result, None

    def _parse_from(self, sql):
        """
        parsing the sql by the grammar
        From ::= T | TT | TTT | ... |
        :return: [From(), states]
        """
        result = []
        result.append(From(0))
        sel_tabs = []

        select = sql['sql']['select'][1]
        for sel in select:
            # now check for the situation with *
            if sel[1][1][1] != 0:
                sel_tabs.append(sql['col_table'][sel[1][1][1]])

        having_tabs = []
        for having_tab in sql['sql']['having']:
            if type(having_tab) is list:
                having_tabs.append(sql['col_table'][having_tab[2][1][1]])

        where_tabs = []
        for where_tab in sql['sql']['where']:
            if type(where_tab) is list:
                where_tabs.append(sql['col_table'][where_tab[2][1][1]])

        groupby_tabs = []
        for groupby_tab in sql['sql']['groupBy']:
            if type(groupby_tab) is list:
                groupby_tabs.append(sql['col_table'][groupby_tab[1]])

        sel_tabs = sel_tabs + having_tabs + where_tabs + groupby_tabs

        from_tabs = []
        for from_tab in sql['sql']['from']['table_units']:
            if from_tab[0] == 'table_unit':
                from_tabs.append(from_tab[1])

        from_cond_tabs = []
        for cond_tab in sql['sql']['from']['conds']:
            if type(cond_tab) is list:
                from_cond_tabs.append(sql['col_table'][cond_tab[3][1]])
                from_cond_tabs.append(sql['col_table'][cond_tab[2][1][1]])

        # all_tabs = from_tabs + from_cond_tabs + having_tabs + where_tabs + groupby_tabs

        all_tabs = from_tabs

        external_from_tabs = set([tab for tab in all_tabs if tab not in sel_tabs and tab != -1])
        for tab in external_from_tabs:
            result.append(T(tab))

        result[0] = From(len(external_from_tabs))
        return result, None

    def _parse_group(self, sql):
        """
        parsing the sql by the grammar
        Group ::= None | N |
        :return: [Group(), states]
        """
        result = []
        result.append(Group(0))
        groupby_cols = []
        groupBys = sql['sql']['groupBy']

        for groupby in groupBys:
            groupby_cols.append(groupby[1])

        if len(groupby_cols) > 0:
            result.append(N(0))
            for gb_col in groupby_cols:
                result.append(A(0))
                self.colSet.add(sql['col_set'].index(sql['names'][gb_col]))
                result.append(C(sql['col_set'].index(sql['names'][gb_col])))
                result.append(T(sql['col_table'][gb_col]))

        if len(result) > 1:
            result[0] = Group(1)
            result[1] = N(len(groupby_cols) -1)
        return result, None

    def _parse_sup(self, sql):
        """
        parsing the sql by the grammar
        Sup ::= Most A | Least A
        A ::= agg column table
        :return: [Sup(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        if sql['sql']['limit'] == None:
            return result, None
        if sql['sql']['orderBy'][0] == 'desc':
            result.append(Sup(0))
        else:
            result.append(Sup(1))

        result.append(A(sql['sql']['orderBy'][1][0][1][0]))
        self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]))
        result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])))
        if sql['sql']['orderBy'][1][0][1][1] == 0:
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []
        # check the where
        if sql['sql']['where'] != [] and sql['sql']['having'] != []:
            result.append(Filter(0))

        if sql['sql']['where'] != []:
            # check the not and/or
            if len(sql['sql']['where']) == 1:
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
            elif len(sql['sql']['where']) == 3:
                if sql['sql']['where'][1] == 'or':
                    result.append(Filter(1))
                else:
                    result.append(Filter(0))
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
            else:
                if sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                elif sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'or':
                    result.append(Filter(1))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                elif sql['sql']['where'][1] == 'or' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(1))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                else:
                    result.append(Filter(1))
                    result.append(Filter(1))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))

        # check having
        if sql['sql']['having'] != []:
            result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql))
        return result, None

    def _parse_order(self, sql):
        """
        parsing the sql by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []

        if 'order' not in sql['query_toks_no_value'] or 'by' not in sql['query_toks_no_value']:
            return result, None
        elif 'limit' in sql['query_toks_no_value']:
            return result, None
        else:
            if sql['sql']['orderBy'] == []:
                return result, None
            else:
                select = sql['sql']['select'][1]
                if sql['sql']['orderBy'][0] == 'desc':
                    result.append(Order(0))
                else:
                    result.append(Order(1))
                result.append(A(sql['sql']['orderBy'][1][0][1][0]))
                self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]))
                result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])))
                if sql['sql']['orderBy'][1][0][1][1] == 0:
                    result.append(self._parser_column0(sql, select))
                else:
                    result.append(T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def parse_one_condition(self, sql_condit, names, sql):
        result = []
        # check if V(root)
        nest_query = True
        if type(sql_condit[3]) != dict:
            nest_query = False

        if sql_condit[0] == True:
            if sql_condit[1] == 9:
                # not like only with values
                fil = Filter(10)
            elif sql_condit[1] == 8:
                # not in with Root
                fil = Filter(19)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else:
            # check for Filter (<,=,>,!=,between, >=,  <=, ...)
            single_map = {1:8,2:2,3:5,4:4,5:7,6:6,7:3}
            nested_map = {1:15,2:11,3:13,4:12,5:16,6:17,7:14}
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if nest_query == False:
                    fil = Filter(single_map[sql_condit[1]])
                else:
                    fil = Filter(nested_map[sql_condit[1]])
            elif sql_condit[1] == 9:
                fil = Filter(9)
            elif sql_condit[1] == 8:
                fil = Filter(18)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")

        result.append(fil)
        result.append(A(sql_condit[2][1][0]))
        self.colSet.add(sql['col_set'].index(sql['names'][sql_condit[2][1][1]]))
        result.append(C(sql['col_set'].index(sql['names'][sql_condit[2][1][1]])))
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql_condit[2][1][1]]))

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
            nest_query['col_set'] = sql['col_set']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['keys'] = sql['keys']
            result.extend(self.parser(nest_query))

        return result

    def _parse_step(self, state, sql):

        if state == 'ROOT':
            return self._parse_root(sql)

        if state == 'SEL':
            return self._parse_select(sql)

        if state == 'FROM':
            return self._parse_from(sql)

        if state == 'GROUP':
            return self._parse_group(sql)

        elif state == 'SUP':
            return self._parse_sup(sql)

        elif state == 'FILTER':
            return self._parse_filter(sql)

        elif state == 'ORDER':
            return self._parse_order(sql)
        else:
            raise NotImplementedError("Not the right state")

    def full_parse(self, query):
        sql = query['sql']
        nest_query = {}
        nest_query['names'] = query['names']
        nest_query['query_toks_no_value'] = ""
        nest_query['col_table'] = query['col_table']
        nest_query['col_set'] = query['col_set']
        nest_query['table_names'] = query['table_names']
        nest_query['question'] = query['question']
        nest_query['query'] = query['query']
        nest_query['keys'] = query['keys']

        if sql['intersect']:
            results = [Root1(0)]
            nest_query['sql'] = sql['intersect']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['union']:
            results = [Root1(1)]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['except']:
            results = [Root1(2)]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        results = [Root1(3)]
        results.extend(self.parser(query))

        return results

    def parser(self, query):
        stack = ["ROOT"]
        result = []
        while len(stack) > 0:
            state = stack.pop()
            step_result, step_state = self._parse_step(state, query)
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()

    parser = Parser()

    # loading dataSets
    datas, table = load_dataSets(args)
    processed_data = []
    from_sql_data_cnt = 0

    for i, d in enumerate(datas):
        if len(datas[i]['sql']['select'][1]) > 6:
            continue
        r = parser.full_parse(datas[i])
        datas[i]['rule_label'] = " ".join([str(x) for x in r])

        # if len(datas[i]['sql']['from']) > 1:
        #     print('#'*100)
        #     print(datas[i]['sql']['groupBy'])
        #     print(datas[i]['query'])

        if 'sql' in datas[i]['sql']['from']['table_units'][0]:
            print('#'*100)
            print(datas[i]['sql']['from'])
            from_sql_data_cnt += 1

        # wrong_sentence = 'List all singer names in concerts in year 2014.'
        # if d['question'] == wrong_sentence:
        #     print('#'*100)
        #     print(datas[i]['rule_label'])
        #     print('#' * 100)
        #     print(datas[i]['sql'])
        #     print('#' * 100)
        #     print(datas[i]['query'])
        #     exit(0)
        processed_data.append(datas[i])
    print(from_sql_data_cnt)

    print('Finished %s datas and failed %s datas' % (len(processed_data), len(datas) - len(processed_data)))
    with open(args.output, 'w', encoding='utf8') as f:
        f.write(json.dumps(processed_data))

