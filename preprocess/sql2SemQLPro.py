# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import copy
from utils import load_dataSets
from src.utils import load_pointed_dataset
from src.rule.semQLPro import Root1, Root, A, C, T, Sel, Filter, Order, From, Group, C1, V


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
        use_order, use_filter, use_group = True, False, False

        if sql['sql']['orderBy'] == []:
            use_order = False

        if sql['sql']['where'] != []:
            use_filter = True

        if  sql['sql']['groupBy'] != []:
            use_group = True

        if use_order and use_filter and use_group:
            return [Root(0)], ['FROM', 'ORDER', 'GROUP', 'FILTER', 'SEL']
        elif use_order and use_group:
            return [Root(1)], ['FROM', 'ORDER', 'GROUP', 'SEL']
        elif use_order and use_filter:
            return [Root(2)], ['FROM', 'ORDER', 'FILTER', 'SEL']
        elif use_filter and use_group:
            return [Root(3)], ['FROM', 'GROUP', 'FILTER', 'SEL']
        elif use_filter:
            return [Root(4)], ['FROM', 'FILTER', 'SEL']
        elif use_group:
            return [Root(5)], ['FROM', 'GROUP', 'SEL']
        elif use_order:
            return [Root(6)], ['FROM', 'ORDER', 'SEL']
        else:
            return [Root(7)], ['FROM', 'SEL']

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        result.append(Sel(len(select)-1))

        for sel in select:
            result.append(A(sel[0]))
            result.append(V(sel[1][0]))
            # if sel[1][0] != 0:
            #     print(sql['sql'])
            #     print(sql['query'])
            #     print(sql['question'].encode("utf8"))
            #     exit(0)

            if sel[1][0] == 0:
                result.append(C1(sel[1][1][0]))
                result.append(C(sel[1][1][1]))
            else:
                result.append(C1(sel[1][1][0]))
                result.append(C(sel[1][1][1]))
                result.append(C1(sel[1][2][0]))
                result.append(C(sel[1][2][1]))
        return result, None

    def _parse_from(self, sql):
        """
        parsing the sql by the grammar
        From ::= ROOT | T | TT | TTT | ... |
        :return: [From(), states]
        """
        result = []
        from_t = sql['sql']['from']['table_units']
        result.append(From(len(from_t)))
        from_sql_falg = False

        for tab_unit in from_t:
            if tab_unit[0] == 'table_unit':
                result.append(T(tab_unit[1]))
            elif tab_unit[0] == 'sql':
                from_sql_falg = True
                nest_query = {}
                nest_query['names'] = sql['names']
                nest_query['query_toks_no_value'] = ""
                nest_query['sql'] = tab_unit[1]
                nest_query['col_table'] = sql['col_table']
                nest_query['col_set'] = sql['col_set']
                nest_query['table_names'] = sql['table_names']
                nest_query['question'] = sql['question']
                nest_query['query'] = sql['query']
                nest_query['keys'] = sql['keys']
                result.extend(self.parser(nest_query))

        if from_sql_falg:
            result[0] = From(0)

        return result, None

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []

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

        return result, None

    def _parse_group(self, sql):
        """
        parsing the sql by the grammar
        Group ::= C | C C | C Filter| C C Filter|
        :return: [Group(), states]
        """

        # check having
        result = []
        groupBys = sql['sql']['groupBy']
        having_filter = sql['sql']['having']

        if len(having_filter) == 0:
            result.append(Group(len(groupBys) -1))
            for groupby in groupBys:
                result.append(C(groupby[1]))
        else:
            if len(groupBys) == 1:
                result.append(Group(2))
                result.append(C(groupBys[0][1]))
                result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql))
            elif len(groupBys) == 2:
                result.append(Group(3))
                result.append(C(groupBys[0][1]))
                result.append(C(groupBys[1][1]))
                result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql))
            else:
                raise NotImplementedError("not implement for the others FIL in _parse_group")

        return result, None

    def _parse_order(self, sql):
        """
        parsing the sql by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []

        orderby_clause = sql['sql']['orderBy']

        is_desc = True if orderby_clause[0] == 'desc' else False
        orderby_col_cnt = len(orderby_clause[1]) if len(orderby_clause[1]) <= 2 else 2
        has_limit = True if sql['sql']['limit'] is not None else False

        if not is_desc and orderby_col_cnt == 2 and has_limit:
            result.append(Order(7))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))
            result.append(C1(orderby_clause[1][1][1][0]))
            result.append(C(orderby_clause[1][1][1][1]))

            return result, None
        elif is_desc and orderby_col_cnt == 2 and has_limit:
            result.append(Order(6))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))
            result.append(C1(orderby_clause[1][1][1][0]))
            result.append(C(orderby_clause[1][1][1][1]))

            return result, None

        elif not is_desc and orderby_col_cnt == 1 and has_limit:
            result.append(Order(5))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))

            return result, None

        elif is_desc and orderby_col_cnt == 1 and has_limit:
            result.append(Order(4))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))

            return result, None

        elif not is_desc and orderby_col_cnt == 2 and not has_limit:
            result.append(Order(3))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))
            result.append(C1(orderby_clause[1][1][1][0]))
            result.append(C(orderby_clause[1][1][1][1]))

            return result, None
        elif is_desc and orderby_col_cnt == 2 and not has_limit:
            result.append(Order(2))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))
            result.append(C1(orderby_clause[1][1][1][0]))
            result.append(C(orderby_clause[1][1][1][1]))

            return result, None

        elif not is_desc and orderby_col_cnt == 1 and not has_limit:
            result.append(Order(1))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))

            return result, None

        elif is_desc and orderby_col_cnt == 1 and not has_limit:
            result.append(Order(0))
            result.append(C1(orderby_clause[1][0][1][0]))
            result.append(C(orderby_clause[1][0][1][1]))

            return result, None

        else:
            print(is_desc)
            print(orderby_col_cnt)
            print(has_limit)
            print(sql['sql'])
            print(sql['query'])
            raise NotImplementedError("not implement for the others FIL in _parse_order")


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
        result.append(V(sql_condit[2][0]))
        if sql_condit[2][0] == 0:
            result.append(C1(sql_condit[2][1][0]))
            result.append(C(sql_condit[2][1][1]))
        else:
            result.append(C1(sql_condit[2][1][0]))
            result.append(C(sql_condit[2][1][1]))
            result.append(C1(sql_condit[2][2][0]))
            result.append(C(sql_condit[2][2][1]))

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
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
    # datas, table = load_pointed_dataset(args.table_path, args.data_path)
    processed_data = []
    complex_rl = 0
    error_tab_set = 0
    for i, d in enumerate(datas):
        if len(datas[i]['sql']['select'][1]) > 6:
            continue
        r = parser.full_parse(datas[i])
        datas[i]['rule_label'] = " ".join([str(x) for x in r])

        rule_label = datas[i]['rule_label'].strip().split(' ')
        gold_col_id_set = []
        gold_tab_id_set = []
        for label_item in rule_label:
            if label_item.find('C(') == 0:
                col_id_s = label_item.find('(') + 1
                col_id_e = label_item.find(')')
                col_id = int(label_item[col_id_s:col_id_e])
                if col_id not in gold_col_id_set and col_id != 0:
                    gold_col_id_set.append(col_id)
            elif label_item.find('T(') == 0:
                tab_id_s = label_item.find('(') + 1
                tab_id_e = label_item.find(')')
                tab_id = int(label_item[tab_id_s:tab_id_e])
                if tab_id not in gold_tab_id_set:
                    gold_tab_id_set.append(tab_id)
            else:
                pass

        col_tab_set = []
        for gcol in gold_col_id_set:
            col_tab_set.append(d['col_table'][gcol])

        if len(set(col_tab_set)) > 0:
            if not set(col_tab_set).issubset(set(gold_tab_id_set)):
                error_tab_set += 1
                print('#'*50, ' {} '.format(str(error_tab_set)), '#'*50)
                print(set(col_tab_set))
                print(set(gold_tab_id_set))
                print(datas[i]['query'])
                print(datas[i]['sql'])

                true_table_units = []
                for t_id in col_tab_set:
                    true_table_units.append(['table_unit', t_id])

                datas[i]['sql']['from']['table_units'] = true_table_units

                r = parser.full_parse(datas[i])
                datas[i]['rule_label'] = " ".join([str(x) for x in r])

        # print('#' * 100)
        # print(datas[i]['query'])
        # print(datas[i]['rule_label'])
        # print(d['question'].encode("utf8"))

        # wrong_sentence = 'give me some good restaurants on buchanan in san francisco for arabic food ?'
        # if d['question'] == wrong_sentence:
        #     print('#'*100)
        #     print(datas[i]['rule_label'])
        #     print('#' * 100)
        #     print(datas[i]['sql'])
        #     print('#' * 100)
        #     print(datas[i]['query'])
        #     exit(0)

        complex_rl += len(r)
        processed_data.append(datas[i])

    print(error_tab_set)
    # exit(0)

    print('AVE SEMQL LEN:', complex_rl/len(datas))

    print('Finished %s datas and failed %s datas' % (len(processed_data), len(datas) - len(processed_data)))
    with open(args.output, 'w', encoding='utf8') as f:
        f.write(json.dumps(processed_data))

    # with open('data/train_link_test3.json', 'r', encoding='utf8') as f1:
    #     link_datas = json.load(f1)
    #
    # with open('data/train_test4.json', 'r', encoding='utf8') as f2:
    #     test_datas = json.load(f2)
    #
    # eq = 0
    # neq = 0
    # for (link, test) in zip(link_datas, test_datas):
    #     link_qt = [' '.join(x) for x in link['question_arg']]
    #     test_qt = [' '.join(x) for x in test['question_arg']]
    #
    #     if link_qt == test_qt:
    #         test['question_col_match'] = link['question_col_match']
    #         test['question_table_match'] = link['question_table_match']
    #         test['schema_linking_matrix'] = link['schema_linking_matrix']
    #
    # print(test_datas[0]['schema_linking_matrix'])
    #
    # with open('data/train_link_test4.json', 'w', encoding='utf8') as f:
    #     f.write(json.dumps(test_datas))
