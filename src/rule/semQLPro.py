# -*- coding: utf-8 -*-
Keywords = ['des', 'asc', 'and', 'or', 'sum', 'min', 'max', 'avg', 'none', '=', '!=', '<', '>', '<=', '>=', 'between', 'like', 'not_like', 'limit'] + [
    'in', 'not_in', 'count', 'intersect', 'union', 'except'] + ['+', '-', '*', '/']


class Grammar(object):
    def __init__(self, is_sketch=False):
        self.begin = 0
        self.type_id = 0
        self.is_sketch = is_sketch
        self.prod2id = {}
        self.type2id = {}
        self._init_grammar(Sel)
        self._init_grammar(Root)
        self._init_grammar(Filter)
        self._init_grammar(Order)
        self._init_grammar(Root1)
        self._init_grammar(From)
        self._init_grammar(Group)
        self._init_grammar(C1)
        self._init_grammar(V)
        self._init_grammar(A)

        # if not self.is_sketch:
        #     self._init_grammar(C1)
        #     self._init_grammar(V)
        #     self._init_grammar(A)

        self._init_id2prod()
        self.type2id[C] = self.type_id
        self.type_id += 1
        self.type2id[T] = self.type_id

    def _init_grammar(self, Cls):
        """
        get the production of class Cls
        :param Cls:
        :return:
        """
        production = Cls._init_grammar()
        for p in production:
            self.prod2id[p] = self.begin
            self.begin += 1
        self.type2id[Cls] = self.type_id
        self.type_id += 1

    def _init_id2prod(self):
        self.id2prod = {}
        for key, value in self.prod2id.items():
            self.id2prod[value] = key

    def get_production(self, Cls):
        return Cls._init_grammar()


class Action(object):
    def __init__(self):
        self.pt = 0
        self.production = None
        self.children = list()

    def get_next_action(self, is_sketch=False):
        actions = list()
        for x in self.production.split(' ')[1:]:
            if x not in Keywords:
                rule_type = eval(x)
                if is_sketch:
                    if rule_type not in [C, T, C1, V, A]:
                        actions.append(rule_type)
                else:
                    actions.append(rule_type)
        return actions

    def set_parent(self, parent):
        self.parent = parent

    def add_children(self, child):
        self.children.append(child)


class Root1(Action):
    def __init__(self, id_c, parent=None):
        super(Root1, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'Root1 intersect Root Root',
            1: 'Root1 union Root Root',
            2: 'Root1 except Root Root',
            3: 'Root1 Root',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Root1(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Root1(' + str(self.id_c) + ')'


class Root(Action):
    def __init__(self, id_c, parent=None):
        super(Root, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'Root Sel Filter Group Order From',
            1: 'Root Sel Group Order From',
            2: 'Root Sel Filter Order From',
            3: 'Root Sel Filter Group From',
            4: 'Root Sel Filter From',
            5: 'Root Sel Group From',
            6: 'Root Sel Order From',
            7: 'Root Sel From',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Root(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Root(' + str(self.id_c) + ')'


class From(Action):
    """
    Number of Columns
    """
    def __init__(self, id_c, parent=None):
        super(From, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'From Root',
            1: 'From T',
            2: 'From T T',
            3: 'From T T T',
            4: 'From T T T T',
            5: 'From T T T T T',
            6: 'From T T T T T T',
            7: 'From T T T T T T T',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'From(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'From(' + str(self.id_c) + ')'


class Group(Action):
    """
    Number of Columns
    """
    def __init__(self, id_c, parent=None):
        super(Group, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Group C',
            1: 'Group C C',
            2: 'Group C Filter',
            3: 'Group C C Filter'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Group(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Group(' + str(self.id_c) + ')'


class C(Action):
    """
    Column
    """
    def __init__(self, id_c, parent=None):
        super(C, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self.production = 'C min'

    def __str__(self):
        return 'C(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'C(' + str(self.id_c) + ')'


class C1(Action):
    """
    Agg Column
    """
    def __init__(self, id_c, parent=None):
        super(C1, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'C1 none C',
            1: 'C1 max C',
            2: "C1 min C",
            3: "C1 count C",
            4: "C1 sum C",
            5: "C1 avg C"
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'C1(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'C1(' + str(self.id_c) + ')'


class V(Action):
    """
    Op C1
    """
    def __init__(self, id_c, parent=None):
        super(V, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'V C1',
            1: 'V - C1 C1',
            2: 'V + C1 C1',
            3: 'V * C1 C1',
            4: 'V / C1 C1'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'V(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'V(' + str(self.id_c) + ')'


class T(Action):
    """
    Table
    """
    def __init__(self, id_c, parent=None):
        super(T, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self.production = 'T min'
        self.table = None

    def __str__(self):
        return 'T(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'T(' + str(self.id_c) + ')'


class A(Action):
    """
    Aggregator V
    """
    def __init__(self, id_c, parent=None):
        super(A, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'A none V',
            1: 'A max V',
            2: "A min V",
            3: "A count V",
            4: "A sum V",
            5: "A avg V"
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'A(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'A(' + str(self.grammar_dict[self.id_c].split(' ')[1]) + ')'


class Sel(Action):
    """
    Select
    """
    def __init__(self, id_c, parent=None):
        super(Sel, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Sel A',
            1: 'Sel A A',
            2: 'Sel A A A',
            3: 'Sel A A A A',
            4: 'Sel A A A A A',
            5: 'Sel A A A A A A',
            6: 'Sel A A A A A A A',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Sel(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Sel(' + str(self.id_c) + ')'


class Filter(Action):
    """
    Filter
    """
    def __init__(self, id_c, parent=None):
        super(Filter, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Filter and Filter Filter',
            1: 'Filter or Filter Filter',
            2: 'Filter = V',
            3: 'Filter != V',
            4: 'Filter < V',
            5: 'Filter > V',
            6: 'Filter <= V',
            7: 'Filter >= V',
            8: 'Filter between V',
            9: 'Filter like V',
            10: 'Filter not_like V',
            # now begin root
            11: 'Filter = V Root',
            12: 'Filter < V Root',
            13: 'Filter > V Root',
            14: 'Filter != V Root',
            15: 'Filter between V Root',
            16: 'Filter >= V Root',
            17: 'Filter <= V Root',
            # now for In
            18: 'Filter in V Root',
            19: 'Filter not_in V Root'

        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Filter(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Filter(' + str(self.grammar_dict[self.id_c]) + ')'


class Order(Action):
    """
    Order
    """
    def __init__(self, id_c, parent=None):
        super(Order, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Order des C1',
            1: 'Order asc C1',
            2: 'Order des C1 C1',
            3: 'Order asc C1 C1',
            4: 'Order des C1 limit',
            5: 'Order asc C1 limit',
            6: 'Order des C1 C1 limit',
            7: 'Order asc C1 C1 limit',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Order(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Order(' + str(self.id_c) + ')'


if __name__ == '__main__':
    print(list(Root._init_grammar()))