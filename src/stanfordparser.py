import json

import copy
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import *

PARSE_LABEL = ['ROOT', 'SBARQ', 'WHNP', 'WHADJP', 'WRB', 'JJ', 'NN', 'PP', 'IN', 'NP', 'SQ', 'VBP', 'ADJP', 'JJR', 'CD', '.', 'FRAG', ',', 'CC', 'VP', 'VBN', 'DT', 'WP', 'VBZ', 'WP$', 'QP', 'SBAR', 'S', "''", 'WHADVP', 'ADVP', 'RBS', 'VBD', 'WHPP', 'WDT', '``', 'UH', 'VBG', 'EX', 'RB', 'INTJ', 'POS', 'VB', 'JJS', 'NML', 'UCP', 'FW', 'PRP', 'NNP', 'ADD', 'PRP$', 'NNS', 'SYM', 'RRC', 'PDT', 'RBR', '-LRB-', '-RRB-', 'PRN', 'MD', 'TO', 'CONJP', 'RP', 'PRT', 'AFX', 'NAC', 'X', 'LST', 'LS', '$', 'SINV', 'GW', ':', 'WORD', 'HYPH']
#
# nlp = StanfordCoreNLP('/slfs1/users/zc825/stanfordnlp/stanford-corenlp-4.0.0', lang='en')
# TRAIN_PATH = "/slfs1/users/zc825/workspace/shadowgnn/data/train_8k_linking_matrix_parse_v3.json"

# def dependency_parser(TRAIN_PATH):
#     with open(TRAIN_PATH) as inf:
#         print('processing ...')
#         sqls = json.load(inf)
#
#     for cnt, sql in enumerate(sqls):
#         if cnt % 200 == 0:
#             print(cnt)
#         query = copy.deepcopy(sql['question_arg'])
#         correst_query = []
#         for token in query:
#             token = ''.join(token).strip()
#             tokened = nlp.word_tokenize(token)
#             if len(tokened) > 1:
#                 token = "".join(filter(str.isalnum, token))
#                 if len(nlp.word_tokenize(token)) > 1:
#                     if len(token) > 2:
#                         token = token[:-2]
#
#             if token == 'didnt':
#                 token = 'did'
#
#             if token == '':
#                 token = ','
#
#             correst_query.append(token)
#
#         query = ' '.join(correst_query).strip()
#
#         try:
#             dependency_parse = nlp.dependency_parse(query)
#         except:
#             print(query)
#             exit(0)
#
#         node_id_set = [0]
#         root_flag = 0
#         single_node_id = [0]
#         root_id = [0]
#         for id, node in enumerate(dependency_parse):
#             if node[0] == 'ROOT':
#                 if id+1 == len(dependency_parse) or dependency_parse[id+1][0] == 'ROOT':
#                     node = ('punct', root_id[-1], max(single_node_id)+1)
#                     dependency_parse[id] = node
#                     single_node_id = []
#                 else:
#                     root_flag = max(node_id_set)
#                     single_node_id = []
#                     root_id.append(node[2])
#                     continue
#
#             single_node_id.append(node[1])
#             single_node_id.append(node[2])
#
#             if root_flag+node[1] not in node_id_set:
#                 node_id_set.append(root_flag+node[1])
#             if root_flag+node[2] not in node_id_set:
#                 node_id_set.append(root_flag+node[2])
#
#         if max(node_id_set) != len(sql['question_arg']):
#             print(max(node_id_set))
#             print(len(sql['question_arg']))
#             print(dependency_parse)
#             print(sql['question_arg'])
#             print(correst_query)
#             exit(0)
#
#         sql['dependency_tree'] = copy.deepcopy(dependency_parse)
#
#     with open(TRAIN_PATH, 'w') as inf:
#         json.dump(sqls, inf)
#
#     query = [''.join(token) for token in sqls[0]['question_arg']]
#     query = ' '.join(query)
#     nlp.close()
#     print(query)
#     print(sqls[0]['dependency_tree'])
#     print('SUC !')
#     exit(0)
#
# dependency_parser(TRAIN_PATH)


# with open(data_path) as inf:
#     print('processing ...')
#     sqls = json.load(inf)
#
# for cnt, sql in enumerate(sqls):
#     if cnt % 200 == 0:
#         print(cnt)


def parser(nlp, query):
    query = copy.deepcopy(query)
    correst_query = []
    for t_id, token in enumerate(query):
        token = ''.join(token).strip()
        tokened = nlp.word_tokenize(token)
        if len(tokened) > 1:
            token = "".join(filter(str.isalnum, token))
            if len(nlp.word_tokenize(token)) > 1:
                if len(token) > 2:
                    token = token[:-2]

        if token == '':
            token = ','

        if token == 'didnt':
            token = 'did'

        if token == '?' or token == '.' or token == '!':
            if t_id != len(query) - 1:
                token = ','

        correst_query.append(token)

    query_str = ' '.join(correst_query).strip()

    parse_tree = nlp.parse(query_str)
    parse_tree = Tree.fromstring(parse_tree)

    if len(correst_query) != len(parse_tree.leaves()):
        print('#'*100)
        print(parse_tree)
        print(correst_query)
        print(parse_tree.leaves())
        print(query)
        exit(0)

    dfs_tree_label = []
    tree_edegs = []
    token_ids = []
    parse_label_set = []
    def dfs_tree_retriever(parent, dfs_tree_label, tree_edegs, token_ids, parse_label_set):
        parent_id = len(dfs_tree_label)
        # dfs_tree_label.append(str(parent.label()))
        dfs_tree_label.append(PARSE_LABEL.index(str(parent.label())))
        if str(parent.label()) not in parse_label_set:
            parse_label_set.append(str(parent.label()))

        for i in range(len(parent)):
            child = parent[i]
            child_id = len(dfs_tree_label)
            tree_edegs.append([parent_id, child_id])

            if type(child) == str:
                # dfs_tree_label.append(str(child))
                dfs_tree_label.append(PARSE_LABEL.index('WORD'))
                token_ids.append(child_id)
                continue
            else:
                dfs_tree_label, tree_edegs, token_ids, parse_label_set = dfs_tree_retriever(child, dfs_tree_label,
                                                                                            tree_edegs, token_ids,
                                                                                            parse_label_set)

        return dfs_tree_label, tree_edegs, token_ids, parse_label_set

    dfs_tree_label, tree_edegs, token_ids, parse_label_set = dfs_tree_retriever(parse_tree, dfs_tree_label, tree_edegs,
                                                                                token_ids, parse_label_set)

    # print(dfs_tree_label)
    # print(tree_edegs)
    # print(token_ids)
    # print(parse_label_set)

    return dfs_tree_label, tree_edegs, token_ids, parse_label_set


if __name__ == '__main__':
    import tqdm
    # stanford_nlp = StanfordCoreNLP('/mnt/lustre/sjtu/home/zc825/stanford-corenlp-4.0.0', lang='en')
    stanford_nlp = StanfordCoreNLP('/slfs1/users/zc825/stanfordnlp/stanford-corenlp-4.0.0', lang='en')
    # query = [['which'], ['department'], ['ha'], ['more'], ['than'], ['1'], ['head'], ['at'], ['a'], ['time'], ['?'], ['list'], ['id'], [','], ['name'], ['and'], ['number'], ['of'], ['head'], ['.']]
    # dfs_tree_label, tree_edegs, token_ids, parse_label_set = parser(stanford_nlp, query)
    # exit(0)

    data_path = "/slfs1/users/zc825/workspace/shadowgnn/data/train_correct_linking_matrix_v3_semql_v4.json"
    save_data_path = "/slfs1/users/zc825/workspace/shadowgnn/data/train_correct_linking_matrix_parse_v3_semql_v4.json"
    with open(data_path) as inf:
        print('processing ...')
        sqls = json.load(inf)

    for cnt, sql in tqdm.tqdm(enumerate(sqls)):
        if cnt % 200 == 0:
            print(cnt)

        query = sql['question_arg']
        dfs_tree_label, tree_edegs, token_ids, parse_label_set = parser(stanford_nlp, query)

        sql['dfs_tree_label'] = dfs_tree_label
        sql['parse_tree_edegs'] = tree_edegs
        sql['parse_token_ids'] = token_ids

    with open(save_data_path, 'w') as inf:
        json.dump(sqls, inf)

    stanford_nlp.close()

    print('DUMP SUC!')
    exit(0)
