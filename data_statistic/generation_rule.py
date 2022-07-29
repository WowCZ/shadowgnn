import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


generation_rule = {
    'INTERSECT': 'and',
    'EXCEPT': 'except that',
    'UNION': 'or',
    'SELECT': 'list',
    'WHERE': 'with',
    '<COLUMN>': '<COLUMN>',
    '<TABLE>': '<TABLE>',
    '<VALUE>': '<VALUE>',
    '*': 'all information',
    'count(*)': 'the total number of <TABLE>',
    '>': 'greater than',
    '<': 'less than',
    '>=': 'greater than or equal to',
    '<=': 'less than or equal to',
    '=': 'equal to',
    '!=': 'not equal to',
    'min(<COLUMN>)': 'the minimum of <COLUMN>',
    'max(<COLUMN>)': 'the maximum of <COLUMN>',
    'avg(<COLUMN>)': 'the average of <COLUMN>',
    'sum(<COLUMN>)': 'the sum of <COLUMN>',
    'count(<COLUMN>)': 'the number of <COLUMN>',
    'GROUP BY': 'for each',
    'HAVING': 'that have',
    'ORDER BY': 'ordered by',
    'ASC': 'ascendingly',
    'DESC': 'descendingly',
    'LIMIT <VALUE>': 'top <VALUE> items of',
    'NOT IN': 'that is not in',
    'like': 'formatted as',
    'NOT LIKE': 'does not formatted as',
    'between <VALUE> AND <VALUE>': 'is between <VALUE> and <VALUE>',
    'AND': 'and',
    'OR': 'or',
    'and': 'and',
    'or': 'or',
    'in': 'that is subset of'
}

ignore_keywords = ['<TABLE>', 'JOIN', 'ON']


def generate_sentence_template(sql_template, nested=False):
    sql_template = sql_template.replace('( ', '(')
    sql_template = sql_template.replace(' )', ')')

    sql_template_split = sql_template.strip().split(' ')
    tables_join = False

    for i, token in enumerate(sql_template_split):
        if token == 'FROM' and i + 2 < len(sql_template_split):
            if sql_template_split[i + 2] == 'JOIN':
                tables_join = True
                break

    new_sql_template_split = []
    j = 0
    nested_flag = False
    nested_token = []
    for i, token in enumerate(sql_template_split):
        if j > 0:
            j -= 1
            continue
        token = token.strip(',')

        if token == '(SELECT':
            nested_flag = True

        # print(nested_token)

        if token.find(')') >= 0 and token.find('(') < 0:
            nested_token.append(token)
            nested_flag = False

        if nested_flag:
            nested_token.append(token)
            continue

        if len(nested_token) > 0:
            new_sql_template_split.append(' '.join(nested_token))
            nested_token = []
            continue

        # print(token)
        # print(nested_flag)

        if token in ['NOT', 'GROUP', 'LIMIT', 'ORDER']:
            j = 2
            new_sql_template_split.append(' '.join(sql_template_split[i:i+j]))
            j -= 1
            continue
        elif token == 'between':
            j = 4
            new_sql_template_split.append(' '.join(sql_template_split[i:i+j]))
            j -= 1
            continue

        new_sql_template_split.append(token)

    if len(nested_token) > 0:
        new_sql_template_split.append(' '.join(nested_token))

    # print(new_sql_template_split)
    # exit(0)

    template_sentence = []
    for i, token in enumerate(new_sql_template_split):
        token = token.strip(',')
        if token in ignore_keywords:
            continue

        if token == 'SELECT' and nested:
            continue

        if token.startswith('(SELECT'):
            token = token.lstrip('(')
            token = token.rstrip(')')
            template_word = generate_sentence_template(token, nested=True)
            template_sentence.append('( ' + template_word + ' )')
            continue

        if token not in generation_rule and token not in ignore_keywords and token != 'FROM':
            print('#'*100)
            print(token)
            print(new_sql_template_split)
            print(sql_template)
            # exit(0)
            continue

        if token == 'FROM' and not tables_join and new_sql_template_split[i-1] != 'count(*)':
            template_word = 'of <TABLE>'
        elif token == 'FROM':
            continue

        if token in generation_rule:
            template_word = generation_rule[token]

        if tables_join:
            template_word = template_word.replace('<COLUMN>', '<COLUMN> of <TABLE>')

        if token == 'LIMIT <VALUE>':
            template_sentence = template_sentence[:1] + [template_word] + template_sentence[1:]
        else:
            template_sentence.append(template_word)

    template_sentence = ' '.join(template_sentence)

    if not nested:
        template_sentence = template_sentence + ' <EOS>'

    return template_sentence


if __name__ == '__main__':
    annotated_template = open(os.path.join('save/', 'annotated_template.txt'), 'r')
    complete_annotated_template = open(os.path.join('save/', 'complete_annotated_template.txt'), 'w')
    for s in annotated_template:
        if s.startswith('SQL:'):
            sql_template = ' '.join(s.strip('\n').split(' ')[1:])
            template_sentence = generate_sentence_template(sql_template)

        if s.startswith('TP:'):
            s = 'TP: ' + template_sentence + '\n'

        complete_annotated_template.write(s)

    complete_annotated_template.close()
    annotated_template.close()
    # sql_template = 'SELECT <COLUMN> FROM <TABLE> WHERE <COLUMN> = (SELECT min(<COLUMN>) FROM <TABLE> WHERE <COLUMN> = <VALUE>) and <COLUMN> = <VALUE>'
    # template_sentence = generate_sentence_template(sql_template)
    # print(template_sentence)
    # exit(0)






