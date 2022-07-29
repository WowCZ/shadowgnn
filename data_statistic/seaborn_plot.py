import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import argparse


def read_bertscore(file):
    f_out = open(file, 'r')

    score = f_out.readline()
    score = score.strip()
    score = score.split(' ')

    real_scores = []
    for s in score:
        real_scores.append(float(s))

    return real_scores


def to_cvs_data(rephrase_score, query_score, save_path):
    assert len(rephrase_score) == len(query_score)

    cvs_data = []
    for i in range(len(rephrase_score)):
        difference_score = rephrase_score[i] - query_score[i]
        data_item = ['SQL_TYPE{}'.format(i), rephrase_score[i], query_score[i], difference_score]
        cvs_data.append(data_item)

    save = pd.DataFrame(cvs_data, columns=['SQL', 'template', 'query', 'difference'])
    save.to_csv(save_path, index=False)


def get_template_ave_score(template_score, query_score, template_len):
    assert len(template_score) == len(query_score) == sum(template_len)

    cur_tem_id = 0
    template_avg_score = []
    query_avg_score = []
    for i, l in enumerate(template_len):
        template_avg_score.append(sum(template_score[cur_tem_id: cur_tem_id + l]) / l)
        query_avg_score.append(sum(query_score[cur_tem_id: cur_tem_id + l]) / l)
        cur_tem_id += l

    return template_avg_score, query_avg_score


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--template_score_path', type=str, help='dataset path', required=True)
    arg_parser.add_argument('--query_score_path', type=str, help='dataset path', required=True)
    arg_parser.add_argument('--output', type=str, help='predicted logical form', required=True)
    args = arg_parser.parse_args()

    template_score = read_bertscore(args.template_score_path)
    query_score = read_bertscore(args.query_score_path)

    with open("example/question.txt") as f:
        questions = [line.strip() for line in f]

    with open("example/query.txt") as f:
        queries = [line.strip() for line in f]

    with open("example/rephrase.txt") as f:
        rephrase_sentences = [line.strip() for line in f]

    template_argscore_idx = np.argsort(np.array(template_score))
    print(template_argscore_idx[:10])

    for i, idx in enumerate(template_argscore_idx[:10]):
        print('='*50, ' {}th Lowest Score '.format(str(i+1)), '='*50)
        print('Question: ' + questions[idx])
        print('SQL: ' + queries[idx])
        print('Template: ' + rephrase_sentences[idx])
    exit(0)

    #
    # template_len = read_bertscore('example/template_lens.txt')
    #
    # template_len_int = []
    # for l in template_len:
    #     template_len_int.append(int(l))
    #
    # template_score, query_score = get_template_ave_score(template_score, query_score, template_len_int)
    #
    # to_cvs_data(template_score, query_score, args.output)
    #
    # print('BERTScore saved in :', args.output)


    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # sns.set_theme(style="whitegrid")
    # # sns.set(font_scale=0.2)
    #
    # # Initialize the matplotlib figure
    # f, ax = plt.subplots(figsize=(6, 15))
    #
    # BERTScore = pd.read_csv('example/score_save.cvs')
    #
    # # # Load the example car crash dataset
    # BERTScore = BERTScore.sort_values("difference", ascending=False)[:200]
    #
    # # Plot the total crashes
    # sns.set_color_codes("pastel")
    # sns.barplot(x="template", y="SQL", data=BERTScore,
    #             label="template", color="b")
    #
    # # Plot the crashes where alcohol was involved
    # sns.set_color_codes("muted")
    # sns.barplot(x="query", y="SQL", data=BERTScore,
    #             label="query", color="b")
    #
    # # Add a legend and informative axis label
    # ax.legend(ncol=2, loc="lower right", frameon=True)
    # ax.set(xlim=(-0.5, 0.8), ylabel="",
    #        xlabel="BERTScore for Evaluating Readability")
    # sns.despine(left=True, bottom=True)
    #
    # plt.yticks(fontsize=2)
    #
    # plt.show()

