# -*- coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itertools
import torch


class relation_types():
    # ======== Schema ======== 16
    C_ID = 1
    T_ID = 2

    # ---- column-column ----
    SAME_T = 3
    FKEY_C_F = 4
    FKEY_C_R = 5
    C_C = 6

    # ---- column-table ----
    PKEY_F = 7
    PKEY_R = 8
    BELONG_F = 9
    BELONG_R = 10
    C_T = 11
    T_C = 12

    # ---- table_table ----
    FKEY_T_F = 13
    FKEY_T_R = 14
    FKEY_T_B = 15
    T_T = 16

    # ======== Question ======== 5
    Q_DIST_L2 = 17
    Q_DIST_L1 = 18
    Q_DIST_0 = 19
    Q_DIST_G1 = 20
    Q_DIST_G2 = 21

    # ======== Linking ======== 12
    Q_C_E = 22
    Q_C_P = 23
    Q_C_N = 24
    C_Q_E = 25
    C_Q_P = 26
    C_Q_N = 27

    Q_T_E = 28
    Q_T_P = 29
    Q_T_N = 30
    T_Q_E = 31
    T_Q_P = 32
    T_Q_N = 33

    Q_C_V = 34
    C_Q_V = 35
    Q_C_NUM = 36
    C_Q_NUM = 37
    Q_C_TIME = 38
    C_Q_TIME = 39

    @classmethod
    def type_num(cls):
        return 39


def get_relation_matrices(entry):
    '''
    :return:
    Q-Q Q-C Q-T
    C-Q C-C C-T
    T-Q T-C T-T
    '''
    q_size, c_size, t_size = len(entry.src_sent), entry.col_num, entry.table_len

    template = np.ones((q_size, q_size))
    new_qq_matr = np.triu(template, 2) * relation_types.Q_DIST_G2 + \
                  np.tril(template, -2) * relation_types.Q_DIST_L2 + \
                  (np.triu(template, 1) - np.triu(template, 2)) * relation_types.Q_DIST_G1 + \
                  (np.tril(template, -1) - np.tril(template, -2)) * relation_types.Q_DIST_L1 + \
                  np.diag(np.diag(template)) * relation_types.Q_DIST_0

    new_qc_matr = np.ones((q_size, c_size)) * relation_types.Q_C_N
    new_cq_matr = np.ones((c_size, q_size)) * relation_types.C_Q_N
    for q_id, c_id in entry.question_col_match['exact']:
        new_qc_matr[q_id][c_id] = relation_types.Q_C_E
        new_cq_matr[c_id][q_id] = relation_types.C_Q_E
    for q_id, c_id in entry.question_col_match['partial']:
        new_qc_matr[q_id][c_id] = relation_types.Q_C_P
        new_cq_matr[c_id][q_id] = relation_types.C_Q_P
    for q_id, c_id in entry.question_col_match['content_exact'] + entry.question_col_match['content_partial']:
        new_qc_matr[q_id][c_id] = relation_types.Q_C_V
        new_cq_matr[c_id][q_id] = relation_types.C_Q_V

    new_qt_matr = np.ones((q_size, t_size)) * relation_types.Q_T_N
    new_tq_matr = np.ones((t_size, q_size)) * relation_types.T_Q_N
    for q_id, t_id in entry.question_table_match['exact']:
        new_qt_matr[q_id][t_id] = relation_types.Q_T_E
        new_tq_matr[t_id][q_id] = relation_types.T_Q_E
    for q_id, t_id in entry.question_table_match['partial']:
        new_qt_matr[q_id][t_id] = relation_types.Q_T_P
        new_tq_matr[t_id][q_id] = relation_types.T_Q_P

    new_cc_matr = np.ones((c_size, c_size)) * relation_types.C_C
    new_tt_matr = np.ones((t_size, t_size)) * relation_types.T_T
    new_cc_matr = new_cc_matr + np.diag(np.diag(np.ones((c_size, c_size)))) * (
            relation_types.C_ID - relation_types.C_C)
    new_tt_matr = new_tt_matr + np.diag(np.diag(np.ones((t_size, t_size)))) * (
            relation_types.T_ID - relation_types.T_T)
    for same_table in list(entry.table_col.values()):
        for id1, id2 in itertools.permutations(same_table, 2):
            new_cc_matr[id1][id2] = relation_types.SAME_T
    for ((fkey_f, fkey_r), (fkey_tab_f, fkey_tab_r)) in zip(entry.foreign_keys, entry.foreign_table_keys):
        new_cc_matr[fkey_f][fkey_r] = relation_types.FKEY_C_F
        new_cc_matr[fkey_r][fkey_f] = relation_types.FKEY_C_R
        new_tt_matr[fkey_tab_f][fkey_tab_r] = relation_types.FKEY_T_F if \
            new_tt_matr[fkey_tab_f][fkey_tab_r] == relation_types.T_T \
            else relation_types.FKEY_T_B
        new_tt_matr[fkey_tab_r][fkey_tab_f] = relation_types.FKEY_T_R if \
            new_tt_matr[fkey_tab_r][fkey_tab_f] == relation_types.T_T \
            else relation_types.FKEY_T_B

    new_ct_matr = np.ones((c_size, t_size)) * relation_types.C_T
    new_tc_matr = np.ones((t_size, c_size)) * relation_types.T_C
    for tid in entry.table_col:
        for cid in entry.table_col[tid]:
            new_ct_matr[cid][tid] = relation_types.BELONG_F if cid not in entry.primary_keys else relation_types.PKEY_F
            new_tc_matr[tid][cid] = relation_types.BELONG_R if cid not in entry.primary_keys else relation_types.PKEY_R

    new_matr = np.vstack([np.hstack([new_qq_matr, new_qc_matr, new_qt_matr]),
                          np.hstack([new_cq_matr, new_cc_matr, new_ct_matr]),
                          np.hstack([new_tq_matr, new_tc_matr, new_tt_matr])])
    new_matr = torch.tensor(new_matr, dtype=torch.long)

    return new_matr
