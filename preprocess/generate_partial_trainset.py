# -*- coding: utf-8 -*-

import os, sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math


def random_partial_trainset(samples, partial_rate):
    total_sample_len = len(samples)
    random_sample_len = math.floor(partial_rate * len(samples))

    random_idx = random.sample(range(0, total_sample_len), random_sample_len)

    partial_sample = []
    for id in random_idx:
        partial_sample.append(samples[id])

    return partial_sample


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()

    # loading samples
    with open(args.data_path, 'r', encoding='utf8') as f:
        raw_samples = json.load(f)

    partial_rate = 0.5
    # process samples
    partial_sample = random_partial_trainset(raw_samples, partial_rate)

    with open(args.output, 'w') as f:
        json.dump(partial_sample, f, indent=4)