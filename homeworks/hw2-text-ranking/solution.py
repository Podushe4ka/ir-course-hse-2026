#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ДЗ 2. Текстовое ранжирование — шаблон решения.

Запуск:
    ./solution.py --submission_file=submission.csv PATH-TO-DATA
"""

import argparse
import csv
import math
import os
import re
from collections import Counter
from timeit import default_timer as timer
import random
import numpy as np


def read_queries(path):
    """QueryId -> QueryText."""
    queries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            query_id, text = line.rstrip('\n').split('\t', 1)
            queries[int(query_id)] = text
    return queries


def read_qrels(path):
    """QueryId -> {DocumentId -> Label}. Формат строки: QueryId 0 DocumentId Label."""
    qrels = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, label = line.split()
            qrels.setdefault(int(query_id), {})[doc_id] = int(label)
    return qrels


def read_docs(path, doc_ids=None):
    """DocumentId -> (URL, Title, Body). Если передан doc_ids, читает только эти документы."""
    docs = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            fields += [''] * (4 - len(fields))
            doc_id, url, title, body = fields[:4]
            if doc_ids is None or doc_id in doc_ids:
                docs[doc_id] = (url, title, body)
    return docs


def main():
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    # Если в решении есть случайность, зафиксируйте seed:
    random.seed(42)
    np.random.seed(42)

    train_queries = read_queries(os.path.join(args.data_dir, 'vkmarco-doctrain-queries.tsv'))
    train_qrels = read_qrels(os.path.join(args.data_dir, 'vkmarco-doctrain-qrels.tsv'))
    test_queries = read_queries(os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv'))
    with open(os.path.join(args.data_dir, 'objects.csv'), encoding='utf-8') as f:
        objects = [(int(r['ObjectId']), int(r['QueryId']), r['DocumentId']) for r in csv.DictReader(f)]

    needed_doc_ids = {doc_id for _, _, doc_id in objects}
    docs = {}
    with open(os.path.join(args.data_dir, 'vkmarco-docs.tsv'), encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t', 3)
            if len(fields) >= 4 and fields[0] in needed_doc_ids:
                docs[fields[0]] = (fields[2], fields[3])

    object_groups = {}
    for object_id, query_id, doc_id in objects:
        object_groups.setdefault(query_id, []).append((object_id, doc_id))

    doc_ids = list(docs)
    token_pattern = re.compile(r'(?u)\b\w\w+\b')
    title_terms, body_terms = {}, {}
    title_lengths, body_lengths = {}, {}
    document_frequency = Counter()
    for doc_id, (title, body) in docs.items():
        title_tf = Counter(token_pattern.findall(title.lower()))
        body_tf = Counter(token_pattern.findall(body.lower()))
        title_terms[doc_id], body_terms[doc_id] = title_tf, body_tf
        title_lengths[doc_id], body_lengths[doc_id] = sum(title_tf.values()), sum(body_tf.values())
        document_frequency.update(title_tf.keys() | body_tf.keys())

    avg_title_length = sum(title_lengths.values()) / max(len(doc_ids), 1)
    avg_body_length = sum(body_lengths.values()) / max(len(doc_ids), 1)
    n_docs = len(doc_ids)
    k1, b_title, b_body, title_weight = 1.2, 0.3, 0.75, 3.0
    scores = {}
    for query_id, candidates in object_groups.items():
        query_tf = Counter(token_pattern.findall(test_queries[query_id].lower()))
        for object_id, doc_id in candidates:
            score = 0.0
            for term, query_frequency in query_tf.items():
                df = document_frequency.get(term, 0)
                if not df:
                    continue
                idf = math.log1p((n_docs - df + 0.5) / (df + 0.5))
                title_tf = title_terms[doc_id].get(term, 0)
                body_tf = body_terms[doc_id].get(term, 0)
                weighted_tf = (
                    title_weight * title_tf / (1 - b_title + b_title * title_lengths[doc_id] / max(avg_title_length, 1))
                    + body_tf / (1 - b_body + b_body * body_lengths[doc_id] / max(avg_body_length, 1))
                )
                if weighted_tf:
                    score += query_frequency * idf * weighted_tf * (k1 + 1) / (weighted_tf + k1)
            scores[object_id] = score

    with open(args.submission_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ObjectId', 'Score'])
        for object_id, _, _ in objects:
            writer.writerow([object_id, scores[object_id]])

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
