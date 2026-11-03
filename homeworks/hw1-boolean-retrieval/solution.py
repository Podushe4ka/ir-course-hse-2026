#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ДЗ 1. Индексация и булев поиск — шаблон решения.

Запуск:
    ./solution.py --build_index --index_dir=index PATH-TO-DATA
    ./solution.py --submission_file=submission.csv --index_dir=index PATH-TO-DATA
"""

import argparse
from timeit import default_timer as timer

from nltk import tokenize
from collections import defaultdict
import csv
from pathlib import Path
import json
import sqlite3
import heapq


def preprocess(text):
    """Токенизация и нормализация. НЕ МЕНЯЙТЕ эту функцию:
    ей же размечен эталон на Kaggle."""
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]


def build_index(data_dir, index_dir):
    # TODO:
    # - прочитать документы из data_dir/vkmarco-docs.tsv
    #   (без заголовка; поля через таб: DocumentId, URL, Title, Body);
    # - разбить Title и Body на термины функцией preprocess() (URL не индексируем);
    # - построить обратный индекс и сохранить его в index_dir.
    index = defaultdict(list)

    data_path = Path(data_dir)
    index_path = Path(index_dir)
    (index_path / 'word_index').parent.mkdir(parents=True, exist_ok=True)

    db_path = index_path / 'word_index.db'
    con = sqlite3.connect(db_path)

    con.execute('CREATE TABLE word_index (word TEXT, doc_id TEXT)')

    with open(data_path / "vkmarco-docs.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            document_id, _, header_text, body_text = row
            words = set(preprocess(header_text) + preprocess(body_text))
            con.executemany(
                'INSERT INTO word_index VALUES (?, ?)',
                ((word, document_id) for word in words),
            )

    con.execute('CREATE INDEX idx_word ON word_index(word)')
    con.commit()
    con.close()


def make_submission(data_dir, index_dir, submission_file):
    # TODO:
    # - загрузить индекс из index_dir;
    # - прочитать запросы из data_dir/vkmarco-doceval-queries.tsv
    #   (без заголовка; поля через таб: QueryId, QueryText) и разбить их функцией preprocess();
    # - для каждого запроса найти документы, в которых есть ВСЕ термины запроса (И-запрос);
    # - для каждой строки data_dir/objects.csv (ObjectId,QueryId,DocumentId) поставить
    #   Label = 1, если документ нашёлся по запросу, иначе 0;
    # - записать submission_file в формате CSV с заголовком ObjectId,Label.
    index_path = Path(index_dir)
    data_path = Path(data_dir)

    db_path = index_path / 'word_index.db'
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    query_doc = defaultdict(set)

    with open(data_path / "vkmarco-doceval-queries.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            query_id, query_text = row
            words = set(preprocess(query_text))

            if not words:
                continue

            doc_ids_per_word = list()
            for word in words:
                cursor.execute(
                    'SELECT doc_id FROM word_index WHERE word = ? ORDER BY doc_id',
                    (word,)
                    )
                doc_ids = [r[0] for r in cursor.fetchall()]
                if not doc_ids:
                    doc_ids_per_word = []
                    break
                doc_ids_per_word.append(doc_ids)

            if not doc_ids_per_word:
                continue

            merged = heapq.merge(*doc_ids_per_word)

            k = len(doc_ids_per_word)
            current_doc = None
            count = 0
            
            for doc_id in merged:
                if doc_id == current_doc:
                    count += 1
                else:
                    if count == k:
                        query_doc[query_id].add(current_doc)
                    current_doc = doc_id
                    count = 1
        
            if count == k:
                query_doc[query_id].add(current_doc)
    
    con.close()

    with open(data_path / "objects.csv", 'r', encoding='utf-8') as f_in, \
         open(submission_file, 'w', encoding='utf-8', newline='') as f_out:
         
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        header = next(reader)
        writer.writerow(['ObjectId', 'Label'])
        
        for row in reader:
            object_id, query_id, document_id = row
            label = 1 if document_id in query_doc[query_id] else 0
            writer.writerow([object_id, label])


def main():
    parser = argparse.ArgumentParser(description='Boolean retrieval homework solution')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('--build_index', action='store_true', help='режим построения индекса')
    parser.add_argument('--index_dir', required=True, help='папка с индексом')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()
    if args.build_index:
        build_index(args.data_dir, args.index_dir)
    else:
        if not args.submission_file:
            parser.error('в режиме генерации сабмишна нужен --submission_file')
        make_submission(args.data_dir, args.index_dir, args.submission_file)
    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
