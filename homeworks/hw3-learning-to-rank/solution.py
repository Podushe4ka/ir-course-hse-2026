#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию — шаблон решения.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold


SEED = 42
N_FOLDS = 5
PARAMETER_CANDIDATES = [
    {'iterations': 200},
    {'iterations': 400, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 5},
    {'iterations': 800, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 3}
]


def mean_ndcg_at_10(labels, scores, query_ids):
    """Средний nDCG@10 по запросам с gain = 2**Label - 1."""
    results = pd.DataFrame({'QueryId': query_ids, 'Label': labels, 'Score': scores})
    return np.mean([
        ndcg_score(
            (np.exp2(group['Label'].to_numpy()) - 1)[None, :],
            group['Score'].to_numpy()[None, :],
            k=10,
        )
        for _, group in results.groupby('QueryId', sort=False)
    ])


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    features = [column for column in data.columns if column not in ('QueryId', 'Label')]

    data = data.sort_values('QueryId', kind='stable').reset_index(drop=True)
    query_ids = data['QueryId'].to_numpy()
    unique_queries = np.unique(query_ids)
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(unique_queries))

    best_score = -np.inf
    best_params = None
    for params in PARAMETER_CANDIDATES:
        fold_scores = []
        for fold_number, (_, valid_query_idx) in enumerate(folds, start=1):
            valid_queries = unique_queries[valid_query_idx]
            valid_mask = np.isin(query_ids, valid_queries)
            train_part = data.loc[~valid_mask]
            valid_part = data.loc[valid_mask]
            model = CatBoostRanker(
                loss_function='YetiRank', random_seed=SEED,
                verbose=False, allow_writing_files=False, **params,
            )
            model.fit(Pool(train_part[features], train_part['Label'], group_id=train_part['QueryId']))
            scores = model.predict(valid_part[features])
            score = mean_ndcg_at_10(
                valid_part['Label'].to_numpy(), scores, valid_part['QueryId'].to_numpy(),
            )
            fold_scores.append(score)
            print(f'{params}, fold {fold_number}: nDCG@10 = {score:.6f}', flush=True)

        mean_score = np.mean(fold_scores)
        print(f'{params}, mean nDCG@10 = {mean_score:.6f}', flush=True)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f'Best parameters: {best_params}; CV nDCG@10 = {best_score:.6f}', flush=True)
    model = CatBoostRanker(
        loss_function='YetiRank', random_seed=SEED,
        verbose=False, allow_writing_files=False, **best_params,
    )
    model.fit(Pool(data[features], data['Label'], group_id=data['QueryId']))
    model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    features = [column for column in test.columns if column not in ('ObjectId', 'QueryId')]

    model = CatBoostRanker()
    model.load_model(model_file)
    scores = model.predict(test[features])

    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
