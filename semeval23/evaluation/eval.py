import argparse
import csv
import os, sys
from copy import deepcopy

import pandas
import numpy as np
# import xgboost
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from sklearn.svm import SVC, LinearSVC

from CalculateIDF import get_lemtokens
from DataConstuctors import NgramColTransformer, ColLengthExtractor, NgramTransformer
from Dataloader import loadDataSplit
from ParallelTokenizer import do_preprocessing
from Training_functions import dummy, get_vocab_idfs


# Classifiers to check in GridSearch
names = ['SVC', 'LinearSVC', 'MultinomNB']  # , 'sagaLR', 'liblinLR', 'restLR'
classifiers = {'SVC': SVC(),
               'MultinomNB': MultinomialNB(),
               'sagaLR': LogisticRegression(max_iter=1000, random_state=2, n_jobs=2),
               }
parameters = {'SVC': {'kernel': ['rbf']},
              'MultinomNB': {'alpha': [2 ** -1],
                             'fit_prior': [True]},
              'sagaLR': {'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [0]},
              }

parameters_regul = {'SVC': {'C': [2 ** 3],
                            'gamma': [2 ** -9]},
                    'sagaLR': {'C': [2 ** -1]},
                    }

# xgnames = ['XGBLinear', 'XGBTree']
# xgclassifiers = [xgboost.XGBClassifier(booster='gblinear', objective='binary:logistic', n_jobs=2),
#                  xgboost.XGBClassifier(objective='binary:logistic', n_jobs=2)]
# xgparameters = [{'updater': ['shotgun', 'coord_descent'],
#                  'feature_selector': ['cyclic', 'shuffle', 'random']},
#                 {'learning_rate': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0],
#                  'n_estimators': [30, 60, 90, 120, 150, 200, 250, 300],
#                  'max_depth': [2, 4, 6, 8, 10, 20, 40], 'grow_policy': ['depthwise', 'lossguide']}]
#
# xgLinSelectingParams = {'updater': ['shotgun', 'coord_descent'],
#                         'feature_selector': ['greedy', 'thrifty'],
#                         'top_k': [100, 500, 1000, 3000, 5000, 10000]},


def results_to_file(results, filename):
    results = pandas.DataFrame(results)
    with open(filename, 'w+') as file:
        results.to_csv(file, sep='\t')

def load_idfs_vocabs(path, tags):
    vocabs = {}
    idfs = {}

    if not tags:
        tags = ['1g']

    if '1g' in tags:
        vocab1, idfs1 = get_vocab_idfs(path, n_grams=1)
        vocabs['1g'] = vocab1
        idfs['1g'] = idfs1

    if '2g' in tags:
        vocab2, idfs2 = get_vocab_idfs(path, n_grams=2)
        vocabs['2g'] = vocab2
        idfs['2g'] = idfs2

    if '3g' in tags:
        vocab3, idfs3 = get_vocab_idfs(path, n_grams=3)
        vocabs['3g'] = vocab3
        idfs['3g'] = idfs3

    if '4g' in tags:
        vocab4, idfs4 = get_vocab_idfs(path, n_grams=4)
        vocabs['4g'] = vocab4
        idfs['4g'] = idfs4

    if 'pos1g' in tags:
        posvocab1, posidfs1 = get_vocab_idfs(path, pos=True, n_grams=1)
        vocabs['pos1g'] = posvocab1
        idfs['pos1g'] = posidfs1

    if 'pos2g' in tags:
        posvocab2, posidfs2 = get_vocab_idfs(path, pos=True, n_grams=2)
        vocabs['pos2g'] = posvocab2
        idfs['pos2g'] = posidfs2

    if 'pos3g' in tags:
        posvocab3, posidfs3 = get_vocab_idfs(path, pos=True, n_grams=3)
        vocabs['pos3g'] = posvocab3
        idfs['pos3g'] = posidfs3

    if 'pos4g' in tags:
        posvocab4, posidfs4 = get_vocab_idfs(path, pos=True, n_grams=4)
        vocabs['pos4g'] = posvocab4
        idfs['pos4g'] = posidfs4

    return vocabs, idfs

def prepare_cb_and_text_tfidf_once(dataframe, test, norm, weights, tags, vocabs, idfs):
    if not tags:
        tags = ['1g', 'cb']
    min_df = 1
    unions_cb = {}
    unions_text = {}

    if norm == 'minmax':
        normalizer = MaxAbsScaler()
    elif norm == 'uniform':
        normalizer = QuantileTransformer(output_distribution='uniform', random_state=2)
    else:
        normalizer = QuantileTransformer(output_distribution='uniform', random_state=2)

    # train
    df = pandas.DataFrame()
    df['cb'] = dataframe['cb_tokens'].values
    df['text'] = dataframe['text_tokens'].values
    df['pos_cb'] = dataframe['cb_tags'].values
    df['pos_text'] = dataframe['text_tags'].values

    # test
    test_df = pandas.DataFrame()
    test_df['cb'] = test['cb_tokens'].values
    test_df['text'] = test['text_tokens'].values
    test_df['pos_cb'] = test['cb_tags'].values
    test_df['pos_text'] = test['text_tags'].values

    # Make ColumnTransformer for each n-gram
    if '1g' in tags:
        datatrans1_1g = NgramColTransformer(1, 'cb')
        datatrans2_1g = NgramColTransformer(1, 'text')

        normalizer1_1g = deepcopy(normalizer)
        normalizer2_1g = deepcopy(normalizer)

        cv1_1g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        vocab1 = vocabs['1g']
        idfs1 = idfs['1g']
        cv2_1g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=vocab1)
        tfidf_1g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_1g.idf_ = idfs1

        _1g_cb = Pipeline([
            ('tokens', datatrans1_1g),
            ('count', cv1_1g),
            ('norm', normalizer1_1g)
        ])

        _1g_text = Pipeline([
            ('tokens', datatrans2_1g),
            ('count', cv2_1g),
            ('tfidf', tfidf_1g)
        ])

        # transform to fit normalizer
        to_fit_1g = _1g_text.transform(df)
        normalizer2_1g.fit(to_fit_1g)
        _1g_text.steps.append(('norm', normalizer2_1g))

        unions_cb['_1g_cb'] = _1g_cb
        unions_text['_1g_text'] = _1g_text

    if '2g' in tags:
        datatrans1_2g = NgramColTransformer(2, 'cb')
        datatrans2_2g = NgramColTransformer(2, 'text')

        normalizer1_2g = deepcopy(normalizer)
        normalizer2_2g = deepcopy(normalizer)

        cv1_2g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        vocab2 = vocabs['2g']
        idfs2 = idfs['2g']
        cv2_2g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=vocab2)
        tfidf_2g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_2g.idf_ = idfs2

        _2g_cb = Pipeline([
            ('tokens', datatrans1_2g),
            ('count', cv1_2g),
            ('norm', normalizer1_2g)
        ])

        _2g_text = Pipeline([
            ('tokens', datatrans2_2g),
            ('count', cv2_2g),
            ('tfidf', tfidf_2g)
        ])
        # transform to fit normalizer and selector
        to_fit_2g = _2g_text.transform(df)
        normalizer2_2g.fit(to_fit_2g)
        _2g_text.steps.append(('norm', normalizer2_2g))

        unions_cb['_2g_cb'] = _2g_cb
        unions_text['_2g_text'] = _2g_text

    if '3g' in tags:
        datatrans1_3g = NgramColTransformer(3, 'cb')
        datatrans2_3g = NgramColTransformer(3, 'text')

        normalizer1_3g = deepcopy(normalizer)
        normalizer2_3g = deepcopy(normalizer)

        cv1_3g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        vocab3 = vocabs['3g']
        idfs3 = idfs['3g']
        cv2_3g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=vocab3)
        tfidf_3g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_3g.idf_ = idfs3

        _3g_cb = Pipeline([
            ('tokens', datatrans1_3g),
            ('count', cv1_3g),
            ('norm', normalizer1_3g)
        ])

        _3g_text = Pipeline([
            ('tokens', datatrans2_3g),
            ('count', cv2_3g),
            ('tfidf', tfidf_3g)
        ])
        # transform to fit normalizer and selector
        to_fit_3g = _3g_text.transform(df)
        normalizer2_3g.fit(to_fit_3g)
        _3g_text.steps.append(('norm', normalizer2_3g))

        unions_cb['_3g_cb'] = _3g_cb
        unions_text['_3g_text'] = _3g_text

    if '4g' in tags:
        datatrans1_4g = NgramColTransformer(4, 'cb')
        datatrans2_4g = NgramColTransformer(4, 'text')

        normalizer1_4g = deepcopy(normalizer)
        normalizer2_4g = deepcopy(normalizer)

        cv1_4g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        vocab4 = vocabs['4g']
        idfs4 = idfs['4g']
        cv2_4g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=vocab4)
        tfidf_4g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_4g.idf_ = idfs4

        _4g_cb = Pipeline([
            ('tokens', datatrans1_4g),
            ('count', cv1_4g),
            ('norm', normalizer1_4g)
        ])

        _4g_text = Pipeline([
            ('tokens', datatrans2_4g),
            ('count', cv2_4g),
            ('tfidf', tfidf_4g)
        ])
        # transform to fit normalizer and selector
        to_fit_4g = _4g_text.transform(df)
        normalizer2_4g.fit(to_fit_4g)
        _4g_text.steps.append(('norm', normalizer2_4g))

        unions_cb['_4g_cb'] = _4g_cb
        unions_text['_4g_text'] = _4g_text

    if 'pos1g' in tags:
        datatranspos1_1g = NgramColTransformer(1, 'pos_cb')
        datatranspos2_1g = NgramColTransformer(1, 'pos_text')

        normalizerpos1_1g = deepcopy(normalizer)
        normalizerpos2_1g = deepcopy(normalizer)

        cv1_pos1g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        pos_vocab1 = vocabs['pos1g']
        pos_idfs1 = idfs['pos1g']
        cv2_pos1g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=pos_vocab1)
        tfidf_pos1g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_pos1g.idf_ = pos_idfs1

        _pos1g_cb = Pipeline([
            ('tokens', datatranspos1_1g),
            ('count', cv1_pos1g),
            ('norm', normalizerpos1_1g)
        ])

        _pos1g_text = Pipeline([
            ('tokens', datatranspos2_1g),
            ('count', cv2_pos1g),
            ('tfidf', tfidf_pos1g)
        ])
        # transform to fit normalizer and selector
        to_fit_pos1g = _pos1g_text.transform(df)
        normalizerpos2_1g.fit(to_fit_pos1g)
        _pos1g_text.steps.append(('norm', normalizerpos2_1g))

        unions_cb['_pos1g_cb'] = _pos1g_cb
        unions_text['_pos1g_text'] = _pos1g_text

    if 'pos2g' in tags:
        datatranspos1_2g = NgramColTransformer(2, 'pos_cb')
        datatranspos2_2g = NgramColTransformer(2, 'pos_text')

        normalizerpos1_2g = deepcopy(normalizer)
        normalizerpos2_2g = deepcopy(normalizer)

        cv1_pos2g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        pos_vocab2 = vocabs['pos2g']
        pos_idfs2 = idfs['pos2g']
        cv2_pos2g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=pos_vocab2)
        tfidf_pos2g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_pos2g.idf_ = pos_idfs2

        _pos2g_cb = Pipeline([
            ('tokens', datatranspos1_2g),
            ('count', cv1_pos2g),
            ('norm', normalizerpos1_2g)
        ])

        _pos2g_text = Pipeline([
            ('tokens', datatranspos2_2g),
            ('count', cv2_pos2g),
            ('tfidf', tfidf_pos2g)
        ])
        # transform to fit normalizer and selector
        to_fit_pos2g = _pos2g_text.transform(df)
        normalizerpos2_2g.fit(to_fit_pos2g)
        _pos2g_text.steps.append(('norm', normalizerpos2_2g))

        unions_cb['_pos2g_cb'] = _pos2g_cb
        unions_text['_pos2g_text'] = _pos2g_text

    if 'pos3g' in tags:
        datatranspos1_3g = NgramColTransformer(3, 'pos_cb')
        datatranspos2_3g = NgramColTransformer(3, 'pos_text')

        normalizerpos1_3g = deepcopy(normalizer)
        normalizerpos2_3g = deepcopy(normalizer)

        cv1_pos3g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        pos_vocab3 = vocabs['pos3g']
        pos_idfs3 = idfs['pos3g']
        cv2_pos3g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=pos_vocab3)
        tfidf_pos3g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_pos3g.idf_ = pos_idfs3

        _pos3g_cb = Pipeline([
            ('tokens', datatranspos1_3g),
            ('count', cv1_pos3g),
            ('norm', normalizerpos1_3g)
        ])

        _pos3g_text = Pipeline([
            ('tokens', datatranspos2_3g),
            ('count', cv2_pos3g),
            ('tfidf', tfidf_pos3g)
        ])
        # transform to fit normalizer and selector
        to_fit_pos3g = _pos3g_text.transform(df)
        normalizerpos2_3g.fit(to_fit_pos3g)
        _pos3g_text.steps.append(('norm', normalizerpos2_3g))

        unions_cb['_pos3g_cb'] = _pos3g_cb
        unions_text['_pos3g_text'] = _pos3g_text

    if 'pos4g' in tags:
        datatranspos1_4g = NgramColTransformer(4, 'pos_cb')
        datatranspos2_4g = NgramColTransformer(4, 'pos_text')

        normalizerpos1_4g = deepcopy(normalizer)
        normalizerpos2_4g = deepcopy(normalizer)

        cv1_pos4g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, min_df=min_df)

        pos_vocab4 = vocabs['pos4g']
        pos_idfs4 = idfs['pos4g']
        cv2_pos4g = CountVectorizer(encoding='utf-8', tokenizer=dummy, preprocessor=dummy, vocabulary=pos_vocab4)
        tfidf_pos4g = TfidfTransformer(smooth_idf=True, use_idf=True)  # False for only tf representation, scaled
        tfidf_pos4g.idf_ = pos_idfs4

        _pos4g_cb = Pipeline([
            ('tokens', datatranspos1_4g),
            ('count', cv1_pos4g),
            ('norm', normalizerpos1_4g)
        ])

        _pos4g_text = Pipeline([
            ('tokens', datatranspos2_4g),
            ('count', cv2_pos4g),
            ('tfidf', tfidf_pos4g)
        ])
        # transform to fit normalizer and selector
        to_fit_pos4g = _pos4g_text.transform(df)
        normalizerpos2_4g.fit(to_fit_pos4g)
        _pos4g_text.steps.append(('norm', normalizerpos2_4g))

        unions_cb['_pos4g_cb'] = _pos4g_cb
        unions_text['_pos4g_text'] = _pos4g_text

    # ColumnTransformers are stored in a dict, get keys to access them
    cb_keys = list(unions_cb.keys())
    text_keys = list(unions_text.keys())
    print('unions_cb: ', cb_keys)
    print('unions_text: ', text_keys)

    pipeline = None

    # Append the calcuated ColumnTransformers to the FeatureUnion
    if 'cb' in tags:
        for cb_key in cb_keys:
            if pipeline:
                pipeline.transformer_list.append((cb_key, unions_cb[cb_key]))
                pipeline.transformer_weights[cb_key] = weights[0]
            else:
                pipeline = FeatureUnion(transformer_list=[
                    (cb_key, unions_cb[cb_key])
                    ],
                    transformer_weights={
                        cb_key: weights[0]
                    })

        # Clickbaittext length normalized to 0-1 range
        cblen_normalizer = deepcopy(normalizer)
        cblen_pipe = Pipeline([
            ('len', ColLengthExtractor('cb')),
            ('norm', cblen_normalizer)
        ])
        cblen_pipe.fit(df)
        pipeline.transformer_list.append(('cb_len', cblen_pipe))

    if 'text' in tags:
        for text_key in text_keys:
            if pipeline:
                pipeline.transformer_list.append((text_key, unions_text[text_key]))
                pipeline.transformer_weights[text_key] = weights[1]
            else:
                pipeline = FeatureUnion(transformer_list=[
                    (text_key, unions_text[text_key])
                    ],
                    transformer_weights={
                        text_key: weights[1]
                    })

        # referenced text length normalized to 0-1 range
        textlen_normalizer = deepcopy(normalizer)
        textlen_pipe = Pipeline([
            ('len', ColLengthExtractor('text')),
            ('norm', textlen_normalizer)
        ])
        textlen_pipe.fit(df)
        pipeline.transformer_list.append(('text_len', textlen_pipe))

    print(len(pipeline.transformer_list))
    print(pipeline.transformer_list)

    if not pipeline.transformer_list:
        raise Exception('Empty FeatureUnion transformer_list: ', pipeline.transformer_list)

    return pipeline, df, test_df, cb_keys, text_keys

def transform_data(df, test_df, pipe, cb_keys, text_keys, tags, percent, scorer, truth):
    print('--------------Append Selector')

    fnames = []
    pipeline = deepcopy(pipe)

    if scorer == 'chi2':
        cb_selector = SelectPercentile(chi2, percentile=percent[0])
        text_selector = SelectPercentile(chi2, percentile=percent[1])
    elif scorer == 'mutual_info':
        cb_selector = SelectPercentile(mutual_info_classif(discrete_features=False), percentile=percent[0])
        text_selector = SelectPercentile(mutual_info_classif(discrete_features=False), percentile=percent[1])
    else:
        cb_selector = SelectPercentile(chi2, percentile=percent[0])
        text_selector = SelectPercentile(chi2, percentile=percent[1])

    if 'cb' in tags:
        # Append selector for clickbait features
        for cb_key in cb_keys:
            selector = deepcopy(cb_selector)
            dict(pipeline.transformer_list)[cb_key].steps.append(('selector' + str(cb_key), selector))
            dict(pipeline.transformer_list)[cb_key].fit(df, truth)

        # Append the calculated feature names after fitting the FeatureUnion
        for cb_key in cb_keys:
            mask = dict(pipeline.transformer_list)[cb_key].named_steps['selector' + str(cb_key)].get_support()
            allnames = dict(pipeline.transformer_list)[cb_key].named_steps['count'].get_feature_names()

            filtered = np.array(allnames)[mask]
            fnames.append(filtered.tolist())
        fnames.append(['cb_len'])

    if 'text' in tags:
        # Append selector for linked text features
        for text_key in text_keys:
            # transform to fit selector
            text_pipe = dict(pipeline.transformer_list)[text_key]

            selector = deepcopy(text_selector)
            to_fit_1g = text_pipe.transform(df)
            selector.fit(to_fit_1g, truth)
            text_pipe.steps.append(('selector' + str(text_key), selector))

        # Append the calculated feature names after fitting the FeatureUnion
        for text_key in text_keys:
            mask = dict(pipeline.transformer_list)[text_key].named_steps['selector' + str(text_key)].get_support()
            allnames = dict(pipeline.transformer_list)[text_key].named_steps['count'].get_feature_names()

            filtered = np.array(allnames)[mask]
            fnames.append(filtered.tolist())
        fnames.append(['text_len'])

    print(pipeline.transformer_list)

    features_names = []
    for namelist in fnames:
        features_names += namelist
    print('feature_names len:', len(features_names))

    features = pipeline.transform(df)
    test_features = pipeline.transform(test_df)

    return features, test_features, features_names

def write_scores_maps(map, wstart, pstart):
    algo_names = map.keys()

    for algo in algo_names:
        map_dicts = map[algo]

        if map_dicts:
            filename = str(algo) + '_w' + str(wstart) + '_p' + str(pstart) + '_map.csv'

            with open(filename, 'w') as csvfile:
                fieldnames = list(map_dicts[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel-tab')

                writer.writeheader()
                writer.writerows(map_dicts)
        else:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idfs', help="Path to idfs .tar.bz2")
    parser.add_argument('--train', help="Path to train .jsonl")
    parser.add_argument('--test', help="Path to train .jsonl")
    parser.add_argument('--out', help="Dir to dump results in")
    parser.add_argument('--njobs', help="Up to how many jobs should be used", type=int,
                        default=2)
    parser.add_argument('--weinum', help="which weights in 'weights' to use", type=int,
                        default=1)
    parser.add_argument('--pernumstart', help="which percents in 'percents' to use", type=int,
                        default=1)
    parser.add_argument('--pernumend', help="which percents in 'percents' to use", type=int,
                        default=1)
    parser.add_argument('--lemma', action='store_true', default=False,
                        help="Lemmatizing words")
    parser.add_argument('--lower', action='store_true', default=False,
                        help="Lowercasing words")
    parser.add_argument('--stop', action='store_true', default=False,
                        help="Clickbait specific stopwords filtering")
    parser.add_argument('--tags', nargs='+', default=['1g', 'cb'],
                        help="Tags of feature sets to use e.g. --tags 1g 2g pos2g")
    arg_norms = parser.add_argument('--norm', choices=['minmax', 'uniform'], default='uniform',
                                    help='Norm algorithm to apply to feature vectors')
    arg_selectalgos = parser.add_argument('--sel', choices=['chi2', 'mutual_info'], default='chi2',
                                          help='Algorithm to use for feature selection score')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)
    os.chdir(args.out)

    log = open(args.out + 'GridSearcher_log_wei' + str(args.weinum) + 'per' + str(args.pernumstart) +
               ''.join(args.tags) + '.txt', 'w')

    print(args.tags)

    # split multi class from passage class
    # full_DataDF = loadDataSplit(args.train)

    DataDF = loadDataSplit(args.train)
    DataDF['label'].replace(to_replace='phrase', value=1, inplace=True)
    DataDF['label'].replace(to_replace='passage', value=0, inplace=True)
    DataDF['label'].replace(to_replace='multi', value=2, inplace=True)
    TestDataDF = loadDataSplit(args.test)
    TestDataDF['label'].replace(to_replace='phrase', value=1, inplace=True)
    TestDataDF['label'].replace(to_replace='passage', value=0, inplace=True)
    TestDataDF['label'].replace(to_replace='multi', value=2, inplace=True)

    # nur clickbait der klasse phrase und passage
    DataDF = DataDF[DataDF['label'] < 2]
    TestDataDF = TestDataDF[TestDataDF['label'] < 2]

    cb_raw = do_preprocessing(DataDF, 'clickbait')
    DataDF['text'] = DataDF[['title', 'text']].agg(' '.join, axis=1)
    text_raw = do_preprocessing(DataDF, 'text')
    test_cb_raw = do_preprocessing(TestDataDF, 'clickbait')
    TestDataDF['text'] = TestDataDF[['title', 'text']].agg(' '.join, axis=1)
    test_text_raw = do_preprocessing(TestDataDF, 'text')

    cb_tokentags = get_lemtokens(cb_raw, lem=args.lemma, stop=args.stop, lower=args.lower)
    text_tokentags = get_lemtokens(text_raw, lem=args.lemma, stop=args.stop, lower=args.lower)
    test_cb_tokentags = get_lemtokens(test_cb_raw, lem=args.lemma, stop=args.stop, lower=args.lower)
    test_text_tokentags = get_lemtokens(test_text_raw, lem=args.lemma, stop=args.stop, lower=args.lower)

    tokentags = pandas.DataFrame()
    tokentags['cb_tokens'] = cb_tokentags['tokens']
    tokentags['cb_tags'] = cb_tokentags['tags']
    tokentags['text_tokens'] = text_tokentags['tokens']
    tokentags['text_tags'] = text_tokentags['tags']
    test_tokentags = pandas.DataFrame()
    test_tokentags['cb_tokens'] = test_cb_tokentags['tokens']
    test_tokentags['cb_tags'] = test_cb_tokentags['tags']
    test_tokentags['text_tokens'] = test_text_tokentags['tokens']
    test_tokentags['text_tags'] = test_text_tokentags['tags']

    # random_state=2 hat 40-60 Balance
    X_train_raw = tokentags
    X_test_raw = test_tokentags
    y_train = DataDF['label']
    y_test = TestDataDF['label']

    njobs = args.njobs
    weights = [[1, 0.25]] #
    percents = [[100, 100], [100, 70]]

    # make map for Algo: [{weights: ..., percent: ..., score_train: ..., score: ..., }, ...]
    scores_map = {}
    for name in names:
        scores_map[name] = []

    weightsnum = int(args.weinum)
    percentsnumstart = int(args.pernumstart)
    percentsnumend = int(args.pernumend)

    # metric = make_scorer(fbeta_score, beta=0.5)
    metric = 'accuracy'

    vocabularies, idf_values = load_idfs_vocabs(args.idfs, args.tags)

    for w in [weights[weightsnum-1]]:#
        print('--------------Weights', w)
        log.write('--------------Weights' + str(w) + '\n')

        # create new directory to dump gridsearch results in
        foldername = 'Weights_' + str(w)

        if not os.path.exists(foldername):
            os.mkdir(foldername)
        os.chdir(foldername)

        train_pipe, X_train, X_test, cb_pipekeys, text_pipekeys = \
            prepare_cb_and_text_tfidf_once(X_train_raw, X_test_raw, args.norm, w, args.tags,
                                           vocabularies, idf_values)

        print(percents[percentsnumstart-1:percentsnumend])
        for percent in percents[percentsnumstart-1:percentsnumend]:#
            print('--------------Top ', percent, '% of features')
            log.write('--------------Top ' + str(percent) + '% of features\n')

            train_vector, test_vector, features_names = \
                transform_data(X_train, X_test, train_pipe, cb_pipekeys, text_pipekeys,
                               args.tags, percent, args.sel, y_train)

            print('train len:', train_vector.shape)
            print('test len:', test_vector.shape)

            print('features: ', len(features_names), '\n')
            log.write('features: ' + str(len(features_names)) + '\n')

            print('--------------Starting Cross-Validation')

            # create new folder for amount of features used
            kfeaturesfolder = str(train_vector.shape[1]) + '_features'
            if not os.path.exists(kfeaturesfolder):
                os.mkdir(kfeaturesfolder)
            os.chdir(kfeaturesfolder)

            for name in names:
                clf = GridSearchCV(classifiers[name], parameters[name], scoring=metric, cv=5, n_jobs=njobs)
                clf.fit(train_vector, y_train)

                best_params = clf.cv_results_['params'][clf.best_index_]
                print('params ' + name + ': ', best_params)
                log.write('params: ' + name + str(best_params) + '\n')
                results_to_file(clf.cv_results_, name + '_all_features.csv')

                # take best parameters model and refit on whole train set
                model = clf.best_estimator_

                if name in ['SVC', 'LinearSVC', 'liblinLR','sagaLR', 'restLR']:
                    print('--------------Regularization')

                    regul_clf = GridSearchCV(model, parameters_regul[name], scoring=metric, cv=5)
                    regul_clf.fit(train_vector, y_train)

                    regul_best_params = regul_clf.cv_results_['params'][regul_clf.best_index_]
                    print('params ' + name + ': ', regul_best_params)
                    log.write('params: ' + name + str(regul_best_params) + '\n')
                    results_to_file(regul_clf.cv_results_, name + '_reguled.csv')

                    train_score = regul_clf.best_score_
                    hyparams = regul_clf.best_params_

                    regul_model = regul_clf.best_estimator_

                    regul_model.fit(train_vector, y_train)
                    predictions = regul_model.predict(test_vector)

                else:
                    train_score = clf.best_score_
                    hyparams = clf.best_params_

                    model.fit(train_vector, y_train)
                    predictions = model.predict(test_vector)

                print('cross_val score: ', train_score)
                log.write('cross_val: ' + str(train_score) + '\n')

                test_accuracy = metrics.accuracy_score(predictions, y_test)
                print('test Accuracy: ', test_accuracy)
                log.write('test Accuracy: ' + str(test_accuracy) + '\n')

                conf_mat = metrics.confusion_matrix(y_test, predictions, labels=[1, 0])
                # TP / TP + FP
                if (conf_mat[0][0] + conf_mat[1][0]) == 0:
                    prec_1 = 0.0
                    prec_2 = conf_mat[1][1] / (conf_mat[0][1] + conf_mat[1][1])
                elif (conf_mat[0][1] + conf_mat[1][1]) == 0:
                    prec_1 = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
                    prec_2 = 0.0
                else:
                    prec_1 = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
                    prec_2 = conf_mat[1][1] / (conf_mat[0][1] + conf_mat[1][1])

                print('precision: ', '{0:.3f}\t{1:.3f}'.format(prec_1, prec_2))
                print(conf_mat)
                log.write('precision: ' + '{0:.3f}\t{1:.3f}'.format(prec_1, prec_2) + '\n')
                log.write(str(conf_mat) + '\n\n')

                # fill scores map
                algo_map = {'weights': w,
                            'percent_cb': percent[0],
                            'percent_text': percent[1],
                            'score_train': train_score,
                            'score': test_accuracy,
                            'hyparams': hyparams}

                scores_map[name].append(algo_map)

            # move up in directory out of '<percent>_features' folder
            path_parent = os.path.dirname(os.getcwd())
            os.chdir(path_parent)

        # move up in directory out of 'Weights' folder
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)

    write_scores_maps(scores_map, weightsnum, percentsnumstart)

    log.close()

    # move up in directory out of 'GridSearch' folder
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
