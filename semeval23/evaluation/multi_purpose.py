import argparse
import collections
import math
import os
import re
import sys, copy
import json, csv

from nltk import word_tokenize

import numpy
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split

from CalculateIDF import get_lemtokens
from Dataloader import loadDataSplit, load_json_asdict
from ParallelTokenizer import do_preprocessing


def load(path_file, key):
    c = 0
    dict = {}
    with open(path_file, 'r', encoding='utf8') as f:
        for line in f:
            obj = json.loads(line)
            dict[obj[key]] = obj
            c += 1

            if c > len(dict):
                print(obj)
                raise Exception('Gotcha! Error: UUID is not unique')
    return dict

def make_scores_dataframe(all_scores_path):
    if str(all_scores_path).endswith('.csv'):
        with open(all_scores_path, 'r', encoding='utf-8') as scores_file:
            scores = pandas.read_csv(scores_file, sep='\t')

        return None, scores

    elif str(all_scores_path).endswith('.json'):
        with open(all_scores_path, 'r', encoding='utf-8') as scores_file:
            scores_json = json.load(scores_file)

            scores = pandas.DataFrame()
            scores['uuid'] = list(scores_json['single_scores'].keys())

            bert_scores = []
            meteor_scores = []
            bleu_scores = []

            for values in scores_json['single_scores'].values():
                bert_scores.append(values['bertscore_score'])
                meteor_scores.append(values['meteor_score'])
                bleu_scores.append(values['bleu4_score'])

            scores['BERTScore'] = bert_scores
            scores['meteor'] = meteor_scores
            scores['bleu-4'] = bleu_scores

        return scores_json, scores
    else:
        raise Exception('Invalid file extension, only .json ,.csv supported')

def make_me_score():
    path_file_instance1 = './train.jsonl'

    path_file_meteor_scores_albert = './meteor_bleu_bert_scores_albert.csv'
    path_file_meteor_scores_allenai = './meteor_bleu_bert_scores_allenai.csv'
    path_file_albert_uuids = './meteor-metric/albert_uuids.txt'
    path_file_allenai_uuids = './meteor-metric/allenai_uuids.txt'
    path_file_ok_postIds = './meteor-metric/ok_postIds.txt'

    cbs20dict = load_json_asdict(path_file_instance1, 'uuid')
    obj_meteor_albert = {}
    obj_meteor_allenai = {}

    with open(path_file_albert_uuids, 'r') as albert_uuids_file:
        albert_uuids = [line.replace('\n', '') for line in albert_uuids_file]

    with open(path_file_allenai_uuids, 'r') as allenai_uuids_file:
        allenai_uuids = [line.replace('\n', '') for line in allenai_uuids_file]

    with open(path_file_meteor_scores_albert, 'r', encoding='utf-8') as alb:
        albert_reader = csv.DictReader(alb, fieldnames=['BERTScore', 'meteor', 'bleu-4', 'uuid'],
                                       dialect='excel-tab')
        for ent in albert_reader:
            obj_meteor_albert[ent['uuid']] = {'meteor': ent['meteor'],
                                              'bleu-4': ent['bleu-4'],
                                              'BERTScore': ent['BERTScore']
                                              }

    with open(path_file_meteor_scores_allenai, 'r', encoding='utf-8') as allen:
        allen_reader = csv.DictReader(allen, fieldnames=['BERTScore', 'meteor', 'bleu-4', 'uuid'],
                                      dialect='excel-tab')
        for ent in allen_reader:
            obj_meteor_allenai[ent['uuid']] = {'meteor': ent['meteor'],
                                               'bleu-4': ent['bleu-4'],
                                               'BERTScore': ent['BERTScore']
                                               }

    albert_ok = []
    albert_notok = []
    allenai_ok = []
    allenai_notok = []
    with open(path_file_ok_postIds, 'r') as ok_file:
        ok_postIds = [line.replace('\n', '') for line in ok_file]

    allenai = False
    BertScore50 = False
    for i in range(1, len(ok_postIds)):
        if 'Allenai' in ok_postIds[i]:
            allenai = True
            BertScore50 = False
            continue

        if 'BertScore' in ok_postIds[i]:
            BertScore50 = True
            continue

        if allenai:
            if BertScore50:
                allenai_notok.append(ok_postIds[i])
            else:
                allenai_ok.append(ok_postIds[i])
        else:
            if BertScore50:
                albert_notok.append(ok_postIds[i])
            else:
                albert_ok.append(ok_postIds[i])

    with open('me_scores_albert.csv', 'w') as out:
        fieldnames = ['uuid', 'score', 'BERTScore']
        writer = csv.DictWriter(out, fieldnames=fieldnames, dialect='excel-tab')
        writer.writeheader()
        for albert_uuid in albert_uuids:
            postId = cbs20dict[albert_uuid]['postId']

            score = 0
            if float(obj_meteor_albert[albert_uuid]['BERTScore']) > 0.5:
                score = 1
                if postId in albert_notok:
                    score = 0
            else:
                score = 0
                if postId in albert_ok:
                    score = 1

            writer.writerow({'uuid': albert_uuid, 'score': score,
                             'BERTScore': obj_meteor_albert[albert_uuid]['BERTScore']})

    with open('me_scores_allenai.csv', 'w') as out2:
        fieldnames = ['uuid', 'score', 'BERTScore']
        writer = csv.DictWriter(out2, fieldnames=fieldnames, dialect='excel-tab')
        writer.writeheader()
        for allenai_uuid in allenai_uuids:
            postId = cbs20dict[allenai_uuid]['postId']

            score = 0
            if float(obj_meteor_allenai[allenai_uuid]['BERTScore']) > 0.5:
                score = 1
                if postId in allenai_notok:
                    score = 0
            else:
                score = 0
                if postId in allenai_ok:
                    score = 1

            writer.writerow({'uuid': allenai_uuid, 'score': score,
                             'BERTScore': obj_meteor_allenai[allenai_uuid]['BERTScore']})


def make_all_csv(args):

    me_scores = {}
    with open(args.me_scores, 'r', encoding='utf-8') as me_file:
        reader = csv.DictReader(me_file, fieldnames=['uuid', 'score'], dialect='excel-tab')
        next(reader)
        for row in reader:
            me_scores[row['uuid']] = row['score']

    meteor_scores = []
    with open(args.meteor_scores, 'r', encoding='utf-8') as meteor_file:
        for i, line in enumerate(meteor_file):
            if i > 0:
                start = line.find('score:') + len('score:') + 1
                meteor_scores.append(float(line[start:].replace('\n', '')))

    BERTScore_scores = []
    with open(args.BERTScore_scores, 'r', encoding='utf-8') as bert_file:
        for j, line in enumerate(bert_file):
            if j > 0:
                BERTScore_scores.append(float(line.replace('\n', '')[18:]))

    with open(args.bleu_scores, 'r', encoding='utf-8') as bleu_file:
        # bleu_scores = [json.loads(line) for line in open(args.bleu_scores, 'r', encoding='utf-8')]
        # uuids = [line.replace('\n', '') for line in open(args.uuids, 'r', encoding='utf-8')]
        bleu_scores = json.load(bleu_file)
        uuids = bleu_scores['single_scores'].keys()

    with open(args.output_dir + 'all_scores.csv', 'w', encoding='utf-8') as ofile:
        fieldnames = ['me', 'BERTScore', 'meteor', 'bleu-4', 'uuid', 'truth', 'prediction']
        writer = csv.DictWriter(ofile, fieldnames=fieldnames, dialect='excel-tab')
        writer.writeheader()

        for i, uuid in enumerate(uuids):
            writer.writerow({#'me': me_scores[uuids[i]],
                             'BERTScore': BERTScore_scores[i],
                             'meteor': meteor_scores[i],
                             'bleu-4': bleu_scores[uuid]['bleu4_score'],
                             'uuid': uuid,
                             'truth': bleu_scores[uuid]['truth'],
                             'prediction': bleu_scores[uuid]['prediction']})

def make_all_json(args):
    output_dir = str(args.output_dir)

    def get_score(str, search_str):
        start = -1
        pos = str.find(search_str)
        if pos < 0:
            return -1.
        else:
            start = pos + len(search_str) + 1
            return float(str[start:].replace('\n', ''))

    print('Reading scores ...')
    with open(args.bleu_scores, 'r', encoding='utf-8') as bleu_file:
        bleu_scores = json.load(bleu_file)
        uuids = bleu_scores['single_scores'].keys()

    meteor_scores = []
    with open(args.meteor_scores, 'r', encoding='utf-8') as meteor_file:
        for i, line in enumerate(meteor_file):
            if line.find('Final score:') >= 0:
                bleu_scores['scores']['meteor_score'] = float(re.findall(r"[-+]?\d*\.\d+", line)[0])
            elif line.find('score:') > 0:
                meteor_scores.append(float(re.findall(r"[-+]?\d*\.\d+", line)[0]))


    BERTScore_scores = []
    with open(args.BERTScore_scores, 'r', encoding='utf-8') as bert_file:
        letsgo = False
        for line in bert_file:
            if line.find('albert-xxlarge-v2_L8_no-idf_version=') >= 0:
                letsgo = True
                start = line.find('F1:') + len('F1:') + 1
                bleu_scores['scores']['bertscore_score'] = float(line.replace('\n', '')[start:])
            else:
                if letsgo and (line.replace('\n', '') != ''):
                    BERTScore_scores.append(float(re.findall(r"[-+]?\d*\.\d+", line)[0]))


    print('Writing all scores to one file...')
    print(len(meteor_scores), 'METEOR scores')
    print(len(BERTScore_scores), 'BERTScore scores')
    with open(output_dir + '/all_scores.json', 'w', encoding='utf-8') as ofile:
        for i, uuid in enumerate(uuids):
            bleu_scores['single_scores'][uuid]['meteor_score'] = meteor_scores[i]
            bleu_scores['single_scores'][uuid]['bertscore_score'] = BERTScore_scores[i]

        json.dump(bleu_scores, ofile, indent=2)

def del_corrected_from_tags():
    count = 0

    corrected_file = './train.jsonl'
    tags_file = './tags_settings_viewer_CBS20_.json'

    first = load(corrected_file, 'uuid')
    tags = load(tags_file, 'name')

    multi = tags['multi']['ids']
    phrase = tags['phrase']['ids']
    passage = tags['passage']['ids']

    print('multi', len(multi), 'phrase', len(phrase), 'passage', len(passage))

    for key in first.keys():
        if key in multi:
            multi.remove(key)
        elif key in phrase:
            phrase.remove(key)
        elif key in passage:
            passage.remove(key)
        else:
            count += 1

    print(count, 'not found keys')

    tags['multi']['ids'] = multi
    tags['phrase']['ids'] = phrase
    tags['passage']['ids'] = passage

    print('multi', len(tags['multi']['ids']), 'phrase', len(tags['phrase']['ids']), 'passage', len(tags['passage']['ids']))

    with open('cleaned_tags.json', 'w', encoding='utf-8') as res:
        res.write(json.dumps(tags['multi']))
        res.write('\n')
        res.write(json.dumps(tags['phrase']))
        res.write('\n')
        res.write(json.dumps(tags['passage']))
        res.write('\n')

def compare_thresholds(args):
    _, scores = make_scores_dataframe(args.meteor_scores)

    def threshold(x, hold):
        if float(x) >= hold:
            return 1
        else:
            return 0

    for thresh in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print('---Threshold', thresh)
        scores['bert_thresh_'+str(thresh * 10)] = scores['BERTScore'].apply(lambda x: threshold(x, thresh))
        scores['meteor_thresh_' + str(thresh * 10)] = scores['meteor'].apply(lambda x: threshold(x, thresh))
        scores['bleu_thresh_' + str(thresh * 10)] = scores['bleu-4'].apply(lambda x: threshold(x, thresh))

        print('BERTScore: {:1.4f}'.format(metrics.accuracy_score(scores['bert_thresh_'+str(thresh*10)], scores['me'])) + '\t' +
              'METEOR: {:1.4f}'.format(metrics.accuracy_score(scores['meteor_thresh_' + str(thresh * 10)], scores['me'])) + '\t\t' +
              'BLEU-4: {:1.4f}'.format(metrics.accuracy_score(scores['bleu_thresh_'+str(thresh*10)], scores['me'])))

        print('FP:' + str((scores['bert_thresh_'+str(thresh*10)] & (scores['me'] == 0)).sum()) + '\t\t\t' +
              'FP:' + str((scores['meteor_thresh_' + str(thresh * 10)] & (scores['me'] == 0)).sum()) + '\t\t\t' +
              'FP:' + str((scores['bleu_thresh_' + str(thresh * 10)] & (scores['me'] == 0)).sum()) + '\n' +
              'FN:' + str(((scores['bert_thresh_'+str(thresh*10)] == 0) & scores['me']).sum()) + '\t\t\t' +
              'FN:' + str(((scores['meteor_thresh_' + str(thresh * 10)] == 0) & scores['me']).sum()) + '\t\t\t' +
              'FN:' + str(((scores['bleu_thresh_' + str(thresh * 10)] == 0) & scores['me']).sum()) + '\n')

def compare_thresholds_passret(scores_path):
    all_scores = pandas.read_csv(scores_path, sep=',')

    def make_num(boolean):
        if boolean:
            return 1
        else:
            return 0
    all_scores['me'] = all_scores['correct'].apply(lambda x: make_num(x))

    def threshold(x, thre):
        if float(x) >= thre:
            return 1
        else:
            return 0

    numbers = []
    for tag in ['phrase', 'passage']:
        scores = copy.deepcopy(all_scores[all_scores['tags'] == tag])
        print('------', tag)
        for thresh in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('---Threshold', thresh)
            scores['bert_thresh_'+str(thresh * 100)] = scores['bertscore_score'].apply(lambda x: threshold(x, thresh))
            scores['meteor_thresh_' + str(thresh * 100)] = scores['meteor_score'].apply(lambda x: threshold(x, thresh))
            scores['bleu_thresh_' + str(thresh * 100)] = scores['bleu4_score'].apply(lambda x: threshold(x, thresh))

            # print('BERTScore Acc: {:1.4f}'.format(metrics.accuracy_score(scores['bert_thresh_'+str(thresh*10)], scores['me'])) + '\t' +
            #       'METEOR Acc: {:1.4f}'.format(metrics.accuracy_score(scores['meteor_thresh_' + str(thresh * 10)], scores['me'])) + '\t\t' +
            #       'BLEU-4 Acc: {:1.4f}'.format(metrics.accuracy_score(scores['bleu_thresh_'+str(thresh*10)], scores['me'])))

            bertscore_acc = metrics.accuracy_score(scores['bert_thresh_'+str(thresh*100)], scores['me'])
            meteor_acc = metrics.accuracy_score(scores['meteor_thresh_' + str(thresh * 100)], scores['me'])
            bleu4_acc = metrics.accuracy_score(scores['bleu_thresh_'+str(thresh*100)], scores['me'])


            if tag == 'phrase':
                factor = 4
            else:
                factor = 5
            bertscore_fp = (scores['bert_thresh_'+str(thresh*100)] & (scores['me'] == 0)).sum() * factor
            meteor_fp = (scores['meteor_thresh_'+str(thresh*100)] & (scores['me'] == 0)).sum() * factor
            bleu4_fp = (scores['bleu_thresh_'+str(thresh*100)] & (scores['me'] == 0)).sum() * factor
            bertscore_fn = ((scores['bert_thresh_'+str(thresh*100)] == 0) & scores['me']).sum() * factor
            meteor_fn = ((scores['meteor_thresh_'+str(thresh*100)] == 0) & scores['me']).sum() * factor
            bleu4_fn = ((scores['bleu_thresh_'+str(thresh*100)] == 0) & scores['me']).sum() * factor

            numbers += [{
                'tags': tag, 'threshold': thresh*100,
                'bleu_fp': bleu4_fp, 'bleu_fn': bleu4_fn, 'bleu_acc': bleu4_acc,
                'mete_fp': meteor_fp, 'mete_fn': meteor_fn, 'mete_acc': meteor_acc,
                'bert_fp': bertscore_fp, 'bert_fn': bertscore_fn, 'bert_acc': bertscore_acc
            }]
    pandas.DataFrame(numbers).to_csv('PR_ble_met_ber_thresholds.csv', sep='\t')

def calc_with_threshs(args):
    if args.threshs == 'passage':
        threshs = [0.3, 0.7, 0.6]
        train_dataframe = loadDataSplit(args.train)
    elif args.threshs == 'phrase':
        threshs = [0.4, 0.5, 0.8]
        train_dataframe = loadDataSplit(args.train)
    elif args.threshs == 'both':
        threshs = [[0.3, 0.7, 0.6], [0.4, 0.5, 0.8]]
        train_dataframe = loadDataSplit(args.train)
    else:
        raise Exception('Invalid value for --threshs parameter choose ( passage | phrase | both )')


    scores_json, scores = make_scores_dataframe(args.meteor_scores)

    spoilers_mask = []
    for uuid in scores_json['single_scores'].keys():
        if uuid in train_dataframe['uuid'].values:
            spoilers_mask.append(True)
        else:
            spoilers_mask.append(False)

    scores = scores[spoilers_mask]

    def threshold(x, hold):
        if float(x) >= hold:
            return 1
        else:
            return 0

    print('---Thresholds', threshs)
    print('BLEU-4' + '\t\t' +
          'METEOR-F1' + '\t' +
          'BERTScore-F1')

    if args.threshs == 'both':
        filtered_dataframe = train_dataframe[train_dataframe['uuid'].isin(scores['uuid'])]
        labels_dataframe = filtered_dataframe[['uuid', 'label']]

        labeled_scores = pandas.merge(scores, labels_dataframe, on='uuid')

        phrase_scores = copy.deepcopy(labeled_scores[labeled_scores['label'] == 'phrase'])
        passage_scores = copy.deepcopy(labeled_scores[labeled_scores['label'] == 'passage'])

        passage_scores['bleu_thresh_' + str(threshs[0][0] * 10)] = \
            passage_scores['bleu-4'].apply(lambda x: threshold(x, threshs[0][0]))
        passage_scores['meteor_thresh_' + str(threshs[0][1] * 10)] \
            = passage_scores['meteor'].apply(lambda x: threshold(x, threshs[0][1]))
        passage_scores['bert_thresh_' + str(threshs[0][2] * 10)] \
            = passage_scores['BERTScore'].apply(lambda x: threshold(x, threshs[0][2]))

        phrase_scores['bleu_thresh_' + str(threshs[1][0] * 10)] \
            = phrase_scores['bleu-4'].apply(lambda x: threshold(x, threshs[1][0]))
        phrase_scores['meteor_thresh_' + str(threshs[1][1] * 10)] \
            = phrase_scores['meteor'].apply(lambda x: threshold(x, threshs[1][1]))
        phrase_scores['bert_thresh_' + str(threshs[1][2] * 10)] \
            = phrase_scores['BERTScore'].apply(lambda x: threshold(x, threshs[1][2]))
    else:
        scores['bleu_thresh_'+str(threshs[0] * 10)] = scores['bleu-4'].apply(lambda x: threshold(x, threshs[0]))
        scores['meteor_thresh_' + str(threshs[1] * 10)] = scores['meteor'].apply(lambda x: threshold(x, threshs[1]))
        scores['bert_thresh_' + str(threshs[2] * 10)] = scores['BERTScore'].apply(lambda x: threshold(x, threshs[2]))

    if scores_json:
        # to cut off 2 decimals from e.g. 0.2345678 -> 23.45
        bleu4_score = math.floor(scores_json['scores']['bleu4_lemma'] * 10000)/100
        meteor_score = math.floor(scores_json['scores']['meteor_score'] * 10000)/100
        bertscore_score = math.floor(scores_json['scores']['bertscore_score'] * 10000)/100

        if args.threshs == 'both':
            tp_bleu4 = phrase_scores['bleu_thresh_' + str(threshs[1][0] * 10)].sum() \
                       + passage_scores['bleu_thresh_' + str(threshs[0][0] * 10)].sum()
            tp_meteor = phrase_scores['meteor_thresh_' + str(threshs[1][1] * 10)].sum() \
                       + passage_scores['meteor_thresh_' + str(threshs[0][1] * 10)].sum()
            tp_bertscore = phrase_scores['bert_thresh_' + str(threshs[1][2] * 10)].sum() \
                           + passage_scores['bert_thresh_' + str(threshs[0][2] * 10)].sum()

            print(str(bleu4_score) + ' TP:' + str(tp_bleu4) + '\t' +
                  str(meteor_score) + ' TP:' + str(tp_meteor) + '\t' +
                  str(bertscore_score) + ' TP:' + str(tp_bertscore) + '\n')
        else:
            print(str(bleu4_score) + ' TP:' + str(
                scores['bleu_thresh_' + str(threshs[0] * 10)].sum()) + '\t' +
                  str(meteor_score) + ' TP:' + str(
                scores['meteor_thresh_' + str(threshs[1] * 10)].sum()) + '\t' +
                  str(bertscore_score) + ' TP:' + str(
                scores['bert_thresh_' + str(threshs[2] * 10)].sum()) + '\n')
    else:
        print('TP:' + str(scores['bleu_thresh_' + str(threshs[0] * 10)].sum()) + '\t\t' +
              'TP:' + str(scores['meteor_thresh_' + str(threshs[1] * 10)].sum()) + '\t\t' +
              'TP:' + str(scores['bert_thresh_' + str(threshs[2] * 10)].sum()) + '\n')

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help="Path to train .jsonl")
    parser.add_argument('--lemma', action='store_true', default=False,
                        help="Lemmatizing words")
    parser.add_argument('--lower', action='store_true', default=False,
                        help="Lowercasing words")
    parser.add_argument('--stop', action='store_true', default=False,
                        help="Clickbait specific stopwords filtering")

    parser.add_argument('--meteor_scores', help="Path to meteor_scores.txt")
    parser.add_argument('--bleu_scores', help="Path to bleu_scores.txt")
    parser.add_argument('--BERTScore_scores', help="Path to BERTScore_scores.txt")
    parser.add_argument('--me_scores', help='Path to me_scores.csv')
    parser.add_argument('--uuids', help="Path to uuids.txt")

    parser.add_argument('--output_dir', help="Directory to write output files to")

    parser.add_argument('--threshs', choices=['passage', 'phrase', 'both'],
                        help='thresholds to choose for eval')
    parser.add_argument('--fun', choices=['split_train', 'make_all_csv', 'make_all_json', 'del_corrected',
                                          'me_score', 'comp_thresh', 'comp_thresh_pr', 'calc_with_thresh',
                                          'plt_points'],
                        help='function to run from this script')


    args = parser.parse_args()

    choice = args.fun
    if choice == 'split_train':
        split_train_set(args)
    elif choice == 'make_all_csv':
        make_all_csv(args)
    elif choice == 'make_all_json':
        make_all_json(args)
    elif choice == 'del_corrected':
        del_corrected_from_tags()
    elif choice == 'me_score':
        make_me_score()
    elif choice == 'comp_thresh':
        compare_thresholds(args)
    elif choice == 'comp_thresh_pr':
        compare_thresholds_passret(args.me_scores)
    elif choice == 'calc_with_thresh':
        calc_with_threshs(args)
    elif choice == 'plt_points':
        plot_points(args.train)

if __name__ == "__main__":
    main(sys.argv[1:])
