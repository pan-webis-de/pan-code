import argparse, os
import csv
import json
import math
import re


# make .csv with scores for every model for every checkpoint
def make_scores_checkpoint_table(models_dir, tune_dir, eval_dir, output_path):
    scores_dict = {}
    checkpoints = set()

    for rootdir, dirnames, filenames in os.walk(models_dir):

        # e.g. squad/checkpoint-6000
        if (tune_dir + '/checkpoint-' in rootdir) and eval_dir in rootdir:
            rootdirname = rootdir.replace(models_dir, '')
            modelname = rootdirname[:rootdirname.find('/')]

            if modelname not in scores_dict.keys():
                scores_dict[modelname] = {}

            checkpoint = -1
            for dir in rootdirname.split('/'):
                if 'checkpoint' in dir:
                    checkpoint = int(re.findall(r"\d+", dir)[0])
                    checkpoints.add(checkpoint)

            print(modelname, checkpoint, '            ', end='\r')

            if checkpoint >= 0:
                if 'eval_results.json' in filenames:
                    for filename in filenames:
                        if filename == 'eval_results.json':
                            with open(os.path.join(rootdir, filename)) as score_file:
                                file_json = json.load(score_file)
                                score = math.floor(file_json['f1'] * 100)/100  # truncate all but 2 decimals

                            scores_dict[modelname]['Checkpoint' + str(checkpoint)] = score
                else:
                    print('WARNING:', '            ')
                    print(str(rootdirname) + ' no file eval_results.json found')

            else:
                print('WARNING:', '            ')
                print(str(modelname) + 'no checkpoints with expected filesystem found\n' +
                                '=> tune_dir/checkpoint-xxxxxx/eval_dir/...')

        # last checkpoint is not in a 'checkpoint'-dir
        elif tune_dir + '/' + eval_dir in rootdir:
            rootdirname = rootdir.replace(models_dir, '')
            modelname = rootdirname[:rootdirname.find('/')]

            if modelname not in scores_dict.keys():
                scores_dict[modelname] = {}

            print(modelname, 'last', '            ')

            if 'eval_results.json' in filenames:
                for filename in filenames:
                    if filename == 'eval_results.json':
                        with open(os.path.join(rootdir, filename)) as score_file:
                            file_json = json.load(score_file)
                            score = math.floor(file_json['f1'] * 100) / 100  # truncate all but 2 decimals

                        scores_dict[modelname]['Checkpointlast'] = score
            else:
                print('WARNING:', '            ')
                print(str(rootdirname) + 'no file eval_results.json found')

    checkpoints = sorted(list(checkpoints))
    print('Checkpoints:', checkpoints)
    checkpoints = list(map( lambda x: 'Checkpoint' + str(x),checkpoints))
    with open(output_path, 'w') as output:
        fieldnames = ['Model'] + checkpoints + ['Checkpointlast']
        writer = csv.DictWriter(output, fieldnames=fieldnames, dialect='excel-tab')
        writer.writeheader()

        for modelname in scores_dict.keys():
            row_to_write = {'Model': modelname}

            for i in range(1, len(fieldnames)):
                if fieldnames[i] in scores_dict[modelname].keys():
                    row_to_write[fieldnames[i]] = scores_dict[modelname][fieldnames[i]]
                else:
                    row_to_write[fieldnames[i]] = '--'

            writer.writerow(row_to_write)

def make_mtscores_checkpoint_table(eval_dir, output_path):
    scores_dict = {}

    with open(eval_dir, 'r') as scores_file:
        for i, line in enumerate(scores_file):
            if (i % 6) == 0:
                tokens = line.split()
                modelname = tokens[0]
                checkpoint = tokens[1]
                if not modelname in scores_dict.keys():
                    scores_dict[modelname] = {}

            elif (i % 6) == 4:
                scores = re.findall(r"[-+]?\d*\.*\d+", line)
                scores_dict[modelname][checkpoint] = {
                    'BLEU-4': scores[0] + ' (' + scores[1] + ')',
                    'METEOR': scores[2] + ' (' + scores[3] + ')',
                    'BERTScore': scores[4] + ' (' + scores[5] + ')'
                }
            else:
                continue

        modelnames = list(scores_dict.keys())
        checkpoints = list(scores_dict[modelnames[0]].keys())
        with open(output_path, 'w') as output:
            fieldnames = ['Model', 'Metric'] + checkpoints
            writer = csv.DictWriter(output, fieldnames=fieldnames, dialect='excel-tab')
            writer.writeheader()

            for modelname in modelnames:
                for k, metric in enumerate(['BLEU-4', 'METEOR', 'BERTScore']):
                    if k == 0:
                        row_to_write = {'Model': modelname, 'Metric': metric}
                    else:
                        row_to_write = {'Model': '', 'Metric': metric}

                    avail_checkpoints = scores_dict[modelname].keys()
                    for i in range(0, len(checkpoints)):
                        if checkpoints[i] in avail_checkpoints:
                            row_to_write[checkpoints[i]] = scores_dict[modelname][checkpoints[i]][metric]
                        else:
                            scores_dict[modelname][checkpoints[i]] = {
                                'BLEU-4': '--',
                                'METEOR': '--',
                                'BERTScore': '--'
                            }
                            row_to_write[checkpoints[i]] = scores_dict[modelname][checkpoints[i]][metric]

                    writer.writerow(row_to_write)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', help="Root directory of model directories")
    parser.add_argument('--tune_dir', choices=['cbs20_phrase', 'cbs20_phrase_only',
                                               'cbs20_passage', 'cbs20_passage_only',
                                               'squad'],
                        help='Name of directory with tuned models which scores should be fetched\n'
                             'choose from [ cbs20_phrase, cbs20_phrase_only,'
                             '              cbs20_passage, cbs20_passage_only,'
                             '              squad]')
    parser.add_argument('--eval_dir', choices=['eval_cbs1ktrain_passage', 'eval_cbs1ktrain_phrase',
                                               'eval_cbstest_passage', 'eval_cbstest_phrase',
                                               'eval_cbs20_both', 'eval_squad', 'eval'],
                        help='Name of eval_dir\n'
                             'choose from [ eval_cbs1ktrain_passage, eval_cbs1ktrain_phrase, '
                             '              eval_cbstest_passage, eval_cbstest_phrase,'
                             '              eval_squad, eval ]')
    parser.add_argument('--output_path', default='model_scores.csv',
                        help="Path to outputfile that's written (.csv)")
    parser.add_argument('--eval_path', default='model_scores.csv',
                        help="Path to eval .txt with thresholded scores for --fun=bmb")
    parser.add_argument('--fun', choices=['squadf1', 'bmb'],
                        help='function to run from this script')

    args = parser.parse_args()

    choice = args.fun
    if choice == 'squadf1':
        make_scores_checkpoint_table(args.models_dir, args.tune_dir, args.eval_dir, args.output_path)
    elif choice == 'bmb':
        make_mtscores_checkpoint_table(args.eval_path, args.output_path)
    else:
        raise Exception('--fun has only choices: ( squadf1 | bmb )')
