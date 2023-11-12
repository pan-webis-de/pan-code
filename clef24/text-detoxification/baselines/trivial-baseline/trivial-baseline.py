#!/usr/bin/env python3
import argparse
from glob import glob
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description='The trivial baseline for the PAN 2023 text detoxification task that removes all/none/or specified stopwrods from given text for detoxification.')

    parser.add_argument('--input', help='The input file, expected a jsonl file.', required=True)
    parser.add_argument('--output', help='The output file, will create a jsonl file.', required=True)
    parser.add_argument('--stopword-directory', help='An optional pointer to a directory containing stopwords.', required=False, default=None)
    parser.add_argument('--remove-all-terms', help='Generate the empty string.', required=False, default=False, type=bool)
    parser.add_argument('--remove-no-terms', help='Output the text without modification.', required=False, default=False, type=bool)

    return parser.parse_args()

def parse_stopwords(stopword_dir):
    ret = set()

    if not stopword_dir:
        return ret

    print('Load stopwords from ' + stopword_dir)

    if stopword_dir:
        for f in glob(stopword_dir + '/*'):
            for t in open(f).read().split('\n'):
                ret.add(t.lower().strip())

    print('Done. Loaded ' + str(len(ret)) + ' stopwords.')

    return ret

def detoxify_text(text, stopword_list=None, remove_all_terms=False, remove_no_terms=True):
    if remove_no_terms:
        return text
    if remove_all_terms:
        return ''
    ret = []
    
    for token in re.split(r'\s+', text):
        if not stopword_list or token.lower().strip() not in stopword_list:
            ret += [token]

    return ' '.join(ret)

def main(args):
    stopwords = parse_stopwords(args.stopword_directory)
    ret = []

    with open(args.input, 'r') as input_file, open(args.output, 'w+') as output_file:
        for l in input_file:
            l = json.loads(l)
            l['text'] = detoxify_text(l['text'], stopwords, args.remove_all_terms, args.remove_no_terms)
            output_file.write(json.dumps(l) + '\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)

