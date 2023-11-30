#!/usr/bin/env python3

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Set, Container, Optional


def parse_stopwords(path: Path) -> Set[str]:
    stopwords: Set[str] = set()

    if not path:
        return stopwords

    logging.info('Loading stopwords from directory: %s', path.absolute())

    for path_file in path.iterdir():
        if path_file.is_file():
            with open(path_file, encoding='UTF-8') as f:
                logging.info('Loading stopwords from file: %s', path_file.name)

                for line in f:
                    stopwords.add(line.strip().lower())

    logging.info('Loaded %d stopword(s)', len(stopwords))

    return stopwords


SPACES = re.compile(r'\s+')


def detoxify(
        text: str,
        stopwords: Optional[Container[str]] = None,
        remove_all_terms: bool = False,
        remove_no_terms: bool = True
) -> str:
    if remove_no_terms:
        return text

    if remove_all_terms:
        return ''

    tokens = []

    for token in SPACES.split(text):
        if not stopwords or token.lower().strip() not in stopwords:
            tokens.append(token)

    return ' '.join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description='The trivial baseline for the PAN 2024 text detoxification task that '
                                                 'removes all/none/or specified stopwrods from given text for '
                                                 'detoxification.')

    parser.add_argument('--input', required=True, type=argparse.FileType('rb'),
                        help='The input file, expected a jsonl file.')
    parser.add_argument('--output', required=True, type=argparse.FileType('w', encoding='UTF-8'),
                        help='The output file, will create a jsonl file.')
    parser.add_argument('--stopword-directory', required=False, type=Path, default=None,
                        help='An optional pointer to a directory containing stopwords.')
    parser.add_argument('--remove-all-terms', required=False, default=False, type=bool,
                        help='Generate the empty string.')
    parser.add_argument('--remove-no-terms', required=False, default=False, type=bool,
                        help='Output the text without modification.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    stopwords = parse_stopwords(args.stopword_directory)

    for line in args.input:
        instance = json.loads(line)
        instance['text'] = detoxify(instance['text'], stopwords, args.remove_all_terms, args.remove_no_terms)

        args.output.write(json.dumps(instance, ensure_ascii=False))
        args.output.write('\n')


if __name__ == '__main__':
    main()
