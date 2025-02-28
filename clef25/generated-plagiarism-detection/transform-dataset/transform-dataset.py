#!/usr/bin/env python3
import click
from glob import glob
import json

SEPERATOR = '\n\n'

def to_plain_text(doc):
    plain_text = doc.get('title', '') + SEPERATOR + 'Abstract' + SEPERATOR + (SEPERATOR.join([i['content'] for i in doc.get('abstract', [])]))

    for section in doc.get('sections', []):
            if 'title' in section:
                plain_text += SEPERATOR + section['title']

            for paragraph in section['paragraphs']:
                plain_text += SEPERATOR
                offset = len(plain_text)
                length = len(paragraph['content'])

                plain_text += paragraph['content']

    plain_text += SEPERATOR + 'References'

    for reference in doc.get('references', []):
        plain_text += SEPERATOR + reference['tag'] + ' ' + reference['ref']

    return plain_text

@click.command()
@click.argument('input_directory')
@click.argument('output_directory')
def main(input_directory, output_directory):
    """Transform a raw dataset into the PAN text alignment format."""

    for i in glob(f'{input_directory}/*.json'):
        i_parsed = json.load(open(i))
        plain_text_source = to_plain_text(i_parsed['source'])
        id_source = i.split('/')[-1].replace('.json', '').replace(':', '-').replace('.', '-') + '-' + i_parsed['source']['doc_id']
        plain_text_candidate = to_plain_text(i_parsed['candidate'])
        id_candidate = i.split('/')[-1].replace('.json', '').replace(':', '-').replace('.', '-') + '-' + i_parsed['candidate']['doc_id']
        

if __name__ == '__main__':
    main()

