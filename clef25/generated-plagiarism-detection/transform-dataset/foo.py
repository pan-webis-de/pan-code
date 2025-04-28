#!/usr/bin/env python3
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm

DIR = '/home/maik/workspace/tira/python-client/tests/3715852/pan12-text-alignment-test-and-training/pan12-text-alignment-test-corpus-2012-08-12/'

if __name__ == '__main__':
    file = '02_no_obfuscation'
    target_dir = Path('..') / file
    (target_dir / 'inputs' / 'src').mkdir(exist_ok=True, parents=True)
    (target_dir / 'truth').mkdir(exist_ok=True, parents=True)
    (target_dir / 'inputs' / 'susp').mkdir(exist_ok=True, parents=True)
    copyfile(f'{DIR}/{file}/pairs', target_dir/ 'inputs' / 'pairs')
    with open(f'{DIR}/{file}/pairs', 'r') as f:
        for l in tqdm(f):
            susp_id, src_id = l.split()
            copyfile(f'{DIR}/src/{src_id}', target_dir/ 'inputs' / 'src' / f'{src_id}')
            copyfile(f'{DIR}/susp/{susp_id}', target_dir/ 'inputs' / 'susp' / f'{susp_id}')
            xml_file = susp_id.replace(".txt","") + '-' + src_id.replace(".txt", "") + '.xml'
            copyfile(f'{DIR}/{file}/{xml_file}', target_dir/ 'truth' / xml_file)
 
    


