#!/usr/bin/python3
from glob import glob
import shutil
import os

def copy_system_inputs(d):
    """I keep it consistent with the structure that was previously uploaded: i.e., the inputs go into a structure like
       easy/{train,test,validation}
         - problem-1.txt
         - ...
       medium/{train,test,validation}
         - problem-1.txt
         - ...
       hard/{train,test,validation}
         - problem-1.txt
         - ...
    """
    for difficulty in ['easy', 'medium', 'hard']:
        target_dir = f'{d}-system-inputs/{difficulty}/{d}/'
        os.makedirs(target_dir)
        c = 0
        for f in glob(f'style_analysis/{difficulty}/{d}/problem-*.txt'):
            c += 1
            shutil.copyfile(f, target_dir + f.split('/')[-1])
        print(f'Copied {c} problem statements to {target_dir}')

def copy_evaluator_inputs(d):
    """I keep it consistent with the structure that was previously uploaded: i.e., the truths directly go into a structure like
       easy/
         - truth-problem-1.json
         - ...
       medium/
         - truth-problem-1.json
         - ...
       hard/
         - truth-problem-1.json
         - ...
    """
    for difficulty in ['easy', 'medium', 'hard']:
        target_dir = f'{d}-truths/{difficulty}/'
        os.makedirs(target_dir)
        c = 0
        for f in glob(f'style_analysis/{difficulty}/{d}/truth*.json'):
            c += 1
            shutil.copyfile(f, target_dir + f.split('/')[-1])
        print(f'Copied {c} truth files to {target_dir}')

for dataset_type in ['test', 'train', 'validation']:
    copy_system_inputs(dataset_type)
    copy_evaluator_inputs(dataset_type)

