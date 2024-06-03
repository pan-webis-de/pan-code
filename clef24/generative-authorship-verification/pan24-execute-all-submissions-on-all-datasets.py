#!/usr/bin/env python3
from tira.rest_api_client import Client
import json
from tqdm import tqdm
from glob import glob
from time import sleep

tira = Client()

task = 'generative-ai-authorship-verification-panclef-2024'
dataset = 'pan24-generative-authorship-test-20240502-test'

TEAM_TO_SUBMISSIONS = json.load(open('team-to-submissions.json'))
SUBMISSION_TO_RESOURCES = json.load(open('submission-to-resources.json'))

DATASETS = ['pan24-generative-authorship-test-20240502-test', 'pan24-generative-authorship-test-b-20240506-test', 'pan24-generative-authorship-news-check-20240523-test', 'pan24-generative-authorship-news-test-c-20240506-test', 'pan24-generative-authorship-news-test-d-20240506-test', 'pan24-generative-authorship-news-test-e-20240506-test', 'pan24-generative-authorship-news-test-f-20240514-test', 'pan24-generative-authorship-news-test-g-20240516-test', 'pan24-generative-authorship-news-test-h-20240521-test', 'pan24-generative-authorship-eloquent-20240523-test']

def create_submission_to_resources(input_dir, output_file):
    run_id_to_software = {}
    ret = {}

    for _, i in tira.submissions(task, dataset).iterrows():
        if i['is_evaluation']:
            continue
        run_id_to_software[i['run_id']] = {'software': i['software'], 'team': i['team']}

    for job in glob(f'{input_dir}/**/**/job-executed-on-*.txt'):
        run_id = job.split('/')[-2]
        if run_id not in run_id_to_software:
            continue

        print(run_id, job)
        gpu = open(job).read().split('TIRA_GPU=')[1].split('\n')[0]
        if gpu == '1-nvidia-1080':
            #xl-resources-gpu
            ret[run_id_to_software[run_id]['software']] = 'xl-resources-gpu'

    for job in glob(f'{input_dir}/**/**/job-executed-on-*.txt'):
        run_id = job.split('/')[-2]
        if run_id not in run_id_to_software or run_id_to_software[run_id]['software'] in ret:
            continue

        gpu = open(job).read().split('TIRA_GPU=')[1].split('\n')[0]

        if gpu == '1-nvidia-a100':
            ret[run_id_to_software[run_id]['software']] = 'a100-resources-gpu'
    
    json.dump(ret, open(output_file, 'w'))


def main():
    software_to_docker_id = {}

    for _, i in tira.submissions(task, dataset).iterrows():
        if i['is_evaluation']:
            continue
        try:
            software_to_docker_id[i['software']] = int(i['docker_software_id'])
        except:
            pass

    dataset_to_executed_software = {}

    for d in DATASETS:
        dataset_to_executed_software[d] = set()
        
        for _, i in tira.submissions(task, d).iterrows():
            if i['is_evaluation']:
                continue
            
            dataset_to_executed_software[d].add(i['software'])

    for team, softwares in TEAM_TO_SUBMISSIONS.items():
        for software in softwares:
            if software not in software_to_docker_id:
                continue
            for d in DATASETS:
                if software in dataset_to_executed_software[d]:
                    print('Skip already executed software', team, software, d)
                    continue
                try:
                    print('Start', team, software, d)
                    tira.run_software(f'{task}/{team}/{software}', d, resources=SUBMISSION_TO_RESOURCES.get(software, 'medium-resources'), software_id=software_to_docker_id[software])
                    print('Done. Started ', team, software, d)
                    sleep(30)
                except Exception as e:
                    print(e)
                    print('Failure, sleep 360 seconds')
                    sleep(180)

if __name__ == '__main__':
    #create_submission_to_resources('../../../generative-ai-authorship-verification-panclef-2024/pan24-generative-authorship-test-20240502-test/', 'submission-to-resources.json')
    main()

