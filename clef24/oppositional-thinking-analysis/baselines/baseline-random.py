import argparse
import json
import random

SPAN_EMPTY_LABEL = 'X'

SPAN_LABELS = [
    'OBJECTIVE', 'AGENT', 'FACILITATOR', 'CAMPAIGNER', 'VICTIM', 'NEGATIVE_EFFECT'
]

def load_json(file_path):
    '''
    Load a JSON file from the given path.
    :param file_path: Path to the JSON file.
    :return: The loaded JSON object.
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def ensure_id_uniqueness(json_data):
    '''
    Ensure that the IDs in the predictions are unique.
    :return:
    '''
    ids = set()
    for txt_data in json_data:
        if txt_data['id'] in ids:
            raise ValueError(f"ID {txt_data['id']} is not unique!")
        ids.add(txt_data['id'])

def ensure_input_data(json_data):
    '''
    Ensure the presence of the necessary data for inference:
    the 'id', 'text', and 'spacy_tokens' fields.
    :return:
    '''
    for txt_data in json_data:
        if 'id' not in txt_data:
            raise ValueError("ID field is missing!")
        if 'text' not in txt_data:
            raise ValueError("Text field is missing!")
        if 'spacy_tokens' not in txt_data:
            raise ValueError("Spacy tokens field is missing!")

def generate_task1(input_path, output_path, p_conspiracy=0.5, verbose=False):
    '''
    Evaluate the predictions for Task 1, the binary classification task.
    :return:
    '''
    # load and validate the input data
    input_json = load_json(input_path)
    ensure_id_uniqueness(input_json)
    ensure_input_data(input_json)
    # generate random predictions
    output_json = []
    for txt_data in input_json:
        out_data = {'id': txt_data['id']}
        out_data['category'] = 'CONSPIRACY' if random.random() < p_conspiracy else 'CRITICAL'
        output_json.append(out_data)
    # save the predictions to the output json file
    json_data = json.dumps(output_json, ensure_ascii=False, indent=2)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(json_data)

def generate_task2(input_path, output_path, verbose=False):
    '''
    Evaluate the predictions for Task 2, the sequence labeling task.
    :return:
    '''
    # load and validate the input data
    input_json = load_json(input_path)
    ensure_id_uniqueness(input_json)
    ensure_input_data(input_json)
    # generate random predictions
    output_json = []
    for txt_data in input_json:
        out_data = {'id': txt_data['id']}
        num_annotations = random.randint(0, 6)
        annotations = []
        for _ in range(num_annotations):
            start_char = random.randint(0, len(txt_data['text']) - 1)
            end_char = random.randint(start_char + 1, len(txt_data['text']))
            span_categ = random.choice(SPAN_LABELS)
            annotations.append({'start_char': start_char, 'end_char': end_char, 'category': span_categ})
        out_data['annotations'] = annotations
        output_json.append(out_data)
    # save the predictions to the output json file
    json_data = json.dumps(output_json, ensure_ascii=False, indent=2)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(json_data)

def main():
    '''
    Entry point for random baseline generator.
    '''
    parser = argparse.ArgumentParser(description="Random baseline for the PAN 2024 'Oppositional' task.")
    # Required arguments
    parser.add_argument("task", type=str, help="Task to evaluate", choices=["task1", "task2"])
    parser.add_argument("--input", required=True, type=str, help="Path of the .json file with input texts.")
    parser.add_argument("--output", required=True, type=str, help="Path of the .json file to which the predictions will be written.")
    args = parser.parse_args()
    if args.task == "task1": generate_task1(args.input, args.output)
    elif args.task == "task2": generate_task2(args.input, args.output)

if __name__ == '__main__':
    main()
