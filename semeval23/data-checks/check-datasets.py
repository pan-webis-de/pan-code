#!/usr/bin/python3
import json
from tqdm import tqdm


def extract_spoiler_from_pos(data_entry, pos):
    pos = data_entry['spoilerPositions'][pos]
    assert len(pos) == 2
    start_paragraph, start_in_paragraph = pos[0]
    end_paragraph, end_in_paragraph = pos[1]

    ret = ''

    for i in range(start_paragraph, end_paragraph +1):
        start_index = 0 if i > start_paragraph else start_in_paragraph
        txt = data_entry['targetParagraphs'][i]
        ret += ('' if len(ret) == 0 else ' ')
        if i < end_paragraph:
            ret += txt[start_index:]
        else:
            ret += txt[start_index:end_in_paragraph]
    
    return ret

def extract_spoiler_from_tag(data_entry, pos):
    return data_entry['spoiler'][pos]

def spoiler_as_expected(entry, pos, ground_truth):
    assert extract_spoiler_from_pos(entry, pos) == ground_truth
    assert extract_spoiler_from_tag(entry, pos) == ground_truth
    assert extract_spoiler_from_tag(entry, pos) == extract_spoiler_from_pos(entry, pos)

test_entries = [json.loads(i) for i in list(open('correct-entries.jsonl', 'r'))]

spoiler_as_expected(test_entries[1], 0, "Let me ask you a question. Are you on a salary? Yes. And that salary is based on the assumption that you work full-time over a 365-day year? That’s right. Well, there are 366 days this year. Today, you’re basically working for free.")

spoiler_as_expected(test_entries[0], 0, "1. Your kid will cry out for you the most on nights you stay up a little later than usual.")
spoiler_as_expected(test_entries[0], 1, "2. At some point you will accidentally hurt your kid and feel like the worst parent ever.")
spoiler_as_expected(test_entries[0], 2, "3. Going to the bathroom in peace will become a thing of the past.")
spoiler_as_expected(test_entries[0], 3, "4. It’ll be way too easy to gain a little weight.")
spoiler_as_expected(test_entries[0], 4, "5. If you co-sleep with your toddler, you shouldn’t expect to have much room.")


#for file_name in ['data/train.jsonl', 'data/validation.jsonl']:
for file_name in ['incorrect-entries.jsonl', 'correct-entries.jsonl']:
    with open(file_name, 'r') as f, open(file_name + '-n', 'w') as t:
        for line in tqdm(list(f)):
            out = line
            line = json.loads(line)
            t.write(out)

            try:
                for i in range(len(line['spoilerPositions'])):
                    assert extract_spoiler_from_pos(line, i) == extract_spoiler_from_tag(line, i)
            except:
                print(f'Invalid entry with uuid {line["uuid"]} in file {file_name}.')

