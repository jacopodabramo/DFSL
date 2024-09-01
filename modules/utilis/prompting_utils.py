from modules.utilis.constant import *
from datasets import Dataset
import json
import os
import re


def normalization(query_pred):
    res = {}
    for query in query_pred:
        query_normalized = query.replace(' ', '').replace('\n', '').lower()
        if query_normalized not in res:
            res[query_normalized] = query
    return list(res.values())


def check_status(code):
    if code == 200:
        return True
    else:
        print(f'Error in the request. Status code: {code}')
        return False


def extraction(query, open_tag, close_tag):
    pattern = re.compile(re.escape(open_tag) + r'(.*?)' + re.escape(close_tag), re.DOTALL)
    matches = pattern.findall(query)
    if matches:
        return matches
    else:
        return []


def extraction_pipe(reponse, open_tag, close_tag):
    query_predicted = extraction(reponse, open_tag, close_tag)
    if query_predicted:
        query_predicted = [query.replace('\n', ' ') for query in query_predicted]
    else:
        print('No match found in the response')

    return query_predicted


def already_exist(path):
    pattern = r"predictions_(\d+)\.json"
    ids = []
    for filename in os.listdir(path):
        # Match the pattern against the filename
        match = re.match(pattern, filename)
        if match:
            # Extract the ID from the matched groups
            id = match.group(1)
            ids.append(id)
    return ids


def format_entities_or_relations(items):
    """Helper function to format entities or relations."""
    formatted = ''
    for item in items:
        if isinstance(item, list):
            formatted += f"{item[0]} ({item[1]}), "
        else:
            formatted += f"{item} "
    return formatted.strip(', ')


def create_example(question_dict, prompt, type=''):
    separator = prompt['separator']
    open_tag = prompt['open_tag']
    close_tag = prompt['close_tag']
    if type == 'all':
        entities = format_entities_or_relations(question_dict['entities'])
        relations = format_entities_or_relations(question_dict['relations'])
        example_prompt = (f"Question: {question_dict['question']}\n\n"
                          f"Gold Entities:\n{entities}\n\n"
                          f"Gold Relations:\n{relations}\n\n"
                          f"Query:\n{open_tag}\n"
                          f"{question_dict['query']}\n"
                          f"{close_tag}\n"
                          f"{separator}\n\n")
    elif type == 'E':
        entities = format_entities_or_relations(question_dict['entities'])
        example_prompt = (f"Question: {question_dict['question']}\n\n"
                          f"Gold Entities:\n{entities}\n\n"
                          f"Query:\n{open_tag}\n"
                          f"{question_dict['query']}\n"
                          f"{close_tag}\n"
                          f"{separator}\n\n")
    elif type == 'R':
        relations = format_entities_or_relations(question_dict['relations'])
        example_prompt = (f"Question: {question_dict['question']}\n\n"
                          f"Gold Relations:\n{relations}\n\n"
                          f"Query:\n{open_tag}\n"
                          f"{question_dict['query']}\n"
                          f"{close_tag}\n"
                          f"{separator}\n\n")
    else:
        example_prompt = (f"Question: {question_dict['question']}\n\n"
                          f"Query:\n{open_tag}\n"
                          f"{question_dict['query']}\n"
                          f"{close_tag}\n"
                          f"{separator}\n\n")
    return example_prompt


def create_input_llm(question_dict, type):
    input = f"Question: {question_dict['question']}\n\n"
    if type == 'all':
        entities = format_entities_or_relations(question_dict['entities'])
        relations = format_entities_or_relations(question_dict['relations'])
        input += f"Gold Entities:\n{entities}\n\n"
        input += f"Gold Relations:\n{relations}\n\n"
        input += f"Query:"
        return input
    elif type == 'E':
        entities = format_entities_or_relations(question_dict['entities'])
        input += f"Gold Entities:\n{entities}\n\n"
        input += f"Query:"
        return input
    elif type == 'R':
        relations = format_entities_or_relations(question_dict['relations'])
        input += f"Gold Relations:\n{relations}\n\n"
        input += f"Query:"
        return input
    else:
        input += "Query:"
    return input


def create_prompt_dfsl(prompt, q_dict, type_exp, similarities, train_data, test_data, id):
    input_config = config.copy()
    prompt_examples = f''
    for sim_id in similarities[id]:
        q_dict = train_data[sim_id]
        prompt_examples += create_example(q_dict, prompt, type_exp)
    q_dict = test_data[id]
    input = create_input_llm(q_dict, type_exp)

    input_config['text'] = prompt['instruction'] + prompt_examples + input
    return input_config


def create_hf_dataset(train_data, test_data, save_path_single_q, prompt, enc_type, similarities, beam):
    all_input = {"id": [], "input": [], "question": [], "query_gold": [], "save_path": [], "url": []}

    # Create input for each question
    restored = already_exist(save_path_single_q)
    print(f"restored: {len(restored)} predictions")
    for id, question_dict in test_data.items():
        if id not in restored:
            # create input configuration
            input_config = create_prompt_dfsl(prompt, question_dict, enc_type, similarities, train_data, test_data, id)
            if beam:
                input_config["return_all_beams"] = True
            all_input["id"].append(id)
            all_input["input"].append(input_config)
            all_input["question"].append(question_dict['question'])
            all_input["query_gold"].append(question_dict['query'])
            save_path_file = os.path.join(save_path_single_q, f'predictions_{id}.json')
            all_input["save_path"].append(save_path_file)
            all_input["url"].append(URL)
    return Dataset.from_dict(all_input)


def merging(save_path):
    all_results = {}
    for filename in os.listdir(save_path):
        if filename.endswith('.json'):
            path = os.path.join(save_path, filename)
            with open(path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            all_results.update(data)
    return all_results


