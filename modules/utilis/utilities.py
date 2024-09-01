from sentence_transformers import util
import json
import os
import torch


def create_dir(path):
    if not os.path.exists(path):
        print(f"Creating directory {path}")
        os.makedirs(path)


def load_data(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def create_input_to_embedd(question_dict, type_enc):
    full_text = question_dict['question']
    if type_enc in ['all', 'E']:
        for ent in question_dict['entities']:
            full_text += f" {ent[0]} " + f"({ent[1]})" if isinstance(ent, list) else f" {ent}"
    if type_enc in ['all', 'R']:
        for rel in question_dict['relations']:
            full_text += f" {rel[0]} " + f"({rel[1]})" if isinstance(rel, list) else f" {rel}"
    return full_text


def embed_questions(model, data, param='all'):
    questions = [create_input_to_embedd(question, param) for question in data.values()]
    embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True, device='cuda')

    return embeddings


def save_tensor(path, tensor):
    torch.save(tensor, path)


def compute_similarity(train_embeddings, test_embeddings, train_data, test_data, top_k=5):
    train_data_ids = list(train_data.keys())
    test_data_ids = list(test_data.keys())

    # Compute cosine-similarity
    cosine_scores = util.cos_sim(train_embeddings, test_embeddings)

    # Initialize dictionary to store results
    similar_questions = {}
    #
    # Iterate over each test question
    for i, test_id in enumerate(test_data_ids):
        # Get similarities for the current test question
        similarities = cosine_scores[:, i]

        # Sort indices based on similarity in descending order
        sorted_indices = torch.argsort(similarities, descending=True)

        # Retrieve train question IDs based on sorted indices
        top_k_train_ids = [train_data_ids[idx.item()] for idx in sorted_indices[:top_k]]

        # Store results in the dictionary
        similar_questions[test_id] = top_k_train_ids

    return similar_questions


def save_similarities(result, train_data, test_data, k, embedding_path):
    res = {}
    for id, similar in result.items():
        res[id] = {'question': test_data[id]['question']}
        res[id]['similar'] = [train_data[sim_id]['question'] for sim_id in similar]
        res[id]['similar_query'] = [train_data[sim_id]['query'] for sim_id in similar]
        # Construct the file path
    file_path = os.path.join(embedding_path, f'similarities_{k}.json')
    save_json(file_path, res)
