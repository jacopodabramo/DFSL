# Dynamic Few-Shot Learning for Knowledge Graph Question Answering

## Introduction

This project introduces a novel approach called Dynamic Few-Shot Learning (DFSL). DFSL combines the efficiency of in-context learning with semantic similarity to offer a generally applicable solution for Knowledge Graph Question Answering (KGQA) that achieves state-of-the-art performance. Extensive evaluations across multiple benchmark datasets and architecture configurations have been conducted to demonstrate the effectiveness of our approach.


## Preamble
To create the storage, the examples must be in a JSON format as follows:
```
{
  "{question_ID}": {
    "question": "{natural language question}",
    "query": "{SPARQL query}",
    "entities": [{ent_1}, {ent_2}, ..., {ent_n}],
    "relations": [{rel_1}, {rel_2}, ..., {rel_n}]
  }
}
```
The test set should follow the same structure as the storage, with an additional entry in each question dictionary for the "golden_answer":
```
{
  "{question_ID}": {
    "question": "{natural language question}",
    "query": "{SPARQL query}",
    "entities": [{ent_1}, {ent_2}, ..., {ent_n}],
    "relations": [{rel_1}, {rel_2}, ..., {rel_n}],
    "golden_answer": [{golden_answer_1}, {golden_answer_2}, ..., {golden_answer_n}]
  }
}
```
For Wikidata entities and relations, it is possible to specify the associated label in the following way:
```
"entities": [[{entity_1}, {label_1}], ..., [{entity_n}, {label_n}]]
```
The same structure applies to relations.

## Installation

To install the required dependencies for this project, please ensure you have Python installed on your system. Then, follow these steps:

1. Clone the repository:
    
    ```bash
    git clone https://gitlab.expert.ai/hybrid/experiments/few-shot-sparql.git
    cd few-shot-sparql
    ```
2. Switch to the Refactor branch of the repository:

   ```bash
   git checkout Refactor
   ```

3. Create a virtual environment (optional but recommended):
    
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. Install the required Python libraries:
    
    ```bash
    pip install -r requirements.txt
    ```

You should now have all the necessary dependencies installed to run the project.

## Usage
To use DFSL, the first step is to set the URL to a VLLM server hosting a Large Language Model (LLM). Navigate to `/modules/utilis/constant.py` and modify the `URL` constant. Then, run `main.py` with the following parameters:

- `storage_data`: Path to the storage set.
- `test_set`: Path to the training set.
- `storage_folder`: Folder to store the storage embeddings.
- `result_folder`: Folder to store the results.
- `dataset`: Type of dataset being used.

The hyperparameters are as follows:

- `enc_type`: Determines what is encoded in the embeddings. Possible values are:
  - `"all"`: Encodes the question, entities, and relations.
  - `"E"`: Encodes only the question and entities.
  - `"R"`: Encodes only the question and relations.
- `beam`: Specifies whether to use beam search or not.
- `--k`: Number of similar questions to be retrieved from the storage.
- `--n_process`: Number of parallel requests to be sent.
- `--eval_type`: Evaluation type, either `FS` (first set) or `LS` (large set).

To run the main script, use the following command:

```bash
python main.py --storage_data <path_to_storage> \
               --test_set <path_to_training_set> \
               --storage_folder <storage_embeddings_folder> \
               --result_folder <result_folder> \
               --dataset <dataset_type> \
               --enc_type <enc_type> \
               --beam <beam_search_boolean> \
               --k <number_of_similar_questions> \
               --n_process <number_of_parallel_requests> \
               --eval_type <evaluation_type>
```


## Contributing
We welcome contributions to the project! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. For any issues, contact [jacopo.dabramo@studio.unibo.it](mailto:jacopo.dabramo@studio.unibo.it).

## License
This project is licensed under the MIT License.