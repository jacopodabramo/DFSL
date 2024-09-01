from modules.storage import Storage
from modules.dfsl import *
from modules.eval import *
from prompt.prompts import *
import argparse

def main(args):
    # path input prameters
    storage_data_path = args.storage_data
    test_set_path = args.test_set
    storage_folder_path = args.storage_folder
    result_folder_path = args.result_folder
    dataset = args.dataset

    # hyperparameters
    type_encoding = args.enc_type
    beam = args.beam
    k = args.k
    n_process = args.n_process
    eval_type = args.eval_type

    url = DB_PEDIA_SPARQL_ENDPOINT if dataset == 'dbpedia' else WIKIDATA_SPARQL_ENDPOINT



    # Create storage object
    storage = Storage(storage_folder_path, storage_data_path, type_encoding)

    # Creating test-storage object
    test_storage = Storage(result_folder_path, test_set_path, type_encoding, name_storage='test_storage')

    # Choice of the prompt
    prompt = dynamic_few_shots_wikidata

    # run DFSL
    dfsl = Dfsl(storage, test_storage, prompt, result_folder_path, type_encoding,eval_type, url, beam, k, n_process)

    # Evaluation
    eval = Eval(dfsl,test_storage,url,eval_type)
    print(eval.final_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--storage_data', type=str, default='C:\\Users\jacop\Desktop\\test\\train_data.json')
    parser.add_argument('--test_set', type=str, default='C:\\Users\jacop\Desktop\\test\\test_data.json')
    parser.add_argument('--storage_folder', type=str, default='./storage')
    parser.add_argument('--result_folder', type=str, default='./results')
    parser.add_argument('--dataset', type=str, default='wikidata', choices=['wikidata', 'dbpedia'])

    parser.add_argument('--enc_type', type=str, default='all', choices=['all', 'E', 'R'])
    parser.add_argument('--beam', action='store_true', default=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_process', type=int, default=8)
    parser.add_argument('--eval_type', type=str, default="FS", choices=["FS", "LS"])

    args = parser.parse_args()

    main(args)
