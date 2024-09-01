from modules.utilis.utilities import *
from modules.utilis.prompting_utils import *
from modules.utilis.eval_utilis import *
import multiprocessing
import requests
import shutil
import sys

class Dfsl:

    def __init__(self,storage, test_storage, prompt, result_folder_path, enc_type, eval_type, url, beam, k = 5, n_process=1):
        self.prediction = None
        self.storage = storage
        self.test_storage = test_storage
        self.prompt = prompt
        self.result_folder_path = result_folder_path
        self.enc_type = enc_type
        self.eval_type = eval_type
        self.url = url
        self.k = k
        self.n_process = n_process
        self.beam = beam
        self.train_data = self.storage.get_storage_data()
        self.test_data = self.test_storage.get_storage_data()


        self.run_dfsl()

    def get_results(self):
        res = load_data(os.path.join(self.result_folder_path, 'predictions.json'))
        return res

    def get_results_folder_path(self):
        return self.result_folder_path

    def call_api(self, example):
        # send request
        response = requests.post(example['url'], json=example['input'])
        results = {}
        # check status
        if check_status(response.status_code):
            json_resp = json.loads(response.text)
            if not self.beam:
                json_resp = json_resp.strip()
                query_pred = extraction_pipe(json_resp,self.prompt['open_tag'],self.prompt['close_tag'])
            else:
                json_resp = ''.join(json_resp['beams'])
                query_pred = extraction_pipe(json_resp,self.prompt['open_tag'],self.prompt['close_tag'])
                query_pred = normalization(query_pred)
            final_query = get_query(self.eval_type,self.url,query_pred)
            results[example['id']] = {'question': example['question'],
                                      'query_gold': example['query_gold'],
                                      'all_query_pred': query_pred,
                                      'query_pred_selected': final_query}
            save_json(example['save_path'], results)

    def LLM_inference(self, test_data, train_data, result_folder_path, prompt, enc_type, similarities, n_process, beam):
        if not os.path.exists(os.path.join(result_folder_path, 'predictions.json')):
            save_path_single_q = os.path.join(result_folder_path, 'single_q')
            create_dir(save_path_single_q)

            df = create_hf_dataset(train_data,
                                   test_data,
                                   save_path_single_q,
                                   prompt,
                                   enc_type,
                                   similarities,
                                   beam)

            print('Starting prompting')
            pool = multiprocessing.Pool(n_process)
            for i, _ in enumerate(pool.imap_unordered(self.call_api, df)):
                sys.stderr.write('\rdone {0:%}'.format(round(i / len(df), 2)))
            print('End of process')
            pool.close()
            pool.join()
            res = merging(save_path_single_q)
            shutil.rmtree(save_path_single_q)
            save_json(os.path.join(result_folder_path, 'predictions.json'), res)
            self.prediction = res

    def run_dfsl(self):
        # Compute similarities
        similarities = compute_similarity(self.storage.get_embedding(), self.test_storage.get_embedding(), self.train_data, self.test_data, self.k)
        save_similarities(similarities, self.train_data, self.test_data, self.k, self.result_folder_path)
        print("Similarities computed and saved successfully.")
        print('-' * 50)

        # Run DFSL
        print("DFSL HYPERPARAMETERS:")
        print(f"Type of encoding: {self.enc_type}")
        print(f"Beam search: {self.beam}")
        print(f"Top-k similar questions: {self.k}")
        print(f"Number of processes: {self.n_process}")
        print('-' * 50)

        self.LLM_inference(self.test_data,self.train_data, self.result_folder_path, self.prompt,self.enc_type, similarities,self.n_process, self.beam)






