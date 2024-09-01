from modules.utilis.eval_utilis import *
from modules.utilis.utilities import *
from tqdm import tqdm


class Eval:

    def __init__(self, dfsl, test_storage, url, eval_type):
        self.dfsl = dfsl
        self.test_storage = test_storage
        self.url = url
        self.eval_type = eval_type

        self.eval_dict = {'precision': 0, 'recall': 0, 'f1': 0, 'total': 0}
        self.eval_dict_not_none = {'precision': 0, 'recall': 0, 'f1': 0, 'total': 0}
        self.final_dict = {}
        self.prediction_with_answers = {}
        self.predictions = self.dfsl.get_results()
        self.exp_path = self.dfsl.get_results_folder_path()
        self.references = self.test_storage.get_storage_data()
        self.evaluate()

    def update_dict(self, precision, recall, f1, golden_answer):
        self.eval_dict['precision'] += precision
        self.eval_dict['recall'] += recall
        self.eval_dict['f1'] += f1
        self.eval_dict['total'] += 1

        if golden_answer is not None:
            self.eval_dict_not_none['precision'] += precision
            self.eval_dict_not_none['recall'] += recall
            self.eval_dict_not_none['f1'] += f1
            self.eval_dict_not_none['total'] += 1

    def calculate_final_dict(self):
        self.final_dict = {
            'all': {
                "precision": float(self.eval_dict['precision'] / self.eval_dict['total']),
                "recall": float(self.eval_dict['recall'] / self.eval_dict['total']),
                "f1": float(self.eval_dict['f1'] / self.eval_dict['total']),
            },
            'non_none': {
                "precision": float(self.eval_dict_not_none['precision'] / self.eval_dict_not_none['total']),
                "recall": float(self.eval_dict_not_none['recall'] / self.eval_dict_not_none['total']),
                "f1": float(self.eval_dict_not_none['f1'] / self.eval_dict_not_none['total']),
            }
        }

    def evaluate(self):
        for id_pred, pred_dict in tqdm(self.predictions.items()):
            golden_reference = self.references[id_pred]
            if pred_dict['query_pred_selected'] is not None:
                result_dict, gold_answer, pred_answer = compute_prf1_one(pred_dict['query_pred_selected'], golden_reference['golden_answer'], self.url)
                p, r, f = result_dict['precision'], result_dict['recall'], result_dict['f1']
                self.update_dict(p, r, f, golden_reference['golden_answer'])
                write_errors(golden_reference['question'], golden_reference['golden_answer'], pred_answer,
                             golden_reference['query'], pred_dict['query_pred_selected'], self.exp_path,
                             id_pred) if f == 0 else None
                pred_answer = list(pred_answer) if isinstance(pred_answer, set) else pred_answer
                self.prediction_with_answers[id_pred] = {'question': golden_reference['question'],
                                                         'query_gold': golden_reference['query'],
                                                         'query_pred': pred_dict['query_pred_selected'],
                                                         'gold_answer': golden_reference['golden_answer'],
                                                         'pred_answer': pred_answer, 'f1': result_dict['f1']}
                self.calculate_final_dict()
            else:
                p, r, f = 0, 0, 0
                self.update_dict(p, r, f, golden_reference['golden_answer'])
        save_json(os.path.join(self.exp_path, 'predictions_with_answers.json'), self.prediction_with_answers)

    def get_final_dict(self):
        return self.final_dict
