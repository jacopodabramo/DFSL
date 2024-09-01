from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import URLError
import time
import re
import os


def change(string):
    string = string.replace('( ', '(').replace(' )', ')').replace('{ ', ' {') \
        .replace(' }', '}').replace(': ', ':').replace(' , ', ', ').replace(" ' ", "'") \
        .replace("' ", "'").replace(" '", "'").replace(' = ', '=').strip()

    rep_dec = re.findall('[0-9] \. [0-9]', string)
    for dec in rep_dec:
        string = string.replace(dec, dec.replace(' . ', '.'))

    return ' '.join(string.split())


def process(string):
    return change(string)


def empty(r):
    if not r:
        return True
    if 'boolean' not in r:
        if 'results' in r:
            if 'bindings' in r['results']:
                if not r['results']['bindings']:
                    return True
                if {} in r['results']['bindings']:
                    return True
    return False


def hitkg2(query, url, typeq="target", max_retries=5, retry_delay=1):
    retries = 0
    while retries < max_retries:
        if retries > 0:
            print('Retrying...')
        try:
            sparql = SPARQLWrapper(url)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            if retries > 0:
                print('Done')
            if empty(results) and typeq == 'target':
                return "Empty"
            else:
                return results
        except URLError as err:
            print("\nURLError occurred")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                print("\nMax retries reached. Giving up.")
                return ''
        except Exception as err:
            return ''


def parse_answer_from_result(result_dict):
    if 'boolean' in result_dict:
        return result_dict['boolean']
    elif 'results' in result_dict:
        raw_results = result_dict['results']['bindings']
        result = [uri['value'] for item in raw_results for uri in list(item.values())]
        result = list(filter(lambda x: x != '', result))
        return result
    else:
        return None


def compute_prf1_one(prediction, gold_answer, url):
    pred_result = hitkg2(process(prediction), url)
    pred_answer = parse_answer_from_result(pred_result)

    if gold_answer is None:
        if pred_answer is None:
            recall, precision, f1 = 1, 1, 1
        else:
            recall, precision, f1 = 0, 0, 0
    elif isinstance(gold_answer, bool):
        if gold_answer == pred_answer:
            recall, precision, f1 = 1, 1, 1
        else:
            recall, precision, f1 = 0, 0, 0
    elif isinstance(pred_answer, bool):
        if gold_answer == pred_answer:
            recall, precision, f1 = 1, 1, 1
        else:
            recall, precision, f1 = 0, 0, 0
    elif isinstance(gold_answer, list):
        if pred_answer is None:
            recall, precision, f1 = 0, 0, 0
        else:
            pred_answer = set(pred_answer)
            gold_answer = set(gold_answer)
            if len(pred_answer.intersection(gold_answer)) != 0:
                precision = len(pred_answer.intersection(gold_answer)) / len(pred_answer)
                recall = len(pred_answer.intersection(gold_answer)) / len(gold_answer)
                f1 = (2 * recall * precision / (recall + precision))
            else:
                recall, precision, f1 = 0, 0, 0
    result_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return result_dict, gold_answer, pred_answer


def get_query_LS(url, queries):
    max_results = -1
    final_query = ""
    first_query = queries[0] if queries else None  # Ensure there's at least one query

    for query in queries:
        pred_result = hitkg2(process(query), url)
        pred_answer = parse_answer_from_result(pred_result)

        if pred_answer is not None:
            if len(pred_answer) > max_results:
                max_results = len(pred_answer)
                final_query = query

    if max_results == -1 and first_query is not None:
        return first_query  # Return the first query if all results were None

    return final_query


def get_query_FS(url,queries):
    first_result = None

    for query in queries:
        pred_result = hitkg2(process(query), url)
        pred_answer = parse_answer_from_result(pred_result)

        if first_result is None:
            first_result = query

        if pred_answer is not None:
            return query

    # If all results are None, return the first result
    return first_result


def write_errors(question, gold_answer, pred_answer, query_golden, query_pred, exp_path, id_pred):
    to_write = (f"Id: {id_pred}\n\n"
                f"Question:\n{question}\n\n"
                f"Golden query:\n{query_golden}\n\n"
                f"Predicted query:\n{query_pred}\n\n"
                f"Golden answer:\n{gold_answer}\n\n"
                f"Predicted answer:\n{pred_answer}\n"
                f"----------------------------------------------------------------------------------------\n\n")

    with open(os.path.join(exp_path, 'errors.txt'), 'a', encoding='utf-8') as file:
        file.write(to_write)


def get_query(type_eval, url, query):
    if type_eval == 'FS':
        return get_query_FS(url,query)
    elif type_eval == 'LS':
        return get_query_LS(url,query)
    else:
        raise ValueError("Type of evaluation not recognized")