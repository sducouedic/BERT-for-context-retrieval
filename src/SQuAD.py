import os.path as path
import pandas as pd
from transformers import BertTokenizer

from util import *

data_path = '../data/'
models_path = '../models/'


def process_data(dataframe):
    # add useful columns and remove irrelevant ones
    df_questions = dataframe.set_index('id') \
                            .drop(columns=['answers', 'is_impossible', 'plausible_answers']) \
                            .rename(columns={'paragraphs.context': 'context', 'title': 'q_title'})
    df_questions['q_title'] = pd.factorize(df_questions['q_title'])[0]
    df_questions['q_context'] = pd.factorize(df_questions['context'])[0]

    # remove duplicates
    df_questions = df_questions.drop_duplicates(keep=False)

    # create separate dataframe for contexts
    df_context = df_questions[['context', 'q_context', 'q_title']].copy() \
        .rename(columns={'q_context': 'context_id', 'q_title': 'c_title'}) \
        .set_index('context_id')
    df_context = df_context.drop_duplicates()
    df_context = df_context.sort_index()

    # remove contexts from df_questions
    df_questions = df_questions.drop(columns=['context'])

    # make a cross join of questions and contexts
    df_questions['qid'] = df_questions.index
    df_context['cid'] = df_context.index
    df_final = df_questions.merge(df_context, how='cross')

    # keep only useful columns and rename them
    df_final['context_in_title'] = df_final['q_title'] == df_final['c_title']
    df_final['context_corresponds'] = df_final['q_context'] == df_final['cid']
    df_final = df_final[['question', 'context', 'context_in_title', 'context_corresponds']]

    # return dataset as numpy record
    data = df_final.to_records(index=False)
    return data


def create_squad_dataset(path):
    """
    :param path: path of the SQuAD train, validation, or test set
    :return: a list of tuples as follow:
            [(str: question, str: context, bool: context_in_title, bool: context_corresponds)]
    """
    data = load_json_data(path)
    df = pd.json_normalize(data, ['paragraphs', 'qas'], ['title', ['paragraphs', 'context']])
    dataset = process_data(df)
    return list(dataset)


def create_squad_torch_input_data(path):
    # load if already computed
    file_name = get_file_name(path)
    bert_data_path = data_path + file_name + '_torch_input.json'
    if path.exists(bert_data_path):
        return load_json_data

    dataset = create_squad_dataset(path)
    tokenizer = BertTokenizer.from_pretrained(models_path + 'tokenizer/vocab.txt')

    bert_data = []
    i = 0
    for question, context, context_in_title, context_corresponds in dataset:
        torch_input = encode_question_context_pair(question, context, tokenizer)
        bert_data.append({'torch_input': torch_input,
                          'context_in_title': 1 if context_in_title else 0,
                          'context_corresponds': 1 if context_corresponds else 0})
        if i % 1000 == 0:
            print(i)
        i += 1
    save_json_data(bert_data_path, {'data': bert_data})
    return bert_data


data = create_squad_torch_input_data('../data/SQuAD-dev-v2.0.json')
