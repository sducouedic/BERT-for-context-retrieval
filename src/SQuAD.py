import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import pickle

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
    return df_questions, df_context


def create_squad_dataset(path):
    """
    :param path: path of the SQuAD train, validation, or test set
    :return: a list of tuples as follow:
            [(str: question, str: context, bool: context_in_title, bool: context_corresponds)]
    """
    # get questions and contexts
    data = load_json_data(path)
    df = pd.json_normalize(data, ['paragraphs', 'qas'], ['title', ['paragraphs', 'context']])
    questions, contexts = process_data(df)

    # save questions and contexts
    file_name = get_file_name(path)
    qa_path = data_path + file_name + '_questions.npy'
    ctx_path = data_path + file_name + '_contexts.npy'
    np.save(qa_path, questions.to_records(index=False))
    np.save(ctx_path, contexts.to_records(index=True))

    # encode and save contexts for pytorch model (BERT)
    contexts_values = contexts['context'].tolist()
    tokenizer = BertTokenizer.from_pretrained(models_path + 'tokenizer/vocab.txt')
    encoded_contexts = encode_contexts(contexts_values, tokenizer)
    encoded_contexts_data_path = data_path + file_name + '_encoded_contexts.json'
    save_json_data(encoded_contexts_data_path, {'data': encoded_contexts})

    return questions, contexts, encoded_contexts


def create_squad_torch_input_data(path):
    # load if already computed
    bert_data_path = data_path + file_name + '_torch_input.json'
    file_name = get_file_name(path)
    if os.path.exists(bert_data_path):
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


data = create_squad_dataset('../data/SQuAD-dev-v2.0.json')
