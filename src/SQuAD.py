import numpy as np
import pandas as pd

from util import *

data_path = '../data/'


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


def create_squad_dataset(path, tokenizer):
    # get questions and contexts
    data = load_json_data(path)
    df = pd.json_normalize(data, ['paragraphs', 'qas'], ['title', ['paragraphs', 'context']])
    questions, contexts = process_data(df)
    questions = questions.to_records(index=False)
    contexts = contexts.to_records(index=True)

    # encode and save contexts for pytorch model (BERT)
    contexts_values = contexts['context'].tolist()
    encoded_contexts = encode_contexts(contexts_values, tokenizer)

    # save questions, contexts and encoded_contexts
    file_name = get_file_name(path)
    qa_path = data_path + file_name + '_questions.npy'
    ctx_path = data_path + file_name + '_contexts.npy'
    np.save(qa_path, questions)
    np.save(ctx_path, contexts)
    encoded_contexts_data_path = data_path + file_name + '_encoded_contexts.json'
    save_json_data(encoded_contexts_data_path, {'data': encoded_contexts})

    # change questions and contexts to lists of tuples
    questions = list(questions)
    contexts = list(contexts)

    return questions, contexts, encoded_contexts


