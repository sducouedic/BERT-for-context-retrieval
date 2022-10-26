import json
import os


def encode_question(question: str, tokenizer):
    question = tokenizer.tokenize(question)
    question = ['[CLS]'] + question + ['[SEP]']
    question = tokenizer.encode(question, add_special_tokens=False)
    return question


def encode_context(context: str, tokenizer):
    context = tokenizer.tokenize(context)
    context = context + ['[SEP]']
    context = tokenizer.encode(context, add_special_tokens=False)
    return context

def encode_contexts(contexts, tokenizer):
    encoded_contexts = []
    for context in contexts:
        encoded_contexts.append(encode_context(context, tokenizer))
    return encoded_contexts


def encode_question_context_pair(question: str, context: str, tokenizer):
    question = encode_question(question, tokenizer)
    context = encode_context(context, tokenizer)
    input = question + context
    if len(input) > 512:
        input = input[:512]
    return input


def get_file_name(path):
    path_no_ext = os.path.splitext(path)[0]
    return os.path.basename(path_no_ext)


def load_json_data(path):
    f = open(path)
    content = json.load(f)
    f.close()
    return content.get('data')


def save_json_data(path, dict_data):
    with open(path, 'w') as write_file:
        json.dump(dict_data, write_file, indent=4)
