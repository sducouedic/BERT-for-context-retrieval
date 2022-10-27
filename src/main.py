from transformers import BertTokenizer
from BERT import BertForPassageRanking
from SQuAD import create_squad_dataset
from eval import Evaluator
import argparse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--question", help="enter question id")
args = parser.parse_args()
print(args.question)

models_path = '../models/'

model = BertForPassageRanking.from_pretrained("../models/BERT_for_passage_ranking/")
tokenizer = BertTokenizer.from_pretrained(models_path + 'tokenizer/vocab.txt')

dataset_path = '../data/SQuAD-dev-v2.0.json'

questions, contexts, encoded_contexts = create_squad_dataset(dataset_path, tokenizer)
evaluator = Evaluator(model, tokenizer, questions, contexts, encoded_contexts)
first_context, target_rank = evaluator.inference(args.question)

print('\n\n\n')
print('first retrieved context:')
print(contexts[first_context][1])
