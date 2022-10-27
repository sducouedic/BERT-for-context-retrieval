import numpy as np
import torch
from util import *


class Evaluator:
    def __init__(self, model, tokenizer, questions, contexts, encoded_contexts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.questions = questions
        self.contexts = contexts
        self.encoded_contexts = encoded_contexts

    def inference(self, question: str, qa_context_id=None):
        self.model.eval()
        encoded_question = encode_question(question, self.tokenizer)

        bert_inputs = []
        for encoded_context in self.encoded_contexts:
            input = torch.concat((torch.tensor(encoded_question), torch.tensor(encoded_context)), 0)
            if input.shape[0] > 512:
                input = input[:512]
            input = input.unsqueeze(0)
            bert_inputs.append(input)

        retrieval_prob = []
        i = 0
        for input in bert_inputs:
            # cheating otherwise it is too slow
            if i > 500:
                break
            i += 1
            output = self.model(input).logits[0, 0].item()
            retrieval_prob.append(output)

        probs = np.array(retrieval_prob)
        sorted_probs = np.argsort(probs)
        if qa_context_id is None:
            return sorted_probs[0], None

        else:
            return sorted_probs[0], np.where(sorted_probs == qa_context_id)[0]+1
