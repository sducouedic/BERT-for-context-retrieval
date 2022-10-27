import torch
from transformers import BertForSequenceClassification


class BertForPassageRanking(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.autograd.Variable(torch.ones(2, config.hidden_size),
                                              requires_grad=True)
        self.bias = torch.autograd.Variable(torch.ones(2), requires_grad=True)
