# SHOULD NOT BE EXECUTED: WONT PRODUCE THE CORRECT RESULT
# (I installed transformers in editable mode and modified it)

from src.BERT import BertForPassageRanking

model = BertForPassageRanking.from_pretrained("../models/BERT_Base_trained_on_MSMARCO",
                                              from_tf=True)
model.save_pretrained("../models/BERT_for_passage_ranking/")
