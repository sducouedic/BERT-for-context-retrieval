# Find Best Context With SQuAD

## Installation

* In the root directory, create a new python environment and install all the requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* Download  and unzip [BERT pretrained for passage reranking](https://drive.google.com/file/d/1tEHjpsAgvTVIqFySiSVl4DKaYVx8ltS5/view?usp=sharing) in `/models/`. You should now have the following files: 
    * `/models/BERT_for_passage_ranking/config.json`
    * `/models/BERT_for_passage_ranking/pytorch_model.bin`

The model is taken from [link](https://github.com/nyu-dl/dl4marco-bert/tree/a75f26a3342a38f146fc9f0958bf458be7a68e15) which adapted BERT for passage ranking. The model is converted to PyTorch using Hugging Face's [transformers](https://huggingface.co/docs/transformers/index) library. 

## Inference

* Go to `src/` folder and run `main.py` with the desired question. The algorithm will print his best guess for the context:

```bash
cd src
python3 main.py --question "In what country is Normandy located?"
```

## Future work 

I didn't have the time to finish all I planned, here is what is missing:

* Fasten the computation by using batches and doing inference using torch tensor (see `src/eval.py`)
* Get feedback on the actual rank of the correct context
* Implement [**MRR**](https://machinelearning.wtf/terms/mean-reciprocal-rank-mrr/) evaluation metric on dataset
* Fine-tuning the model
* Add new arguments to the parser to handle test dataset
