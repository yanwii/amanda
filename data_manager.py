import json
from itertools import chain

import nltk
import torch
from torch.utils.data.dataset import Dataset


class DataManager(Dataset):
    
    def __init__(self, batch_size=1):
        self.passage_vocab = {}
        self.question_vocab = {
            "what": 0,
            "who": 1,
            "how": 2,
            "when": 3,
            "which": 4,
            "where": 5,
            "why": 6
        }
        self.passage_vec = []
        self.question_vec = []
        self.answer_spans = []

        self.load_data()

    def load_data(self):
        # with open("data/train.json", encoding="utf-8") as fopen:
            # datas = json.load(fopen)

        with open("data/search.dev.json", encoding="utf-8") as fopen:
            for data in fopen:
                data = json.loads(data.strip())
                if not data["answer_docs"]:
                    continue
                answer_doc = data["answer_docs"][0]
                document = data["documents"][answer_doc]

                passage = document["segmented_paragraphs"]
                passage = list(chain.from_iterable(passage))
                question = data["segmented_question"]

                self.passage_vec.append(
                    self.update_vocab(passage, self.passage_vocab)
                )
                self.question_vec.append(
                    self.update_vocab(question, self.passage_vocab)
                )
                self.answer_spans.append(
                    data["answer_spans"][0]
                )

    def update_vocab(self, segments, vocab):
        # segments = nltk.word_tokenize(sentence)
        vec = []
        for segment in segments:
            if segment not in vocab:
                vocab[segment] = len(vocab.keys())
            vec.append(
                vocab[segment]
            )
        return vec

    def __getitem__(self, index):
        passage = torch.LongTensor(self.passage_vec[index])
        question = torch.LongTensor(self.question_vec[index])
        spans = torch.LongTensor(self.answer_spans[index])
        return passage, question, spans

    def __len__(self):
        return len(self.passage_vec)
