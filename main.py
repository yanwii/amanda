import torch
from data_manager import DataManager
from torch.utils.data import DataLoader
from model import AmandaModel

class Amanda(object):

    def __init__(self):
        self.max_epoch = 50

    def train(self):
        data_manager = DataManager()
        passage_vocab_size = len(data_manager.passage_vocab.values())
        question_vocab_size = len(data_manager.passage_vocab.values())
        data_loader = DataLoader(
            data_manager, batch_size=1, shuffle=True, num_workers=4
        )
        model = AmandaModel(
            passage_vocab_size=passage_vocab_size,
            question_vocab_size=question_vocab_size
            
        )
        optimizer = torch.optim.Adamax(model.parameters())
        crt = torch.nn.CrossEntropyLoss()
        for epoch in range(self.max_epoch):
            print("epcoh {}".format(epoch))
            
            for index, batch in enumerate(data_loader):
                model.zero_grad()

                passage, question, spans = batch
                Pra = model(passage, question)
                loss = 0.
                for b_index in range(passage.shape[0]):
                    loss += crt(Pra[b_index], spans[b_index])
                print("loss ", loss)
                loss.backward()
                optimizer.step()
        pred = model(passage, question)
            
if __name__ == "__main__":
    amanda = Amanda()
    amanda.train()

