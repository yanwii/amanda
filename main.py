import torch
from data_manager import DataManager
from torch.utils.data import DataLoader
from model import AmandaModel

class Amanda(object):

    def __init__(self):
        self.max_epoch = 200

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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        
        for epoch in range(self.max_epoch):
            print("epcoh {}".format(epoch))
            
            for index, batch in enumerate(data_loader):
                model.zero_grad()

                passage, question = batch
                loss = model(passage, question)
                print("loss ", loss)
                loss.backward()
                optimizer.step()

            
if __name__ == "__main__":
    amanda = Amanda()
    amanda.train()

