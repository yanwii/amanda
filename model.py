import torch
from torch import nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=100):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = 0.5

        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim // 2,
                        num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        lstm_out, hidden = self.lstm(embeddings)
        lstm_out = lstm_out.view(batch_size, -1, self.hidden_dim)
        return lstm_out, hidden

class MultiFactor(nn.Module):

    def __init__(self, factor_dim=10, hidden_dim=128):
        super(MultiFactor, self).__init__()

        self.factor_dim = factor_dim
        self.hidden_dim = hidden_dim

        self.context_matrix = nn.Parameter(
             torch.randn(self.factor_dim, self.hidden_dim, self.hidden_dim)
        )
    
    def forward(self, context_vectors):
        time_steps = context_vectors.shape[1]
        batch_size = context_vectors.shape[0]
        multi_factors = []
        for batch_index in range(batch_size):
            context_vector = context_vectors[batch_index]
            factors = []
            for index_i in range(time_steps):
                vi = context_vector[index_i, :].view(1, -1, self.hidden_dim)
                row_factors = []
                for index_j in range(time_steps):
                    vj = context_vector[index_j, :].view(1, -1, self.hidden_dim)

                    vi = vi.expand(self.factor_dim, 1, self.hidden_dim)
                    vj = vj.expand(self.factor_dim, 1, self.hidden_dim)

                    scores = torch.matmul(torch.matmul(vi, self.context_matrix), vj.transpose(2, 1)).view(self.factor_dim)
                    max_index = torch.argmax(scores.view(self.factor_dim))
                    row_factors.append(scores[max_index].tolist())
                factors.append(row_factors)
            factors = torch.tensor(factors)
            factors_softmax = torch.softmax(factors, -1)
            factors = torch.matmul(factors_softmax, context_vector)
            multi_factors.append(factors.view(1, -1, self.hidden_dim))
        multi_factors = torch.cat(multi_factors)
        return multi_factors


class FeedForward(nn.Module):
    def __init__(self, hidden_dim=128):
        super(FeedForward, self).__init__()
        self.hidden_dim = hidden_dim

        self.Wg = nn.Parameter(
            torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim)
        )
        self.bg = nn.Parameter(
            torch.randn(2 * self.hidden_dim)
        )
    
    def forward(self, multi_factors):
        batch_size = multi_factors.shape[0]

        W = self.Wg.expand(1, 2*self.hidden_dim, 2*self.hidden_dim)
        b = self.bg.expand(1, 2*self.hidden_dim)

        mul = torch.matmul(multi_factors, W) + b
        mul_sig = torch.sigmoid(mul)
        Y = multi_factors * mul_sig
        return Y

class QuestionRep(nn.Module):

    def __init__(self, hidden_dim=128):
        super(QuestionRep, self).__init__()

        self.hidden_dim = hidden_dim

        self.Wq = nn.Parameter(
            torch.randn(3 * self.hidden_dim, self.hidden_dim)
        )
        self.bq = nn.Parameter(
            torch.randn(self.hidden_dim)
        )

    def forward(self, question_context):
        batch_size = question_context.shape[0]
        Wq = self.Wq.expand(batch_size, 3 * self.hidden_dim, self.hidden_dim)
        bq = self.bq.expand(batch_size, self.hidden_dim)

        question_rep = torch.tanh(
            torch.matmul(question_context, self.Wq) + bq
        )
        return question_rep

class AmandaModel(nn.Module):

    def __init__(self, passage_vocab_size=10, question_vocab_size=10):
        super(AmandaModel, self).__init__()
        self.hidden_dim = 128
        self.batch_size = 1
        self.dropout = 0.5

        self.passage_embedd_dim = 100
        self.question_embedd_dim = 100

        self.passage_vocab_size = passage_vocab_size
        self.question_vocab_size = question_vocab_size

        self.share_embedd = nn.Embedding(self.passage_vocab_size, self.passage_embedd_dim)

        self.share_lstm = BiLSTM(
            hidden_dim=self.hidden_dim,
            embedding_dim=self.passage_embedd_dim
        ) 
        self.pass_encoder_lstm = BiLSTM(
            hidden_dim=self.hidden_dim,
            embedding_dim=self.hidden_dim * 2
        )
        self.multi_factor = MultiFactor(
            hidden_dim=self.hidden_dim
        )
        self.feed_forward = FeedForward(
            hidden_dim=self.hidden_dim
        )
        self.begin_pointer_lstm = BiLSTM(
            hidden_dim=self.hidden_dim,
            embedding_dim=self.hidden_dim * 2
        )
        self.end_pointer_lstm = BiLSTM(
            hidden_dim=self.hidden_dim,
            embedding_dim=self.hidden_dim
        )
        self.question_rep_forward = QuestionRep(
            hidden_dim=self.hidden_dim
        )
        
    def forward(self, passages, questions):
        passage_length = passages.shape[1]
        questions_length = questions.shape[1]
        self.batch_size = passages.shape[0]

        passage_embeddings = self.share_embedd(passages).view(self.batch_size, passage_length, self.passage_embedd_dim)
        question_embeddings = self.share_embedd(questions).view(self.batch_size, questions_length, self.question_embedd_dim)

        # P = [T x H]
        pass_lstm_out, pass_hidden = self.share_lstm(passage_embeddings)
        # Q = [U x H]
        ques_lstm_out, ques_hidden = self.share_lstm(question_embeddings)

        # A = PQ(T) 
        # [T x H] * [U x H](t) => [T x U]
        ques_lstm_out_t = ques_lstm_out.transpose(2, 1)
        attention_layer = torch.matmul(pass_lstm_out, ques_lstm_out_t)

        # R = row-wise softmax(A) [T x U]
        relation_layer = torch.softmax(attention_layer, -1)
        # G = RQ   
        # [T x U] * [U x H] => [T x H]
        que_vectors =  torch.matmul(relation_layer, ques_lstm_out)
        
        # S = [P|G] => [T x 2H]
        context = torch.cat([pass_lstm_out, que_vectors], -1)

        # V [T x H]
        context_vectors, context_hidden = self.pass_encoder_lstm(context)

        # M = FV
        multi_factors = self.multi_factor(context_vectors)

        # M = [M|V] => [T x 2H]
        multi_factors = torch.cat([multi_factors, context_vectors], -1)
        
        # Y => [T x 2H]
        Y = self.feed_forward(multi_factors)

        # B & E
        begin_pointer_out, begin_pointer_hidden = self.begin_pointer_lstm(Y)
        end_pointer_out, end_pointer_hidden = self.end_pointer_lstm(begin_pointer_out)


        # Question-focused Attensional Pointing
        max_index = torch.argmax(attention_layer, 1)
        maxcol_attention = torch.gather(attention_layer, 1, max_index.view(self.batch_size, 1, -1))
        maxcol_attention = maxcol_attention.view(self.batch_size, 1, -1)
        qma = torch.matmul(maxcol_attention, ques_lstm_out).view(self.batch_size, -1)

        qf = ques_lstm_out[:, :2, :].contiguous().view(self.batch_size, 2 * self.hidden_dim)
        q = torch.cat([qma, qf], 1)

        question_rep = self.question_rep_forward(q).view(self.batch_size, 1, self.hidden_dim)

        sb = torch.matmul(question_rep, begin_pointer_out.transpose(2, 1))
        se = torch.matmul(question_rep, end_pointer_out.transpose(2, 1))
        
        Prb = torch.softmax(sb, -1)
        Pre = torch.softmax(se, -1)
        Pra = Prb * Pre
        loss = - torch.sum(torch.log(Pra))
        return loss
