
import torch
import torch.nn.functional as F


class SkipGram(torch.nn.Module):

    def __init__(self, embed_size: int, vocab_size: int, learn_rate: float):
        super(SkipGram, self).__init__()
        self.learn_rate = learn_rate
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.w_mat_to_hidden = torch.autograd.Variable(torch.randn(embed_size, vocab_size).float(),
                                                       requires_grad=True)  # one_hot -> embedding
        self.w_mat_to_out = torch.autograd.Variable(torch.randn(vocab_size, embed_size).float(),
                                                       requires_grad=True)  # embedding -> out

    def forward(self, word_one_hot: torch.Tensor) -> torch.Tensor:
        """
        :param word_one_hot: one-hot vector representation of the word
        :return: a probability distribution over all the words in the vocab.
        """
        embedding = self.as_embedding(word_one_hot)
        out = torch.matmul(self.w_mat_to_out, embedding)   # (V, N) * (N, 1) -> (V, 1) (linear transformation)
        prob_dist = F.log_softmax(out, dim=0)  # (1, V) -> (1, V) (non-linear transformation)
        return prob_dist

    def as_embedding(self, word_one_hot: torch.Tensor) -> torch.Tensor:
        """
        If you just want to get the embedding for the word, just use this output.
        :param word_one_hot: one-hot vector representation of the word
        :return: the embedding representation of
        """
        # (N, V) * (V, 1) *  -> (N, 1) (linear transformation)
        embedding = torch.matmul(self.w_mat_to_hidden, word_one_hot)
        return embedding

    def optimise(self):
        self.w_mat_to_hidden.data -= self.learn_rate * self.w_mat_to_out.grad.data
        self.w_mat_to_out.data -= self.learn_rate * self.w_mat_to_out.grad.data

    def clear_grads(self):
        self.w_mat_to_hidden.grad.data.zero_()
        self.w_mat_to_out.grad.data.zero_()
