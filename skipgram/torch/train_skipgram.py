import torch
import numpy as np
import torch.nn.functional as F
from skipgram.torch.model import SkipGram
from skipgram.torch.preprocess import tokenise_corpus, build_vocab, build_id_pairs

# nano corpus to use
CORPUS = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

# hyper parameters
NUM_EPOCHS: int = 100
LR: float = 0.001
WINDOW_SIZE: int = 2
EMBED_SIZE: int = 10


def to_one_hot(word_id: int, vocab_size: int) -> torch.Tensor:
    x = torch.zeros(vocab_size).float()
    x[word_id] = 1.0
    return x


def main():
    global CORPUS, EMBED_SIZE, LR

    tokenised_corpus = tokenise_corpus(CORPUS)
    vocab = build_vocab(tokenised_corpus)
    id_pairs = build_id_pairs(WINDOW_SIZE, vocab, tokenised_corpus)
    vocab_size = len(vocab)

    skip_gram = SkipGram(embed_size=EMBED_SIZE, learn_rate=LR, vocab_size=vocab_size)
    
    for epo in range(NUM_EPOCHS):
        loss_val = 0
        for centre_id, context_id in id_pairs:
            centre_one_hot = to_one_hot(centre_id, vocab_size)
            # F.n11_loss creates one-hot representation by itself. That's quirky.
            y_true = torch.autograd.Variable(torch.from_numpy(np.array([context_id])).long())
            log_softmax = skip_gram.forward(word_one_hot=centre_one_hot)
            loss = F.nll_loss(log_softmax.view(1, -1), y_true)
            loss_val += loss.data.item()
            loss.backward()
            skip_gram.optimise()
            skip_gram.clear_grads()
        if epo % 10 == 0:
            print(f'Loss at epo {epo}: {loss_val / len(id_pairs)}')


if __name__ == '__main__':
    main()
