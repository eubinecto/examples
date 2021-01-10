from typing import List, Dict
import numpy as np


class Vocab:
    def __init__(self, vocab_list: List[str],
                 word2id: Dict[str, int],
                 id2word: Dict[int, str]):
        self.vocab_list = vocab_list
        self.word2id = word2id
        self.id2word = id2word

    def __len__(self):
        return len(self.vocab_list)


def tokenise_corpus(corpus: List[str]) -> List[List[str]]:
    return [
        sent.split()
        for sent in corpus
    ]


def build_vocab(tokenised_corpus) -> Vocab:
    vocab_list = list()
    for sent in tokenised_corpus:
        for token in sent:
            if token not in vocab_list:
                vocab_list.append(token)

    word2id = {
        word: idx
        for (idx, word) in enumerate(vocab_list)
    }
    id2word = {
        idx: word
        for (idx, word) in enumerate(vocab_list)
    }
    return Vocab(vocab_list, word2id, id2word)


def build_id_pairs(window_size: int, vocab: Vocab,
                   tokenised_corpus: List[List[str]]) -> np.ndarray:
    id_pairs = []
    # for each sentence
    for sentence in tokenised_corpus:
        indices = [vocab.word2id[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                id_pairs.append((indices[center_word_pos], context_word_idx))
    return np.array(id_pairs)
