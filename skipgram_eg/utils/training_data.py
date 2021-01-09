import re
from typing import List, Dict, Tuple
import numpy as np


def tokenize(text: str) -> List[str]:
    """
    obtains tokens with a least 1 alphabet.
    * note that tokenisation process is largely context dependent - this is just a simple example.
    * you might want to include numbers in certain cases (e.g. 1984, Iphone 12, >_< (what if you want emoji's?)
    * customize your tokenisation pipeline: use spaCy!
    :param text: untokenized text
    :return: list of tokens
    """
    # this is rather unwieldy way of doing it..
    # pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    # return pattern.findall(text.lower())
    # this is how I'd do it:
    return [
        token
        for token in text.split(" ")
        # filter out
        if re.match(r'.*[a-zA-Z]+.*', token)
    ]


def mapping(tokens: List[str]) -> Tuple[dict, dict]:
    """

    :param tokens:
    :return:
    """
    # we maintain both.
    word_to_id = dict()
    id_to_word = dict()
    # set(tokens) creates a "vocabulary"
    for idx, token in enumerate(set(tokens)):
        word_to_id[token] = idx
        id_to_word[idx] = token

    return word_to_id, id_to_word


def generate_training_data(tokens: List[str],
                           word_to_id: Dict[str, int],
                           window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param tokens:
    :param word_to_id:
    :param window_size:
    :return:
    e.g.
    for tokens = [I, love, you]
    X = [   I,   I, love, love,  you,  you]
    Y = [love, you,    I,  you,    I, love]
    """
    N = len(tokens)
    X, Y = [], []

    # indices for all tokens
    for i in range(N):
        # TODO: what does this line do?
        # I think... it create the indices for the window?
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])

    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)

    return X, Y
