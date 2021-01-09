import numpy as np
# (functional) implementation of skip-gram model.


def initialize_wrd_emb(vocab_size: int, emb_size: int) -> np.ndarray:
    """
    vocab_size: int. vocabulary size of your corpus or training data
    emb_size: int. word embedding size. How many dimensions to represent each vocabulary (e.g. 100, 200, 300)
    * note: the bigger this is, the more training it takes, the better the quality of the word vectors.
    """
    # (vocab_size=ONE_HOT_SIZE, EMB_SIZE)
    # TODO: why multiply 0.01?
    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    return WRD_EMB


def initialize_dense(input_size: int, output_size: int) -> np.ndarray:
    """
    input_size: int. size of the input to the dense layer
    output_size: int. size of the output of the dense layer
    * here, the dense layer = "the projection layer."
    """
    # TODO: bias가 있는 경우 vs. 없는 경우. 선형변환이 유지되는가? 증명해보기.
    # note that there is no biases here. This is what keeps the linearity.
    W = np.random.randn(output_size, input_size) * 0.01
    return W


def initialize_parameters(vocab_size, emb_size) -> dict:
    """
    initialize all the training parameters
    """
    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)

    parameters = {'WRD_EMB': WRD_EMB, 'W': W}

    return parameters
