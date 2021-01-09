from skipgram_numpy_eg.utils.backward import backward_propagation, update_parameters
from skipgram_numpy_eg.utils.cost_calculation import cross_entropy
from skipgram_numpy_eg.utils.forward import forward_propagation
from skipgram_numpy_eg.utils.initialisation import initialize_parameters
from skipgram_numpy_eg.utils.training_data import tokenize, mapping, generate_training_data
import numpy as np
from matplotlib import pyplot as plt

# sample document to train skip-gram on
DOC = "After the deduction of the costs of investing, " \
      "beating the stock market is a loser's game."


def skipgram_model_training(X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None,
                            print_cost=True, plot_cost=True):
    """
    X: Input word indices. shape: (1, m)
    Y: One-hot encodeing of output word indices. shape: (vocab_size, m)
    vocab_size: vocabulary size of your corpus or training data
    emb_size: word embedding size. How many dimensions to represent each vocabulary
    learning_rate: alaph in the weight update formula
    epochs: how many epochs to train the model
    batch_size: size of mini batch
    parameters: pre-trained or pre-initialized parameters
    print_cost: whether or not to print costs during the training process
    """
    costs = []
    m = X.shape[1]

    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size)

    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0, m, batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds:
            X_batch = X[:, i:i + batch_size]
            Y_batch = Y[:, i:i + batch_size]

            softmax_out, caches = forward_propagation(X_batch, parameters)
            gradients = backward_propagation(Y_batch, softmax_out, caches)
            update_parameters(parameters, caches, gradients, learning_rate)
            cost = cross_entropy(softmax_out, Y_batch)
            epoch_cost += np.squeeze(cost)

        costs.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98

    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
    return parameters


def main():
    global DOC

    tokens = tokenize(DOC)
    word_to_id, id_to_word = mapping(tokens)
    X, Y = generate_training_data(tokens, word_to_id, window_size=3)
    vocab_size = len(id_to_word)
    m = Y.shape[1]

    # turn Y into one hot encoding
    Y_one_hot = np.zeros((vocab_size, m))
    Y_one_hot[Y.flatten(), np.arange(m)] = 1

    # this line executes the training
    paras = skipgram_model_training(X, Y_one_hot,
                                    vocab_size, 50, 0.05, 5000,
                                    batch_size=128, parameters=None, print_cost=True)


if __name__ == '__main__':
    main()
