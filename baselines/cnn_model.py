"""
Defines a CNN Model class.
Adapted from:
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AttrDict(dict):
    """
    Attribute dictionary.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class IBMCNN(nn.Module):
    """
    CNN Model from IBM paper.
    """

    def __init__(self, params):
        """
        params:             hyperparameters / settings of the model
        train_embeddings:   train new embeddings?
        num_embeddings:     number of embeddings (i.e. # of tokens)
        embedding dim:      word embedding dimension        (200)
        positional_embedding dim (50) (optional)
        POS embedding dim (20) (optional)
        section embedding size (10)
        kernel_sizes:       size of window over embeddings  (2, 3, 4)
        num_filters:        number of filters               (100)
        output_dim:         number of expansions for one abbreviation
        dropout:            dropout probability             (0.5)
        """
        super(IBMCNN, self).__init__()

        self.params = params

        if params.train_embeddings:
            self.embedding = nn.Embedding(num_embeddings=params.num_embeddings,
                                          embedding_dim=params.embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=params.num_filters,
                      kernel_size=(fs, params.embedding_dim))
            for fs in params.kernel_sizes
        ])

        self.dropout = nn.Dropout(params.dropout)

        self.fcn = nn.Linear(in_features=len(params.kernel_sizes) * params.num_filters,
                             out_features=params.output_dim
                             )

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the model.
        x: Tokenized sentence (to be passed into word embedding matrix)
        """
        batch_size, max_sen_len, embedding_dim = x.shape

        assert x.shape == (batch_size, max_sen_len, self.params.embedding_dim)

        # shape (batch_size, sen_len, embedding_dim) -> (batch_size, 1, sentence_len, embedding_dim)
        # so that each example has 1 "input channel" for conv layers
        x = x.unsqueeze(1)
        assert x.shape == (batch_size, 1, max_sen_len, self.params.embedding_dim)

        # self.conved[i].shape = (batch_size, num_filters, sent_len - kernel_sizes[i])
        self.conved = [F.relu(conv(x.float())).squeeze(3) for conv in self.convs]
        assert all([self.conved[i].shape ==
                    (batch_size, self.params.num_filters, max_sen_len + 1 - self.params.kernel_sizes[i])
                    for i in range(len(self.params.kernel_sizes))])

        # self.pooled[i].shape = (batch_size, num_filters)
        self.pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in self.conved]
        assert all([self.pooled[i].shape == (batch_size, self.params.num_filters)
                    for i in range(len(self.params.kernel_sizes))])

        # self.cat.shape = (batch_size, num_filters * len(filter_sizes))
        self.cat = self.dropout(torch.cat(self.pooled, dim=1))
        assert self.cat.shape == (batch_size, self.params.num_filters * len(self.params.kernel_sizes))

        self.fced = self.fcn(self.cat)
        assert self.fced.shape == (batch_size, self.params.output_dim)

        self.out_final = self.softmax(self.fced)
        assert self.out_final.shape == (batch_size, self.params.output_dim)

        return self.out_final


def train(model, data, hyperparams):
    """
    Train the model.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # unpack data
    x_train, y_train = data["train_src"], data["train_tgt"]
    x_val, y_val = data["val_src"], data["val_tgt"]
    x_test, y_test = data["test_src"], data["test_tgt"]

    # other useful params
    x_train_len, y_train_len = len(x_train), len(y_train)
    x_val_len, y_val_len = len(x_val), len(y_val)
    x_test_len, y_test_len = len(x_test), len(y_test)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    train_losses = [0 for _ in range(hyperparams.num_epochs)]
    train_accs = [0 for _ in range(hyperparams.num_epochs)]
    val_losses = [0 for _ in range(hyperparams.num_epochs)]
    val_accs = [0 for _ in range(hyperparams.num_epochs)]

    for epoch in range(hyperparams.num_epochs):

        # train mode; in particular turn dropout on
        model.train()

        num_batches = int(x_train_len / hyperparams.batch_size)
        if x_train_len % hyperparams.batch_size != 0:
            num_batches += 1

        current_index = 0  # index in train set
        for _ in range(num_batches):
            # get batch
            current_x = x_train[current_index:current_index + min(x_train_len - current_index, hyperparams.batch_size)]
            current_y = y_train[current_index:current_index + min(y_train_len - current_index, hyperparams.batch_size)]
            current_index += hyperparams.batch_size

            optimizer.zero_grad()

            # pad batches
            current_x = nn.utils.rnn.pad_sequence(current_x, batch_first=True)
            predictions = model(current_x).squeeze(1)

            loss = criterion(predictions, current_y)
            correct = num_correct(predictions, current_y)

            loss.backward()
            optimizer.step()

            train_losses[epoch] += loss.item()
            train_accs[epoch] += correct.item()

        train_losses[epoch] /= num_batches
        train_accs[epoch] /= x_train_len

        # evaluate mode; in particular turn dropout off
        model.eval()

        with torch.no_grad():

            num_batches = int(x_val_len / hyperparams.batch_size)
            if x_val_len % hyperparams.batch_size != 0:
                num_batches += 1

            current_index = 0  # index in val set
            for _ in range(num_batches):
                # get batch
                current_x = x_val[
                            current_index:current_index + min(x_val_len - current_index, hyperparams.batch_size)]
                current_y = y_val[
                            current_index:current_index + min(y_val_len - current_index, hyperparams.batch_size)]
                current_index += hyperparams.batch_size

                # pad batches
                current_x = nn.utils.rnn.pad_sequence(current_x, batch_first=True)
                predictions = model(current_x).squeeze(1)

                loss = criterion(predictions, current_y)
                correct = num_correct(predictions, current_y)

                val_losses[epoch] += loss.item()
                val_accs[epoch] += correct.item()

            val_losses[epoch] /= num_batches
            val_accs[epoch] /= x_val_len

            if epoch % 5 == 0:
                print("Epoch: %d, Train Loss: %.4f, Val Loss: %.4f" % (epoch, train_losses[epoch], val_losses[epoch]))

    test_acc = 0

    # evaluate mode; in particular turn dropout off
    model.eval()

    with torch.no_grad():

        num_batches = int(x_test_len / hyperparams.batch_size)
        if x_test_len % hyperparams.batch_size != 0:
            num_batches += 1

        current_index = 0  # index in test set
        for _ in range(num_batches):
            # get batch
            current_x = x_test[
                        current_index:current_index + min(x_test_len - current_index, hyperparams.batch_size)]
            current_y = y_test[
                        current_index:current_index + min(y_test_len - current_index, hyperparams.batch_size)]
            current_index += hyperparams.batch_size

            # pad batches
            current_x = nn.utils.rnn.pad_sequence(current_x, batch_first=True)
            predictions = model(current_x).squeeze(1)

            correct = num_correct(predictions, current_y)

            test_acc += correct.item()

        test_acc /= x_test_len

    train_acc = 0

    # evaluate mode; in particular turn dropout off
    model.eval()

    with torch.no_grad():

        num_batches = int(x_train_len / hyperparams.batch_size)
        if x_train_len % hyperparams.batch_size != 0:
            num_batches += 1

        current_index = 0  # index in test set
        for _ in range(num_batches):
            # get batch
            current_x = x_train[
                        current_index:current_index + min(x_train_len - current_index, hyperparams.batch_size)]
            current_y = y_train[
                        current_index:current_index + min(y_train_len - current_index, hyperparams.batch_size)]
            current_index += hyperparams.batch_size

            # pad batches
            current_x = nn.utils.rnn.pad_sequence(current_x, batch_first=True)
            predictions = model(current_x).squeeze(1)

            correct = num_correct(predictions, current_y)

            train_acc += correct.item()

        train_acc /= x_train_len

    out = {
        "train_acc": train_acc,
        "train_corr": train_acc * x_train_len,
        "train_tot": x_train_len,
        "val_acc": val_accs[-1],
        "val_corr": val_accs[-1] * x_val_len,
        "val_tot": x_val_len,
        "test_acc": test_acc,
        "test_corr": test_acc * x_test_len,
        "test_tot": x_test_len,
    }
    return out


def num_correct(preds, y):
    """
    Returns number of correct predictions on a batch.
    """
    num_correct = (torch.argmax(preds, dim=1) == y).float().sum()
    return num_correct


if __name__ == "__main__":
    # a simple unit test to see if we can compute forward pass
    args = AttrDict()
    args_dict = {
        "train_embeddings": True,
        "num_embeddings": 2,
        "embedding_dim": 2,
        "num_filters": 100,
        "kernel_sizes": [2, 3, 4],
        "dropout": 0.5,
        "output_dim": 4
    }
    args.update(args_dict)
    model = IBMCNN(args)

    x = torch.tensor([0, 1, 1, 1, 0, 0, 1]).unsqueeze(1)
    model.forward(x)
