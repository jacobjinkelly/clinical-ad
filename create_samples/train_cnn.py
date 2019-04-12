"""
Train a CNN on word embeddings.
"""
import argparse
import pickle

import torch

import load_data
import numpy as np

import cnn_model

from localglobalembed import AbbrRep


def train_model(mimic_train, mimic_val, casi_test):
    """
    Train a CNN model.
    """
    n_classes = len(mimic_train[0].onehot)
    embedding_dim = 100

    train_src = []
    train_tgt = []
    for i in mimic_train:
        train_src.append(torch.tensor(i.embedding))
        train_tgt.append(np.argmax(i.onehot))
    # train_tgt = np.array(train_tgt).reshape(-1, n_classes)

    val_src = []
    val_tgt = []
    for i in mimic_val:
        val_src.append(torch.tensor(i.embedding))
        val_tgt.append(np.argmax(i.onehot))
    # val_tgt = np.array(val_tgt).reshape(-1, n_classes)

    test_src = []
    test_tgt = []
    for i in casi_test:
        test_src.append(torch.tensor(i.embedding))
        test_tgt.append(np.argmax(i.onehot))
    # test_tgt = np.array(test_tgt).reshape(-1, n_classes)

    args = cnn_model.AttrDict()
    args_dict = {
        "train_embeddings": False,
        "embedding_dim": embedding_dim,
        "num_filters": 100,
        "kernel_sizes": [2, 3, 4],
        "dropout": 0.5,
        "output_dim": n_classes
    }
    args.update(args_dict)
    model = cnn_model.IBMCNN(args)

    data = {
        "train_src": train_src,
        "train_tgt": torch.tensor(train_tgt).long(),
        "val_src": val_src,
        "val_tgt": torch.tensor(val_tgt).long(),
        "test_src": test_src,
        "test_tgt": torch.tensor(test_tgt).long()
    }

    hyperparams = cnn_model.AttrDict()
    hyperparams_dict = {
        "num_epochs": 40,
        "batch_size": 64,
    }
    hyperparams.update(hyperparams_dict)

    train_loss, val_loss, test_acc = cnn_model.train(model, data, hyperparams)
    return train_loss, val_loss, test_acc


def dataset_type(data):
    """
    Trains models on different datasets.
    """
    casi_test = load_data.casi_process(data)

    mimic_train, mimic_val = load_data.reverse_sub(data)
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))

    mimic_train, mimic_val = load_data.reverse_sub_sim(data)
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS+3N: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))

    mimic_train, mimic_val = load_data.reverse_sub_unlabelled(data, "gda")
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS+U-gda: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))

    mimic_train, mimic_val = load_data.reverse_sub_sim_unlabelled(data, "gda")
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS+3N+U-gda: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))

    mimic_train, mimic_val = load_data.reverse_sub_unlabelled(data, "knn")
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS+U-knn: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))

    mimic_train, mimic_val = load_data.reverse_sub_sim_unlabelled(data, "knn")
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS+3N+U-knn: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))


def main():
    """
    Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help="pickle file which contains one AbbrRep object per line")

    opt = parser.parse_args()

    pickle_in = open(opt.dataset, 'rb')
    data = pickle.load(pickle_in)

    dataset_type(data)


if __name__ == "__main__":
    main()
