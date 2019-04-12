"""
Train a CNN on word embeddings.
"""
import argparse
import pickle

import torch

import load_data
import numpy as np

import cnn_model


def train_model(mimic_train, mimic_val, casi_test, opt):
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

    val_src = []
    val_tgt = []
    for i in mimic_val:
        val_src.append(torch.tensor(i.embedding))
        val_tgt.append(np.argmax(i.onehot))

    test_src = []
    test_tgt = []
    for i in casi_test:
        test_src.append(torch.tensor(i.embedding))
        test_tgt.append(np.argmax(i.onehot))

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
        "num_epochs": opt.num_epochs,
        "batch_size": 64,
    }
    hyperparams.update(hyperparams_dict)

    out = cnn_model.train(model, data, hyperparams)
    return out


def dataset_type(data, opt):
    """
    Trains models on different datasets.
    """
    # load datasets
    casi_test = load_data.casi_process(data)
    mimic_train, mimic_val = load_data.reverse_sub(data, opt)

    # statistics on class imbalances
    def count_labels(dataset):
        """
        Create histogram of labels for dataset.
        """
        labels = np.zeros(dataset[0].onehot.shape)
        for x in dataset:
            labels += x.onehot
        return labels / len(dataset)

    mimic_train_labels = list(count_labels(mimic_train))
    mimic_val_labels = list(count_labels(mimic_val))
    casi_test_labels = list(count_labels(casi_test))

    # format for printing
    f_mimic_train_labels = ["%.2f" % percent for percent in mimic_train_labels]
    f_mimic_val_labels = ["%.2f" % percent for percent in mimic_val_labels]
    f_casi_test_labels = ["%.2f" % percent for percent in casi_test_labels]

    out = train_model(mimic_train, mimic_val, casi_test, opt)
    print("Histogram of MIMIC train labels:\t", f_mimic_train_labels)
    print("Histogram of MIMIC val labels:\t\t", f_mimic_val_labels)
    print("Histogram of CASI test labels:\t\t", f_casi_test_labels)
    print("Final MIMIC Train Accuracy:\t\t%.4f (%d of %d)" % (out["train_acc"], out["train_corr"], out["train_tot"]))
    print("Final MIMIC Val Accuracy:\t\t%.4f (%d of %d)" % (out["val_acc"], out["val_corr"], out["val_tot"]))
    print("CASI Accuracy:\t\t\t\t%.4f (%d of %d)" % (out["test_acc"], out["test_corr"], out["test_tot"]))


def main():
    """
    Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help="pickle file which contains one AbbrRep object per line")
    parser.add_argument('-num_epochs', required=True, type=int, help="number of epochs to train for")
    parser.add_argument('-ns', required=True, type=int, help="max num. samples of expansions to take per abbreviation")

    opt = parser.parse_args()

    pickle_in = open(opt.dataset, 'rb')
    data = pickle.load(pickle_in)

    print("Dataset: %s" % opt.dataset)

    dataset_type(data, opt)


if __name__ == "__main__":
    main()
