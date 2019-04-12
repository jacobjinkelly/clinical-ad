"""
Train a MFS classifier.
"""
import argparse
import pickle

import torch

import load_data
import numpy as np

import cnn_model


def eval_mfs(mimic_train, mimic_val, casi_test, mfs_pred):
    """
    Train a CNN model.
    """

    train_tgt = []
    for i in mimic_train:
        train_tgt.append(np.argmax(i.onehot))

    val_tgt = []
    for i in mimic_val:
        val_tgt.append(np.argmax(i.onehot))

    test_tgt = []
    for i in casi_test:
        test_tgt.append(np.argmax(i.onehot))

    train_tgt = torch.tensor(train_tgt).long()
    val_tgt = torch.tensor(val_tgt).long()
    test_tgt = torch.tensor(test_tgt).long()

    def num_correct(preds, y):
        """
        Returns number of correct predictions on a batch.
        """
        num_correct = (preds == y).float().sum()
        return num_correct

    train_num_corr = num_correct(mfs_pred, train_tgt)
    val_num_corr = num_correct(mfs_pred, val_tgt)
    test_num_corr = num_correct(mfs_pred, test_tgt)

    x_train_len = len(mimic_train)
    x_val_len = len(mimic_val)
    x_test_len = len(casi_test)

    out = {
        "train_acc": train_num_corr / x_train_len,
        "train_corr": train_num_corr,
        "train_tot": x_train_len,
        "val_acc": val_num_corr / x_val_len,
        "val_corr": val_num_corr,
        "val_tot": x_val_len,
        "test_acc": test_num_corr / x_test_len,
        "test_corr": test_num_corr,
        "test_tot": x_test_len,
    }
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

    # construct prediction of MFS model
    mfs_model = np.argmax(count_labels(mimic_train))

    mimic_train_labels = list(count_labels(mimic_train))
    mimic_val_labels = list(count_labels(mimic_val))
    casi_test_labels = list(count_labels(casi_test))

    # format for printing
    f_mimic_train_labels = ["%.2f" % percent for percent in mimic_train_labels]
    f_mimic_val_labels = ["%.2f" % percent for percent in mimic_val_labels]
    f_casi_test_labels = ["%.2f" % percent for percent in casi_test_labels]

    out = eval_mfs(mimic_train, mimic_val, casi_test, mfs_model)
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
    parser.add_argument('-ns', required=True, type=int, help="max num. samples of expansions to take per abbreviation")

    opt = parser.parse_args()

    pickle_in = open(opt.dataset, 'rb')
    data = pickle.load(pickle_in)

    print("Dataset: %s" % opt.dataset)

    dataset_type(data, opt)


if __name__ == "__main__":
    main()
