"""
Defines FCN model class.
"""

import argparse
import numpy as np
from fcn_model import FCN
import pickle
from localglobalembed import AbbrRep
import load_data
from abbrrep_class import AbbrRep


def train_model(mimic_train, mimic_val, casi_test, opt):
    """
    Train the model.
    """
    n_classes = len(mimic_train[0].onehot)
    columns = 200

    train_src = []
    train_tgt = []
    for i in mimic_train:
        train_src.append(i.embedding[0])
        train_tgt.append(i.onehot)
    train_tgt = np.array(train_tgt).reshape(-1, n_classes)

    val_src = []
    val_tgt = []
    for i in mimic_val:
        val_src.append(i.embedding[0])
        val_tgt.append(i.onehot)
    val_tgt = np.array(val_tgt).reshape(-1, n_classes)

    test_src = []
    test_tgt = []
    for i in casi_test:
        test_src.append(i.embedding[0])
        test_tgt.append(i.onehot)
    test_tgt = np.array(test_tgt).reshape(-1, n_classes)

    data = {
        "x_train": train_src,
        "y_train": train_tgt,
        "x_val": val_src,
        "y_val": val_tgt,
        "x_test": test_src,
        "y_test": test_tgt,
    }

    model = FCN(n_classes, columns)
    out = model.train_nn(data, opt)

    return out


def dataset_type(data, opt):
    """
    Trains model on different datasets.
    """
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
    Main function.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help="pickle file which contains one AbbrRep obecjt per line")
    parser.add_argument('-num_epochs', required=True, type=int, help="number of epochs to train for")
    parser.add_argument('-ns', required=True, type=int, help="max num. samples of expansions to take per abbreviation")

    opt = parser.parse_args()

    pickle_in = open(opt.dataset, 'rb')
    data = pickle.load(pickle_in)

    print("Dataset: %s" % opt.dataset)

    dataset_type(data, opt)


if __name__ == "__main__":
    main()
