"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""

import argparse
import conceptEmbedModel_encoder
import numpy as np
import os
import json
import fastText
import pickle
import tensorflow as tf
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--datafiles', help="address to the datafile(s) (split by ',' if multiple")
    parser.add_argument('--sm',
                        help="address to sparse matrix")
    parser.add_argument('--fasttext', help="address to the fasttext word bin file")
    parser.add_argument('--output', help="address to the directory where the trained model will be stored")
    parser.add_argument('--flat', action="store_true")
    parser.add_argument('--cl1', type=int, help="cl1", default=1024)
    parser.add_argument('--cl2', type=int, help="cl2", default=1024)
    parser.add_argument('--lr', type=float, help="lr", default=0.002)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=256)
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=16)
    parser.add_argument('--epochs', type=int, help="epochs", default=50)
    args = parser.parse_args()

    word_model = fastText.load_model(args.fasttext)

    p_in = open(args.sm, 'rb')
    sm = pickle.load(p_in)
    p_in.close()

    data = {}
    datafiles = args.datafiles.split(",")
    for fname in datafiles:
        with open(fname, 'rb') as file_handle:
            print("file loaded.............. " + fname)
            fdata = pickle.load(file_handle)
            data.update(fdata)

    print("All files loaded! Number of samples: " + str(len(data)))

    model = conceptEmbedModel_encoder.ConceptEmbedModel(args, sm, data, word_model)
    model.init_training()

    param_dir = args.output
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    for epoch in tqdm(range(args.epochs)):
        print("Epoch :: " + str(epoch))
        model.train_epoch(verbose=True)
        if epoch > 0 and (epoch % 5) == 0:
            top1, top5 = model.check_val_set(5)
            print("Top1Acc: " + str(top1))
            print("Top5Acc: " + str(top5))
        if epoch > 0 and (epoch % 10) == 0:
            model.save_params(epoch=epoch, repdir=param_dir)

    with open(param_dir + '/config.json', 'w') as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    main()
