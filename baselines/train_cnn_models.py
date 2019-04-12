"""
Trains CNN models on variable length embeddings.
"""
import argparse
import os


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-prefix', required=True, help="prefix for all datasets we wish to train on")
    parser.add_argument('-num_epochs', required=True, type=int, help="number of epochs to train for")
    parser.add_argument('-ns', required=True, type=int, help="max num. samples of expansions to take per abbreviation")

    opt = parser.parse_args()

    # look for all pickle files
    for file in os.listdir("."):
        if file.endswith(".pickle") and file.startswith(opt.prefix):
            # build command
            command = "python train_cnn.py"
            command += " -dataset=%s" % file
            command += " -num_epochs=%d" % opt.num_epochs
            command += " -ns=%s" % opt.ns
            print(command)
            os.system(command)


if __name__ == "__main__":
    main()
