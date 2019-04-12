"""
Trains MFS models.
"""
import argparse
import os


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-prefix', default="", help="prefix for all datasets we wish to train on")
    parser.add_argument('-ns', required=True, type=int, help="max num. samples of expansions to take per abbreviation")

    opt = parser.parse_args()

    # look for all pickle files
    for file in os.listdir("."):
        if file.endswith(".pickle") and file.startswith(opt.prefix):
            # build command
            command = "python train_mfs.py"
            command += " -dataset=%s" % file
            command += " -ns=%s" % opt.ns
            print(command)
            os.system(command)


if __name__ == "__main__":
    main()
