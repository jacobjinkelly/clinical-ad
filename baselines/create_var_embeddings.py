"""
Creates variable-length (local context) embeddings from fixed-length (local context) embeddings.
"""
import argparse
import os


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-prefix', required=True, help="prefix to add to generated embedding files")
    parser.add_argument('-ignore', default="", help="ignore all pickle files w/ this prefix")
    parser.add_argument('-window', required=True, help="max number of words to consider in local_context")
    parser.add_argument('-g', action="store_true", help="if true, consider global context")
    parser.add_argument('-variable_local', action="store_true", help="if true, have variable length local context")

    opt = parser.parse_args()

    # look for all pickle files
    for file in os.listdir("."):
        if file.endswith(".pickle"):
            if opt.ignore == "" or not file.startswith(opt.ignore):
                # build command
                command = "python localglobalembed.py"
                command += " -dataset=%s" % file
                command += " -outputfile=%s" % (opt.prefix + file)
                command += " -window=%s" % opt.window
                if opt.g:
                    command += " -g"
                if opt.variable_local:
                    command += " -variable_local"
                print(command)
                os.system(command)


if __name__ == "__main__":
    main()
