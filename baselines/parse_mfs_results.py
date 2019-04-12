"""
Parse results of experiments.
"""


def parse(file):
    """
    Parse one file.
    """

    def get_acc(line, num_tabs):
        """
        Given line, parses accuracy from it.
        """
        # strip away label (e.g. "Final MIMIC Train Accuracy")
        nums = line.split("\t" * num_tabs)[1]

        # get accuracy and fraction
        acc, frac = nums.split("(")[0], nums.split("(")[1]

        # get numerator and denominator of fraction
        num, denom = frac.split(" of ")[0], frac.split(" of ")[1].split(")")[0]

        return float(acc), float(num), float(denom)

    mimic_train_macro_num, mimic_train_macro_denom = 0, 0
    mimic_val_macro_num, mimic_val_macro_denom = 0, 0
    casi_macro_num, casi_macro_denom = 0, 0

    mimic_train_micro_num, mimic_train_micro_denom = 0, 0
    mimic_val_micro_num, mimic_val_micro_denom = 0, 0
    casi_micro_num, casi_micro_denom = 0, 0

    # calculate micro and macro accuracies
    with open(file, "r") as fp:
        for _, line in enumerate(fp):
            if line.startswith("Final MIMIC Train Accuracy"):
                acc, num, denom = get_acc(line, 2)
                mimic_train_macro_num += acc
                mimic_train_macro_denom += 1
                mimic_train_micro_num += num
                mimic_train_micro_denom += denom
            elif line.startswith("Final MIMIC Val Accuracy"):
                acc, num, denom = get_acc(line, 2)
                mimic_val_macro_num += acc
                mimic_val_macro_denom += 1
                mimic_val_micro_num += num
                mimic_val_micro_denom += denom
            elif line.startswith("CASI Accuracy"):
                acc, num, denom = get_acc(line, 4)
                casi_macro_num += acc
                casi_macro_denom += 1
                casi_micro_num += num
                casi_micro_denom += denom

    print("MIMIC Train Macro: %.4f" % (mimic_train_macro_num / mimic_train_macro_denom))
    print("MIMIC Train Micro: %.4f" % (mimic_train_micro_num / mimic_train_micro_denom))
    print("MIMIC Val Macro: %.4f" % (mimic_val_macro_num / mimic_val_macro_denom))
    print("MIMIC Val Micro: %.4f" % (mimic_val_micro_num / mimic_val_micro_denom))
    print("Casi Macro: %.4f" % (casi_macro_num / casi_macro_denom))
    print("Casi Micro: %.4f" % (casi_micro_num / casi_micro_denom))


def main():
    num_samples = [100, 500, 1000]
    for ns in num_samples:
        file = "mfs_ns%d.txt" % ns
        print("Parsing %s" % file)
        parse(file)


if __name__ == "__main__":
    main()
