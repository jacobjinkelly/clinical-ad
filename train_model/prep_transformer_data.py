import load_data
import argparse
import pickle
from localglobalembed import AbbrRep

def save_data(train, val, test, file_extension, abbr):

    f1 = open(abbr + "_" + file_extension +"_train_src.txt", 'w')
    f2 = open(abbr + "_" + file_extension + "_train_tgt.txt", 'w')
    f3 = open(abbr + "_" + file_extension +  "_train_lc.txt", 'w')
    for i in train:
        f1.write(' '.join(i.features_left) + ' '
                 + ' '.join(i.features_right) + '\n')
        f2.write(i.label + '\n')
        f3.write(str([i.features_left, i.features_right]))
    g1 = open(abbr + "_" + file_extension +  "_val_src.txt", 'w')
    g2 = open(abbr + "_" + file_extension +  "_val_tgt.txt", 'w')
    g3 = open(abbr + "_" + file_extension +  "_val_lc.txt", 'w')
    for i in val:
        g1.write(' '.join(i.features_left) + ' '
                 + ' '.join(i.features_right) + '\n')
        g2.write(i.label + '\n')
        g3.write(str([i.features_left, i.features_right]))
    h1 = open(abbr + "_casi_test_src.txt", 'w')
    h2 = open(abbr + "_casi_test_tgt.txt", 'w')
    h3 = open(abbr + "_casi_test_lc.txt", 'w')
    for i in test:
        h1.write(' '.join(i.features_left) + ' '
                 + ' '.join(i.features_right) + '\n')
        h2.write(i.label + '\n')
        h3.write(str([i.features_left, i.features_right]))
def prep_transformer_data(data,abbr):

    mimic_train, mimic_val = load_data.reverse_sub(data)
    casi_test = load_data.casi_process(data)

    file_extension = "mimic_rs"
    save_data(mimic_train, mimic_val, casi_test, file_extension,abbr)

    mimic_train, mimic_val = load_data.reverse_sub_sim(data)
    file_extension = "mimic_rs_sim"
    save_data(mimic_train, mimic_val, casi_test, file_extension,abbr)


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True)

    opt = parser.parse_args()
    abbr = opt.dataset.split("_")[0]
    pickle_in = open(opt.dataset, 'rb')
    data = pickle.load(pickle_in)
    prep_transformer_data(data,abbr)

if __name__ == "__main__":
   main()