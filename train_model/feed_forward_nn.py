import argparse
import numpy as np
import train_nn_relu
import pickle
from localglobalembed import AbbrRep
import load_data
import os
from abbrrep_class import AbbrRep

src_dir = "/Users/Marta/80k_abbreviations/allacronyms"
o = open(os.path.join(src_dir, "allacronyms_meta2name_20190318.pickle"), 'rb')
meta2name = pickle.load(o)
o.close()

def train_model(mimic_train, mimic_val, casi_test):
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
    model = train_nn_relu.LocalModel(n_classes, columns)
    loss, accuracy, casi_accuracy = model.train_nn(train_src, train_tgt, val_src, val_tgt, test_src, test_tgt)

    return loss, accuracy, casi_accuracy

def dataset_type(data, abbr, limit):
    casi_test = load_data.casi_process(data)
    mimic_train, mimic_val = load_data.reverse_sub(data, key="mimic_rs", abbr=abbr, cap=500, limit=limit)
    loss, accuracy, casi_accuracy = train_model(mimic_train, mimic_val, casi_test)
    print("RS: loss=" + str(loss) + " val_accuracy=" + str(accuracy) + " casi_accuracy=" + str(casi_accuracy))

    mimic_train, mimic_val = load_data.unlabelled(data, key="mimic_abbr")
    loss2, accuracy2, casi_accuracy2 = train_model(mimic_train, mimic_val, casi_test)
    print("unlabelled: loss=" + str(loss2) + " val_accuracy=" + str(accuracy2) + " casi_accuracy=" + str(casi_accuracy2))

    #mimic_train, mimic_val = load_data.reverse_sub_unlabelled(data, limit=limit, abbr=abbr)
    #loss3, accuracy3, casi_accuracy3 = train_model(mimic_train, mimic_val, casi_test)
    #print("RS + unlabelled: loss=" + str(loss3) + " val_accuracy=" + str(accuracy3) + " casi_accuracy=" + str(casi_accuracy3))


    X = load_data.reverse_sub_test(data)
    loss3, accuracy3, casi_accuracy3 = train_model(mimic_train, mimic_val, X)
    print("unlabelled on MIMIC RS: loss=" + str(loss3) +  " mimic_rs_accuracy=" + str(
        casi_accuracy3))

    return loss, accuracy, casi_accuracy, loss2, accuracy2, casi_accuracy2, loss3, accuracy3, casi_accuracy3


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', required=True, help="pickle file which contains one AbbrRep obecjt per line")
    parser.add_argument('-limit', action="store_true")

    opt = parser.parse_args()

    accuracy_rs = 0
    accuracy_rs_casi = 0
    accuracy_unlabelled = 0
    accuracy_unlabelled_casi = 0
    accuracy_rs_unlabelled = 0
    accuracy_rs_unlabelled_casi = 0
    #src_dir = "/Volumes/terminator/hpf/abbr_dataset_mimic_casi_20190319_w2_ns500/abbr_dataset_mimic_casi_pickle_20190319/"
    src_dir = "/Users/Marta/80k_abbreviations/abbr_dataset_mimic_casi_LABELLED_CASI_w5_ns500_g_20190402/"
    #src_dir += opt.src + '/'
    counter = 0
    print("SOURCE DIR:")
    print(src_dir)
    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            if ".pickle" in file and "abmimic" not in file:

                print(file)
                pickle_in = open(os.path.join(subdir,file), 'rb')
                data = pickle.load(pickle_in)
                abbr = file.split("_")[0]

                loss, accuracy, casi_accuracy, loss2, accuracy2, casi_accuracy2, \
                loss3, accuracy3, casi_accuracy3 = dataset_type(data, abbr=abbr, limit=opt.limit)
                accuracy_rs += accuracy
                accuracy_rs_casi += casi_accuracy
                accuracy_unlabelled += accuracy2
                accuracy_unlabelled_casi += casi_accuracy2
                accuracy_rs_unlabelled += accuracy3
                accuracy_rs_unlabelled_casi += casi_accuracy3
                counter += 1



    print(str(counter) + " files total")
    accuracy_rs/=counter
    accuracy_rs_casi/=counter
    accuracy_unlabelled/=counter
    accuracy_unlabelled_casi/=counter
    accuracy_rs_unlabelled /= counter
    accuracy_rs_unlabelled_casi /= counter

    print("GLOBAL STATs:")
    print("RS: val_accuracy=" + str(accuracy_rs) + " casi_accuracy=" + str(accuracy_rs_casi))
    print("unlabelled: val_accuracy=" + str(accuracy_unlabelled) + " casi_accuracy=" + str(accuracy_unlabelled_casi))
    print("unlabelled on MIMIC RS:  mimic_rs_accuracy=" + str(accuracy_rs_unlabelled_casi))

if __name__ == "__main__":
   main()
