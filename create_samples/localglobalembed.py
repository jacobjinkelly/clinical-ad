import ast
import fastText
import numpy as np
from sklearn.utils import shuffle
import pickle
import argparse

FASTTEXT_MODEL = None
IDF_WEIGHTS = None
hash_word_vectors = {}
abbr_sense_dict = {}
label2idx = {}


def load_models():
    global FASTTEXT_MODEL, IDF_WEIGHTS

    FASTTEXT_MODEL = fastText.load_model("./word_embeddings.bin")
    IDF_WEIGHTS = read_word_weightings("./word_weighting_dict.txt")

def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip

class AbbrRep:
    def __init__(self):
        self.label = ""
        self.features_left = []         # context left
        self.features_right = []        # context right
        self.features_doc = []          # entire document
        self.features_doc_left = []     # entire document to left of curr word
        self.source = ""
        self.embedding = []
        self.onehot = None

def load_data(input):
    source = "_".join(input.split("/")[-1][:-4].split("_")[1:])
    global abbr_sense_dict
    global label2idx
    if source not in abbr_sense_dict:
        abbr_sense_dict[source] = {}
    with open(input) as f:
        for line in f:
            if line[:-1] == "None":
                continue
            try:
                content = ast.literal_eval(line)
            except ValueError:
                print("STOPPED HERE")
                print(input)
                print(line)

            label = content[0]
            sample = AbbrRep()

            if label not in abbr_sense_dict[source]:
                abbr_sense_dict[source][label] = []

            if label not in label2idx:
                label2idx[label] = len(label2idx)
            sample.label = content[0]
            sample.features_doc = content[1].split()
            sample.features_doc_left = content[2].split()
            sample.features_left = content[3].split()
            sample.features_right = content[4].split()
            sample.source = source
            abbr_sense_dict[source][label].append(sample)


def get_local_context(opt, x):

    features_left = x.features_left     # words to the left
    features_right = x.features_right   # words to the right
    total_number_embeds = 0
    left_window_len = len(features_left) - max(0, len(features_left) - int(opt.window)) + 1
    right_window_len = min(len(features_right), int(opt.window))

    if opt.variable_local:
        local_context = np.zeros((left_window_len + right_window_len, 100))
    else:
        local_context = np.zeros((1, 100))

    start_ind = max(0, len(features_left) - int(opt.window))
    for j in range(start_ind, len(features_left)):
        if opt.variable_local:
            local_context[j - start_ind] = FASTTEXT_MODEL.get_word_vector(features_left[j])
        else:
            local_context = np.add(local_context, FASTTEXT_MODEL.get_word_vector(features_left[j]))
        total_number_embeds += 1

    for k in range(0, right_window_len):
        z = features_right[k]
        try:
            word_vector = hash_word_vectors[z]
        except KeyError:
            hash_word_vectors[z] = FASTTEXT_MODEL.get_word_vector(z)
            word_vector = hash_word_vectors[z]
        if opt.variable_local:
            local_context[left_window_len + k] = word_vector
        else:
            local_context = np.add(local_context, word_vector)
        total_number_embeds += 1

    if total_number_embeds > 0 and not opt.variable_local:
        local_context = local_context / total_number_embeds
    return local_context

def get_global_context(x):
    total_weighting = 0
    global_context = np.zeros(100)
    doc = x.features_doc

    for z in doc:
        try:
            current_word_weighting = IDF_WEIGHTS[z]
        except:
            current_word_weighting = 0
        try:
            word_vector = hash_word_vectors[z]
        except KeyError:
            hash_word_vectors[z] = FASTTEXT_MODEL.get_word_vector(z)
            word_vector = hash_word_vectors[z]
        global_context = np.add(global_context, (current_word_weighting * word_vector))
        total_weighting += current_word_weighting

    if total_weighting > 0:
        global_context = global_context / total_weighting

    return global_context


def shuffle_data(abbr_sense_we_dict):
    shuffled_abbr_sense_we_dict = {}
    for key in abbr_sense_we_dict:
        shuffled_abbr_sense_we_dict[key] = {}
        for subkey in abbr_sense_we_dict[key]:
            shuffled_abbr_sense_we_dict[key][subkey] = shuffle(abbr_sense_we_dict[key][subkey], random_state=42)
    return shuffled_abbr_sense_we_dict


def create_embed(opt):
    n_sne = int(opt.ns)

    if opt.mimic_rs:
        load_data(opt.mimic_rs)
    if opt.mimic_abbr:
        load_data(opt.mimic_abbr)
    if opt.mimic_rs_sim:
        load_data(opt.mimic_rs_sim)
    if opt.casi_abbr:
        load_data(opt.casi_abbr)

    for key in abbr_sense_dict:
        for subkey in abbr_sense_dict[key]:
            for item in abbr_sense_dict[key][subkey]:
                y_onehot = np.zeros(len(label2idx)+1)
                y_onehot[label2idx[item.label]] = 1
                item.onehot = y_onehot

    shuffled_abbr_sense_we_dict = shuffle_data(abbr_sense_dict)
    training_samples = {}

    columns = 100
    if opt.g:
        columns += 100

    for key in shuffled_abbr_sense_we_dict:
        training_samples[key] = {}
        for subkey in shuffled_abbr_sense_we_dict[key]:
            training_samples[key][subkey] = []
            if opt.all:
                num_samples = len(shuffled_abbr_sense_we_dict[key][subkey])
            else:
                num_samples = min(n_sne, len(shuffled_abbr_sense_we_dict[key][subkey]))
            for i in range(num_samples):
                training_samples[key][subkey].append(shuffled_abbr_sense_we_dict[key][subkey][i])
                if opt.variable_local:
                    x = get_local_context(opt, training_samples[key][subkey][i])
                    if opt.g:
                        global_context = get_global_context(training_samples[key][subkey][i])
                        x = np.concatenate((x, np.expand_dims(global_context, axis=0)), axis=0)
                    training_samples[key][subkey][i].embedding = x
                else:
                    x = np.zeros((1, columns))
                    x[0][0:100] = get_local_context(opt, training_samples[key][subkey][i])
                    if opt.g:
                        x[0][100:200] = get_global_context(training_samples[key][subkey][i])
                    training_samples[key][subkey][i].embedding = x

    training_samples["label2idx"] = label2idx

    if opt.outputfile:
        outputfile = opt.outputfile
    elif opt.mimic_abbr and opt.mimic_rs_sim and opt.casi_abbr:
        outputfile = opt.abbr + "_dataset_rs_sim_abbr_casi.pickle"
    elif opt.mimic_abbr and opt.mimic_rs_sim:
        outputfile = opt.abbr + "_dataset_rs_sim_abbr.pickle"
    elif opt.mimic_abbr:
        outputfile = opt.abbr + "_dataset_rs_abbr.pickle"
    else:
        outputfile = opt.abbr + "_dataset_rs.pickle"

    pickle_out = open(outputfile, "wb")
    pickle.dump(training_samples, pickle_out)
    pickle_out.close()

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-abbr', help="target abbr (e.g. mg)")
    parser.add_argument('-mimic_rs', required=True,
                        help=".txt file where each line has format: [expansion, global_context, "
                             "left_global_context, left_local_context, right_local_context]")
    parser.add_argument('-mimic_abbr',
                        help=".txt file where each line has format: [abbreviation, global_context,"
                             " left_global_context, left_local_context, right_local_context]")
    parser.add_argument('-mimic_rs_sim',
                        help=".txt file where each line has format: [expansion, global_context, "
                             "left_global_context, left_local_context, right_local_context]")
    parser.add_argument('-casi_abbr',
                        help=".txt file where each line has format: [abbreviation, global_context, "
                             "left_global_context, left_local_context, right_local_context]")
    parser.add_argument('-window', required=True, help="max number of words to consider in local_context")
    parser.add_argument('-ns', default=500, help="max number of random samples PER EXPANSION to "
                                                 "create training set from")
    parser.add_argument('-all', action="store_true", help="if true, create training set with ALL samples "
                                                          "PER EXPANSION (overrides -ns)")
    parser.add_argument('-g', action="store_true", help="if true, consider global context")
    parser.add_argument('-variable_local', action="store_true", help="if true,  have variable length local context")
    parser.add_argument('-outputfile', help="name of pickle file to store embeddings in")

    opt = parser.parse_args()
    load_models()
    create_embed(opt)


if __name__ == "__main__":
   main()
