import ast
import fastText
import numpy as np
from sklearn.utils import shuffle
import pickle
import argparse
from abbrrep_class import AbbrRep
import os

FASTTEXT_MODEL = None
IDF_WEIGHTS = None
hash_word_vectors = {}


NUM_ID = 100
NUM_CUIS = 0

def get_abbrs():
    m = open("cleaned_allacronyms_dict_20190318.pickle", 'rb')
    abbr_dict = pickle.load(m)
    m.close()
    return list(abbr_dict.keys())

def load_models():
    global FASTTEXT_MODEL, IDF_WEIGHTS
    src_dir = "/hpf/projects/brudno/marta/mimic_rs_collection"
    FASTTEXT_MODEL = fastText.load_model(os.path.join(src_dir, "word_embeddings.bin"))
    IDF_WEIGHTS = read_word_weightings(os.path.join(src_dir, "word_weighting_dict.txt"))

def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip


def load_data(abbr_samples, abbr):
    n = open("allacronyms_name2meta_20190318.pickle", 'rb')
    name2meta = pickle.load(n)
    n.close()

    o = open("allacronyms_meta2name_20190318.pickle", 'rb')
    meta2name = pickle.load(o)
    o.close()

    abbr_sense_dict = {}
    all_abbr_expansions = name2meta[abbr]
    for i in abbr_samples:
        content = i.split("|")
        label = content[0]
        if label == abbr:
            source = "mimic_abbr"
        else:
            source = "mimic_rs"

        if source not in abbr_sense_dict:
            abbr_sense_dict[source] = {}

        sample = AbbrRep()
        sample.label = label
        sample.features_doc = content[1].split()
        sample.features_doc_left = content[2].split()
        sample.features_left = content[3].split()
        sample.features_right = content[4].split()
        sample.source = source

        y_onehot = np.zeros(len(meta2name[abbr]) + 1)

        if sample.label == abbr:
            y_onehot[len(meta2name[abbr])] = 1
        else:
            meta_id = int(name2meta[abbr][sample.label])
            print(meta_id)
            y_onehot[meta_id] = 1
        sample.onehot = y_onehot

        if sample.label != abbr:
            meta_id = name2meta[abbr][sample.label]
            try:
                abbr_sense_dict[source][meta_id].append(sample)
            except:
                abbr_sense_dict[source][meta_id] = [sample]
        else:
            try:
                abbr_sense_dict[source][abbr].append(sample)
            except:
                abbr_sense_dict[source][abbr] = [sample]

    return abbr_sense_dict


def get_local_context(opt, x):

    features_left = x.features_left
    features_right = x.features_right
    total_number_embeds = 0
    local_context = np.zeros((1, 100))

    for j in range(max(0, len(features_left) - int(opt.window)), len(features_left)):
        local_context = np.add(local_context, FASTTEXT_MODEL.get_word_vector(features_left[j]))
        total_number_embeds += 1

    len_window = min(len(features_right), int(opt.window))
    for k in range(0, len_window):
        z = features_right[k]
        try:
            word_vector = hash_word_vectors[z]
        except KeyError:
            hash_word_vectors[z] = FASTTEXT_MODEL.get_word_vector(z)
            word_vector = hash_word_vectors[z]
        local_context = np.add(local_context, word_vector)
        total_number_embeds += 1

    if (total_number_embeds > 0):
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

    if (total_weighting > 0):
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
    abbr_files = get_abbrs()
    global NUM_CUIS
    NUM_CUIS = len(abbr_files)

    job_id = int(opt.id)
    chunk = int(NUM_CUIS // NUM_ID)
    start = int(chunk * (job_id - 1))
    end = int(start + chunk)
    if job_id == NUM_ID:
        end = NUM_CUIS

    print("Covering indices: " + str(start) + " to " + str(end))
    src_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/abbr_dataset_20190318/"

    for j in range(start, end):
        abbr = abbr_files[j]
        try:
            fname = abbr + ".txt"
            abbr_samples = open(os.path.join(src_dir, fname), 'r').readlines()
        except:
            continue
        abbr_sense_dict = load_data(abbr_samples, abbr)

        n_sne = int(opt.ns)

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
                    training_samples[key][subkey].append(abbr_sense_dict[key][subkey][i])
                    x = np.zeros((1, columns))
                    x[0][0:100] = get_local_context(opt, training_samples[key][subkey][i])
                    if opt.g:
                        x[0][100:200] = get_global_context(training_samples[key][subkey][i])
                    training_samples[key][subkey][i].embedding = x


        dest_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/abbr_dataset_pickle_w5_20190319/"
        outputfile = abbr + "_w" + str(opt.window) + "_ns" + str(opt.ns) + ".pickle"
        pickle_out = open(os.path.join(dest_dir, outputfile), "wb")
        pickle.dump(training_samples, pickle_out)
        pickle_out.close()

    print("Done writing CUI embeddings: " + str(abbr_files[start]) + " to " + str(abbr_files[end - 1]))

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of cui files to generate embeddings "
                                                   "from; e.g.: '1'")
    parser.add_argument('-window', required=True, help="max number of words to consider in local_context")
    parser.add_argument('-ns', default=500, help="max number of random samples PER EXPANSION to "
                                                 "create training set from")
    parser.add_argument('-all', action="store_true", help="if true, create training set with ALL samples "
                                                          "PER EXPANSION (overrides -ns)")
    parser.add_argument('-g', action="store_true", help="if true, consider global context")

    opt = parser.parse_args()
    load_models()
    create_embed(opt)

    print("Done creating abbr embeddings!!! \U0001F388 \U0001F33B")

if __name__ == "__main__":
   main()
