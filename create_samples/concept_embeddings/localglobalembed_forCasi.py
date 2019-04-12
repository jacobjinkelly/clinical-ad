import ast
import fastText
import numpy as np
from sklearn.utils import shuffle
import pickle
import argparse
from abbrrep_class import AbbrRep
import os
import sys
import datetime

FASTTEXT_MODEL = None
IDF_WEIGHTS = None
hash_word_vectors = {}

date = datetime.datetime.today().strftime('%Y%m%d')
def get_casi_abbrs():
    casi_files = []
    #root_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/casi_sentences_reformatted_20190319/"
    root_dir = "/Users/Marta/80k_abbreviations/create_samples/casi_sentences_reformatted_20190319/"
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if ".txt" in file:
                casi_files.append(file)
    return casi_files

def load_models():
    global FASTTEXT_MODEL, IDF_WEIGHTS
    src_dir = "/hpf/projects/brudno/marta/mimic_rs_collection"
    src_dir = "/Users/Marta/80k_abbreviations/create_samples/"
    FASTTEXT_MODEL = fastText.load_model(os.path.join(src_dir, "word_embeddings.bin"))
    IDF_WEIGHTS = read_word_weightings(os.path.join(src_dir, "word_weighting_dict.txt"))

def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip


def load_data(abbr_samples, abbr):
    src_dir = "/Users/Marta/80k_abbreviations/allacronyms"
    #n = open(os.path.join(src_dir,"allacronyms_name2meta_20190318.pickle"), 'rb')
    n = open(os.path.join(src_dir, "allacronyms_name2meta_20190402_NEW.pickle"), 'rb')
    name2meta = pickle.load(n)
    n.close()

    #o = open(os.path.join(src_dir,"allacronyms_meta2name_20190318.pickle"), 'rb')
    o = open(os.path.join(src_dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
    meta2name = pickle.load(o)
    o.close()

    abbr_sense_dict = {}
    all_abbr_expansions = name2meta[abbr]
    for i in abbr_samples:
        content = i.split("|")
        label = content[0]
        if label not in all_abbr_expansions:
            #print(label + " does not exist")
            continue
        source = "casi_abbr"
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
        meta_id = int(name2meta[abbr][sample.label])
        y_onehot[meta_id] = 1

        sample.onehot = y_onehot

        meta_id = name2meta[abbr][sample.label]
        #print(sample.label, meta_id)
        try:
            abbr_sense_dict[source][meta_id].append(sample)
        except:
            abbr_sense_dict[source][meta_id] = [sample]

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

    #abbr_files = get_casi_abbrs()
    #job_id = int(opt.id)-1

    #print("Covering indices: " + str(job_id))
    #print("File: " + abbr_files[job_id])
    #src_dir = "/Users/Marta/80k_abbreviations/create_samples/i2b2_sentences_20190327/"
    src_dir = "/Users/Marta/80k_abbreviations/create_samples/casi_sentences_reformatted_20190319/"
    #abbr = abbr_files[job_id].split("_")[0]
    #abbrs = ['pa', 'ra', 'dc', 'dm', 'op', 'dt', 'rt', 'le', 'pcp', 'pda', 'ivf']
    abbrs = []
    if opt.abbrlist:
        abbrs_list = open(opt.abbrlist, 'r').readlines()
        for abbr in abbrs_list:
            # opt.abbr = abbr[:-1]
            abbrs.append(abbr[:-1])

    for abbr in abbrs:
        fname = abbr+"_casi_20190319.txt"
        #fname = abbr_files[job_id]
        try:
            abbr_samples = open(os.path.join(src_dir, fname), 'r').readlines()
        except:
            print("exiting script because file does not exist")
            print(fname)
            sys.exit(1)
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

        root = "w" + str(opt.window) + "_ns" + str(opt.ns) + "_"
        if opt.g:
            root += "g_"
        root += "20190408"
        dest_dir = "/Users/Marta/80k_abbreviations/abbr_dataset_mimic_casi" + root + "/"
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        mimic_dir = "/Volumes/terminator/hpf/abbr_dataset_pickle_" + root + '/'
        #mimic_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/concept_embedding_dataset_" + root
        # mimic_file = abbr + "_w5_ns500_g_LABELLED.pickle"
        mimic_file = abbr + "_" + root + ".pickle"
        p_in = open(os.path.join(mimic_dir, mimic_file), 'rb')
        mimic_dict = pickle.load(p_in)
        p_in.close()
        try:
            mimic_dict["casi_abbr"] = training_samples["casi_abbr"]
        except KeyError:
            print("NO CASI ABBRS for abbr: " + abbr)
            continue

        joined_file = abbr + "_mimic_casi_" + root + ".pickle"
        p_out = open(os.path.join(dest_dir, joined_file), 'wb')
        pickle.dump(mimic_dict, p_out)
        p_out.close()

        # casi_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/abbr_dataset_casi_pickle_20190402_w5_g/"
        # if not os.path.exists(casi_dir):
        #     os.mkdir(casi_dir)
        # outputfile = abbr + "casi_w" + str(opt.window) + "_ns" + str(opt.ns) + "_g.pickle"
        # pickle_out = open(os.path.join(casi_dir, outputfile), "wb")
        # pickle.dump(training_samples, pickle_out)
        # pickle_out.close()

        print("Done getting CASI embeddings for file: " + abbr)

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-window', required=True, help="max number of words to consider in local_context")
    parser.add_argument('-ns', default=500, help="max number of random samples PER EXPANSION to "
                                                 "create training set from")
    parser.add_argument('-all', action="store_true", help="if true, create training set with ALL samples "
                                                          "PER EXPANSION (overrides -ns)")
    parser.add_argument('-g', action="store_true", help="if true, consider global context")
    parser.add_argument('-abbrlist', help='list of abbrs to get embeddings for')



    opt = parser.parse_args()
    load_models()
    create_embed(opt)

    print("Done creating abbr embeddings!!! \U0001F388 \U0001F33B")

if __name__ == "__main__":
   main()
