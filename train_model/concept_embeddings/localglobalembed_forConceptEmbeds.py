import ast
import fastText
import numpy as np
import pickle
import argparse
from abbrrep_class import AbbrRep
import datetime
import os
from tqdm import tqdm
import sys

date = datetime.datetime.today().strftime('%Y%m%d')
NUM_ID = 14
NUM_CUIS = 0

FASTTEXT_MODEL = None
IDF_WEIGHTS = None
hash_word_vectors = {}
abbr_sense_dict = {}


def get_files():
    files = open("cuis_in_mimic.txt", 'r').readlines()
    for i in range(len(files)):
        files[i] = files[i][:-5] + "_500.txt"
    return files


def get_specific_files(opt, abbr):
    dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/sparse_matrices_20190406"
    #dir = "/hpf/projects/brudno/marta/mimic_rs_collection/make_abbr_dataset/sparse_matrices"

    #sm = opt.abbr + "_umls_ancestor_sparseMatrix_20190322.pickle"
    sm = abbr + "_umls_ancestor_sparseMatrix_20190406_full.pickle"
    #sm = "all_umls_ancestor_sparseMatrix_20190322.pickle"
    print("Getting sparse matrix..................")
    print(os.path.join(dir, sm))
    p_in = open(os.path.join(dir, sm), 'rb')
    umls_rel = pickle.load(p_in)
    p_in.close()

    concepts_to_get = umls_rel["all_concepts"]
    files = []
    for concept in concepts_to_get:
        file_to_get = concept + ".txt"
        files.append(file_to_get)
    print(files)
    return files

def load_models():
    global FASTTEXT_MODEL, IDF_WEIGHTS
    wordEmbed_model = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "word_embeddings.bin")
    FASTTEXT_MODEL = fastText.load_model(wordEmbed_model)
    idfWeights_dict = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "word_weighting_dict.txt")
    IDF_WEIGHTS = read_word_weightings(idfWeights_dict)


def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip


def load_data(cui_samples, cui):
    source = cui
    global abbr_sense_dict
    if source not in abbr_sense_dict:
        abbr_sense_dict[source] = []
    for i in cui_samples:
        try:
            content = i.split("|")
        except ValueError:
            print("STOPPED HERE")
            print(cui, i)

        label = content[0]
        sample = AbbrRep()

        sample.label = label
        sample.features_doc = content[1].split()
        sample.features_doc_left = content[2].split()
        sample.features_left = content[3].split()
        sample.features_right = content[4].split()
        sample.source = source
        abbr_sense_dict[source].append(sample)


def get_local_context(opt, x):
    features_left = x.features_left
    features_right = x.features_right
    total_number_embeds = 0
    local_context = np.zeros((1, 100))
    #local_context = np.zeros((int(opt.window)*2, 100))
    counter = 0

    for j in range(max(0, len(features_left) - int(opt.window)), len(features_left)):
        #local_context = np.add(local_context, FASTTEXT_MODEL.get_word_vector(features_left[j]))
        z = features_left[j]
        try:
            word_vector = hash_word_vectors[z]
        except KeyError:
            hash_word_vectors[z] = FASTTEXT_MODEL.get_word_vector(z)
            word_vector = hash_word_vectors[z]
        #local_context[counter] = word_vector
        local_context = np.add(local_context, word_vector)
        total_number_embeds += 1
        counter += 1

    len_window = min(len(features_right), int(opt.window))
    for k in range(0, len_window):
        z = features_right[k]
        try:
            word_vector = hash_word_vectors[z]
        except KeyError:
            hash_word_vectors[z] = FASTTEXT_MODEL.get_word_vector(z)
            word_vector = hash_word_vectors[z]
        local_context = np.add(local_context, word_vector)
        #local_context[counter] = word_vector
        total_number_embeds += 1
        counter += 1

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


def create_embed(opt, abbr):
    if opt.sm:
        file_names = get_specific_files(opt, abbr)
    else:
        file_names = get_files()
    global NUM_CUIS
    NUM_CUIS = len(file_names)
    print(NUM_CUIS)

    start = 0
    end = NUM_CUIS
    print("Covering indices: " + str(start) + " to " + str(end))

    counter = 0
    src_dir = "/Volumes/terminator/hpf/casi_cuis_pars_gpars_1000S_20190408/" + abbr + "_cuis_gpars_1000S_20190408"
    #src_dir = "/Volumes/terminator/hpf/casi_cuis_pars_gpars_20190402/" + opt.abbr + "_cuis_gpars_20190403"
    print("Getting cuis from..................")
    print(src_dir)
    for i in tqdm(range(start, end)):
        file = file_names[i]
        try:
            cui_samples = open(os.path.join(src_dir, file), 'r').readlines()
        except:
            counter += 1
            continue

        #cui = file.split("_")[0]
        cui = file[:-4]
        load_data(cui_samples, cui)


    print("num files not found: " + str(counter))
    columns = 100
    if opt.g:
        columns += 100
    counter_master = len(abbr_sense_dict)
    print(counter_master)
    counter = 0
    training_samples = {}
    for key in tqdm(abbr_sense_dict):
        # if counter == int(counter_master*0.2):
        #     outputfile = opt.abbr + "_embedding_dataset_rs_gpars_20190406_1.pickle"
        #     print("outputfile1..................")
        #     pickle_out = open(outputfile, "wb")
        #     pickle.dump(training_samples, pickle_out)
        #     pickle_out.close()
        #     print(len(training_samples))
        #     training_samples = {}
        # if counter == int(counter_master*0.4):
        #     outputfile = opt.abbr + "_embedding_dataset_rs_gpars_20190406_2.pickle"
        #     print("outputfile2..................")
        #     pickle_out = open(outputfile, "wb")
        #     pickle.dump(training_samples, pickle_out)
        #     pickle_out.close()
        #     print(len(training_samples))
        #     training_samples = {}
        # if counter == int(counter_master*0.6):
        #     outputfile = opt.abbr + "_embedding_dataset_rs_gpars_20190406_3.pickle"
        #     print("outputfile3..................")
        #     pickle_out = open(outputfile, "wb")
        #     pickle.dump(training_samples, pickle_out)
        #     pickle_out.close()
        #     print(len(training_samples))
        #     training_samples = {}
        # if counter == int(counter_master*0.8):
        #     outputfile = opt.abbr + "_embedding_dataset_rs_gpars_20190406_4.pickle"
        #     print("outputfile4..................")
        #     pickle_out = open(outputfile, "wb")
        #     pickle.dump(training_samples, pickle_out)
        #     pickle_out.close()
        #     print(len(training_samples))
        #     training_samples = {}

        training_samples[key] = []
        if opt.all:
            num_samples = len(abbr_sense_dict[key])
        else:
            num_samples = min(int(opt.ns), len(abbr_sense_dict[key]))
        for i in range(num_samples):
            training_samples[key].append(abbr_sense_dict[key][i])
            x = np.zeros((1, columns))
            x[0][0:100] = get_local_context(opt, training_samples[key][i])
            if opt.g:
                x[0][100:200] = get_global_context(training_samples[key][i])
            training_samples[key][i].embedding = x
        counter += 1

    outputfile = abbr + "_embedding_dataset_rs_gpars_20190408.pickle"
    root = "w" + str(opt.window) + "_ns" + str(opt.ns) + "_"
    if opt.g:
        root += "g_"
    root += "20190408"
    dest_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/concept_embedding_dataset_" + root
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    #outputfile = opt.abbr + "_embedding_dataset_rs_gpars_" + root + ".pickle"
    if opt.outputfile:
        outputfile = opt.outputfile
    print("outputfile..................")
    print(outputfile)
    pickle_out = open(os.path.join(dest_dir, outputfile), "wb")
    print(len(training_samples))
    pickle.dump(training_samples, pickle_out)
    pickle_out.close()

    print("Done writing CUI embeddings: " + str(file_names[start]) + " to " + str(file_names[end - 1]))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-sm', action="store_true", help="if only getting subset of abbreviations, input sparse matrix name")
    parser.add_argument('-window', required=True, help="max number of words to consider in local_context")
    parser.add_argument('-ns', default=1000, help="max number of random samples PER EXPANSION to "
                                                 "create training set from")
    parser.add_argument('-all', action="store_true", help="if true, create training set with ALL samples "
                                                          "PER EXPANSION (overrides -ns)")
    parser.add_argument('-g', action="store_true", help="if true, consider global context")

    parser.add_argument('-outputfile')
    parser.add_argument('-abbrlist', help='list of abbrs to get embeddings for')
    parser.add_argument('-abbr', action="store_true")
    opt = parser.parse_args()


    load_models()
    #if opt.abbrlist:

        #abbrs = open(opt.abbrlist, 'r').readlines()
    if opt.abbr:
        abbr_dict = ["bmp", "cvp", "fsh", "mom", "ac", "ald", "ama", "asa", "av", "avr", "bal", "bm", "ca", "cea", "cr",
                     "cva", "cvs", "dc", "dip",
                     "dm", "dt", "er", "et", "gt", "im", "ir", "it", "ivf", "le", "mp", "mr", "ms", "na", "np", "op",
                     "or", "otc", "pa", "pac", "pcp", "pd", "pda", "pe", "pr", "pt", "ra", "rt", "sbp", "sma", "vad"]
        for abbr in abbr_dict:
            global abbr_sense_dict
            abbr_sense_dict = {}
            #opt.abbr = abbr[:-1]
            print(opt.abbr)
            create_embed(opt, abbr)
    else:
        create_embed(opt)

    print("Done creating CUI embeddings!!! \U0001F388 \U0001F33B")


if __name__ == "__main__":
    main()
