import getopt
import sys
import ast

import fastText
import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import hexify
import matplotlib.pyplot as plt
import pandas as pd
import pickle

FASTTEXT_MODEL = None
IDF_WEIGHTS = None
hash_word_vectors = {}
WINDOW = 2
PERPLEXITY = 50
PRINT_SAMPLES = False
NS = 0


def load_models():
    global FASTTEXT_MODEL, IDF_WEIGHTS

    FASTTEXT_MODEL = fastText.load_model("word_embeddings.bin")
    IDF_WEIGHTS = read_word_weightings("word_weighting_dict.txt")

def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip

class AbbrRep:
    def __init__(self):
        self.features_left = []
        self.features_right = []
        self.features_doc = []
        self.features_doc_left = []

def load_data(input):
    abbr_distr = {}
    abbr_sense_dict = {}
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
            if label == "dipyridamole":
                continue
            sample = AbbrRep()

            try:
                abbr_distr[label] += 1
            except KeyError:
                abbr_distr[label] = 1
                abbr_sense_dict[label] = []

            sample.features_doc = content[1].split()
            sample.features_doc_left = content[2].split()
            sample.features_left = content[3].split()
            sample.features_right = content[4].split()
            abbr_sense_dict[label].append(sample)

    return abbr_sense_dict, abbr_distr

def get_local_context(x):

    features_left = x.features_left
    features_right = x.features_right
    total_number_embeds = 0
    local_context = np.zeros((1,100))

    for j in range(max(0, len(features_left) - WINDOW), len(features_left)):
        local_context = np.add(local_context, FASTTEXT_MODEL.get_word_vector(features_left[j]))
        total_number_embeds += 1

    len_window = min(len(features_right), WINDOW)
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
        shuffled_abbr_sense_we_dict[key] = shuffle(abbr_sense_we_dict[key], random_state=42)
    return shuffled_abbr_sense_we_dict

def graph_abbr_representation(inputfile, n_sne, g=True):

    abbr_sense_dict, abbr_distr = load_data(inputfile)
    shuffled_abbr_sense_we_dict = shuffle_data(abbr_sense_dict)
    X_data = []
    y_data = []
    total = 0
    for key in shuffled_abbr_sense_we_dict:
        num_samples = min(n_sne, len(shuffled_abbr_sense_we_dict[key]))
        total += num_samples
        X_data.extend(shuffled_abbr_sense_we_dict[key][:num_samples])
        y_data.extend([key] * num_samples)

    columns = 200
    num_examples = len(X_data)
    X_data_embeddings = np.zeros((num_examples, columns))
    for i in range(num_examples):
        x = np.zeros((1, columns))
        x[0][0:100] = get_local_context(X_data[i])
        x[0][100:200] = get_global_context(X_data[i])
        X_data_embeddings[i] = x

    return X_data, X_data_embeddings, y_data, abbr_distr

def graph_word_embeddings(inputfile):
    abbr_sense_dict, abbr_distr = load_data(inputfile)
    labels = []
    for key in abbr_sense_dict:
        labels.append(key)
    x = np.zeros((len(labels), 100))
    for i in range(len(labels)):
        z = labels[i]
        try:
            word_vector = hash_word_vectors[z]
        except KeyError:
            hash_word_vectors[z] = FASTTEXT_MODEL.get_word_vector(z)
            word_vector = hash_word_vectors[z]
        x[i] = word_vector

    y = labels
    return x, y, abbr_distr

def get_distinct_labels(y):
    labels = {}
    counter = 0
    for i in y:
        try:
            labels[i] += 1
        except KeyError:
            labels[i] = 1
            counter += 1
    return counter

def visualize(files, abbr, dr, type):

    n_sne = 1000
    all_X_words = []
    all_X = []
    all_y = []
    style = []
    abbrs_dicts = []

    # ----------------------- represent words using embeddings or local + global context ---------------------------- #
    for file in files:
        if type == "we":
            X_data_embeddings, y_data, abbr_distr = graph_word_embeddings(file)
        else:
            X_data, X_data_embeddings, y_data, abbr_distr = graph_abbr_representation(file, n_sne)
            all_X_words.extend(X_data)

        all_X.extend(X_data_embeddings)
        all_y.extend(y_data)

        file_words = file.split("_")
        if file_words[1] != "expanded":
            label = file_words[0] + " (ABBR) from " + file_words[1][:-4]
        else:
            label = "RS from " + file_words[2][:-4]
        style.extend([label] * len(y_data))
        abbrs_dicts.append((file, abbr_distr))

    if type == "we":
        title = " visualization of word embeddings for abbreviation: " + abbr
        fig_name = abbr+"_wordembeddings_"
    else:
        title = " visualization of l + g representations for abbreviation: " + abbr
        fig_name = abbr + "_lgcontext_w" + str(WINDOW) + "_"

    # ---------------------------------------------------------------------------------------------------------------- #

    # --------------------------------- map data to lower dimension using PCA or t-SNE ------------------------------- #
    if int(dr) == 0:
        pca = PCA(n_components=2)
        results = pca.fit_transform(all_X)
        title = "PCA" + title
        fig_name += "pca.png"
    else:
        tsne = TSNE(n_components=2, verbose=0, perplexity=PERPLEXITY, n_iter=1000, random_state=42)
        results = tsne.fit_transform(all_X)
        title = "t-SNE" + title
        fig_name += "tsne_p" + str(PERPLEXITY) + ".png"

    # format components
    vis_x = results[:, 0]
    vis_y = results[:, 1]

    if PRINT_SAMPLES:
        coord_file_name = "COORDINATES_" + fig_name[:-4] + ".pickle"
        print(coord_file_name)
        pickle_out = open(coord_file_name, "wb")
        coord_info = [vis_x, vis_y, all_X_words]
        pickle.dump(coord_info, pickle_out)
        pickle_out.close()


    # ---------------------------------------------------------------------------------------------------------------- #
    num_distinct_labels = get_distinct_labels(all_y)
    flatui = hexify.random_rbg(num_distinct_labels, NS)
    colors = sns.set_palette(flatui)
    fig = plt.figure()
    plt.title(title, fontsize=20, wrap=True)
    d = {'Sense': all_y, 'Dataset': style}
    df = pd.DataFrame(data=d)
    sns.scatterplot(x=vis_x, y=vis_y, hue=df["Sense"], style=df["Dataset"], palette=colors,s=100)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., fontsize=16)

    # Parameter info to put under image
    parameters = "# points/sense = min(1000, # samples)"
    image_info = "PLOT PARAMS: " + parameters + " | local_window = " + str(WINDOW)
    if dr == 1:
        image_info += " | perplexity = " + str(PERPLEXITY)
    fig.text(.5, .05, str(image_info), ha='center', wrap=True, fontsize=18)
    fig.text(.5, .02, str("SENSE DISTRIBUTION: " + str(abbrs_dicts)), ha='center', wrap=True, fontsize=12)

    fig.set_size_inches(40, 20)
    plt.savefig(fig_name)

def main(argv):
    dr = 1  # dimensionality reduction; 0 = PCA, 1 = t-sne
    global NS # new random seed to change color scheme; 0 = No, 1 = Yes
    global PERPLEXITY #t-sne parameter
    global PRINT_SAMPLES, WINDOW
    plot_embeddings = False

    params_msg = 'python3 visualize_data.py -t <dim_red_method> -p <perplexity> -s <new_seed> -c <save_coords> ' \
                 '-w <window_size> -e <plot_of_embeddings> [STR of FILES to plot]'
    try:
        opts, args = getopt.getopt(argv, "t:p:s:c:w:e:")
    except getopt.GetoptError:
        print(params_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-t":
            dr = int(arg)
        elif opt == "-p":
            PERPLEXITY = int(arg)
        elif opt == "-s":
            NS = int(arg)
        elif opt == "-c":
            if int(arg) == 1:
                PRINT_SAMPLES = True
        elif opt == "-w":
            WINDOW = int(arg)
        elif opt == "-e":
            if int(arg) == 1:
                plot_embeddings = True

    load_models()
    files = args
    abbr = files[0].split("_")[0].lower()

    if plot_embeddings:
        visualize(files, abbr, dr, "we")

    visualize(files, abbr, dr, "lg")

if __name__ == "__main__":
    main(sys.argv[1:])
