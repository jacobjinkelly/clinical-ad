"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""

import tensorflow as tf
import numpy as np
import random
import json
import pickle
import fastText
import os
import argparse
import re
from tqdm import tqdm
from abbrrep_class import AbbrRep
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


results = {}
results["mimic"] = {}
results["casi"] = {}

class ConceptEmbedModel_Encoder():
    def phrase2vec(self, phrase_list, max_length):
        phrase_vec_list = []
        phrase_seq_lengths = []
        for phrase in phrase_list:
            start_left = int(max(len(phrase.features_left)-(max_length/2), 0))
            tokens = phrase.features_left[start_left:]
            end_right = int(min(len(phrase.features_right),(max_length/2)))
            tokens_right = phrase.features_left[:end_right]
            tokens.extend(tokens_right)
            phrase_vec_list.append([self.word_model.get_word_vector(tokens[i]) if i<len(tokens) else [0]*100 for i in range(max_length)])
            phrase_seq_lengths.append(len(tokens))
        return np.array(phrase_vec_list), np.array(phrase_seq_lengths)

    # ************** Initialize model ***************** #

    def __init__(self, config, sm, word_model):
        #print("Creating the model graph")
        tf.reset_default_graph()
        self.sm = sm
        self.word_model = word_model
        self.abbr = config.abbr
        config.concepts_size = len(self.sm['all_concepts']) +1

        self.config = config

        # ************** Initialize matrix dimensions ***************** #
        self.label = tf.placeholder(tf.int32, shape=[None])
        self.class_weights = tf.Variable(tf.ones([config.concepts_size]), False)

        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, 100])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.lr = tf.Variable(0.002, trainable=False)
        self.is_training = tf.placeholder(tf.bool)


        # ************** Compute dense ancestor matrix from LIL matrix format ***************** #

        sparse_ancestrs = np.zeros((self.sm['matrix_hierarchy'].getnnz(), 2))
        counter = 0
        for row in range(self.sm['matrix_hierarchy'].shape[0]):
            for col in self.sm['matrix_hierarchy'].rows[row]:
                sparse_ancestrs[counter][0] = row
                sparse_ancestrs[counter][1] = col
                counter += 1

        self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(indices=sparse_ancestrs, values=[1.0]*len(sparse_ancestrs),
                                                                        dense_shape=[config.concepts_size, config.concepts_size]))

        # ************** Encoder for sentence embeddings ***************** #

        layer1 = tf.layers.conv1d(self.seq, self.config.cl1, 5, activation=tf.nn.elu,\
                kernel_initializer=tf.random_normal_initializer(0.0,0.1),\
                bias_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=True)

        layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), self.config.cl2, activation=tf.nn.relu,\
                kernel_initializer=tf.random_normal_initializer(0.0,stddev=0.1),
                bias_initializer=tf.random_normal_initializer(0.0,stddev=0.01), use_bias=True)

        self.seq_embedding = tf.nn.l2_normalize(layer2, axis=1)

        # ************** Concept embeddings ***************** #
        self.embeddings = tf.get_variable("embeddings", shape=[self.config.concepts_size, self.config.cl2],
                                          initializer=tf.random_normal_initializer(stddev=0.1))


        self.aggregated_embeddings = tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, self.embeddings)

        last_layer_b = tf.get_variable('last_layer_bias', shape = [self.config.concepts_size], initializer = tf.random_normal_initializer(stddev=0.001))

        self.score_layer = tf.matmul(self.seq_embedding, tf.transpose(self.aggregated_embeddings)) + last_layer_b

        # ************** Loss ***************** #
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label, self.score_layer))
        self.pred = tf.nn.softmax(self.score_layer)
        self.agg_pred, _ =  tf.nn.top_k(tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.ancestry_sparse_tensor), tf.transpose(self.pred))), 2)

        # ************** Backprop ***************** #
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())


    def load_params(self, repdir='.'):
        tf.train.Saver().restore(self.sess, (repdir+'.ckpt').replace('//','/'))

        # dir = "/Users/Marta/80k_abbreviations/allacronyms"
        # # n = open(os.path.join(dir,"allacronyms_cui2meta_20190318.pickle"), 'rb')
        #
        #
        # # m = open(os.path.join(dir, "allacronyms_meta2cui_20190318.pickle"), 'rb')
        # m = open(os.path.join(dir, "allacronyms_meta2cui_20190402_NEW.pickle"), 'rb')
        # meta2cui = pickle.load(m)
        # m.close()
        #
        # q = open(os.path.join(dir, "allacronyms_name2meta_20190402_NEW.pickle"), 'rb')
        # name2meta = pickle.load(q)
        # q.close()
        #
        # word2idx = self.sm['word2idx']

        # pr = meta2cui["pr"][name2meta["pr"]["progesterone receptor"]]
        # pr_idx = word2idx[pr]
        #
        # p = meta2cui["pr"][name2meta["pr"]["progesterone"]]
        # p_idx = word2idx[p]
        #
        # prim = meta2cui["pr"][name2meta["pr"]["primary"]]
        # prim_idx = word2idx[prim]
        #
        # prod = meta2cui["pr"][name2meta["pr"]["production"]]
        # prod_idx = word2idx[prod]
        # musc = meta2cui["ms"][name2meta["ms"]["musculoskeletal"]]
        # musc_idx = word2idx[musc]
        #
        # ms = meta2cui["ms"][name2meta["ms"]["multiple sclerosis"]]
        # ms_idx = word2idx[ms]
        #
        # print(self.embeddings)
        # v = self.sess.run(tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, self.embeddings))
        #
        # print("musculoskeletal: " + str(np.linalg.norm(v[musc_idx])))
        # print("multiple sclerosis: " + str(np.linalg.norm(v[ms_idx])))

    def get_probs(self, val_samples):
        seq, seq_len = self.phrase2vec(val_samples, self.config.max_sequence_length)
        querry_dict = {self.seq: seq, self.seq_len: seq_len}
        res_querry = self.sess.run([self.pred, self.agg_pred], feed_dict=querry_dict)
        return res_querry

    def label_data(self, data, cui2meta, meta2name, name2meta, abbr, source):
        possible_expansions = list(cui2meta[abbr].keys())
        batch_size = 512
        head = 0
        while head < len(data):
            querry_subset = data[head:min(head + batch_size, len(data))]
            res_tmp, agg_pred_tmp = self.get_probs(querry_subset)
            if head == 0:
                res_querry = res_tmp  # self.get_probs(querry_subset)
                agg_pred = agg_pred_tmp  # self.get_probs(querry_subset)
            else:
                res_querry = np.concatenate((res_querry, res_tmp))
                agg_pred = np.concatenate((agg_pred, agg_pred_tmp))
            head += batch_size

        closest_concepts = []
        for s in range(len(data)):
            indecies_querry = np.argsort(-res_querry[s, :])
            tmp_res = []
            for i in indecies_querry:
                if i == len(self.sm['all_concepts']):
                    tmp_res.append(('None'))
                else:
                    tmp_res.append((self.sm['idx2word'][i], res_querry[s, i], agg_pred[s, 1]))

            closest_concepts.append(tmp_res)

        counter = 0
        score = 0
        total = 0
        for tmp_res in closest_concepts:
            seen = False
            for i in range(len(tmp_res)):
                if not seen and tmp_res[i][0] in possible_expansions:
                    print(data[counter].features_left, data[counter].features_right)
                    label = cui2meta[abbr][tmp_res[i][0]]
                    ground_truth = name2meta[abbr][data[counter].label]
                    print(meta2name[abbr][label], meta2name[abbr][ground_truth])
                    if label == ground_truth:
                        score +=1
                    break
            total += 1
            counter += 1
        print(abbr + " from " + source + ": CORRECT="+ str(score) + " TOTAL=" + str(total))
        results[source][abbr] = [score, total]


def load_data(opt):
    from sklearn.utils import shuffle
    import math
    root = "w5_ns1000_g_20190408"
    src_dir = "/Users/Marta/80k_abbreviations/abbr_dataset_mimic_casi" + root + "/"
    fname = opt.abbr + "_mimic_casi_" + root + ".pickle"
    p_in = open(os.path.join(src_dir, fname), "rb")
    data = pickle.load(p_in)
    p_in.close()

    abbr = opt.abbr
    # unlabelled_data = data['mimic_abbr'][abbr]
    casi_data = data['casi_abbr']
    casi_test = []
    for subkey in casi_data:
        casi_test.extend(casi_data[subkey])
    casi_test = shuffle(casi_test, random_state=42)

    full_data = []
    for subkey in data["mimic_rs"]:
        curr_data = data["mimic_rs"][subkey]
        curr_data = shuffle(curr_data, random_state=42)
        full_data.extend(curr_data[:min(int(opt.ns), len(data["mimic_rs"][subkey]))])
    full_data = shuffle(full_data, random_state=42)
    split = math.floor(len(full_data) * 0.7)
    mimic_test = full_data[split:]

    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    n = open(os.path.join(dir, "allacronyms_cui2meta_20190402_NEW.pickle"), 'rb')
    cui2meta = pickle.load(n)
    n.close()
    m = open(os.path.join(dir, "allacronyms_meta2cui_20190402_NEW.pickle"), 'rb')
    meta2cui = pickle.load(m)
    m.close()
    o = open(os.path.join(dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
    meta2name = pickle.load(o)
    o.close()
    q = open(os.path.join(dir, "allacronyms_name2meta_20190402_NEW.pickle"), 'rb')
    name2meta = pickle.load(q)
    q.close()

    return abbr, data, casi_test, mimic_test, cui2meta, meta2cui, meta2name, name2meta


def label_unlabelled_data(args, sm, word_model):
    abbr, data, unlabelled_data, mimic_test, cui2meta, meta2cui, meta2name, name2meta = load_data(args)
    model = ConceptEmbedModel_Encoder(args, sm, word_model)
    model.load_params(repdir=args.param_dir)
    model.label_data(unlabelled_data, cui2meta, meta2name, name2meta, abbr, source="casi")
    model.label_data(mimic_test, cui2meta, meta2name, name2meta, abbr, source="mimic")




def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--abbr', help="abbreviation to label")
    parser.add_argument('--sm',
                        help="address to sparse matrix")
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=50)
    parser.add_argument('--cl1', type=int, help="cl1", default=1024)
    parser.add_argument('--cl2', type=int, help="cl2", default=1024)
    parser.add_argument('--flat', action="store_true")
    parser.add_argument('--ns', help="number mimic to test", default=500)

    parser.add_argument('--param_dir', help="model to load")

    args = parser.parse_args()

    word_model = fastText.load_model(args.fasttext)

    #abbr_dict = ["ivf", "dm", "le", "ra", "pcp", "op", "dt", "dc", "pa", "pda", "rt", "sma", "ac", "pe",
     #            "otc", "im", "pac", "pr", "asa", "ir", "sbp", "cea", "ca", "er", "bal", "avr", "cvp", "av"]
    # abbr_dict = ["ivf", "dm","le",]
    abbr_dict = ["bmp", "cvp", "fsh", "mom","ac", "ald", "ama", "asa", "av", "avr", "bal", "bm",  "ca", "cea", "cr", "cva", "cvs", "dc", "dip",
     "dm", "dt", "er", "et", "gt", "im", "ir", "it", "ivf", "le",  "mp", "mr", "ms", "na", "np", "op",
     "or", "otc", "pa", "pac", "pcp", "pd", "pda", "pe", "pr", "pt", "ra", "rt", "sbp", "sma", "vad"]
    #abbr_dict = ["ms"]
    if not args.abbr:
        p_in = open(args.sm, 'rb')
        sm = pickle.load(p_in)
        p_in.close()
        for abbr in abbr_dict:
            args.abbr = abbr
            label_unlabelled_data(args, sm, word_model)
    else:
        dir_num = args.abbr
        for abbr in abbr_dict:
            args.sm = "sparse_matrices_20190406/" + abbr + "_umls_ancestor_sparseMatrix_20190406_full.pickle"
            p_in = open(args.sm, 'rb')
            sm = pickle.load(p_in)
            p_in.close()
            args.param_dir = "/Volumes/terminator/hpf/20190408_tf_modelcheckpoints/" \
                             "tf_model_indiv_ns" + dir_num + "_w" + str(args.max_sequence_length)+"_20190408/tf_model_20190408_" \
                             +str(args.max_sequence_length) + "w_" +dir_num + "S_"  + abbr + "/e49_params"
            args.abbr = abbr
            print("HELLO")
            label_unlabelled_data(args, sm, word_model)

    micro_acc_casi = 0
    total_casi = 0
    score_casi = 0

    micro_acc_mimic = 0
    total_mimic = 0
    score_mimic = 0

    for abbr in results['casi']:
        correct = results['casi'][abbr][0]
        total = results['casi'][abbr][1]
        micro_acc_casi += correct/total
        total_casi += total
        score_casi += correct
    micro_acc_casi /= len(results['casi'])
    macro_acc_casi = score_casi/total_casi

    for abbr in results['mimic']:
        correct = results['mimic'][abbr][0]
        total = results['mimic'][abbr][1]
        micro_acc_mimic += correct/total
        total_mimic += total
        score_mimic += correct
    micro_acc_mimic /= len(results['mimic'])
    macro_acc_mimic = score_mimic/total_mimic
    print("MICRO ACC CASI: " + str(micro_acc_casi))
    print("MACRO ACC CASI: " + str(macro_acc_casi))

    print("MICRO ACC MIMIC: " + str(micro_acc_mimic))
    print("MACRO ACC MIMIC: " + str(macro_acc_mimic))

        
if __name__ == "__main__":
    main()