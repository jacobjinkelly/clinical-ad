"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""

import tensorflow as tf
import numpy as np
import random
import json
import pickle
import fastText
import re
from tqdm import tqdm
from abbrrep_class import AbbrRep


class ConceptEmbedModel():
    def sent2vec(self, phrase_list, max_length):
        phrase_vec_list = []
        phrase_seq_lengths = []
        for phrase in tqdm(phrase_list):
            start_left = int(max(len(phrase.features_left)-(max_length/2), 0))
            tokens = phrase.features_left[start_left:]
            end_right = int(min(len(phrase.features_right), (max_length/2)))
            tokens_right = phrase.features_left[:end_right]
            tokens.extend(tokens_right)
            phrase_vec_list.append([self.word_model.get_word_vector(tokens[i]) if i<len(tokens) else [0]*100 for i in range(max_length)])
            phrase_seq_lengths.append(len(tokens))
        return np.array(phrase_vec_list), np.array(phrase_seq_lengths)

    def __init__(self, config, sm, data, word_model):
        # ************** Initialize variables ***************** #
        tf.reset_default_graph()
        self.sm = sm
        self.word_model = word_model
        self.data = data
        config.concepts_size = len(self.sm['all_concepts']) +1
        self.config = config

        # ************** Initialize matrix dimensions ***************** #
        self.label = tf.placeholder(tf.int32, shape=[None])
        self.class_weights = tf.Variable(tf.ones([config.concepts_size]), False)

        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, 100])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.lr = tf.Variable(config.lr, trainable=False)
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
        layer1 = tf.layers.conv1d(self.seq, self.config.cl1, 1, activation=tf.nn.elu,\
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
        if config.flat:
            aggregated_w = self.embeddings
        else:
            aggregated_w = self.aggregated_embeddings

        last_layer_b = tf.get_variable('last_layer_bias', shape = [self.config.concepts_size], initializer = tf.random_normal_initializer(stddev=0.001))

        self.score_layer = tf.matmul(self.seq_embedding, tf.transpose(aggregated_w)) + last_layer_b

        # ************** Loss ***************** #
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label, self.score_layer)) # + reg_constant*tf.reduce_sum(reg_losses)

        self.pred = tf.nn.softmax(self.score_layer)
        self.agg_pred, _ =  tf.nn.top_k(tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.ancestry_sparse_tensor), tf.transpose(self.pred))), 2)

        # ************** Backprop ***************** #
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())


    def save_params(self, epoch, repdir='.'):
        tf.train.Saver().save(self.sess, (repdir + "/e" + str(epoch) +'_params.ckpt').replace('//','/'))


    def init_training(self):
        raw_samples = []
        labels = []
        for key in self.data:
            for item in self.data[key]:
                raw_samples.append(item)
                labels.append(self.sm['word2idx'][key])

        shuffled_idx = list(range(len(raw_samples)))
        random.shuffle(shuffled_idx)
        train_idx = shuffled_idx[:int(len(shuffled_idx)*0.7)]
        val_idx = shuffled_idx[int(len(shuffled_idx)*0.95):]

        training_samples = []
        training_labels = []
        for i in train_idx:
            training_samples.append(raw_samples[i])
            training_labels.append(labels[i])

        self.val_samples = []
        self.val_labels = []
        for i in val_idx:
            self.val_samples.append(raw_samples[i])
            self.val_labels.append(labels[i])
        self.val_labels = np.array(self.val_labels)

        self.training_samples = {}
        self.training_samples['seq'], self.training_samples['seq_len'] = self.sent2vec(training_samples, self.config.max_sequence_length)
        self.training_samples['label'] = np.array(training_labels)


    def train_epoch(self, verbose=True):
        ct = 0
        report_loss = 0
        total_loss = 0
        report_len = 20
        start = 0
        training_size = self.training_samples['seq'].shape[0]
        shuffled_idx = list(range(training_size))
        random.shuffle(shuffled_idx)
        while start < training_size:
            end = min(training_size, start + self.config.batch_size)
            batch = {}
            for cat in self.training_samples:
                batch[cat] = self.training_samples[cat][shuffled_idx[start:end]]
            start += self.config.batch_size
            batch_feed = {self.seq:batch['seq'],\
                    self.seq_len:batch['seq_len'],\
                    self.label:batch['label'],
                    self.is_training:True}
            _ , batch_loss = self.sess.run([self.train_step, self.loss], feed_dict=batch_feed)
            report_loss += batch_loss
            total_loss += batch_loss
            if verbose and ct % report_len == report_len-1:
                print("Step = "+str(ct+1)+"\tLoss ="+str(report_loss/report_len))
                report_loss = 0
            ct += 1

        print("Epoch loss: " + str(total_loss/training_size))
        return total_loss/ct


    def get_probs(self, val_samples):
        seq, seq_len = self.sent2vec(val_samples, self.config.max_sequence_length)
        input = {self.seq: seq, self.seq_len: seq_len, self.is_training: False}
        predictions = self.sess.run([self.pred, self.agg_pred], feed_dict=input)
        return predictions

    def check_val_set(self, count=5):
        batch_size = 512
        head = 0
        while head < len(self.val_samples):
            minibatch = self.val_samples[head:min(head + batch_size, len(self.val_samples))]
            res_tmp, agg_pred_tmp = self.get_probs(minibatch)
            if head == 0:
                res_querry = res_tmp
                agg_pred = agg_pred_tmp
            else:
                res_querry = np.concatenate((res_querry, res_tmp))
                agg_pred = np.concatenate((agg_pred, agg_pred_tmp))
            head += batch_size

        results = []
        for s in range(len(self.val_samples)):
            closest_concepts = np.argsort(-res_querry[s, :])
            tmp_res = []
            for i in closest_concepts:
                if i == len(self.sm['all_concepts']):
                    tmp_res.append(('None', res_querry[self.val_labels[s], i]))
                else:
                    tmp_res.append((self.sm['idx2word'][i], res_querry[s, i], agg_pred[s, 1]))
                if len(tmp_res) >= count:
                    break
            results.append(tmp_res)

        missed1 = [x for i, x in enumerate(self.val_labels) if
                   self.sm['idx2word'][self.val_labels[i]] not in [results[i][0][0]]]
        top1 = (len(self.val_labels) - len(missed1)) / len(self.val_labels)
        missed5 = [x for i, x in enumerate(self.val_labels) if
                   self.sm['idx2word'][self.val_labels[i]] not in [r[0] for r in results[i]]]
        top5 = (len(self.val_labels) - len(missed5)) / len(self.val_labels)
        return top1, top5
