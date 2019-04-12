#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 00:49:44 2018

@author: Marta
"""


import tensorflow as tf


class LocalModel():
    #############################
    ##### Creates the model #####
    #############################
    def __init__(self, n_classes, columns):

        self.n_nodes_hl1 = 100
        self.n_classes = n_classes
        self.batch_size = 10
        self.columns = columns
        
        self.x = tf.placeholder('float', [None,self.columns])
        self.y = tf.placeholder('float')
  
    def local_nn(self, data):

        #initializer = tf.contrib.layers.xavier_initializer(seed=42) #XAVIER
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=42)  #HE
        self.hl1 = {'W': tf.Variable(initializer([self.columns, self.n_nodes_hl1])),
                    'b': tf.Variable(initializer([self.n_nodes_hl1]))}
        
        self.out = {'W': tf.Variable(initializer([self.n_nodes_hl1, self.n_classes])),
                    'b': tf.Variable(initializer([self.n_classes]))}
        
        l1 = tf.add(tf.matmul(data,self.hl1['W']), self.hl1['b'])
        l1 = tf.nn.relu(l1)
        
        output = tf.matmul(l1, self.out['W']) + self.out['b']    
    
        return output
    
    def train_nn(self, X_train, y_train, X_val, y_val, X_mimic, y_mimic):
        print(y_train.shape)
        prediction = self.local_nn(self.x)
        if(self.n_classes > 2):
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y) )
        else:
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        #optimizer = tf.train.AdamOptimizer()
        #gvs = optimizer.compute_gradients(cost)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        #train_op = optimizer.apply_gradients(capped_gvs)
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        gvs = optimizer.compute_gradients(cost)
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad,1e-10,100.0) + 1e-10
        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(clipped_gradients)
        '''
        
        total_epochs = 100
    
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            last_epoch_loss = 0
            current_index = 0
            all_loss = []
            for epoch in range(total_epochs):
                epoch_loss = 0
                current_index = 0
                num_batches = int(len(X_train)/self.batch_size)
                if(len(X_train)%self.batch_size != 0):
                    num_batches += 1
                for _ in range(num_batches):
                    current_x = X_train[current_index:current_index+min(len(X_train)-current_index, self.batch_size)]
                    current_y = y_train[current_index:current_index+min(len(y_train)-current_index, self.batch_size)]
                    current_index += self.batch_size
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: current_x, self.y: current_y})
                    epoch_loss += c
                if(epoch == total_epochs-1):
                    last_epoch_loss = epoch_loss
                # print('Epoch ' + str(epoch) + ' completed out of ' + str(total_epochs) + " LOSS: " + str(epoch_loss))
                all_loss.append(epoch_loss)
                
            #return if index of max value of array are same (hopefully they are the same index)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            cv_set_accuracy = accuracy.eval(feed_dict={self.x: X_val, self.y: y_val})

            # print(last_epoch_loss, cv_set_accuracy)

            #if(y_mimic != ""):
            mimic_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y,1))
            mimic_eval = tf.reduce_mean(tf.cast(mimic_correct, 'float'))


            mimic_accuracy = mimic_eval.eval(feed_dict={self.x: X_mimic, self.y: y_mimic})

            # pred_model = tf.argmax(prediction, 1)
            # ground_truth = tf.argmax(self.y, 1)
            # best = sess.run([pred_model], feed_dict={self.x: X_mimic})
            # truth = sess.run([ground_truth], feed_dict={self.y: y_mimic})
            # print(best, truth)


            return last_epoch_loss, cv_set_accuracy, mimic_accuracy

            
            
           
            
    
