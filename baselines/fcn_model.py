"""
Defines a FCN Model Class
"""


import tensorflow as tf


class FCN:
    """
    FCN Model.
    """
    def __init__(self, n_classes, columns):

        self.n_nodes_hl1 = 100
        self.n_classes = n_classes
        self.batch_size = 10
        self.columns = columns

        self.x = tf.placeholder('float', [None, self.columns])
        self.y = tf.placeholder('float')

    def forward(self, x):
        """
        Defines forward pass of network.
        """

        initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=42)
        self.hl1 = {'W': tf.Variable(initializer([self.columns, self.n_nodes_hl1])),
                    'b': tf.Variable(initializer([self.n_nodes_hl1]))}

        self.out = {'W': tf.Variable(initializer([self.n_nodes_hl1, self.n_classes])),
                    'b': tf.Variable(initializer([self.n_classes]))}

        l1 = tf.add(tf.matmul(x, self.hl1['W']), self.hl1['b'])
        l1 = tf.nn.relu(l1)

        output = tf.matmul(l1, self.out['W']) + self.out['b']

        return output

    def train_nn(self, data, hyperparams):
        """
        Train the network.
        """
        # unpack data
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_val = data["x_val"]
        y_val = data["y_val"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        prediction = self.forward(self.x)

        if self.n_classes > 2:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        else:
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=self.y))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        total_epochs = hyperparams.num_epochs

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            all_loss = []
            for epoch in range(total_epochs):
                epoch_loss = 0
                current_index = 0
                num_batches = int(len(x_train)/self.batch_size)
                if len(x_train) % self.batch_size != 0:
                    num_batches += 1
                for _ in range(num_batches):
                    current_x = x_train[
                                current_index: current_index + min(len(x_train) - current_index, self.batch_size)]
                    current_y = y_train[
                                current_index: current_index + min(len(y_train) - current_index, self.batch_size)]
                    current_index += self.batch_size
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: current_x, self.y: current_y})
                    epoch_loss += c
                all_loss.append(epoch_loss)

            # set up evaluation of the model
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            num_correct = tf.reduce_sum(tf.cast(correct, 'float'))
            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            # train_acc = accuracy.eval(feed_dict={self.x: x_train, self.y: y_train})
            # val_acc = accuracy.eval(feed_dict={self.x: x_val, self.y: y_val})
            # test_acc = accuracy.eval(feed_dict={self.x: x_test, self.y: y_test})

            train_num_corr = num_correct.eval(feed_dict={self.x: x_train, self.y: y_train})
            val_num_corr = num_correct.eval(feed_dict={self.x: x_val, self.y: y_val})
            test_num_corr = num_correct.eval(feed_dict={self.x: x_test, self.y: y_test})

            x_train_len = len(x_train)
            x_val_len = len(x_val)
            x_test_len = len(x_test)

            out = {
                "train_acc": train_num_corr / x_train_len,
                "train_corr": train_num_corr,
                "train_tot": x_train_len,
                "val_acc": val_num_corr / x_val_len,
                "val_corr": val_num_corr,
                "val_tot": x_val_len,
                "test_acc": test_num_corr / x_test_len,
                "test_corr": test_num_corr,
                "test_tot": x_test_len,
            }

            return out
