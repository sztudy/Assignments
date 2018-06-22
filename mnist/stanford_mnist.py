# encoding = utf-8

import os
import time

import tensorflow as tf

import utils

class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.momentum = 0.9
        self.batch_size = 128
        self.drop_prob = tf.constant(0.4)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = False
    
    def get_data(self):
        with tf.name_scope('data'):
            train, val, test = utils.read_mnist('../data/mnist', flatten=False)
            
            train_data = tf.data.Dataset.from_tensor_slices(train)
            train_data = train_data.shuffle(10000).repeat().batch(self.batch_size)

            test_data = tf.data.Dataset.from_tensor_slices(test)
            test_data = test_data.batch(self.batch_size)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, [-1, 28, 28, 1])
            self.train_data_init = iterator.make_initializer(train_data)
            self.test_data_init = iterator.make_initializer(test_data)

    def inference(self):
        conv1 = tf.layers.conv2d(inputs=self.img, filters=32, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')

        pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
        fc = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='fc')
        dropout = tf.layers.dropout(inputs=fc, rate=self.drop_prob, training=self.training, name='dropout')
        self.logits = tf.layers.dense(inputs=dropout, units=self.n_classes, name='logits')

    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        self.opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum).minimize(self.loss, global_step=self.gstep)

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        with tf.name_scope('evaluate'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))

                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'model/stanford_mnist', step)
        print('Average loss at epoch{0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1}'.format(epoch, total_correct_preds / self.n_test))
        print('Took:{0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        utils.safe_mkdir('model')
        utils.safe_mkdir('model/stanford_mnist')
        writer = tf.summary.FileWriter('./graph/stanford_mnist', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/stanford_mnist/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_data_init, writer, epoch, step)
                self.eval_once(sess, self.test_data_init, writer, epoch, step)

        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=15)
