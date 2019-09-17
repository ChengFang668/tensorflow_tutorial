import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("D:/Users/T00014140/PycharmProjects/vipshop/mnist_data", one_hot=True)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    w = tf.Variable(tf.random_normal([784, 10], stddev=0.5, seed=1))
    biases = tf.Variable(tf.constant(0.1, shape=[10]))
    y = tf.nn.relu(tf.matmul(x, w) + biases)
    # y = tf.matmul(x, w) + biases
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1)))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    # 训练数据
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(20000):
            xs, ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={x: xs, y_: ys})

        # 计算正确率
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_score = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
        print("After %d training step(s), validation accuracy is %g." % (i, accuracy_score))


def main(argv=None):
    mnist = input_data.read_data_sets("D:/Users/T00014140/PycharmProjects/vipshop/mnist_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

