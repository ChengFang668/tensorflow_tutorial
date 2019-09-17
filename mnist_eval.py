import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载 mnist_inference.py 和 mnist_train.py 中定义的常量和函数
import mnist_inference
import mnist_train

# 每10秒加载一次最新的模型，并在验证数据集上测试正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:

        # 定义输入输出数据格式
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 调用函数计算前向传播结果
        y = mnist_inference.inference(x, None)
        # 通过预测结果计算正确率
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名方式加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔 EVAL_INTERVAL_SECS 秒调用一次计算正确率的过程，以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名找到当前迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("After %s training step(s), validation accuracy = %g." % (global_step, accuracy_score))
        else:
            print("No checkpoint file found.")
            return
    time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("D:/Users/T00014140/PycharmProjects/vipshop/mnist_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
