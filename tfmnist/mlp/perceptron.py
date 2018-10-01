import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


learning_rate = 0.001
training_epoches = 1000
batch_size = 100
step = 50
dropout = 0.75


input_size = 784
hidden_first_size = 256
hidden_secnd_size = 256
hidden_third_size = 256
output_size = 10


X = tf.placeholder("float", [None, input_size])
Y = tf.placeholder("float", [None, output_size])
keep_prob = tf.placeholder(tf.float32)


weights = {
    'first'  : tf.get_variable("w_first", shape=[input_size, hidden_first_size],
                        initializer=tf.contrib.layers.variance_scaling_initializer()),
    'secnd'  : tf.get_variable("w_secnd", shape=[hidden_first_size, hidden_secnd_size],
                        initializer=tf.contrib.layers.variance_scaling_initializer()),
    'third'  : tf.get_variable("w_third", shape=[hidden_secnd_size, hidden_third_size],
                        initializer=tf.contrib.layers.variance_scaling_initializer()),
    'output' : tf.Variable(tf.random_normal([hidden_third_size, output_size]))
}

biases = {
    'first'  : tf.Variable(tf.random_normal([hidden_first_size])),
    'secnd'  : tf.Variable(tf.random_normal([hidden_secnd_size])),
    'third'  : tf.Variable(tf.random_normal([hidden_third_size])),
    'output' : tf.Variable(tf.random_normal([output_size]))
}


def multilayer_perceptron(x):
    first_layer = tf.nn.relu_layer(x, weights['first'], biases['first'])
    secnd_layer = tf.nn.relu_layer(first_layer, weights['secnd'], biases['secnd'])
    third_layer = tf.nn.relu_layer(secnd_layer, weights['third'], biases['third'])

    first_layer = tf.nn.dropout(first_layer, dropout)
    secnd_layer = tf.nn.dropout(secnd_layer, dropout)
    third_layer = tf.nn.dropout(third_layer, dropout)

    out_layer = tf.add(tf.matmul(third_layer, weights['output']), biases['output'])

    return out_layer


logits = multilayer_perceptron(X)

saver = tf.train.Saver(max_to_keep=1)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoches):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y,
                                                            keep_prob: 0.8})

            avg_cost += c / total_batch

        if epoch % step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    print("Optimization Finished!")

    saver.save(sess, "./mlp_model")

    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy (learn): ", accuracy.eval({X: mnist.train.images,
                                               Y: mnist.train.labels,
                                               keep_prob: 1.0}))
    print("Accuracy (test):  ", accuracy.eval({X: mnist.test.images,
                                               Y: mnist.test.labels,
                                               keep_prob: 1.0}))