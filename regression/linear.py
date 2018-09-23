import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

learn_rate = 0.001
epoch_num = 1000
display_step = 50


def linear():
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    w_histo = tf.summary.histogram('weights', W)
    b_histo = tf.summary.histogram('biases', b)

    x_input = rng.sample((100, 1))
    y_input = rng.sample((100, 1))
    n_samples = x_input.shape[0]

    X = tf.placeholder(tf.float32, name="input_x")
    Y = tf.placeholder(tf.float32, name="input_y")

    y_predict = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_sum(tf.pow(y_predict-Y, 2))/(2*n_samples)
    tf.summary.scalar('cost_function', cost)

    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    merged = tf.summary.merge_all()

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_writer = tf.summary.FileWriter('E:/py_devel/train', graph_def=session.graph_def)

        for epoch in range(epoch_num):
            for (x_val, y_val) in zip(x_input, y_input):
                session.run(optimizer, feed_dict={X: x_val, Y: y_val})

            if (epoch + 1) % 50 == 0:
                c = session.run(cost, feed_dict={X: x_input, Y: y_input})

                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                      "W=", session.run(W), "b=", session.run(b))
                summary_str = session.run(merged, feed_dict={X: x_input, Y: y_input})
                train_writer.add_summary(summary_str, epoch)

        print("Optimization Finished!")
        training_cost = session.run(cost, feed_dict={X: x_input, Y: y_input})
        print("Training cost=", training_cost, "W=", session.run(W), "b=", session.run(b), '\n')

        # Graphic display
        plt.plot(x_input, y_input, 'ro', label='Original data')
        plt.plot(x_input, session.run(W) * x_input + session.run(b), label='Fitted line')
        plt.legend()
        plt.show()

    return 0


if __name__ == "__main__":
    linear()
