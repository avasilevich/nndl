import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

learn_rate = 0.001
epoch_num = 1000
display_step = 50


def generate_data(example=False):
    x_input = 0
    y_input = 0

    if example:
        x_input = rng.rand(100, 1)
        noise = rng.normal(scale=0.1, size=(100, 1))
        y_input = np.reshape(5 * x_input  + 2 + noise, (-1))

    else:
        x_input = rng.sample((100, 1))
        y_input = rng.sample((100, 1))

    return x_input, y_input


def linear():
    W = tf.Variable(tf.random_normal(shape=[1, 1]), name="weight")
    b = tf.Variable(tf.random_normal(shape=[1, 1]), name="bias")

    w_histo = tf.summary.histogram('weights', W)
    b_histo = tf.summary.histogram('biases', b)

    x_input, y_input = generate_data()

    X = tf.placeholder(tf.float32, name="input_x")
    Y = tf.placeholder(tf.float32, name="input_y")

    model_output = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(Y - model_output))
    tf.summary.scalar('cost_function', cost)

    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    merged = tf.summary.merge_all()

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_writer = tf.summary.FileWriter('train', graph=session.graph)

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
