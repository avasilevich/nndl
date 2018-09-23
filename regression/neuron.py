import tensorflow as tf


epoch_number = 1500
learn_rate = 0.01


def single_neuron():
    x = tf.constant(1.0, name="input")
    w = tf.Variable(0.9, name="weight")
    y = tf.multiply(x, w, name="output")

    y_learn = tf.constant(0.0)

    loss = (y - y_learn) ** 2

    summary_writer = tf.summary.FileWriter("log_simple_graph", graph=tf.get_default_graph())

    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        print("Input:  {}".format(session.run(x)))
        print("Weight: {:.1f}".format(session.run(w)))

        for i in range(epoch_number):
            session.run(train_step)

        print(session.run(y))

        summary_y = tf.summary.scalar("output", y)
        summary_writer = tf.summary.FileWriter("log_simple_stats")
        session.run(tf.global_variables_initializer())

        for i in range(epoch_number):
            summary_str = session.run(summary_y)
            summary_writer.add_summary(summary_str, i)
            session.run(train_step)


single_neuron()