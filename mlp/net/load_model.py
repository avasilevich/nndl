import tensorflow as tf


tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("mlp_model.meta") 


with tf.Session() as sess:  
    imported_meta.restore(sess, tf.train.latest_checkpoint('./'))
    h_est2 = sess.run('hor_estimate:0')
    v_est2 = sess.run('ver_estimate:0')
    print("h_est: %.2f, v_est: %.2f" % (h_est2, v_est2))