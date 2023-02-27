import tensorflow as tf
import math


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W, s):
    return tf.nn.conv2d(x, W, strides=s, padding='SAME')


def pooling_layer_parameterized(pool_method, h_conv, pool_kernel, pool_stride):
    if pool_method == 1:
        return tf.nn.max_pool(h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1], padding='SAME')
    elif pool_method == 2:
        return tf.nn.avg_pool(h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1], padding='SAME')


def variable_summaries(var, name, collection):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries') as r:
        mean = tf.reduce_mean(var)
        tf.add_to_collection(collection, tf.scalar_summary('mean/' + name, mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.add_to_collection(collection, tf.scalar_summary('stddev/' + name, stddev))
        tf.add_to_collection(collection, tf.scalar_summary('max/' + name, tf.reduce_max(var)))
        tf.add_to_collection(collection, tf.scalar_summary('min/' + name, tf.reduce_min(var)))
        tf.add_to_collection(collection, tf.histogram_summary(name, var))


def build_model(hyperparameters):
    learning_rate = hyperparameters['learning_rate']
    l2_regularization_penalty = hyperparameters['l2_regularization_penalty']
    fc1_n_neurons = hyperparameters['fc1_n_neurons']
    fc2_1_n_neurons = hyperparameters['fc2_1_n_neurons']
    fc2_2_n_neurons = hyperparameters['fc2_2_n_neurons']
    fc2_3_n_neurons = hyperparameters['fc2_3_n_neurons']
    conv1_kernel = hyperparameters['conv1_kernel']
    conv2_kernel = hyperparameters['conv2_kernel']
    conv1_filters = hyperparameters['conv1_filters']
    conv2_filters = hyperparameters['conv2_filters']
    conv1_stride = hyperparameters['conv1_stride']
    conv2_stride = hyperparameters['conv2_stride']
    pool1_kernel = hyperparameters['pool1_kernel']
    pool2_kernel = hyperparameters['pool2_kernel']
    pool1_stride = hyperparameters['pool1_stride']
    pool2_stride = hyperparameters['pool2_stride']
    pool1_method = 1
    pool2_method = 1

    INPUT_SIZE = 400
    tfo = {}    # Tensorflow objects

    x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
    label_classifier = tf.placeholder(tf.float32, shape=[None], name='label_classifier')
    label_offset = tf.placeholder(tf.float32, shape=[None], name='label_offset')
    label_coldensity = tf.placeholder(tf.float32, shape=[None], name='label_coldensity')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # First Convolutional Layer
    # Kernel size (16,1)
    # Stride (4,1)
    # number of filters = 4 (features?)
    # Neuron activation = ReLU (rectified linear unit)

    W_conv1 = weight_variable([conv1_kernel, 1, 1, conv1_filters])
    b_conv1 = bias_variable([conv1_filters])
    x_4d = tf.reshape(x, [-1, INPUT_SIZE, 1, 1])

    # https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution
    # out_height = ceil(float(in_height) / float(strides[1])) = ceil(1024./4.) = 256
    # out_width = ceil(float(in_width) / float(strides[2])) = 1
    # shape of h_conv1: [-1, 256, 1, 4]
    stride1 = [1, conv1_stride, 1, 1]
    h_conv1 = tf.nn.relu(conv1d(x_4d, W_conv1, stride1) + b_conv1)

    # Kernel size (8,1)
    # Stride (2,1)
    # Pooling type = Max Pooling
    # out_height = ceil(float(in_height) / float(strides[1])) = ceil(256./2.) = 128
    # out_width = ceil(float(in_width) / float(strides[2])) = 1
    # shape of h_pool1: [-1, 128, 1, 4]
    h_pool1 = pooling_layer_parameterized(pool1_method, h_conv1, pool1_kernel, pool1_stride)

    # Second Convolutional Layer
    # Kernel size (16,1)
    # Stride (2,1)
    # number of filters=8
    # Neuron activation = ReLU (rectified linear unit)
    W_conv2 = weight_variable([conv2_kernel, 1, conv1_filters, conv2_filters])
    b_conv2 = bias_variable([conv2_filters])
    # out_height = ceil(float(in_height) / float(strides[1])) = ceil(128./2.) = 64
    # out_width = ceil(float(in_width) / float(strides[2])) = 1
    # shape of h_conv1: [-1, 64, 1, 8]
    stride2 = [1, conv2_stride, 1, 1]
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2, stride2) + b_conv2)
    h_pool2 = pooling_layer_parameterized(pool2_method, h_conv2, pool2_kernel, pool2_stride)

    # FC1: first fully connected layer, shared
    inputsize_fc1 = int(math.ceil(math.ceil(math.ceil(math.ceil(
        INPUT_SIZE / conv1_stride) / pool1_stride) / conv2_stride) / pool2_stride)) * conv2_filters
    h_pool2_flat = tf.reshape(h_pool2, [-1, inputsize_fc1])

    W_fc1 = weight_variable([inputsize_fc1, fc1_n_neurons])
    b_fc1 = bias_variable([fc1_n_neurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout FC1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2_1 = weight_variable([fc1_n_neurons, fc2_1_n_neurons])
    b_fc2_1 = bias_variable([fc2_1_n_neurons])
    W_fc2_2 = weight_variable([fc1_n_neurons, fc2_2_n_neurons])
    b_fc2_2 = bias_variable([fc2_2_n_neurons])
    W_fc2_3 = weight_variable([fc1_n_neurons, fc2_3_n_neurons])
    b_fc2_3 = bias_variable([fc2_3_n_neurons])

    # FC2 activations
    h_fc2_1 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_1) + b_fc2_1)
    h_fc2_2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_2) + b_fc2_2)
    h_fc2_3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_3) + b_fc2_3)

    # FC2 Dropout [1-3]
    h_fc2_1_drop = tf.nn.dropout(h_fc2_1, keep_prob)
    h_fc2_2_drop = tf.nn.dropout(h_fc2_2, keep_prob)
    h_fc2_3_drop = tf.nn.dropout(h_fc2_3, keep_prob)

    # Readout Layer
    W_fc3_1 = weight_variable([fc2_1_n_neurons, 1])
    b_fc3_1 = bias_variable([1])
    W_fc3_2 = weight_variable([fc2_2_n_neurons, 1])
    b_fc3_2 = bias_variable([1])
    W_fc3_3 = weight_variable([fc2_3_n_neurons, 1])
    b_fc3_3 = bias_variable([1])

    # y_fc4 = tf.add(tf.matmul(h_fc3_drop, W_fc4), b_fc4)
    # y_nn = tf.reshape(y_fc4, [-1])
    y_fc4_1 = tf.add(tf.matmul(h_fc2_1_drop, W_fc3_1), b_fc3_1)
    y_nn_classifier = tf.reshape(y_fc4_1, [-1], name='y_nn_classifer')
    y_fc4_2 = tf.add(tf.matmul(h_fc2_2_drop, W_fc3_2), b_fc3_2)
    y_nn_offset = tf.reshape(y_fc4_2, [-1], name='y_nn_offset')
    y_fc4_3 = tf.add(tf.matmul(h_fc2_3_drop, W_fc3_3), b_fc3_3)
    y_nn_coldensity = tf.reshape(y_fc4_3, [-1], name='y_nn_coldensity')

    # Train and Evaluate the model
    loss_classifier = tf.add(tf.nn.sigmoid_cross_entropy_with_logits(y_nn_classifier, label_classifier),
                             l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                          tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
                             name='loss_classifier')
    loss_offset_regression = tf.add(tf.reduce_sum(tf.nn.l2_loss(y_nn_offset - label_offset)),
                                    l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                                 tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_2)),
                                    name='loss_offset_regression')
    epsilon = 1e-6
    loss_coldensity_regression = tf.reduce_sum(
        tf.mul(tf.square(y_nn_coldensity - label_coldensity),
               tf.div(label_coldensity,label_coldensity+epsilon)) +
        l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                     tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
       name='loss_coldensity_regression')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    cost_all_samples_lossfns_AB = loss_classifier + loss_offset_regression
    cost_pos_samples_lossfns_ABC = loss_classifier + loss_offset_regression + loss_coldensity_regression
    # train_step_AB = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all_samples_lossfns_AB, global_step=global_step, name='train_step_AB')
    train_step_ABC = optimizer.minimize(cost_pos_samples_lossfns_ABC, global_step=global_step, name='train_step_ABC')
    # train_step_C = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_coldensity_regression, global_step=global_step, name='train_step_C')
    output_classifier = tf.sigmoid(y_nn_classifier, name='output_classifier')
    prediction = tf.round(output_classifier, name='prediction')
    correct_prediction = tf.equal(prediction, label_classifier)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    rmse_offset = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_nn_offset,label_offset))), name='rmse_offset')
    rmse_coldensity = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_nn_coldensity,label_coldensity))), name='rmse_coldensity')

    variable_summaries(loss_classifier, 'loss_classifier', 'SUMMARY_A')
    variable_summaries(loss_offset_regression, 'loss_offset_regression', 'SUMMARY_B')
    variable_summaries(loss_coldensity_regression, 'loss_coldensity_regression', 'SUMMARY_C')
    variable_summaries(accuracy, 'classification_accuracy', 'SUMMARY_A')
    variable_summaries(rmse_offset, 'rmse_offset', 'SUMMARY_B')
    variable_summaries(rmse_coldensity, 'rmse_coldensity', 'SUMMARY_C')
    # tb_summaries = tf.merge_all_summaries()

    return train_step_ABC, tfo #, accuracy , loss_classifier, loss_offset_regression, loss_coldensity_regression, \
           #x, label_classifier, label_offset, label_coldensity, keep_prob, prediction, output_classifier, y_nn_offset, \
           #rmse_offset, y_nn_coldensity, rmse_coldensity