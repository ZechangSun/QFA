""" TensorFlow Models for DLA finder"""

"""
0.  Convert the model to TF 2.1

used the tf_upgrade_v2 --intree model_v1  --outtree model_v2 to update this code to TF2.1
But tf.summary.scalar requires manual check. 
The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* 
and must be migrated manually.
The method I have used can be found at https://www.tensorflow.org/guide/upgrade
The report.txt is at https://docs.google.com/document/d/1l07KDAfWWZyv9eps71q1d5Qqa46SMUm73u0FZcU-tlY/edit?usp=sharing

"""

"""
tf.random.truncated_normal:Outputs random values from a truncated normal distribution.(shape:The shape of the output tensor.
stddev:The standard deviation of the normal distribution, before truncation)
tf.Variable:
tf.constant:generating constant
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
"""

def weight_variable(shape):
   
    """

    Generating random number according to the tensor shape,The standard deviation is 0.1，define a variable function
    parameter_shape:the shape of the output tensor，A 1-D integer Tensor or Python array
    return:tf.Variable (initial_value is random number）A tensor of the specified shape filled with random truncated normal values.

    """
    initial = tf.random.truncated_normal(shape, stddev=0.1) #Outputs random values from a truncated normal distribution，shape:The shape of the output tensor.stddev:The standard deviation of the normal distribution, before truncation
    return tf.Variable(initial)


def bias_variable(shape):
    
    """

    Generating constant 0 according to the tensor shape,define a variable function
    parameter_shape:the shape of the output tensor，A 1-D integer Tensor or Python array
    return:tf.Variable（initial_value is a constant）A tensor of the specified shape filled with random truncated normal values.

    """
    initial = tf.constant(0.0, shape=shape) #generating constant
    return tf.Variable(initial)


def conv1d(x, W, s):
    
    """

    achieve the convolution
    input:
    parameter_x:the input image need to do the convolution,must a tensor format
    parameter_W:the core of the CNN,a tensor format
    parameter_s:the batch_size on each dimension,a one-dimensional vector
    return:tf.nn.conv2d(padding defines which convolution method will be used,"SAME"or"VALID")

    """
    return tf.nn.conv2d(input=x, filters=W, strides=s, padding='SAME')#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)


def pooling_layer_parameterized(pool_method, h_conv, pool_kernel, pool_stride):
    
    """

    define pooling parameter,which method should be used
    input: pool_method 1 or 2
    parameter_pool_method:1 or 2,determines use max_pool or avg_pool
    parameter_h_conv:the input need to be pooled,shape:[batch, height, width, channels]
    parameter_pool_kernel:int format,the height of the pool window.     ksize:[1,height,width,1],1-D CNN,width=1
    parameter_pool_stride:int format,size each window slides on each dimension.    strides:[1,stride,stride,1]
    return: pool_method=1,use the max set
            pool_method=2,use the average set

    """
    if pool_method == 1:
        return tf.nn.max_pool2d(input=h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1], padding='SAME')
    elif pool_method == 2:
        return tf.nn.avg_pool2d(input=h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1], padding='SAME')

def variable_summaries(var, name, collection):
    """
    
    Attach a lot of summaries to a Tensor.
    
    parameter_var:tensor format
    parameter_name:string format,the name of information
    parameter_collection:value=None
    
    display information of the tensor and draw a histogram about tensor and name
    """
    writer = tf.summary.create_file_writer("summary_file")
    #tf.summary.experimental.set_step()
    with tf.name_scope('summaries') as r:
        #sets a default value for the step parameter
        tf.summary.experimental.set_step(step=1) #An int64-castable default step value, or None to unset.
        # I make it 1 to test,thsi value is variable
        mean = tf.reduce_mean(input_tensor=var) #reduce the dimension of input tensor
        with writer.as_default():
            tf.summary.scalar('mean/' + name,mean,step=step)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        with writer.as_default():                                 #display the information
            tf.summary.scalar('stddev/' + name, stddev,step=step) #tf.summary.scalar(tags, values, step)
            tf.summary.scalar('max/' + name, tf.reduce_max(input_tensor=var),step=step)
            tf.summary.scalar('min/' + name, tf.reduce_min(input_tensor=var),step=step)
            tf.summary.histogram(name, var)
       

     """
    every tf.summary.scalar command needs a new parameter "step" but I have no idea how to pass value to this parameter,
    maybe use tf.summary.experimental.set_step()? 
    the parameter "collection" is not used, I changed this part using writer.as_default(), but I have no idea whether I keep the logic correct
    if these summary commands do not work well,we can use the following code:
    
    def variable_summaries(var, name, collection):
        with tf.compat.v1.name_scope('summaries') as r:
            mean = tf.reduce_mean(input_tensor=var)
            tf.compat.v1.add_to_collection(collection,tf.compat.v1.summary.scalar('mean/'+name,mean))
            with tf.compat.v1.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            tf.compat.v1.add_to_collection(collection,tf.compat.v1.summary.scalar('stddev/'+name, stddev))
            tf.compat.v1.add_to_collection(collection,tf.compat.v1.summary.scalar('max/'+name, tf.reduce_max(input_tensor=var)))
            tf.compat.v1.add_to_collection(collection,tf.compat.v1.summary.scalar('min/'+name, tf.reduce_min(input_tensor=var)))
            tf.compat.v1.add_to_collection(collection,tf.compat.v1.summary.histogram(name, var))
            
    tf.summary has many differences with tf.add_to_collection,using tf.compat.v1. can make tf.add_to_collection work with TF2.1
    this may need to be changed into TF2.1 totally in the future
            
    """

        

def build_model(hyperparameters):
    """
    This is Model_v5 from SDSS
    
    three conv layers,three pool layers,two full connected layers

    Parameters
    ----------
    hyperparameters：use the hyperparameters in dla_cnn/models/model_gensample_v7.1_hyperparams.json

    Returns
    -------

●   

    """
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    l2_regularization_penalty = hyperparameters['l2_regularization_penalty']
    fc1_n_neurons = hyperparameters['fc1_n_neurons']
    fc2_1_n_neurons = hyperparameters['fc2_1_n_neurons']
    fc2_2_n_neurons = hyperparameters['fc2_2_n_neurons']
    fc2_3_n_neurons = hyperparameters['fc2_3_n_neurons']
    conv1_kernel = hyperparameters['conv1_kernel']
    conv2_kernel = hyperparameters['conv2_kernel']
    conv3_kernel = hyperparameters['conv3_kernel']
    conv1_filters = hyperparameters['conv1_filters']
    conv2_filters = hyperparameters['conv2_filters']
    conv3_filters = hyperparameters['conv3_filters']
    conv1_stride = hyperparameters['conv1_stride']
    conv2_stride = hyperparameters['conv2_stride']
    conv3_stride = hyperparameters['conv3_stride']
    pool1_kernel = hyperparameters['pool1_kernel']
    pool2_kernel = hyperparameters['pool2_kernel']
    pool3_kernel = hyperparameters['pool3_kernel']
    pool1_stride = hyperparameters['pool1_stride']
    pool2_stride = hyperparameters['pool2_stride']
    pool3_stride = hyperparameters['pool3_stride']
    pool1_method = 1        # Removed from hyperparameter search because it never chose anything but this method
    pool2_method = 1
    pool3_method = 1

    INPUT_SIZE = hyperparameters['INPUT_SIZE'] # the pixel of input image,need a new hyperparameter
    tfo = {}    # Tensorflow objects

    x = tf.Variable(tf.ones(shape=[None, INPUT_SIZE]), name='x')
    label_classifier = tf.Variable(tf.zeros(shape=[None]), name='label_classifier')
    label_offset = tf.Variable(tf.zeros(shape=[None]), name='label_offset')
    label_coldensity = tf.Variable(tf.zeros(shape=[None]), name='label_coldensity')
    keep_prob = tf.Variable( name='keep_prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # I converted "tf.placeholder" to "tf.Variable" because tf.placeholder is not used anymore in TF2.1,but if tf.Variable is
    #incompatible with the nump version,you can still use tf.placeholder like the following code 
    """"
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
    label_classifier = tf.compat.v1.placeholder(tf.float32, shape=[None], name='label_classifier')
    label_offset = tf.compat.v1.placeholder(tf.float32, shape=[None], name='label_offset')
    label_coldensity = tf.compat.v1.placeholder(tf.float32, shape=[None], name='label_coldensity')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    """"
    

    x_4d = tf.reshape(x, [-1, INPUT_SIZE, 1, 1])

    # First Convolutional Layer
    # Kernel size (16,1)
    # Stride (4,1)
    # number of filters = 4 (features?)
    # Neuron activation = ReLU (rectified linear unit)
    W_conv1 = weight_variable([conv1_kernel, 1, 1, conv1_filters])
    b_conv1 = bias_variable([conv1_filters])

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
    stride2 = [1, conv2_stride, 1, 1]
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2, stride2) + b_conv2)
    h_pool2 = pooling_layer_parameterized(pool2_method, h_conv2, pool2_kernel, pool2_stride)

    # Third convolutional layer
    W_conv3 = weight_variable([conv3_kernel, 1, conv2_filters, conv3_filters])
    b_conv3 = bias_variable([conv3_filters])
    stride3 = [1, conv3_stride, 1, 1]
    h_conv3 = tf.nn.relu(conv1d(h_pool2, W_conv3, stride3) + b_conv3)
    h_pool3 = pooling_layer_parameterized(pool3_method, h_conv3, pool3_kernel, pool3_stride)


    # FC1: first fully connected layer, shared
    inputsize_fc1 = int(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(
        INPUT_SIZE / conv1_stride) / pool1_stride) / conv2_stride) / pool2_stride) / conv3_stride) / pool3_stride)) * conv3_filters
    # batch_size = x.get_shape().as_list()[0]
    h_pool3_flat = tf.reshape(h_pool3, [-1, inputsize_fc1])

    W_fc1 = weight_variable([inputsize_fc1, fc1_n_neurons])
    b_fc1 = bias_variable([fc1_n_neurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout FC1
    h_fc1_drop = tf.nn.dropout(h_fc1, 1 - (keep_prob))

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
    h_fc2_1_drop = tf.nn.dropout(h_fc2_1, 1 - (keep_prob))
    h_fc2_2_drop = tf.nn.dropout(h_fc2_2, 1 - (keep_prob))
    h_fc2_3_drop = tf.nn.dropout(h_fc2_3, 1 - (keep_prob))

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
    loss_classifier = tf.add(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_nn_classifier, labels=label_classifier),
                             l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                          tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
                             name='loss_classifier')
    loss_offset_regression = tf.add(tf.reduce_sum(input_tensor=tf.nn.l2_loss(y_nn_offset - label_offset)),
                                    l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                                 tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_2)),
                                    name='loss_offset_regression')
    epsilon = 1e-6
    loss_coldensity_regression = tf.reduce_sum(
        input_tensor=tf.multiply(tf.square(y_nn_coldensity - label_coldensity),
                    tf.compat.v1.div(label_coldensity,label_coldensity+epsilon)) +
        l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                     tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
        name='loss_coldensity_regression')

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    cost_all_samples_lossfns_AB = loss_classifier + loss_offset_regression
    cost_pos_samples_lossfns_ABC = loss_classifier + loss_offset_regression + loss_coldensity_regression
    # train_step_AB = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all_samples_lossfns_AB, global_step=global_step, name='train_step_AB')
    train_step_ABC = optimizer.minimize(cost_pos_samples_lossfns_ABC, global_step=global_step, name='train_step_ABC')
    # train_step_C = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_coldensity_regression, global_step=global_step, name='train_step_C')
    output_classifier = tf.sigmoid(y_nn_classifier, name='output_classifier')
    prediction = tf.round(output_classifier, name='prediction')
    correct_prediction = tf.equal(prediction, label_classifier)
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32), name='accuracy')
    rmse_offset = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(tf.subtract(y_nn_offset,label_offset))), name='rmse_offset')
    rmse_coldensity = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(tf.subtract(y_nn_coldensity,label_coldensity))), name='rmse_coldensity')

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

