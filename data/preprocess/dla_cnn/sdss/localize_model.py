""" This code was for TF 1.X and the SDSS model"""
#!python
from __future__ import print_function, absolute_import, division, unicode_literals

from dla_cnn.sdss.Model_v5 import build_model
import tensorflow as tf
import numpy as np
import random, os, sys, traceback, math, json, timeit, gc, multiprocessing, gzip, pickle
import peakutils, re, scipy, getopt, argparse, fasteners

# Mean and std deviation of distribution of column densities
# COL_DENSITY_MEAN = 20.488289796394628
# COL_DENSITY_STD = 0.31015769579662766
tensor_regex = re.compile('.*:\d*')



# def predictions_ann_multiprocess(param_tuple):
#     return predictions_ann(param_tuple[0], param_tuple[1], param_tuple[2], param_tuple[3])


def predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE=''):
    timer = timeit.default_timer()
    BATCH_SIZE = 4000
    n_samples = flux.shape[0]
    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred)

    with tf.Graph().as_default():
        build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.Session() as sess:
            tf.train.Saver().restore(sess, checkpoint_filename+".ckpt")
            for i in range(0,n_samples,BATCH_SIZE):
                pred[i:i+BATCH_SIZE], conf[i:i+BATCH_SIZE], offset[i:i+BATCH_SIZE], coldensity[i:i+BATCH_SIZE] = \
                    sess.run([t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                             feed_dict={t('x'):                 flux[i:i+BATCH_SIZE,:],
                                        t('keep_prob'):         1.0})

    print("Localize Model processed {:d} samples in chunks of {:d} in {:0.1f} seconds".format(
          n_samples, BATCH_SIZE, timeit.default_timer() - timer))

    # coldensity_rescaled = coldensity * COL_DENSITY_STD + COL_DENSITY_MEAN
    return pred, conf, offset, coldensity


def train_ann(hyperparameters, train_dataset, test_dataset, save_filename=None, load_filename=None, tblogs = "../tmp/tblogs", TF_DEVICE=''):
    training_iters = hyperparameters['training_iters']
    batch_size = hyperparameters['batch_size']
    dropout_keep_prob = hyperparameters['dropout_keep_prob']

    # Predefine variables that need to be returned from local scope
    best_accuracy = 0.0
    best_offset_rmse = 999999999
    best_density_rmse = 999999999
    test_accuracy = None
    loss_value = None

    with tf.Graph().as_default():
        train_step_ABC, tb_summaries = build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.Session() as sess:
            # Restore or initialize model
            if load_filename is not None:
                tf.train.Saver().restore(sess, load_filename+".ckpt")
                print("Model loaded from checkpoint: %s"%load_filename)
            else:
                print("Initializing variables")
                sess.run(tf.initialize_all_variables())

            summary_writer = tf.train.SummaryWriter(tblogs, sess.graph)

            for i in range(training_iters):
                batch_fluxes, batch_labels_classifier, batch_labels_offset, batch_col_density = train_dataset.next_batch(batch_size)
                sess.run(train_step_ABC, feed_dict={t('x'):                batch_fluxes,
                                                    t('label_classifier'): batch_labels_classifier,
                                                    t('label_offset'):     batch_labels_offset,
                                                    t('label_coldensity'): batch_col_density,
                                                    t('keep_prob'):        dropout_keep_prob})

                if i % 200 == 0:   # i%2 = 1 on 200ths iteration, that's important so we have the full batch pos/neg
                    train_accuracy, loss_value, result_rmse_offset, result_loss_offset_regression, \
                    result_rmse_coldensity, result_loss_coldensity_regression \
                        = train_ann_test_batch(sess, np.random.permutation(train_dataset.fluxes.shape[0])[0:10000],
                                               train_dataset.data, summary_writer=summary_writer)
                        # = train_ann_test_batch(sess, batch_ix, data_train)  # Note this batch_ix must come from train_step_ABC
                    print("step {:06d}, classify-offset-density acc/loss - RMSE/loss      {:0.3f}/{:0.3f} - {:0.3f}/{:0.3f} - {:0.3f}/{:0.3f}".format(
                          i, train_accuracy, float(np.mean(loss_value)), result_rmse_offset,
                             result_loss_offset_regression, result_rmse_coldensity,
                             result_loss_coldensity_regression))
                if i % 5000 == 0 or i == training_iters - 1:
                    test_accuracy, _, result_rmse_offset, _, result_rmse_coldensity, _ = train_ann_test_batch(
                        sess, np.arange(test_dataset.fluxes.shape[0]), test_dataset.data)
                    best_accuracy = test_accuracy if test_accuracy > best_accuracy else best_accuracy
                    best_offset_rmse = result_rmse_offset if result_rmse_offset < best_offset_rmse else best_offset_rmse
                    best_density_rmse = result_rmse_coldensity if result_rmse_coldensity < best_density_rmse else best_density_rmse
                    print("             test accuracy/offset RMSE/density RMSE:     {:0.3f} / {:0.3f} / {:0.3f}".format(
                          test_accuracy, result_rmse_offset, result_rmse_coldensity))
                    save_checkpoint(sess, (save_filename + "_" + str(i)) if save_filename is not None else None)

           # Save checkpoint
            save_checkpoint(sess, save_filename)

            return best_accuracy, test_accuracy, np.mean(loss_value), best_offset_rmse, result_rmse_offset, \
                   best_density_rmse, result_rmse_coldensity


def save_checkpoint(sess, save_filename):
    if save_filename is not None:
        tf.train.Saver().save(sess, save_filename + ".ckpt")
        with open(checkpoint_filename + "_hyperparams.json", 'w') as fp:
            json.dump(hyperparameters, fp)
        print("Model saved in file: %s" % save_filename + ".ckpt")


# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.get_default_graph().get_tensor_by_name(tensor_name)


# Called from train_ann to perform a test of the train or test data, needs to separate pos/neg to get accurate #'s
def train_ann_test_batch(sess, ixs, data, summary_writer=None):    #inputs: label_classifier, label_offset, label_coldensity
    MAX_BATCH_SIZE = 40000.0

    classifier_accuracy = 0.0
    classifier_loss_value = 0.0
    result_rmse_offset = 0.0
    result_loss_offset_regression = 0.0
    result_rmse_coldensity = 0.0
    result_loss_coldensity_regression = 0.0

    # split x and labels into pos/neg
    pos_ixs = ixs[data['labels_classifier'][ixs]==1]
    neg_ixs = ixs[data['labels_classifier'][ixs]==0]
    pos_ixs_split = np.array_split(pos_ixs, math.ceil(len(pos_ixs)/MAX_BATCH_SIZE)) if len(pos_ixs) > 0 else []
    neg_ixs_split = np.array_split(neg_ixs, math.ceil(len(neg_ixs)/MAX_BATCH_SIZE)) if len(neg_ixs) > 0 else []

    sum_samples = float(len(ixs))       # forces cast of percent len(pos_ix)/sum_samples to a float value
    sum_pos_only = float(len(pos_ixs))

    # Process pos and neg samples separately
    # pos
    for pos_ix in pos_ixs_split:
        tensors = [t('accuracy'), t('loss_classifier'), t('rmse_offset'), t('loss_offset_regression'),
                        t('rmse_coldensity'), t('loss_coldensity_regression'), t('global_step')]
        if summary_writer is not None:
            tensors.extend(tf.get_collection('SUMMARY_A'))
            tensors.extend(tf.get_collection('SUMMARY_B'))
            tensors.extend(tf.get_collection('SUMMARY_C'))
            tensors.extend(tf.get_collection('SUMMARY_shared'))

        run = sess.run(tensors, feed_dict={t('x'):                data['fluxes'][pos_ix],
                                           t('label_classifier'): data['labels_classifier'][pos_ix],
                                           t('label_offset'):     data['labels_offset'][pos_ix],
                                           t('label_coldensity'): data['col_density'][pos_ix],
                                           t('keep_prob'):        1.0})
        weight_wrt_total_samples = len(pos_ix)/sum_samples
        weight_wrt_pos_samples = len(pos_ix)/sum_pos_only
        # print "DEBUG> Processing %d positive samples" % len(pos_ix), weight_wrt_total_samples, sum_samples, weight_wrt_pos_samples, sum_pos_only
        classifier_accuracy += run[0] * weight_wrt_total_samples
        classifier_loss_value += np.sum(run[1]) * weight_wrt_total_samples
        result_rmse_offset += run[2] * weight_wrt_total_samples
        result_loss_offset_regression += run[3] * weight_wrt_total_samples
        result_rmse_coldensity += run[4] * weight_wrt_pos_samples
        result_loss_coldensity_regression += run[5] * weight_wrt_pos_samples

        for i in range(7, len(tensors)):
            summary_writer.add_summary(run[i], run[6])

    # neg
    for neg_ix in neg_ixs_split:
        tensors = [t('accuracy'), t('loss_classifier'), t('rmse_offset'), t('loss_offset_regression'), t('global_step')]
        if summary_writer is not None:
            tensors.extend(tf.get_collection('SUMMARY_A'))
            tensors.extend(tf.get_collection('SUMMARY_B'))
            tensors.extend(tf.get_collection('SUMMARY_shared'))

        run = sess.run(tensors, feed_dict={t('x'):                data['fluxes'][neg_ix],
                                           t('label_classifier'): data['labels_classifier'][neg_ix],
                                           t('label_offset'):     data['labels_offset'][neg_ix],
                                           t('keep_prob'):        1.0})
        weight_wrt_total_samples = len(neg_ix)/sum_samples
        # print "DEBUG> Processing %d negative samples" % len(neg_ix), weight_wrt_total_samples, sum_samples
        classifier_accuracy += run[0] * weight_wrt_total_samples
        classifier_loss_value += np.sum(run[1]) * weight_wrt_total_samples
        result_rmse_offset += run[2] * weight_wrt_total_samples
        result_loss_offset_regression += run[3] * weight_wrt_total_samples

        for i in range(5,len(tensors)):
            summary_writer.add_summary(run[i], run[4])

    return classifier_accuracy, classifier_loss_value, \
           result_rmse_offset, result_loss_offset_regression, \
           result_rmse_coldensity, result_loss_coldensity_regression


def load_and_save_current_best_hyperparameters(parameters, batch_results_file, best_param_ix):
    # Filename based on batch_results_file
    best_filename = os.path.splitext(batch_results_file)[0] + '_bestparams.pickle'

    # Lock file
    with fasteners.InterProcessLock(best_filename):
        if os.path.getsize(best_filename) == 0:
            # Create the file if it doesn't exist
            with open(best_filename, 'w') as fh:
                r = np.array([])
                for i in range(0,100):
                    r = np.hstack((r,np.random.permutation(len(parameters))))
                best_params = [p[0] for p in parameters]
                pickle.dump((best_params, r), fh)

        # Read current best parameters (pickle)
        with open(best_filename, 'r') as fh:
            (best_params, r) = pickle.load(fh)  # load the best parameters currently on disk

        # take all parameters from the version loaded from file except for the parameter we were updating here
        for j in range(len(best_params)):
            parameters[j][0] = best_params[j] if j != best_param_ix else parameters[j][0]

        # take the next index from file and roll it
        next_param_index = r[0]
        r = np.roll(r,1)

        # Save best parameters file (pickle)
        with open(best_filename, 'w') as fh:
            best_params_out = [p[0] for p in parameters]                # grab all best params
            best_params_out[best_param_ix] = parameters[best_param_ix][0]  # update the file with this rounds best choice param
            pickle.dump((best_params_out, r), fh)

    # Return updated parameters and next parameter index
    return parameters, int(next_param_index)


if __name__ == '__main__':
    #
    # Execute batch mode
    #
    from Dataset import Dataset

    DEFAULT_TRAINING_ITERS = 30000
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--hyperparamsearch', help='Run hyperparam search', required=False, action='store_true', default=False)
    parser.add_argument('-o', '--output_file', help='Hyperparam csv result file location', required=False, default='../tmp/batch_results.csv')
    parser.add_argument('-i', '--iterations', help='Number of training iterations', required=False, default=DEFAULT_TRAINING_ITERS)
    parser.add_argument('-l', '--loadmodel', help='Specify a model name to load', required=False, default=None)
    parser.add_argument('-c', '--checkpoint_file', help='Name of the checkpoint file to save (without file extension)', required=False, default="../models/training/current")
    parser.add_argument('-r', '--train_dataset_filename', help='File name of the training dataset without extension', required=False, default="../data/gensample/train_*.npz")
    parser.add_argument('-e', '--test_dataset_filename', help='File name of the testing dataset without extension', required=False, default="../data/gensample/test_mix_23559.npz")
    args = vars(parser.parse_args())

    RUN_SINGLE_ITERATION = not args['hyperparamsearch']
    checkpoint_filename = args['checkpoint_file'] if RUN_SINGLE_ITERATION else None
    batch_results_file = args['output_file']
    tf.logging.set_verbosity(tf.logging.DEBUG)

    train_dataset = Dataset(args['train_dataset_filename'])
    test_dataset = Dataset(args['test_dataset_filename'])

    exception_counter = 0
    iteration_num = 0

    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty", "dropout_keep_prob",
                       "fc1_n_neurons", "fc2_1_n_neurons", "fc2_2_n_neurons", "fc2_3_n_neurons",
                       "conv1_kernel", "conv2_kernel", "conv3_kernel",
                       "conv1_filters", "conv2_filters", "conv3_filters",
                       "conv1_stride", "conv2_stride", "conv3_stride",
                       "pool1_kernel", "pool2_kernel", "pool3_kernel",
                       "pool1_stride", "pool2_stride", "pool3_stride"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try

        # learning_rate
        [0.00002,         0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070],
        # training_iters
        [int(args['iterations'])],
        # batch_size
        [700,           400, 500, 600, 700, 850, 1000],
        # l2_regularization_penalty
        [0.005,         0.01, 0.008, 0.005, 0.003],
        # dropout_keep_prob
        [0.98,          0.75, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [350,           200, 350, 500, 700, 900, 1500],
        # fc2_1_n_neurons
        [200,           200, 350, 500, 700, 900, 1500],
        # fc2_2_n_neurons
        [350,           200, 350, 500, 700, 900, 1500],
        # fc2_3_n_neurons
        [150,           200, 350, 500, 700, 900, 1500],
        # conv1_kernel
        [32,            20, 22, 24, 26, 28, 32, 40, 48, 54],
        # conv2_kernel
        [16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv3_kernel
        [16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv1_filters
        [100,           64, 80, 90, 100, 110, 120, 140, 160, 200],
        # conv2_filters
        [96,            80, 96, 128, 192, 256],
        # conv3_filters
        [96,            80, 96, 128, 192, 256],
        # conv1_stride
        [3,             2, 3, 4, 5, 6, 8],
        # conv2_stride
        [1,             1, 2, 3, 4, 5, 6],
        # conv3_stride
        [1,             1, 2, 3, 4, 5, 6],
        # pool1_kernel
        [7,             3, 4, 5, 6, 7, 8, 9],
        # pool2_kernel
        [6,             4, 5, 6, 7, 8, 9, 10],
        # pool3_kernel
        [6,             4, 5, 6, 7, 8, 9, 10],
        # pool1_stride
        [4,             1, 2, 4, 5, 6],
        # pool2_stride
        [4,             1, 2, 3, 4, 5, 6, 7, 8],
        # pool3_stride
        [4,             1, 2, 3, 4, 5, 6, 7, 8]
    ]

    # Random permutation of parameters out some artibrarily long distance
    r = np.random.permutation(1000)

    # Write out CSV header
    os.remove(batch_results_file) if os.path.exists(batch_results_file) else None
    with open(batch_results_file, "a") as csvoutput:
        csvoutput.write("iteration_num,normalized_score,best_accuracy,last_accuracy,last_objective,best_offset_rmse,last_offset_rmse,best_coldensity_rmse,last_coldensity_rmse," + ",".join(parameter_names) + "\n")

    while (RUN_SINGLE_ITERATION and iteration_num < 1) or not RUN_SINGLE_ITERATION:
        iteration_best_result = 9999999
        i = -1      # index of next parameter to change

        # # For distributed HP search we save the list of current best hyperparameters & load results that might
        # # have been generated by another iteration using the same batch_results_file
        # if not RUN_SINGLE_ITERATION:
        #     (parameters, i) = load_and_save_current_best_hyperparameters(parameters, batch_results_file, i)

        i = r[0]%len(parameters)    # Choose a random parameter to change
        r = np.roll(r,1)
        hyperparameters = {}    # Dictionary that stores all hyperparameters used in this iteration

        for j in (range(1,len(parameters[i])) if not RUN_SINGLE_ITERATION else range(0,1)):
            iteration_num += 1

            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][j] if i == k else parameters[k][0]

            try:
                # Print parameters during training
                print("----------------------------------------------------------------------------------------------")
                print("PARAMETERS (varying %s): " % parameter_names[i])
                for k in range(0,len(parameters)):
                    sys.stdout.write("{:<30}{:<15}\n".format( parameter_names[k], parameters[k][j] if k==i else parameters[k][0] ))

                # ANN Training
                (best_accuracy, last_accuracy, last_objective, best_offset_rmse, last_offset_rmse, best_coldensity_rmse,
                 last_coldensity_rmse) = train_ann(hyperparameters, train_dataset, test_dataset,
                                                   save_filename=checkpoint_filename, load_filename=args['loadmodel'])

                # These mean & std dev are used to normalize the score from all 3 loss functions for hyperparam optimize
                mean_best_accuracy = 0.02
                std_best_accuracy = 0.0535
                mean_best_offset_rmse = 4.0
                std_best_offset_rmse = 1.56
                mean_best_coldensity_rmse = 0.33
                std_best_coldensity_rmse = 0.1
                normalized_score = (1-best_accuracy-mean_best_accuracy)/std_best_accuracy + \
                                   (best_offset_rmse-mean_best_offset_rmse)/std_best_offset_rmse + \
                                   (best_coldensity_rmse-mean_best_coldensity_rmse)/std_best_coldensity_rmse

                # Save results and parameters to CSV
                with open(batch_results_file, "a") as csvoutput:
                    csvoutput.write("%d,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f" % \
                                    (iteration_num, normalized_score, best_accuracy, last_accuracy,
                                     float(last_objective), best_offset_rmse, last_offset_rmse,
                                     best_coldensity_rmse, last_coldensity_rmse) )
                    for hp in parameter_names:
                        csvoutput.write(",%07f" % hyperparameters[hp])
                    csvoutput.write("\n")

                # Keep a running tab of the best parameters based on overall accuracy
                if normalized_score <= iteration_best_result:
                    iteration_best_result = normalized_score
                    parameters[i][0] = parameters[i][j]
                    print("Best result for parameter [%s] with iteration_best_result [%0.2f] now set to [%f]" % (parameter_names[i], iteration_best_result, parameters[i][0]))

                exception_counter = 0   # reset exception counter on success

            # Log and ignore exceptions
            except Exception as e:
                exception_counter += 1
                with open(batch_results_file, "a") as csvoutput:
                    csvoutput.write("%d,error,error,error,error,error,error,error,error" % iteration_num)
                    for hp in parameter_names:
                        csvoutput.write(",%07f" % hyperparameters[hp])
                    csvoutput.write("\n")

                traceback.print_exc()
                if exception_counter > 20:
                    exit()                  # circuit breaker

