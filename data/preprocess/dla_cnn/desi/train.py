""" Methods for training the Model"""
#!python

"""
0. Update all methods to TF 2.1 including using tf.Dataset
"""
import numpy as np
import math
import re

from dla_cnn.desi.model import build_model


tensor_regex = re.compile('.*:\d*')
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



def train_ann(hyperparameters, train_dataset, test_dataset, save_filename=None, load_filename=None, tblogs = "../tmp/tblogs", TF_DEVICE=''):
    """
    Perform training

    Parameters
    ----------
    hyperparameters
    train_dataset
    test_dataset
    save_filename
    load_filename
    tblogs
    TF_DEVICE

    Returns
    -------

    """
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
                # Grab a batch
                batch_fluxes, batch_labels_classifier, batch_labels_offset, batch_col_density = train_dataset.next_batch(batch_size)
                # Train
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
                # Run on test
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



def calc_normalized_score(best_accuracy, best_offset_rmse, best_coldensity_rmse):
    """
    Generate the normalized score from the 3 loss functions

    This should be used for a hyperparameter search

    Parameters
    ----------
    best_accuracy
    best_offset_rmse
    best_coldensity_rmse

    Returns
    -------

    """
    # These mean & std dev are used to normalize the score from all 3 loss functions for hyperparam optimize
    # TODO -- Reconsider these for DESI
    mean_best_accuracy = 0.02
    std_best_accuracy = 0.0535
    mean_best_offset_rmse = 4.0
    std_best_offset_rmse = 1.56
    mean_best_coldensity_rmse = 0.33
    std_best_coldensity_rmse = 0.1
    normalized_score = (1 - best_accuracy - mean_best_accuracy) / std_best_accuracy + \
                       (best_offset_rmse - mean_best_offset_rmse) / std_best_offset_rmse + \
                       (best_coldensity_rmse - mean_best_coldensity_rmse) / std_best_coldensity_rmse
    #
    return normalized_score


def main():
    # ANN Training
    (best_accuracy, last_accuracy, last_objective, best_offset_rmse, last_offset_rmse, best_coldensity_rmse,
     last_coldensity_rmse) = train_ann(hyperparameters, train_dataset, test_dataset,
                                       save_filename=checkpoint_filename, load_filename=args['loadmodel'])

