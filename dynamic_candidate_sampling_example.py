import unittest

import numpy as np
import tensorflow as tf

from dynamic_sampled_softmax_loss import DynamicSampledSoftmaxLoss


class TestDynamicCandidateSamplingLoss(unittest.TestCase):
    def test_uniform_sampler_single_element(self):
        with tf.Session() as sess:
            dSampledLoss = DynamicSampledSoftmaxLoss("./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib")
            true_classes = tf.Variable([1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7], trainable=False, dtype=tf.int64)

            num_true = tf.Variable([1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1], trainable=False, dtype=tf.int64)
            samples, true_expected_count, samples_expected_count = \
                dSampledLoss._uniform_candidate_sampler(true_classes, num_true,
                                                        num_sampled=13, unique=True,
                                                        range_max=50, seed=12)

            tf_samples, tf_true_expected_count, tf_samples_expected_count = \
                tf.nn.uniform_candidate_sampler(true_classes=tf.expand_dims(true_classes, 1),
                                                num_true=1,
                                                num_sampled=13,
                                                unique=True,
                                                range_max=50,
                                                seed=12)

            sess.run(tf.global_variables_initializer())

            s, tf_s, ec, tf_ec, sc, tf_sc = sess.run([samples, tf_samples, true_expected_count,
                                                      tf_true_expected_count, samples_expected_count,
                                                      tf_samples_expected_count])

            np.testing.assert_equal(s, tf_s, err_msg="sampled_indices check")
            np.testing.assert_almost_equal(ec, np.reshape(tf_ec, [-1]), err_msg="true_expected_count check")
            np.testing.assert_almost_equal(sc, tf_sc, err_msg="samples_expected_count check")

    def test_uniform_sampler_multiple_element(self):
        with tf.Session() as sess:
            dSampledLoss = DynamicSampledSoftmaxLoss("./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib")
            true_classes = tf.Variable([1, 11,
                                        2, 21,
                                        3, 31,
                                        4, 41,
                                        5, 51,
                                        6, 61,
                                        7, 71], trainable=False, dtype=tf.int64)

            num_true = tf.Variable([2,
                                    2,
                                    2,
                                    2,
                                    2,
                                    2,
                                    2], trainable=False, dtype=tf.int64)
            samples, true_expected_count, samples_expected_count = \
                dSampledLoss._uniform_candidate_sampler(true_classes, num_true,
                                                        num_sampled=13, unique=True,
                                                        range_max=50, seed=12)

            tf_samples, tf_true_expected_count, tf_samples_expected_count = \
                tf.nn.uniform_candidate_sampler(true_classes=tf.reshape(true_classes, [-1, 2]),
                                                num_true=2,
                                                num_sampled=13,
                                                unique=True,
                                                range_max=50,
                                                seed=12)

            sess.run(tf.global_variables_initializer())

            s, tf_s, ec, tf_ec, sc, tf_sc = sess.run([samples, tf_samples, true_expected_count,
                                                      tf_true_expected_count, samples_expected_count,
                                                      tf_samples_expected_count])

            np.testing.assert_equal(s, tf_s, err_msg="sampled_indices check")
            np.testing.assert_almost_equal(ec, np.reshape(tf_ec, [-1]), err_msg="true_expected_count check")
            np.testing.assert_almost_equal(sc, tf_sc, err_msg="samples_expected_count check")

    @unittest.skip("TODO: implement dynamic n_element sampler test.")
    def test_uniform_sampler_dynamic_element(self):
        pass

    def test_compute_accidental_hit_single_element(self):
        with tf.Session() as sess:
            dSampledLoss = DynamicSampledSoftmaxLoss("./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib")
            true_classes = tf.Variable([1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7], trainable=False, dtype=tf.int64)

            num_true = tf.Variable([1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1], trainable=False, dtype=tf.int64)
            samples, true_expected_count, samples_expected_count = \
                dSampledLoss._uniform_candidate_sampler(true_classes, num_true,
                                                        num_sampled=13, unique=True,
                                                        range_max=50, seed=12)

            tf_samples, tf_true_expected_count, tf_samples_expected_count = \
                tf.nn.uniform_candidate_sampler(true_classes=tf.expand_dims(true_classes, 1),
                                                num_true=1,
                                                num_sampled=13,
                                                unique=True,
                                                range_max=50,
                                                seed=12)

            sess.run(tf.global_variables_initializer())

            s, tf_s, ec, tf_ec, sc, tf_sc = sess.run([samples, tf_samples, true_expected_count,
                                                      tf_true_expected_count, samples_expected_count,
                                                      tf_samples_expected_count])

            np.testing.assert_equal(s, tf_s, err_msg="sampled_indices check")
            np.testing.assert_almost_equal(ec, np.reshape(tf_ec, [-1]), err_msg="true_expected_count check")
            np.testing.assert_almost_equal(sc, tf_sc, err_msg="samples_expected_count check")

            tf_acc_hits = tf.nn.compute_accidental_hits(true_classes=tf.expand_dims(true_classes, 1),
                                                        sampled_candidates=tf_s,
                                                        num_true=1,
                                                        seed=13)

            acc_hits = dSampledLoss._compute_accidental_hits(true_classes=true_classes,
                                                             num_true=num_true,
                                                             sampled_candidates=s,
                                                             seed=13)

            indices, ids, weights = sess.run(acc_hits)
            tf_indices, tf_ids, tf_weights = sess.run(tf_acc_hits)

            np.testing.assert_equal(indices, tf_indices)
            np.testing.assert_equal(ids, tf_ids)
            np.testing.assert_almost_equal(weights, tf_weights)

    def test_compute_accidental_hit_multiple_element(self):
        with tf.Session() as sess:
            dSampledLoss = DynamicSampledSoftmaxLoss("./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib")
            true_classes = tf.Variable([1, 11,
                                        2, 21,
                                        3, 31,
                                        4, 41,
                                        5, 51,
                                        6, 61,
                                        7, 71], trainable=False, dtype=tf.int64)

            num_true = tf.Variable([2,
                                    2,
                                    2,
                                    2,
                                    2,
                                    2,
                                    2], trainable=False, dtype=tf.int64)
            samples, true_expected_count, samples_expected_count = \
                dSampledLoss._uniform_candidate_sampler(true_classes, num_true,
                                                        num_sampled=13, unique=True,
                                                        range_max=50, seed=12)

            tf_samples, tf_true_expected_count, tf_samples_expected_count = \
                tf.nn.uniform_candidate_sampler(true_classes=tf.reshape(true_classes, [-1, 2]),
                                                num_true=2,
                                                num_sampled=13,
                                                unique=True,
                                                range_max=50,
                                                seed=12)

            sess.run(tf.global_variables_initializer())

            s, tf_s, ec, tf_ec, sc, tf_sc = sess.run([samples, tf_samples, true_expected_count,
                                                      tf_true_expected_count, samples_expected_count,
                                                      tf_samples_expected_count])

            np.testing.assert_equal(s, tf_s, err_msg="sampled_indices check")
            np.testing.assert_almost_equal(ec, np.reshape(tf_ec, [-1]), err_msg="true_expected_count check")
            np.testing.assert_almost_equal(sc, tf_sc, err_msg="samples_expected_count check")

            tf_acc_hits = tf.nn.compute_accidental_hits(true_classes=tf.reshape(true_classes, [-1, 2]),
                                                        sampled_candidates=tf_s,
                                                        num_true=2,
                                                        seed=13)

            acc_hits = dSampledLoss._compute_accidental_hits(true_classes=true_classes,
                                                             num_true=num_true,
                                                             sampled_candidates=s,
                                                             seed=13)

            indices, ids, weights = sess.run(acc_hits)
            tf_indices, tf_ids, tf_weights = sess.run(tf_acc_hits)

            np.testing.assert_equal(indices, tf_indices)
            np.testing.assert_equal(ids, tf_ids)
            np.testing.assert_almost_equal(weights, tf_weights)

    def test_compute_accidental_hit_dynamic_element(self):
        pass

    @unittest.skip("")
    def test_optimize_single_element(self):
        with tf.Session() as sess:
            dSampledLoss = DynamicSampledSoftmaxLoss("./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib")

            true_classes = tf.Variable([1,
                                        2, 3,
                                        1, 2, 4,
                                        3, 4], trainable=False, dtype=tf.int64)
            num_true = tf.Variable([1,
                                    2,
                                    3,
                                    2], trainable=False, dtype=tf.int64)

            softmax_weights = tf.get_variable("w_opt_single", [10, 5], dtype=tf.float32)
            softmax_biases = tf.get_variable("b_opt_single", [10], dtype=tf.float32)
            inputs = tf.get_variable("inputs_opt_single", [4, 5], dtype=tf.float32)

            loss = dSampledLoss.sampled_softmax_loss(weights=softmax_weights,
                                                     biases=softmax_biases, labels=true_classes, num_true=num_true,
                                                     inputs=inputs,
                                                     num_sampled=5, num_classes=10,
                                                     sampled_values=dSampledLoss._uniform_candidate_sampler(
                                                         true_classes,
                                                         num_true,
                                                         num_sampled=5,
                                                         unique=True,
                                                         range_max=10,
                                                         seed=12),
                                                     remove_accidental_hits=True)

            opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            opt_op = opt.minimize(tf.reduce_mean(loss), var_list=[softmax_weights, softmax_biases, inputs])

            sess.run(tf.global_variables_initializer())

            for i in range(10000):
                l, _ = sess.run([tf.reduce_mean(loss), opt_op])
                print("%d: %.4f" % (i, l))


def test_compute_gradient_single_element(self):
    with tf.Session() as sess:
        dSampledLoss = DynamicSampledSoftmaxLoss("./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib")

        true_classes = tf.Variable([1, 2], trainable=False, dtype=tf.int64)

        num_true = tf.Variable([1, 1], trainable=False, dtype=tf.int64)

        softmax_weights = tf.get_variable("w_grad_single", [4, 5], dtype=tf.float32)
        softmax_biases = tf.get_variable("b_grad_single", [4], dtype=tf.float32)
        inputs = tf.get_variable("inputs_grad_single", [2, 5], dtype=tf.float32)

        loss = dSampledLoss.sampled_softmax_loss(weights=softmax_weights,
                                                 biases=softmax_biases, labels=true_classes, num_true=num_true,
                                                 inputs=inputs,
                                                 num_sampled=2, num_classes=4,
                                                 sampled_values=dSampledLoss._uniform_candidate_sampler(true_classes,
                                                                                                        num_true,
                                                                                                        num_sampled=2,
                                                                                                        unique=True,
                                                                                                        range_max=4,
                                                                                                        seed=12),
                                                 remove_accidental_hits=True)

        tf_loss = tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                             biases=softmax_biases,
                                             labels=tf.expand_dims(true_classes, 1),
                                             inputs=inputs,
                                             num_sampled=2,
                                             num_classes=4,
                                             num_true=1,
                                             sampled_values=tf.nn.uniform_candidate_sampler(
                                                 true_classes=tf.expand_dims(true_classes, 1),
                                                 num_true=1,
                                                 num_sampled=2,
                                                 unique=True,
                                                 range_max=4,
                                                 seed=12),
                                             remove_accidental_hits=True)

        sess.run(tf.global_variables_initializer())

        loss_res, tf_loss_res = sess.run([tf.reduce_mean(x) for x in [loss, tf_loss]])

        w_grad, b_grad, i_grad = tf.gradients(tf.reduce_mean(loss),
                                              [softmax_weights, softmax_biases, inputs])

        tf_w_grad, tf_b_grad, tf_i_grad = tf.gradients(tf.reduce_mean(tf_loss),
                                                       [softmax_weights, softmax_biases, inputs])

        w_grad_res, b_grad_res, i_grad_res, \
        tf_w_grad_res, tf_b_grad_res, tf_i_grad_res = sess.run(
            [w_grad, b_grad, i_grad, tf_w_grad, tf_b_grad, tf_i_grad])

        np.testing.assert_almost_equal(loss_res, tf_loss_res)

        for i, (grad_w_row, tf_grad_w_row) in enumerate(zip(w_grad_res, tf_w_grad_res)):
            for c, (elem, tf_elem) in enumerate(zip(grad_w_row, tf_grad_w_row)):
                np.testing.assert_almost_equal(elem, tf_elem,
                                               err_msg=(
                                                   "w_grad_res[%d][%d] and tf_w_grad_res[%d][%d] does not match" % (
                                                       i, c, i, c)))

        for c, (elem, tf_elem) in enumerate(zip(b_grad_res, tf_b_grad_res)):
            np.testing.assert_almost_equal(elem, tf_elem,
                                           err_msg=("b_grad_res[%d] and tf_b_grad_res[%d] does not match" % (c, c)))

            # Current test show that tf.reduce_sum(i_grad_res, axis=1) equals to tf_i_grad_res


# for i, (grad_input_row, tf_grad_input_row) in enumerate(zip(i_grad_res, tf_i_grad_res)):
#        for c, (elem, tf_elem) in enumerate(zip(grad_input_row, tf_grad_input_row)):
#          np.testing.assert_almost_equal(elem, tf_elem,
#                                         err_msg=(
#                                           "i_grad_res[%d][%d] and tf_i_grad_res[%d][%d] does not match" % (
#                                           i, c, i, c)))


if __name__ == '__main__':
    unittest.main()
