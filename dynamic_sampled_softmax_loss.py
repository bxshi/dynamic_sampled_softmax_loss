import tensorflow as tf


def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = tf.shape(x)[1]
    ones_shape = tf.stack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])


class DynamicSampledSoftmaxLoss(object):
    def __init__(self, lib_path="./cmake-build-debug/libdynamic_sampled_softmax_loss.dylib"):
        self.__so_module = tf.load_op_library(lib_path)

    def _uniform_candidate_sampler(self, true_classes, num_true, num_sampled, unique,
                                   range_max, seed=None, name=None):
        """Generates labels for candidate sampling with a uniform distribution.

        For each batch, this op picks a single set of sampled candidate labels.
        The difference between this function and the UniformCandidateSampler in
        TensorFlow is this function supports dynamic number of true classes per
        training instance or entry. This is sometimes useful when predicting one
        to many relations such as predicting cities in <city, located_at, country>.

        An example is, suppose we have three training instances, where

        1. the first one has two positive labels 1 and 2
        2. the second one has five positive labels 1,2,3,4,5
        3. the third one has one positive label 6

        Then `true_classes` is the flatten vector of all training instances,
        `[1,2,1,2,3,4,5,6]`. And `num_true` is a 1-D vector contains the
        number of true classes per instance. In this case `num_true` equals
        `[2,5,1]`.

        Args:
          true_classes: A flatten 1-D vector, in which contains the IDs of target_classes.
          num_true: A 1-D vector that contains number of true labels per context.
          num_sampled: Number of candidates to randomly sample per batch.
          unique: If unique is true, we sample with rejection, so that all sampled
            candidates in a batch are unique. This requires some approximation to
            estimate the post-rejection sampling probabilities.
          range_max: The sampler will sample integers from the interval [0, range_max).
          seed: If either seed or seed2 are set to be non-zero, the random number
            generator is seeded by the given seed.  Otherwise, it is seeded by a
            random seed.

        Returns:
          A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
          sampled_candidates: A `Tensor` of type `int64`. A vector of length `num_sampled`.
          true_expected_count: A `Tensor` of type `float32`. A `true_classes.shape()[0]` length vector.
          sampled_expected_count: A `Tensor` of type `float32`. A vector of length `num_sampled`.
        """

        seed1, seed2 = tf.get_seed(seed)
        if true_classes.dtype != tf.int64:
            true_classes = tf.cast(true_classes, tf.int64)
        if num_true.dtype != tf.int64:
            num_true = tf.cast(num_true, tf.int64)
        return self.__so_module.dynamic_uniform_candidate_sampler(true_classes=true_classes,
                                                                  num_true=num_true,
                                                                  num_sampled=num_sampled,
                                                                  unique=unique,
                                                                  range_max=range_max,
                                                                  seed=seed1,
                                                                  seed2=seed2,
                                                                  name=name)

    def _compute_accidental_hits(self, true_classes, num_true, sampled_candidates, seed=None, name=None):
        """ Compute accidental hits in sampled_candidates that in true_classes.
        Args:
          true_classes: A 1-D `Tensor` with type `int64`
          num_true: A 1-D `Tensor` with type `int64`
          sampled_candidates: A 1-D `Tensor` with type `iint64`
          seed: Random seed
          name:

        Returns:
          indices: The id of the row in the batch.
          ids: The index of the hit sampled_candidate.
          weights: Weights that will be applied onto the ids.

        """
        seed1, seed2 = tf.get_seed(seed)
        return self.__so_module.compute_dynamic_accidental_hits(true_classes=true_classes,
                                                                num_true=num_true,
                                                                sampled_candidates=sampled_candidates,
                                                                seed=seed1, seed2=seed2,
                                                                name=name)

    def _compute_sampled_logits(self,
                                weights,
                                biases,
                                labels,
                                num_true,
                                inputs,
                                num_sampled,
                                num_classes,
                                sampled_values=None,
                                subtract_log_q=True,
                                remove_accidental_hits=False,
                                partition_strategy="mod",
                                name=None):
        """Helper function for sampled_softmax_loss function.

        Computes sampled output training logits and labels suitable for implementing
        sampled softmax.

        Note: In the case where num_true[i] > 1, we assign to each target class the
        target probability 1 / num_true[i] so that the target probabilities sum to
        1 per-example.

        Args:
          weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
            objects whose concatenation along dimension 0 has shape `[num_classes, dim]`.
          biases:
          labels: A `Tensor` vector of type `int64`.
          num_true: A `Tensor` vector of type `int64` with length `batch_size`.
            `tf.reduce_sum(num_true)` should equals to `labels.shape()[0]`.
          inputs: A `Tensor` of shape `[batch_size, dim]`.
          num_sampled: An `int`.
          num_classes: An `int`.
          sampled_values: Keep this None right now
          subtract_log_q:
          remove_accidental_hits:
          partition_strategy:
          name:
        Returns:
          true_logits: A `Tensor` vector of shape [total_num_true]
          sampled_logits: A `Tensor` matrix of shape [batch_size, num_sampled]
        """

        if not isinstance(weights, list):
            weights = [weights]

        with tf.name_scope(name, 'compute_dynamic_sampled_logits', weights + [biases, inputs, labels]):
            if labels.dtype != tf.int64:
                labels = tf.cast(labels, tf.int64)
            labels_flat = tf.reshape(labels, [-1])

            if num_true.dtype != tf.int32:
                num_true = tf.cast(num_true, tf.int32)
            num_true = tf.reshape(num_true, [-1])

            # tf.slice(all_ids, num_true_start[i], num_true[i]) will get the true classes of training instance i
            num_true_start = tf.cumsum(num_true, axis=0, exclusive=True)

            total_num_true = tf.reduce_sum(num_true)

            # Create a [total_num_true] index vector containing indices in inputs
            empty_input_idx = tf.zeros_like(labels_flat, dtype=tf.int32)

            def _compute_input_idx(a, x):
                start_idx, current_len, current_val = tf.unstack(x, num=3, axis=0)
                update_indices = tf.range(start_idx,
                                          limit=start_idx + current_len,
                                          name="update_indices_range")
                return a + tf.sparse_to_dense(update_indices,
                                              tf.shape(a),
                                              current_val,
                                              default_value=0,
                                              validate_indices=True)

            input_idices_elems = tf.stack([num_true_start,
                                           num_true,
                                           tf.range(0, tf.shape(inputs)[0],
                                                    dtype=tf.int32)],
                                          axis=1)

            # an index vector with shape [total_num_true]
            input_idices = tf.foldl(_compute_input_idx, input_idices_elems, initializer=empty_input_idx)

            if sampled_values is None:
                sampled_values = self._uniform_candidate_sampler(true_classes=labels,
                                                                 num_true=num_true,
                                                                 num_sampled=num_sampled,
                                                                 unique=True,
                                                                 range_max=num_classes)

            sampled, true_expected_count, sampled_expected_count = sampled_values

            all_ids = tf.concat_v2([labels_flat, sampled], 0)

            all_w = tf.nn.embedding_lookup(weights, all_ids, partition_strategy=partition_strategy)
            all_b = tf.nn.embedding_lookup(biases, all_ids)

            # true_w shape is [total_num_true, dim]
            true_w = tf.slice(all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
            # true_b shape is [total_num_true]
            true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

            # new_inputs, the shape is [total_num_true, dim]
            new_inputs = tf.nn.embedding_lookup(inputs, input_idices, partition_strategy=partition_strategy)
            row_wise_dots = tf.multiply(
                new_inputs,  # [total_num_true, dim]
                true_w  # [total_num_true, dim]
            )  # [total_num_true, dim]

            true_logits_flat = tf.reshape(_sum_rows(row_wise_dots), [-1])  # [total_num_true]
            true_logits_flat += true_b

            sampled_w = tf.slice(all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
            sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

            sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True) + sampled_b

            if remove_accidental_hits:
                acc_hits = self._compute_accidental_hits(labels_flat, tf.cast(num_true, tf.int64), sampled)
                acc_indices, acc_ids, acc_weights = acc_hits
                acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
                acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
                sparse_indices = tf.concat_v2([acc_indices_2d, acc_ids_2d_int32], 1, 'sparse_indices')

                sampled_logits_shape = tf.concat_v2([tf.shape(num_true), tf.expand_dims(num_sampled, 0)], 0);
                if sampled_logits.dtype != acc_weights.dtype:
                    acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
                sampled_logits += tf.sparse_to_dense(
                    sparse_indices,
                    sampled_logits_shape,
                    acc_weights,
                    default_value=0.0,
                    validate_indices=False)

            if subtract_log_q:
                true_logits_flat -= tf.log(true_expected_count)
                sampled_logits -= tf.log(sampled_expected_count)

            return true_logits_flat, sampled_logits

    def _softmax_corss_entropy_with_logits(self, true_logits, sampled_logits, num_true):

        num_true = tf.cast(num_true, tf.int32)
        num_true_start = tf.cumsum(num_true, axis=0, exclusive=True)

        zero_labels = tf.zeros(tf.shape(sampled_logits)[1:2])

        true_logits = tf.reshape(true_logits, [-1])

        def softmax_cross_entropy_helper(x):
            sampled_logits_row, start_idx, num_true_len = x

            true_logits_row = tf.slice(true_logits, start_idx, num_true_len)

            logits = tf.concat_v2([true_logits_row, sampled_logits_row], axis=0)
            labels = tf.concat_v2([tf.ones_like(true_logits_row) / tf.cast(num_true_len, tf.float32),
                                   zero_labels], axis=0)

            return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        return tf.map_fn(softmax_cross_entropy_helper, [sampled_logits, num_true_start, num_true],
                         dtype=tf.float32,
                         parallel_iterations=10, back_prop=True, swap_memory=True,
                         name="dynamic_softmax_cross_entropy")

    def sampled_softmax_loss(self, weights, biases, labels,
                             num_true, inputs, num_sampled,
                             num_classes, sampled_values=None,
                             remove_accidental_hits=True,
                             partition_strategy='mod',
                             name="sampled_softmax_loss"):

        true_logits_flat, sampled_logits = self._compute_sampled_logits(weights=weights, biases=biases,
                                                                        labels=labels, num_true=num_true,
                                                                        inputs=inputs, num_sampled=num_sampled,
                                                                        num_classes=num_classes,
                                                                        sampled_values=sampled_values,
                                                                        subtract_log_q=True,
                                                                        remove_accidental_hits=remove_accidental_hits,
                                                                        partition_strategy=partition_strategy,
                                                                        name=name)
        return self._softmax_corss_entropy_with_logits(true_logits=true_logits_flat, sampled_logits=sampled_logits,
                                                       num_true=num_true)
