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

  def _tf_compute_sampled_logits_helper(self, weights, biases, inputs, labels, num_sampled, num_classes, num_true=1,
                                        sampled_values=None, subtract_log_q=True, remove_accidental_hits=True,
                                        partition_strategy="mod", name=None):
    if labels.dtype != tf.int64:
      labels = tf.cast(labels, tf.int64)

    labels_flat = tf.reshape(labels, [-1])

    sampled, true_expected_count, sampled_expected_count = sampled_values

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = tf.concat([labels_flat, sampled], 0)

    # weights shape is [num_classes, dim]
    all_w = tf.nn.embedding_lookup(
      weights, all_ids, partition_strategy=partition_strategy)
    all_b = tf.nn.embedding_lookup(biases, all_ids)
    # true_w shape is [batch_size * num_true, dim]
    # true_b is a [batch_size * num_true] tensor
    true_w = tf.slice(
      all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
    true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    row_wise_dots = tf.multiply(
      tf.expand_dims(inputs, 1),
      tf.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = tf.reshape(row_wise_dots,
                                tf.concat([[-1], dim], 0))
    true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = tf.reshape(true_b, [-1, num_true])
    true_logits += true_b

    # Lookup weights and biases for sampled labels.
    #   sampled_w shape is [num_sampled, dim]
    #   sampled_b is a [num_sampled] float tensor
    sampled_w = tf.slice(
      all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
    sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # sampled_b has shape [num_sampled]
    # Apply X*W'+B, which yields [batch_size, num_sampled]
    sampled_logits = tf.matmul(
      inputs, sampled_w, transpose_b=True) + sampled_b

    if remove_accidental_hits:
      acc_hits = tf.nn.compute_accidental_hits(
        labels, sampled, num_true=num_true, seed=44)
      acc_indices, acc_ids, acc_weights = acc_hits

      # This is how SparseToDense expects the indices.
      acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = tf.reshape(
        tf.cast(acc_ids, tf.int32), [-1, 1])
      sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                 "sparse_indices")
      # Create sampled_logits_shape = [batch_size, num_sampled]
      sampled_logits_shape = tf.concat(
        [tf.shape(labels)[:1], tf.expand_dims(num_sampled, 0)],
        0)
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += tf.sparse_to_dense(
        sparse_indices,
        sampled_logits_shape,
        acc_weights,
        default_value=0.0,
        validate_indices=False)

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= tf.log(true_expected_count)
      sampled_logits -= tf.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = tf.concat([true_logits, sampled_logits], 1)
    # true_logits is a float tensor, ones_like(true_logits) is a float tensor
    # of ones. We then divide by num_true to ensure the per-example labels sum
    # to 1.0, i.e. form a proper probability distribution.
    out_labels = tf.concat([
      tf.ones_like(true_logits) / num_true,
      tf.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_labels, true_expected_count, sampled_expected_count

  def _compute_sampled_logits_helper(self,
                                     weights,
                                     biases,
                                     labels,
                                     num_true,
                                     inputs,
                                     num_sampled,
                                     num_classes,
                                     sampled_values=None,
                                     subtract_log_q=True,
                                     remove_accidental_hits=True,
                                     partition_strategy="mod",
                                     name=None):

    if labels.dtype != tf.int64:
      labels = tf.cast(labels, tf.int64)
    labels_flat = tf.reshape(labels, [-1])

    if num_true.dtype != tf.int64:
      num_true = tf.cast(num_true, tf.int64)
    num_true = tf.reshape(num_true, [-1])
    # tf.slice(all_ids, num_true_start[i], num_true[i]) will get the true classes of training instance i
    num_true_start = tf.cumsum(num_true, axis=0, exclusive=True)

    sampled, true_expected_count, sampled_expected_count = sampled_values

    all_ids = tf.concat([labels_flat, sampled], 0)

    all_w = tf.nn.embedding_lookup(weights, all_ids, partition_strategy=partition_strategy)
    all_b = tf.nn.embedding_lookup(biases, all_ids)

    # true_w shape is [total_num_true, dim]
    true_w = tf.slice(all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
    # true_b shape is [total_num_true]
    true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

    # print(inputs.get_shape(), num_true_start.get_shape(), num_true.get_shape())

    def _calc_true_logit(_, x):
      # used_input shape is [dim]
      # start_idx shape is [1]
      # num_true_val shape is [1]
      used_input, start_idx, num_true_len = x
      used_input = tf.reshape(used_input, [-1])

      # used_w shape is [num_true_val, dim]
      used_w = tf.slice(true_w, tf.stack([start_idx, 0]), tf.stack([num_true_len, -1]))

      # used_true_logits shape is [num_true_val]
      used_true_logits = tf.matmul(tf.reshape(used_input, [1, -1]), used_w, transpose_b=True)

      # create an update vector that updates [start_idx : (start_idx + num_true_len)]'s logits
      logit_update = tf.sparse_to_dense(sparse_indices=tf.range(start_idx, start_idx + num_true_len),
                                        output_shape=tf.shape(labels),
                                        sparse_values=tf.reshape(used_true_logits, [-1]),
                                        default_value=0.0,
                                        validate_indices=True)

      return logit_update

    # true_logits shape is [total_num_true]
    scan_res = tf.scan(_calc_true_logit,
                       [inputs,
                        tf.cast(num_true_start, tf.int32),
                        tf.cast(num_true, tf.int32)],
                       initializer=tf.zeros_like(labels, dtype=tf.float32),
                       parallel_iterations=10, back_prop=True, swap_memory=True,
                       name="scan_dynamic_softmax_true_logits")
    # print("scan res", scan_res.get_shape())
    true_logits = tf.reduce_sum(scan_res, axis=0)
    # print("reduce_sum true_logits", true_logits.get_shape())
    true_logits += true_b

    sampled_w = tf.slice(all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
    sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

    sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True) + sampled_b

    if remove_accidental_hits:
      acc_hits = self._compute_accidental_hits(labels_flat, num_true, sampled, seed=44)
      acc_indices, acc_ids, acc_weights = acc_hits
      acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
      sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1, 'sparse_indices')

      sampled_logits_shape = tf.concat([tf.shape(num_true), tf.expand_dims(num_sampled, 0)], 0);
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += tf.sparse_to_dense(
        sparse_indices,
        sampled_logits_shape,
        acc_weights,
        default_value=0.0,
        validate_indices=False)

    if subtract_log_q:
      true_logits -= tf.log(true_expected_count)
      sampled_logits -= tf.log(sampled_expected_count)
    return true_logits, sampled_logits, true_expected_count, sampled_expected_count

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

      if num_true.dtype != tf.int64:
        num_true = tf.cast(num_true, tf.int64)
      num_true = tf.reshape(num_true, [-1])
      # tf.slice(all_ids, num_true_start[i], num_true[i]) will get the true classes of training instance i
      num_true_start = tf.cumsum(num_true, axis=0, exclusive=True)

      if sampled_values is None:
        sampled_values = self._uniform_candidate_sampler(true_classes=labels,
                                                         num_true=num_true,
                                                         num_sampled=num_sampled,
                                                         unique=True,
                                                         range_max=num_classes)

      sampled, true_expected_count, sampled_expected_count = sampled_values

      all_ids = tf.concat([labels_flat, sampled], 0)

      all_w = tf.nn.embedding_lookup(weights, all_ids, partition_strategy=partition_strategy)
      all_b = tf.nn.embedding_lookup(biases, all_ids)

      # true_w shape is [total_num_true, dim]
      true_w = tf.slice(all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
      # true_b shape is [total_num_true]
      true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

      # print(inputs.get_shape(), num_true_start.get_shape(), num_true.get_shape())

      def _calc_true_logit(_, x):
        # used_input shape is [dim]
        # start_idx shape is [1]
        # num_true_val shape is [1]
        used_input, start_idx, num_true_len = x
        used_input = tf.reshape(used_input, [-1])

        # used_w shape is [num_true_val, dim]
        used_w = tf.slice(true_w, tf.stack([start_idx, 0]), tf.stack([num_true_len, -1]))

        # used_true_logits shape is [num_true_val]
        used_true_logits = tf.matmul(tf.reshape(used_input, [1, -1]), used_w, transpose_b=True)

        # create an update vector that updates [start_idx : (start_idx + num_true_len)]'s logits
        logit_update = tf.sparse_to_dense(sparse_indices=tf.range(start_idx, start_idx + num_true_len),
                                          output_shape=tf.shape(labels),
                                          sparse_values=tf.reshape(used_true_logits, [-1]),
                                          default_value=0.0,
                                          validate_indices=True)

        return logit_update

      # true_logits shape is [total_num_true]
      scan_res = tf.scan(_calc_true_logit,
                         [inputs,
                          tf.cast(num_true_start, tf.int32),
                          tf.cast(num_true, tf.int32)],
                         initializer=tf.zeros_like(labels, dtype=tf.float32),
                         parallel_iterations=10, back_prop=True, swap_memory=True,
                         name="scan_dynamic_softmax_true_logits")

      # print("scan res", scan_res.get_shape())
      true_logits = tf.reduce_sum(scan_res, axis=0)
      # print("reduce_sum true_logits", true_logits.get_shape())
      true_logits += true_b

      sampled_w = tf.slice(all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
      sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

      sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True) + sampled_b

      if remove_accidental_hits:
        acc_hits = self._compute_accidental_hits(labels_flat, num_true, sampled)
        acc_indices, acc_ids, acc_weights = acc_hits
        acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
        acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
        sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1, 'sparse_indices')

        sampled_logits_shape = tf.concat([tf.shape(num_true), tf.expand_dims(num_sampled, 0)], 0);
        if sampled_logits.dtype != acc_weights.dtype:
          acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
        sampled_logits += tf.sparse_to_dense(
          sparse_indices,
          sampled_logits_shape,
          acc_weights,
          default_value=0.0,
          validate_indices=False)

      if subtract_log_q:
        true_logits -= tf.log(true_expected_count)
        sampled_logits -= tf.log(sampled_expected_count)
      # print("true_logits ", true_logits.get_shape())

      return true_logits, sampled_logits

  def _softmax_corss_entropy_with_logits(self, true_logits, sampled_logits, num_true):

    num_true = tf.cast(num_true, tf.int32)
    num_true_start = tf.cumsum(num_true, axis=0, exclusive=True)

    zero_labels = tf.zeros(tf.shape(sampled_logits)[1:2])

    true_logits = tf.reshape(true_logits, [-1])

    def softmax_cross_entropy_helper(x):
      sampled_logits_row, start_idx, num_true_len = x

      # print(sampled_logits_row.get_shape(), start_idx.get_shape(), num_true_len.get_shape())
      # print(true_logits.get_shape())
      true_logits_row = tf.slice(true_logits, [start_idx], [num_true_len])
      # print(true_logits_row.get_shape())

      logits = tf.concat([true_logits_row, sampled_logits_row], axis=0)
      labels = tf.concat([tf.ones_like(true_logits_row) / tf.cast(num_true_len, tf.float32),
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
    # The softmax cross entropy is correct, the problem arises from compute_sampled_logits
    return self._softmax_corss_entropy_with_logits(true_logits=true_logits_flat, sampled_logits=sampled_logits,
                                                   num_true=num_true)
