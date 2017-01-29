#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using ::shape_inference::DimensionHandle;
using ::shape_inference::InferenceContext;
using ::shape_inference::ShapeHandle;

namespace {
  Status DynamicCandidateSamplerShapeFn(InferenceContext *c) {
    int64 num_sampled;
    TF_RETURN_IF_ERROR(c->GetAttr("num_sampled", &num_sampled));

    ShapeHandle true_classes_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &true_classes_shape));

    ShapeHandle num_sampled_v = c->Vector(num_sampled);

    c->set_output(0, num_sampled_v);
    c->set_output(1, c->input(0));
    c->set_output(2, num_sampled_v);
    return Status::OK();
  }

  Status ComputeDynamicAccidentalHitsShapeFn(InferenceContext *c) {
    ShapeHandle true_classes;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &true_classes));

    ShapeHandle num_true;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &num_true));

    ShapeHandle v = c->Vector(InferenceContext::kUnknownDim);

    c->set_output(0, v);
    c->set_output(1, v);
    c->set_output(2, v);
    return Status::OK();
  }
}


REGISTER_OP("DynamicUniformCandidateSampler")
    .Input("true_classes: int64")
    .Input("num_true: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(DynamicCandidateSamplerShapeFn)
    .Doc(R"doc(
Generates labels for candidate sampling with a uniform distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

true_classes: A flatten 1-D vector, in which contains the IDs of target_classes.
num_true: A 1-D vector that contains number of true labels per context.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_sampled: Number of candidates to randomly sample per batch.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
range_max: The sampler will sample integers from the interval [0, range_max).
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("ComputeDynamicAccidentalHits")
    .Input("true_classes: int64")
    .Input("num_true: int64")
    .Input("sampled_candidates: int64")
    .Output("indices: int32")
    .Output("ids: int64")
    .Output("weights: float")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(ComputeDynamicAccidentalHitsShapeFn)
    .Doc(R"doc(
Computes the ids of the positions in sampled_candidates that match true_labels.

When doing log-odds NCE, the result of this op should be passed through a
SparseToDense op, then added to the logits of the sampled candidates. This has
the effect of 'removing' the sampled labels that match the true labels by
making the classifier sure that they are sampled labels.

true_classes: The true_classes output of UnpackSparseLabels.
num_true: Number of true labels per context.
sampled_candidates: The sampled_candidates output of CandidateSampler.
indices: A vector of indices corresponding to rows of true_candidates.
ids: A vector of IDs of positions in sampled_candidates that match a true_label
  for the row with the corresponding index in indices.
weights: A vector of the same length as indices and ids, in which each element
  is -FLOAT_MAX.
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");
