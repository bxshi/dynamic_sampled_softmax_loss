#define EIGEN_USE_THREADS

#include <cfloat>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "dynamic_range_sampler.h"
#include "tensorflow/core/util/guarded_philox_random.h"

using namespace tensorflow;

class BaseDynamicCandidateSamplerOp : public OpKernel {
public:
    explicit BaseDynamicCandidateSamplerOp(OpKernelConstruction *context)
        : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("num_sampled", &num_sampled_));
      OP_REQUIRES_OK(context, context->GetAttr("unique", &unique_));
      OP_REQUIRES_OK(context, generator_.Init(context));
    }

    void Compute(OpKernelContext *context) override {
      const Tensor &true_classes = context->input(0);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(true_classes.shape()),
                  errors::InvalidArgument("true_classes must be a vector"));

      // total number of true classes for this batch
      // this should equals to sum(num_true).
      const uint64 n_true_classes = uint64(true_classes.dim_size(0));

      const Tensor &num_true = context->input(1);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(num_true.shape()),
                  errors::InvalidArgument("num_true must be a vector"));
      const int32 batch_size = num_true.dim_size(0);

      Tensor *out_sampled_candidates = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({num_sampled_}),
                                              &out_sampled_candidates));

      Tensor *out_true_expected_count = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(1, true_classes.shape(),
                                              &out_true_expected_count));

      Tensor *out_sampled_expected_count = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(2, TensorShape({num_sampled_}),
                                              &out_sampled_expected_count));

      gtl::ArraySlice<int64> true_candidate(true_classes.vec<int64>().data(), n_true_classes);
      gtl::MutableArraySlice<int64> sampled_candidate(out_sampled_candidates->vec<int64>().data(),
                                                      uint64(num_sampled_));
      gtl::MutableArraySlice<float> true_expected_count(out_true_expected_count->vec<float>().data(), n_true_classes);
      gtl::MutableArraySlice<float> sampled_expected_count(out_sampled_expected_count->vec<float>().data(),
                                                           num_sampled_);

      CHECK(sampler_) << "CandidateSamplerOp did not set sampler_";

      // Approximately conservatively estimate the number of samples required.
      // In cases where rejection sampling is used we may occasionally use more
      // samples than expected, which will result in reused random bits.
      const int64 samples32 = 2048 * num_sampled_;

      // Pick sampled candidates.
      auto local_gen = generator_.ReserveSamples32(samples32);
      random::SimplePhilox random(&local_gen);
      sampler_->SampleBatchGetExpectedCount(&random, unique_, &sampled_candidate,
                                            &sampled_expected_count,
                                            true_candidate, &true_expected_count);

      if (sampler_->NeedsUpdates()) {
        sampler_->Update(true_candidate);
      }

    }

protected:
    void set_sampler(DynamicRangeSampler *sampler) { sampler_.reset(sampler); }

private:
    int32 num_sampled_;
    bool unique_;
    std::unique_ptr<DynamicRangeSampler> sampler_;
    GuardedPhiloxRandom generator_;
};

template<class DynamicRangeSamplerType>
class SimpleDynamicCandidateSamplerOp : public BaseDynamicCandidateSamplerOp {
public:
    explicit SimpleDynamicCandidateSamplerOp(OpKernelConstruction *context) : BaseDynamicCandidateSamplerOp(context) {
      int64 range_max;
      OP_REQUIRES_OK(context, context->GetAttr("range_max", &range_max));
      set_sampler(new DynamicRangeSamplerType(range_max));
    }
};

REGISTER_KERNEL_BUILDER(Name("DynamicUniformCandidateSampler").Device(DEVICE_CPU),
                        SimpleDynamicCandidateSamplerOp<DynamicUniformSampler>);

class ComputeDynamicAccidentalHitsOp : public OpKernel {
public:
    explicit ComputeDynamicAccidentalHitsOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
      const Tensor &in_true_candidates = context->input(0);
      const TensorShape &in_true_candidates_shape = in_true_candidates.shape();
      OP_REQUIRES(context, TensorShapeUtils::IsVector(in_true_candidates_shape),
                  errors::InvalidArgument("true_candidates must be a vector"));

      const Tensor &num_true = context->input(1);

      const int64 batch_size = num_true.dim_size(0);
      const int64 total_true = in_true_candidates.dim_size(0);

      const Tensor &in_sampled_candidates = context->input(2);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(in_sampled_candidates.shape()),
                  errors::InvalidArgument("sampled_candidates must be a vector, which is typically "
                                              "an output from CandidateSampler"));

      std::unordered_map<int64, int64> sampled_candidate_to_pos;
      for (int64 i = 0; i < in_sampled_candidates.dim_size(0); ++i) {
        // candidate_id -> idx in in_sampled_candidates
        sampled_candidate_to_pos[in_sampled_candidates.vec<int64>()(i)] = i;
      }

      std::vector<int> indices;
      std::vector<int64> ids;
      std::vector<float> weights;

      int64 element_counter = 0;
      int batch_id = 0;
      int64 elements_in_batch = num_true.vec<int64>()(batch_id);

      // check all true candidates, if they are selected in sampled_candidate
      // 1. push_back the batch id into indices
      // 2. push_back the index of such candidate in sampled_candidate into ids
      // 3. push -FLT_MAX into weights
      for (int64 i = 0; i < total_true; ++i) {
        if (element_counter == elements_in_batch) {
          element_counter = 0;
          ++batch_id;
          elements_in_batch = num_true.vec<int64>()(batch_id);
        }

        const int64 true_candidate = in_true_candidates.vec<int64>()(i);
        const auto look = sampled_candidate_to_pos.find(true_candidate);

        if (look != sampled_candidate_to_pos.end()) {
          indices.push_back(batch_id);
          ids.push_back(look->second);
          weights.push_back(-FLT_MAX);
        }

        ++element_counter;
      }

      Tensor *out_indices = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({static_cast<int>(indices.size())}), &out_indices));

      Tensor *out_ids = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(
          1, TensorShape({static_cast<int>(ids.size())}), &out_ids));

      Tensor *out_weights = nullptr;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(
              2, TensorShape({static_cast<int>(weights.size())}), &out_weights));

      for (size_t i = 0; i < indices.size(); ++i) {
        out_indices->vec<int32>()(i) = indices[i];
        out_ids->vec<int64>()(i) = ids[i];
        out_weights->vec<float>()(i) = weights[i];
      }
    }

};

REGISTER_KERNEL_BUILDER(Name("ComputeDynamicAccidentalHits").Device(DEVICE_CPU),
                        ComputeDynamicAccidentalHitsOp);