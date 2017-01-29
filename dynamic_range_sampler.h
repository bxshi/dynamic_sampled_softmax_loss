#ifndef DYNAMIC_SAMPLED_SOFTMAX_LOSS_DYNAMICRANGESAMPLER_H
#define DYNAMIC_SAMPLED_SOFTMAX_LOSS_DYNAMICRANGESAMPLER_H

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/weighted_picker.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

class Env;

class DynamicRangeSampler {
public:
    explicit DynamicRangeSampler(int64 range) : range_(range) { CHECK_GT(range_, 0); }

    virtual ~DynamicRangeSampler();

    // Sample a single value
    virtual int64 Sample(random::SimplePhilox *rnd) const = 0;

    // The probability that a single call to Sample() returns the given value.
    // Assumes that value is in [0, range).  No range checking is done.
    virtual float Probability(int64 value) const = 0;

    // Fill "batch" with samples from the distribution.
    // If unique=true, then we re-pick each element until we get a
    // value distinct from all previously picked values in the batch.
    void SampleBatch(random::SimplePhilox *rnd, bool unique,
                     gtl::MutableArraySlice<int64> batch) const;

    // Fill "batch" with samples from the distribution, and report
    // "expected counts".
    //
    // The "expected count" of a value is an estimate of the expected
    // number of occurrences of the value in the batch returned by a
    // call to this function with the given parameters.  If unique=true,
    // the expected count is an inclusion probability.  For details on
    // this estimation, see the comment to "ExpectedCountHelper" in the
    // .cc file.
    //
    // Expected counts for the elements of the returned "batch" are reported
    // in the aligned array "batch_expected_count".
    //
    // The user can optionally provide "extras", containing values in the range.
    // The expected counts for the extras are reported in the aligned array
    // "extras_expected_count".
    //
    // "batch_expected_count" must have size equal to 0 or to the size of "batch".
    // "extras" and "extras_expected_count" must have equal size.
    void SampleBatchGetExpectedCount(
        random::SimplePhilox *rnd, bool unique,
        gtl::MutableArraySlice<int64> batch,
        gtl::MutableArraySlice<float> batch_expected_count,
        gtl::ArraySlice<int64> extras,
        gtl::MutableArraySlice<float> extras_expected_count) const;

    // Same as SampleBatchGetExpectedCount (see above), but with avoided values.
    // We repick to avoid all of the values in "avoided_values".
    // "avoided_values" is only supported with unique=true.  If
    // unique=false, then avoided_values must be empty.
    virtual void SampleBatchGetExpectedCountAvoid(
        random::SimplePhilox *rnd, bool unique,
        gtl::MutableArraySlice<int64> batch,
        gtl::MutableArraySlice<float> batch_expected_count,
        gtl::ArraySlice<int64> extras,
        gtl::MutableArraySlice<float> extras_expected_count,
        gtl::ArraySlice<int64> avoided_values) const;

    // Does this sampler need to be updated with values, e.g. UnigramSampler
    virtual bool NeedsUpdates() const { return false; }

    // Updates the underlying distribution
    virtual void Update(gtl::ArraySlice<int64> values) {
      LOG(FATAL) << "Update not supported for this sampler type.";
    }

    int64 range() { return range_; }


protected:
    const int64 range_;
};

class DynamicUniformSampler : public DynamicRangeSampler {
public:
    explicit DynamicUniformSampler(int64 range);
    ~DynamicUniformSampler() override {}

    int64 Sample(random::SimplePhilox* rnd) const override;

    float Probability(int64 value) const override;

private:
    const float inv_range_;
};

//TODO: Implement other samplers.

#endif //DYNAMIC_SAMPLED_SOFTMAX_LOSS_DYNAMICRANGESAMPLER_H
