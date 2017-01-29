#include "dynamic_range_sampler.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"


using namespace tensorflow;

using ::gtl::ArraySlice;
using ::gtl::MutableArraySlice;

DynamicRangeSampler::~DynamicRangeSampler() {}

void DynamicRangeSampler::SampleBatch(random::SimplePhilox *rnd, bool unique,
                                      gtl::MutableArraySlice<int64> batch) const {
  SampleBatchGetExpectedCount(
      rnd, unique, batch, gtl::MutableArraySlice<float>(),
      gtl::ArraySlice<int64>(), gtl::MutableArraySlice<float>());
}

void DynamicRangeSampler::SampleBatchGetExpectedCount(
    random::SimplePhilox *rnd, bool unique,
    gtl::MutableArraySlice<int64> batch,
    gtl::MutableArraySlice<float> batch_expected_count,
    gtl::ArraySlice<int64> extras,
    gtl::MutableArraySlice<float> extras_expected_count) const {
  SampleBatchGetExpectedCountAvoid(rnd, unique, batch, batch_expected_count,
                                   extras, extras_expected_count,
                                   gtl::ArraySlice<int64>());
}

namespace {
  // Approximates the expected count of a value in the output of SampleBatch.
//
// If unique=false, then this is (Probability(value) * batch_size)
//
// We use batch_size and num_tries, where num_tries is the observed number of
// tries it took to get batch_size unique values.
//
// Assuming (falsely) that the number of tries to get a batch of batch_size
// distinct values is _always_ num_tries, the probability that the value
// is in a batch is (1 - (1-p)^num_tries)
  static float ExpectedCountHelper(float p, int batch_size, int num_tries) {
    if (num_tries == batch_size) {
      // This shortcut will always be taken if unique=false
      return p * batch_size;
    }
    // numerically stable version of (1 - (1-p)^num_tries)
    return float(-expm1(num_tries * log1p(-p)));
  }
}

void DynamicRangeSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox *rnd, bool unique,
    gtl::MutableArraySlice<int64> batch,
    gtl::MutableArraySlice<float> batch_expected_count,
    gtl::ArraySlice<int64> extras,
    gtl::MutableArraySlice<float> extras_expected_count,
    gtl::ArraySlice<int64> avoided_values) const {
  const int batch_size = batch.size();
  int num_tries;

  if (unique) {
    CHECK_LE(batch_size + avoided_values.size(), range_);
    std::unordered_set<int64> used(batch_size);
    used.insert(avoided_values.begin(), avoided_values.end());
    int num_picked = 0;
    num_tries = 0;
    while (num_picked < batch_size) {
      num_tries++;
      CHECK_LT(num_tries, kint32max);
      int64 value = Sample(rnd);
      if (gtl::InsertIfNotPresent(&used, value)) {
        batch[num_picked++] = value;
      }
    }
  } else {
    CHECK_EQ(avoided_values.size(), size_t{0})
      << "avoided_values only supported with unique=true";
    for (int i = 0; i < batch_size; i++) {
      batch[i] = Sample(rnd);
    }
    num_tries = batch_size;
  }
  // Compute the expected counts of the batch and the extra values
  if (batch_expected_count.size() > 0) {
    CHECK_EQ(batch_size, batch_expected_count.size());
    for (int i = 0; i < batch_size; i++) {
      batch_expected_count[i] =
          ExpectedCountHelper(Probability(batch[i]), batch_size, num_tries);
    }
  }
  CHECK_EQ(extras.size(), extras_expected_count.size());
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] =
        ExpectedCountHelper(Probability(extras[i]), batch_size, num_tries);
  }
}

DynamicUniformSampler::DynamicUniformSampler(int64 range)
    : DynamicRangeSampler(range), inv_range_(float(1.0 / range)) {}

int64 DynamicUniformSampler::Sample(random::SimplePhilox* rnd) const {
  return rnd->Uniform64(uint64(range_));
}

float DynamicUniformSampler::Probability(int64 value) const { return inv_range_;}