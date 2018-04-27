## TensorFlow op for dynamic number of true classes per instance.


### Usage:

Make sure you have TensorFlow installed, and if you are using an virtualenv, source your virtualenv first before compile the code.

```
cmake .
cmake --build .
```

You will get a `.so` or `.dylib` under the project root depending on your OS.

An example and test cases can be found in `dynamic_candidate_sampling_example.py`.

A modified version of this is used in [ConMask](https://github.com/bxshi/ConMask). If you use this code, please consider cite
`Shi B., Weninger T., Open-World Knowledge Graph Completion, AAAI 2018`. Thank you!
