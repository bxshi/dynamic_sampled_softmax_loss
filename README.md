## TensorFlow op for dynamic number of true classes per instance.


### Usage:

Make sure you have TensorFlow installed, and if you are using an virtualenv, source your virtualenv first before compile the code.

```
cmake .
cmake --build .
```

You will get a `.so` or `.dylib` under the project root depending on your OS.

An example and test cases can be found in `dynamic_candidate_sampling_example.py`.

This is created as a submodule for [ProjC](https://github.com/nddsg/ProjC), feel free to check that project!