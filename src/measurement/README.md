## Measurement Overview
This folder consists of our measurement infrastructure. The most important file is ```measure_single.cpp``` and ```measure_multi.cpp```. Both contain a ```main()``` function that let's the user execute either a single- or multi-layer measurement for a particular strategy (i.e., implementation of TNN). For this purpose one needs to uncomment the correct lines in ```../CMakeLists.txt``` to build the correct binary. Further instructions can be found in the respective sections below.

### measure_GEMM.cpp
Contains the files that can be used to perform measurements on the implementations of GEMM. Follow these steps:
- In ```main()``` look for the configuration section.
- Specify the input sizes M, N, K and the function to test. The meaning of M, N, K is also explained.
- Go to ```../CMakeLists.txt``` and uncomment the gemm-relevent lines to build the ```measure_GEMM``` binary.
- run the code after compiling the correct file through ```./gemm (rep) (num)``` (where rep, num are optional).

### measure_multi.cpp
File used to perform multi-layer measurements. In order to perform a certain experiment, follow these steps:
- In ```main()``` look for the configuration section.
- Specify the input sizes. They can correspond, e.g., to the channels/KN and later be used to plot runtimes.
- There are different options, one can en- or disable: ```check_validity``` (check for correctness), ```verbose``` (verbose outputs), ```separators``` (separators for thousands), ```reps``` (number of times, the experiment is repeated and the median is taken), ```timing_type``` (type for the timing. One of: NANO, MICRO, MILLI).
- Set the values of the test case that are input independent. E.g., h = w = 1 in the 1x1 kernel case.
- Go to the section of the code that creates the test cases and create them according to your needs.
- run the code after compiling the correct file through ```./measure```

### measure_single.cpp
File used to perform single-layer measurements. In order to perform a certain experiment, follow these steps:
- In ```main()``` look for the configuration section.
- Specify the input sizes. They can correspond, e.g., to the channels/KN and later be used to plot runtimes.
- There are different options, one can en- or disable: ```check_validity``` (check for correctness), ```verbose``` (verbose outputs), ```separators``` (separators for thousands), ```reps``` (number of times, the experiment is repeated and the median is taken), ```timing_type``` (type for the timing. One of: NANO, MICRO, MILLI), ```warmup``` (measures the execution time and executes the code several times to achieve predefined total execution time).
- Set the values of the test case that are input independent. E.g., h = w = 1 in the 1x1 kernel case.
- Go to the section of the code that creates the test cases and create them according to your needs.
- run the code after compiling the correct file through ```./measure```

### measurment.cpp
Some shared code used by single- and multi-layer measurments.

### strategies.cpp
In this file we register end-to-end version of the TNN algorithm. They are used for timing measurements by the measurement infrastructure. 

### tsc_x86.h
Helper for measuring cycles. 