#include <iomanip>
#include <stdio.h>
#include <locale>

#include "tsc_x86.h"
#include "measurement.h"

#include "../TAB/utility.h"

#define CYCLES_REQUIRED (1*1e9)  // number of cycles required for an experiment

/* Global vars, used to keep track of custom functions */
std::vector<comp_func> userFuncs;
std::vector<std::string> funcNames;
std::vector<size_t> funcOps;

/* output formatting */
struct thousands_sep : std::numpunct<char> {
protected:
  char do_thousands_sep() const override { return '\''; }
  std::string do_grouping() const override { return "\3"; }
};

/* ground truth implementation (direct conv in FP32) */
void kernel_base(TestCase tc, float* tensor_x, float* tensor_w, float* res) {  
  int64_t padded_h = tc.height + 2 * tc.padding;
  int64_t padded_w = tc.width + 2 * tc.padding;
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1;
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1;

  std::vector<float> px = DirectPad(tensor_x, tc.padding, tc.padding, tc.batch_size, tc.channels, tc.height, tc.width);
  DirectConv2d_FP32(px.data(), tensor_w, tc.stride, tc.stride, tc.batch_size, tc.channels, 
                    padded_h, padded_w, tc.num_kernels, tc.kernel_height, tc.kernel_width, res);
}

/*
  Registers a user function to be tested by the driver program.
  Registers a string description of the function as well
*/
void add_function(comp_func f, std::string name, size_t ops) {
  userFuncs.push_back(f);
  funcNames.push_back(name);
  funcOps.push_back(ops);
}

/*
  Computes, reports and returns the median timing information.
*/
Timing perf_test(comp_func f, TestCase tc, bool warmup, int reps) {
  long num_runs = 1;

  const int64_t total_size_x = tc.batch_size * tc.channels * tc.height * tc.width;
  const int64_t total_size_w = tc.num_kernels * tc.channels * tc.kernel_height * tc.kernel_width;

  const int64_t paddedh = tc.height + 2 * tc.padding; // height after zero padding
  const int64_t paddedw = tc.width + 2 * tc.padding;  // width  after zero padding

  const int64_t outh = (paddedh - tc.kernel_height) / tc.stride + 1; // The output height of y
  const int64_t outw = (paddedw - tc.kernel_width) / tc.stride + 1;  // The output width  of y

  const int64_t total_size_y = tc.batch_size * tc.num_kernels * outh * outw;

  float* x = (float*) malloc(total_size_x * sizeof(float));
  float* w = (float*) malloc(total_size_w * sizeof(float));
  float* y = (float*) malloc(total_size_y * sizeof(float));

  fill_rands(x, total_size_x);
  fill_rands(w, total_size_w);

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  if (warmup) {
    double multiplier = 1;
    double cycles = 0.;
    myInt64 start, end;

    do {
      num_runs = num_runs * multiplier;
      start = start_tsc();
      for (int i = 0; i < num_runs; i++) {
        f(tc, x, w, y);           
      }
      end = stop_tsc(start);

      cycles = (double) end;
      multiplier = (CYCLES_REQUIRED) / (cycles);
          
    } while (multiplier > 2);
  }

  std::vector<Timing> timings(reps, {0, 0, 0, 0});
  std::cout << "Runs per experiment: " << num_runs << " (" << (warmup ? "warmup" : "static" ) << ")" << std::endl;

  // Actual performance measurements repeated <reps> times.
  // We simply store all results and get median.
  Timing curr_timing;
  for (int j = 0; j < reps; j++) {
    for (int i = 0; i < num_runs; ++i) {
      curr_timing = f(tc, x, w, y);

      timings[j].quantize_ns += curr_timing.quantize_ns;    
      timings[j].img_2_row_ns += curr_timing.img_2_row_ns;
      timings[j].gemm_ns += curr_timing.gemm_ns;
      timings[j].prelu_ns += curr_timing.prelu_ns;
    }

    timings[j].quantize_ns /= num_runs;
    timings[j].img_2_row_ns /= num_runs;
    timings[j].gemm_ns /= num_runs;
    timings[j].prelu_ns /= num_runs;
  }

  destroy(x);
  destroy(w);
  destroy(y);

  return computeMedianTiming(timings);
}

int main(int argc, char **argv) {
  register_functions();
  std::cout << "Number of registered functions: " << userFuncs.size() << std::endl;

  /* Configuration of the test case(s) */
  std::vector<int64_t> inputs = {16, 28, 56, 112, 256, 386, 512, 750, 1024};  // enable for 3x3 cases
  // std::vector<int64_t> inputs = { 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000 };  // enable for 1x1 cases
  bool warmup = false;
  bool check_validity = true;
  bool verbose = true;
  bool separators = false;
  int reps = 5;
  TimingType timing_type = TimingType::NANO;

  int64_t bn = 1, c = 64, kn = 16, kh = 3, kw = 3, p = 1, s = 1;  // enable for 3x3
  // int64_t bn = 32, kh = 1, kw = 1, p = 0, s = 1, height = 1, width = 1;  // enable for 1x1

  const float relu_alpha = 1;
  const float q_threshold = 0.5;
  /* --------------------------------- */

  if (separators) {
    std::locale custom_locale(std::locale(""), new thousands_sep());

    // Use the custom locale for output
    std::cout.imbue(custom_locale);
  }

  std::string unit = "ms";
  if (timing_type == TimingType::MICRO) {
    unit = "microseconds";
  } else if (timing_type == TimingType::NANO) {
    unit = "ns";
  }

  if (!check_validity) {
    std::cout << "\033[1;31m" << "Warning! Running without validation." << "\033[0m" << std::endl;
  }

  int amount_inputs = inputs.size();
  int numFuncs = userFuncs.size();
  int64_t quantize_ns[numFuncs][amount_inputs];
  int64_t img2row_ns[numFuncs][amount_inputs];
  int64_t gemm_ns[numFuncs][amount_inputs];
  int64_t prelu_ns[numFuncs][amount_inputs];
  int64_t total_ns[numFuncs][amount_inputs];

  for (int i = 0; i < amount_inputs; i++) {
    int64_t input = inputs[i];
    std::cout << "\nInput size = " << input << std::endl;

    /* Create test cases */
    int64_t height = input;  // enable for 3x3
    int64_t width = input;  // enable for 3x3
    // int64_t c = input;  // enable for 1x1
    // int64_t kn = input;  // enable for 1x1

    TestCase tc = {bn, c, height, width, kn, kh, kw, p, s, relu_alpha, q_threshold, timing_type};  // enable for 3x3
    // TestCase tc = {bn, c, height, width, kn, kh, kw, p, s, relu_alpha, q_threshold, timing_type};  // enable for 1x1
      
    // Check validity of functions.
    if (check_validity) {
      const int64_t total_size_x = tc.batch_size * tc.channels * tc.height * tc.width;
      const int64_t total_size_w = tc.num_kernels * tc.channels * tc.kernel_height * tc.kernel_width;

      const int64_t paddedh = tc.height + 2 * tc.padding; // height after zero padding
      const int64_t paddedw = tc.width + 2 * tc.padding; // width  after zero padding

      const int64_t outh = (paddedh - tc.kernel_height + 1) / tc.stride; // The output height of y
      const int64_t outw = (paddedw - tc.kernel_width + 1) / tc.stride; // The output width  of y

      const int64_t total_size_y = tc.batch_size * tc.num_kernels * outh * outw;

      float* x = (float*) malloc(total_size_x * sizeof(float));
      float* w = (float*) malloc(total_size_w * sizeof(float));
      float* y = (float*) malloc(total_size_y * sizeof(float));
      float* y_ref = (float*) malloc(total_size_y * sizeof(float));

      fill_rands(x, total_size_x);
      fill_rands(w, total_size_w);

      kernel_base(tc, x, w, y_ref); 

      int cmp;
      for (int j = 0; j < numFuncs; j++) {
        comp_func f = userFuncs[j];
        zeros(y, total_size_y);
        f(tc, x, w, y);

        cmp = compare_tensors(y, y_ref, tc.batch_size, tc.num_kernels, outh, outw);

        if (cmp < 0) {
          std::cout << "\033[1;31m" << "The result of function " << funcNames[j] << " is not correct." << "\033[0m" << std::endl;
        }
      }

      destroy(x);
      destroy(w);
      destroy(y);
      destroy(y_ref);
    }

    Timing perf;
    for (int j = 0; j < numFuncs; j++) {
      std::cout << std::endl << "Running: " << funcNames[j] << std::endl;
      perf = perf_test(userFuncs[j], tc, warmup, reps);

      int64_t total = perf.quantize_ns + perf.img_2_row_ns + perf.gemm_ns + perf.prelu_ns;

      if (verbose) {
        std::cout << std::setw(10) << std::left << "quantize:" << std::setw(12) << std::right << perf.quantize_ns << " " << unit << std::endl;
        std::cout << std::setw(10) << std::left << "img2row:" << std::setw(12) << std::right << perf.img_2_row_ns << " " << unit << std::endl;
        std::cout << std::setw(10) << std::left << "gemm:" << std::setw(12) << std::right << perf.gemm_ns << " " << unit << std::endl;
        std::cout << std::setw(10) << std::left << "prelu:" << std::setw(12) << std::right << perf.prelu_ns << " " << unit << std::endl;
        std::cout << "-----------------------------" << std::endl;
        std::cout << std::setw(10) << std::left << "Total:" << std::setw(12) << std::right << total << " " << unit << std::endl;
      }
      
      quantize_ns[j][i] = perf.quantize_ns;
      img2row_ns[j][i] = perf.img_2_row_ns;
      gemm_ns[j][i] = perf.gemm_ns;
      prelu_ns[j][i] = perf.prelu_ns;
      total_ns[j][i] = total;
    }
  }

  std::cout << std::endl << "[----- Data Summary -----]" << std::endl;
  std::cout << "inputs: ";
  printArray(inputs.data(), amount_inputs);
  for (int j = 0; j < numFuncs; j++) {
    std::cout << std::endl << funcNames[j] << ":" << std::endl;
    std::cout << "quantize: ";
    printArray(quantize_ns[j], amount_inputs);
    std::cout << "img2row: ";
    printArray(img2row_ns[j], amount_inputs);
    std::cout << "gemm: ";
    printArray(gemm_ns[j], amount_inputs);
    std::cout << "prelu: ";
    printArray(prelu_ns[j], amount_inputs);
    std::cout << "total: ";
    printArray(total_ns[j], amount_inputs);
  }
  std::cout << "[------------------------]" << std::endl;
}