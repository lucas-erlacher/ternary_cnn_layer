#include <iomanip>
#include <stdio.h>
#include <locale>

#include "measurement.h"

#include "../TAB/utility.h"

/* Global vars, used to keep track of custom functions */
std::vector<std::pair<comp_func, comp_func>> funcs;
std::vector<std::string> funcNames;

/* output formatting */
struct thousands_sep : std::numpunct<char> {
protected:
  char do_thousands_sep() const override { return '\''; }
  std::string do_grouping() const override { return "\3"; }
};

/*
  Registers a user function pair to be tested by the driver program.
  Registers a string description of the function as well
*/
void addPair(comp_func f1, comp_func f2, std::string name) {
  funcs.push_back(std::make_pair(f1, f2));
  funcNames.push_back(name);
}

/*
  Computes, reports and returns the median timing information.
*/
Timing perf_test(std::pair<comp_func, comp_func> funcPair, std::vector<TestCase> cases, int num_cases, int reps) {
  const long num_runs = 1;

  if (num_cases <= 0) {
    return {0, 0, 0, 0};
  }

  int64_t total_size_x = 0;
  int64_t total_size_w = 0;
  for (int i = 0; i < num_cases; i++) {
    TestCase tc = cases[i];
    total_size_x += tc.batch_size * tc.channels * tc.height * tc.width;
    total_size_w += tc.num_kernels * tc.channels * tc.kernel_height * tc.kernel_width;
  }

  // size of y only depends on last test case
  const int64_t paddedh = cases[num_cases - 1].height + 2 * cases[num_cases - 1].padding; // height after zero padding
  const int64_t paddedw = cases[num_cases - 1].width + 2 * cases[num_cases - 1].padding;  // width  after zero padding

  const int64_t outh = (paddedh - cases[num_cases - 1].kernel_height) / cases[num_cases - 1].stride + 1; // The output height of y
  const int64_t outw = (paddedw - cases[num_cases - 1].kernel_width) / cases[num_cases - 1].stride + 1;  // The output width  of y

  const int64_t total_size_y = cases[num_cases - 1].batch_size * cases[num_cases - 1].num_kernels * outh * outw;

  float* x = (float*) malloc(total_size_x * sizeof(float));
  float* w = (float*) malloc(total_size_w * sizeof(float));
  float* y = (float*) malloc(total_size_y * sizeof(float));

  fill_rands(x, cases[0].batch_size * cases[0].channels * cases[0].height * cases[0].width);
  fill_rands(w, total_size_w);

  std::vector<Timing> timings(reps, {0, 0, 0, 0});
  std::cout << "Runs per experiment: " << num_runs << " (static)" << std::endl;

  // Actual performance measurements repeated <reps> times.
  // We simply store all results and get median.
  for (int j = 0; j < reps; j++) {
    for (int i = 0; i < num_runs; ++i) {
      comp_func f = funcPair.first;
      int64_t start_x = 0;
      int64_t start_w = 0;

      for (int k = 0; k < num_cases; k++) {
        TestCase tc = cases[i];
        const int64_t total_size_x = tc.batch_size * tc.channels * tc.height * tc.width;
        const int64_t total_size_w = tc.num_kernels * tc.channels * tc.kernel_height * tc.kernel_width;

        Timing curr_timing;
        if (k == num_cases - 1) {
          curr_timing = f(tc, x + start_x, w + start_w, y);
        } else {
          curr_timing = f(tc, x + start_x, w + start_w, x + start_x + total_size_x);
        }
        f = funcPair.second;

        timings[j].quantize_ns += curr_timing.quantize_ns;
        timings[j].img_2_row_ns += curr_timing.img_2_row_ns;
        timings[j].gemm_ns += curr_timing.gemm_ns;
        timings[j].prelu_ns += curr_timing.prelu_ns;
      }
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
  register_function_pairs();
  std::cout << "Number of registered functions: " << funcs.size() << std::endl;

  /* Configuration of the test case(s) */
  std::vector<int64_t> inputs = {16, 28, 56, 112, 256, 386, 512, 600}; // enable for 3x3
  // std::vector<int64_t> inputs = { 200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000}; // enable for 1x1
  bool check_validity = false;
  bool verbose = true;
  bool separators = false;
  int reps = 5;
  TimingType timing_type = TimingType::MICRO;

  int64_t bn = 1, c = 64, kn = 64, kh = 3, kw = 3, p = 1, s = 1;  // enable for 3x3
  // int64_t bn = 32, kh = 1, kw = 1, p = 0, s = 1, height = 1, width = 1;  // enable for 1x1

  const float relu_alpha = 1;
  const float q_threshold = 0.5;

  const int num_layers = 3;
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

  int numFuncs = funcs.size();
  if (!check_validity || numFuncs < 2) {
    std::cout << "\033[1;31m" << "Warning! Running without validation." << "\033[0m" << std::endl;
  }

  int amount_inputs = inputs.size();
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

    std::vector<TestCase> tcs = std::vector<TestCase>();
    for (int k = 0; k < num_layers; k++) {
      tcs.push_back({bn, c, height, width, kn, kh, kw, p, s, relu_alpha, q_threshold, timing_type});
    }
      
    // Check validity of functions.
    if (check_validity) {
      int64_t total_size_x = 0;
      int64_t total_size_w = 0;
      int64_t size_last_y = 0;

      for (int k = 0; k < tcs.size(); k++) {
        TestCase tc = tcs[k];
        const int64_t size_x = tc.batch_size * tc.channels * tc.height * tc.width;

        if (k > 0 && size_x != size_last_y) {
          std::cout << "\033[1;31m" << "Dimensions from layer " << k+1 << " to " << k+2 << " do not match" << "\033[0m" << std::endl;
        }

        total_size_x += size_x;
        total_size_w += tc.num_kernels * tc.channels * tc.kernel_height * tc.kernel_width;

        const int64_t paddedh = tc.height + 2 * tc.padding; // height after zero padding
        const int64_t paddedw = tc.width + 2 * tc.padding; // width  after zero padding

        const int64_t outh = (paddedh - tc.kernel_height + 1) / tc.stride; // The output height of y
        const int64_t outw = (paddedw - tc.kernel_width + 1) / tc.stride; // The output width  of y

        size_last_y = tc.batch_size * tc.num_kernels * outh * outw;
      }

      float* x = (float*) malloc(total_size_x * sizeof(float));
      float* w = (float*) malloc(total_size_w * sizeof(float));
      float* y = (float*) malloc(size_last_y * sizeof(float));
      float* y_ref = (float*) malloc(size_last_y * sizeof(float));

      fill_rands(x, tcs[0].batch_size * tcs[0].channels * tcs[0].height * tcs[0].width);
      fill_rands(w, total_size_w);

      int cmp;
      for (int j = 0; j < numFuncs; j++) {
        std::pair<comp_func, comp_func> fPair = funcs[j];
        zeros(y, size_last_y);

        comp_func f = fPair.first;
        int64_t start_x = 0;
        int64_t start_w = 0;

        for (int k = 0; k < num_layers; k++) {
          TestCase tc = tcs[k];
          const int64_t total_size_x = tc.batch_size * tc.channels * tc.height * tc.width;
          const int64_t total_size_w = tc.num_kernels * tc.channels * tc.kernel_height * tc.kernel_width;

          if (k == num_layers - 1) {
            f(tc, x + start_x, w + start_w, y);
            if (j == 0) {

              // We perform cross-checking, i.e. the first output is considered as ground-truth.
              f(tc, x + start_x, w + start_w, y_ref);
            }
          } else {
            f(tc, x + start_x, w + start_w, x + start_x + total_size_x);
          }

          start_x += total_size_x;
          start_w += total_size_w;
          f = fPair.second;
        }

        const int64_t paddedh = tcs[num_layers - 1].height + 2 * tcs[num_layers - 1].padding; // height after zero padding
        const int64_t paddedw = tcs[num_layers - 1].width + 2 * tcs[num_layers - 1].padding; // width  after zero padding

        const int64_t outh = (paddedh - tcs[num_layers - 1].kernel_height + 1) / tcs[num_layers - 1].stride; // The output height of y
        const int64_t outw = (paddedw - tcs[num_layers - 1].kernel_width + 1) / tcs[num_layers - 1].stride; // The output width  of y

        cmp = compare_tensors(y, y_ref, tcs[num_layers - 1].batch_size, tcs[num_layers - 1].num_kernels, outh, outw);

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
      perf = perf_test(funcs[j], tcs, num_layers, reps);

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