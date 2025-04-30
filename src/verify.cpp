#include <iostream>
#include <fstream>
#include <vector>
#include "TAB/common.h"
#include "TAB/Quantize.h"
#include "TAB/TAB_CPU.h"
#include "TAB/utility.h"

std::vector<float> read_binary_file(const std::string& file_path, int array_size) {
  // Open the binary file for reading
  std::ifstream infile(file_path, std::ios::binary);
  if (!infile) {
    std::cerr << "Failed to open file for reading: " << file_path << std::endl;
    return {};
  }

  // Determine the file size
  infile.seekg(0, std::ios::end);
  std::streampos file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // Check if the file size matches the expected size
  if (file_size != array_size * sizeof(float)) {
    std::cerr << "File size doesn't match the expected size (" << array_size << " floats)" << std::endl;
    return {};
  }

  // Read the contents of the file into a vector
  std::vector<float> float_vector(array_size);
  infile.read(reinterpret_cast<char*>(float_vector.data()), file_size);

  // Close the file
  infile.close();

  return float_vector;
}

std::vector<float> quantize_manually(const std::vector<float>& float_vector, float threshold) {
  std::vector<float> result;

  // Reserve space to avoid unnecessary reallocations
  result.reserve(float_vector.size());

  for (float value : float_vector) {
    if (value < -threshold) {
      result.push_back(-1);
    } else if (value > threshold) {
      result.push_back(1);
    } else {
      result.push_back(0);
    }
  }

  return result;
}

int main(int argc, char* argv[]) {
  const int ReLU_alpha = 1;
  const float threshold = 0.5;

  // Check if correct number of command-line arguments are provided
  if (argc != 11) {
    std::cerr << "Usage: " << argv[0] << " <file_path> <bs> <c> <h> <w> <kn> <kh> <kw> <p> <s>" << std::endl;
    return 1;
  }

  // Get command-line arguments
  std::string file_path = argv[1];
  int batch_size = std::stoi(argv[2]);
  int channels = std::stoi(argv[3]);
  int height = std::stoi(argv[4]);
  int width = std::stoi(argv[5]);

  int num_kernels = std::stoi(argv[6]);
  int kernel_height = std::stoi(argv[7]);
  int kernel_width = std::stoi(argv[8]);
  int padding = std::stoi(argv[9]);
  int stride = std::stoi(argv[10]);

  int input_size = batch_size * channels * height * width;
  int weights_size = num_kernels * kernel_height * kernel_width;

  // Read binary file and create float vector
  std::vector<float> random_array = read_binary_file(file_path, input_size + weights_size);
  if (!random_array.empty()) {
    std::cout << "Float vector read successfully from file." << std::endl;
  } else {
    return 1;
  }

  std::vector<float> input(random_array.begin(), std::next(random_array.begin(), input_size));
  std::vector<float> weights(std::next(random_array.begin(), input_size), random_array.end());

  // Quantization threshold for ternarization
  std::vector<float> Q_Threshold = std::vector<float>(std::max(batch_size, num_kernels), threshold);

  // Ternarize weights
  std::vector<int64_t> QW = Ternarize_NCHW_to_NHWCB(weights.data(), 0, 0, Q_Threshold.data(), num_kernels, channels, kernel_height, kernel_width);
  
  // Compute TNN
  std::vector<float> y = TAB_Conv(input.data(), Q_Threshold.data(), QW.data(), NULL, ConvType::TNN, padding, padding, stride, stride, batch_size, channels, height, width, num_kernels, kernel_height, kernel_width, ReLU_alpha);

  // Now check, whether the result matches the floating-point solution
  std::vector<float> quantized_x = quantize_manually(input, threshold);
  std::vector<float> quantized_w = quantize_manually(weights, threshold);

  std::vector<float> px = DirectPad(quantized_x.data(), padding, padding, batch_size, channels, height, width);
  int paddedh = height + 2 * padding; // height after zero padding
  int paddedw = width + 2 * padding; // width  adter zero padding
  
  std::vector<float> ref_y = DirectConv2d_FP32(px.data(), quantized_w.data(), stride, stride, batch_size, channels, paddedh, paddedw, num_kernels, kernel_height, kernel_width);

  // Compare the conv results to ensure the functions are correct
  int cmp;
  int outh = (height + 2 * padding - kernel_height + 1) / stride; // The output height of y
  int outw = (width + 2 * padding - kernel_width + 1) / stride; // The output width  of y

  cmp = Compare_Tensor_NHWC(y.data(), ref_y.data(), batch_size, num_kernels, outh, outw);
  
  if(cmp>0)
    std::cout << "Test Case Passed!" << std::endl;    
  else 
    std::cout << "Test Case Failed!" << std::endl;

  return 0;
}