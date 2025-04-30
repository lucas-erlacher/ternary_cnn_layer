#include "measurement.h"

#include "../TAB/common.h"
#include "../TAB/Quantize.h"
#include "../TAB/TAB_CPU.h"
#include "../TAB/Img2Row.h"
#include "../TAB/Activation.h"

#include "../optimization/GEMM_opt.h"
#include "../optimization/fused_nhwc.h"
#include "../optimization/yuliia_lucas_functions.h"
#include "../optimization/fused_1x1.h"

// helper function for timing
Timing get_timing(TimePoint p1, TimePoint p2, TimePoint p3, TimePoint p4, TimePoint p5, TimingType timing_type) {
  if (timing_type == NANO) {
    return {
      (p2 - p1).count(),
      (p3 - p2).count(),
      (p4 - p3).count(),
      (p5 - p4).count()
    };
  } else if (timing_type == MICRO) {
    return {
      std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count(),
      std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count(),
      std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count(),
      std::chrono::duration_cast<std::chrono::microseconds>(p5 - p4).count()
    };
  } else {
    // MILLI
    return {
      std::chrono::duration_cast<std::chrono::milliseconds>(p2 - p1).count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(p3 - p2).count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(p4 - p3).count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(p5 - p4).count()
    };
  }
}

Timing baseline(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  std::vector<int64_t> quantized_x = Ternarize_NCHW_to_NHWCB(tensor_x, tc.padding, tc.padding, tc.q_threshold, tc.batch_size, 
                                                    tc.channels, tc.height, tc.width);
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();  

  quantized_x = Img2Row_NHWCB_to_N_OHOW_KHKWC(quantized_x.data(), tc.batch_size, PackedC * BITS, padded_h, padded_w, 
                                              tc.kernel_height, tc.kernel_width, tc.stride, tc.stride);
  
  TimePoint end_img_2_row = std::chrono::high_resolution_clock::now();

  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img_2_row, end_gemm, end_prelu, tc.timing_type);
}

Timing conv_1x1_v1(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  TimePoint start_time = std::chrono::high_resolution_clock::now();

  specialized_1x1_v1(tensor_x, tensor_w, res, tc.batch_size, tc.channels, tc.num_kernels, tc.q_threshold, tc.relu_alpha);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, start_time, start_time, start_time, end_prelu, tc.timing_type);
}

Timing conv_1x1_v2(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  TimePoint start_time = std::chrono::high_resolution_clock::now();

  specialized_1x1_v2(tensor_x, tensor_w, res, tc.batch_size, tc.channels, tc.num_kernels, tc.q_threshold, tc.relu_alpha);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, start_time, start_time, start_time, end_prelu, tc.timing_type);
}


// same as baseline but taking in x in  N_H_W_C (instead of N_C_H_W which is what baseline takes)
Timing baseline_N_H_W_C(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  // reshape iput s.t. we can reuse baseline function
  std::vector<float> reshaped_input = convert_NHWC_to_NCHW(tensor_x, tc.batch_size, tc.height, tc.width, tc.channels);
  return baseline(tc, reshaped_input.data(), tensor_w, res);
}

Timing ly_our_fusing_1(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);

  TimePoint end_quantize = std::chrono::high_resolution_clock::now();   
  std::vector<int64_t> quantized_x = our_fusing_1(tensor_x, tc.q_threshold, tc.padding, tc.stride, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, output_h, output_w);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}

Timing ly_our_fusing_2(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        

  std::vector<int64_t> quantized_x = our_fusing_2(tensor_x, tc.q_threshold, tc.padding, tc.stride, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, output_h, output_w);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}


Timing ly_our_fusing_3(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        

  std::vector<int64_t> quantized_x = our_fusing_3(tensor_x, tc.q_threshold, tc.padding, tc.stride, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, output_h, output_w);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}

Timing ly_paper_fusing_1(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        

  std::vector<int64_t> quantized_x = paper_fusing_1(tensor_x, tc.q_threshold, tc.padding, tc.stride, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}

Timing ly_paper_fusing_2(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  // std::vector<int64_t> quantized_w = quantize_NCHW_to_NHWCB(tensor_w, 0, tc.q_threshold, tc.num_kernels, 
  //                                                           tc.channels, tc.kernel_height, tc.kernel_width);  // for end-to-end
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        

  std::vector<int64_t> quantized_x = paper_fusing_2(tensor_x, tc.q_threshold, tc.padding, tc.stride, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}

Timing ly_paper_fusing_3(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  // std::vector<int64_t> quantized_w = quantize_NCHW_to_NHWCB(tensor_w, 0, tc.q_threshold, tc.num_kernels, 
  //                                                           tc.channels, tc.kernel_height, tc.kernel_width);  // for end-to-end
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        

  std::vector<int64_t> quantized_x = paper_fusing_3(tensor_x, tc.q_threshold, tc.padding, tc.stride, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}


Timing fused(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        
  // std::vector<float> cvt_x = convert_NCHW_to_NHWC(tensor_x, tc.batch_size, tc.channels, tc.height, tc.width); // for correctness

  std::vector<int64_t> quantized_x = fused_NHWC_to_NHWCB(tensor_x, tc.padding, tc.q_threshold, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, tc.stride);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}


Timing fused_simd(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  // std::vector<int64_t> quantized_w = quantize_NCHW_to_NHWCB(tensor_w, 0, tc.q_threshold, tc.num_kernels, 
  //                                                           tc.channels, tc.kernel_height, tc.kernel_width);  // for end-to-end
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        
  // std::vector<float> cvt_x = convert_NCHW_to_NHWC(tensor_x, tc.batch_size, tc.channels, tc.height, tc.width); // for correctness

  std::vector<int64_t> quantized_x = fused_NHWC_to_NHWCB_simd(tensor_x, tc.padding, tc.q_threshold, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, tc.stride);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}


Timing fused_buffer(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  // std::vector<int64_t> quantized_w = quantize_NCHW_to_NHWCB(tensor_w, 0, tc.q_threshold, tc.num_kernels, 
  //                                                           tc.channels, tc.kernel_height, tc.kernel_width);  // for end-to-end
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        
  // std::vector<float> cvt_x = convert_NCHW_to_NHWC(tensor_x, tc.batch_size, tc.channels, tc.height, tc.width); // for correctness

  std::vector<int64_t> quantized_x = fused_NHWC_to_NHWCB_buffer(tensor_x, tc.padding, tc.q_threshold, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, tc.stride);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}


Timing fused_blocked(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  // std::vector<int64_t> quantized_w = quantize_NCHW_to_NHWCB(tensor_w, 0, tc.q_threshold, tc.num_kernels, 
  //                                                           tc.channels, tc.kernel_height, tc.kernel_width);  // for end-to-end
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        
  // std::vector<float> cvt_x = convert_NCHW_to_NHWC(tensor_x, tc.batch_size, tc.channels, tc.height, tc.width); // for correctness

  std::vector<int64_t> quantized_x = fused_NHWC_to_NHWCB_blocked(tensor_x, tc.padding, tc.q_threshold, tc.batch_size, 
                                                         tc.channels, tc.height, tc.width, tc.kernel_height, 
                                                         tc.kernel_width, tc.stride);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}

Timing specialized_3x3(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  int64_t padded_h = tc.height + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = tc.width + 2 * tc.padding; // Width  after bit-packing
  int64_t output_h = (padded_h - tc.kernel_height) / tc.stride + 1; // Output Height
  int64_t output_w = (padded_w - tc.kernel_width) / tc.stride + 1; // Output Width

  int64_t PackedC = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits); // The channel after bit-packing
  
  std::vector<int> yi = std::vector<int>(tc.batch_size * output_h * output_w * tc.num_kernels, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, tc.num_kernels, 
                                                             tc.channels, tc.kernel_height, tc.kernel_width);
  TimePoint end_quantize = std::chrono::high_resolution_clock::now();                                                        
  // std::vector<float> cvt_x = convert_NCHW_to_NHWC(tensor_x, tc.batch_size, tc.channels, tc.height, tc.width); // for correctness

  std::vector<int64_t> quantized_x = fused_NHWC_to_NHWCB_blocked3x3(tensor_x, tc.q_threshold, tc.batch_size, 
                                                                    tc.channels, tc.height, tc.width);

  TimePoint end_img2row = std::chrono::high_resolution_clock::now();
  
  TNNGEMM_baseline(quantized_x.data(), quantized_w.data(), yi.data(), tc.batch_size * output_h * output_w, 
                   tc.num_kernels, PackedC * tc.kernel_height * tc.kernel_width);
  
  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(yi.data(), tc.batch_size, tc.num_kernels, output_h, output_w, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_img2row, end_gemm, end_prelu, tc.timing_type);
}

// Direct Convolution approach
// Assume s = 1
Timing direct_conv(TestCase tc, float* tensor_x, float* tensor_w, float* res) {
  const int64_t N = tc.batch_size;
  const int64_t C = (tc.channels % cntbits) ? ((tc.channels / cntbits) + 1) : (tc.channels / cntbits);
  const int64_t H = tc.height;
  const int64_t W = tc.width;
  const int64_t KN = tc.num_kernels;
  const int64_t KH = tc.kernel_height;
  const int64_t KW = tc.kernel_width;

  // output dimensions (assuming s = 1)
  int64_t padded_h = H + 2 * tc.padding; // Height after bit-packing
  int64_t padded_w = W + 2 * tc.padding; // Width  after bit-packing

  const int64_t OH = (padded_h - KH) + 1;
  const int64_t OW = (padded_w - KW) + 1;
  const int64_t KC = C * BITS;

  std::vector<int> y = std::vector<int>(N * OH * OW * KN, 0);

  TimePoint start_time = std::chrono::high_resolution_clock::now();
    
  std::vector<int64_t> quantized_w = Ternarize_NCHW_to_NHWCB(tensor_w, 0, 0, tc.q_threshold, KN, 
                                                             tc.channels, KH, KW);
  std::vector<int64_t> quantized_x = Ternarize_NCHW_to_NHWCB(tensor_x, tc.padding, tc.padding, tc.q_threshold, N, 
                                                    tc.channels, H, W);

  TimePoint end_quantize = std::chrono::high_resolution_clock::now();

  int64_t* x = quantized_x.data();
  int64_t* w = quantized_w.data();

  // N
  for (int64_t on = 0; on < N; on++) {
    for (int64_t oh = 0; oh < OH; oh++) {
      for (int64_t ow = 0; ow < OW; ow++) {

        // each output is computed through sum of weighted entries of input
        for (int64_t kn = 0; kn < KN; kn++) {
          int cntp1 = 0;
          int cntp2 = 0;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              for (int64_t kc = 0; kc < KC; kc += BITS) {
                // Use N_H_W_C_B
                int64_t p1 = x[on * (padded_h * padded_w * KC) + (oh + kh) * (padded_w * KC) + (ow + kw) * KC + kc + 0] ^ w[kn * (KH * KW * KC) + kh * (KW * KC) + kw * KC + kc + 0];
                int64_t p2 = x[on * (padded_h * padded_w * KC) + (oh + kh) * (padded_w * KC) + (ow + kw) * KC + kc + 1] & w[kn * (KH * KW * KC) + kh * (KW * KC) + kw * KC + kc + 1];
                int64_t p3 = p1 & p2;
                cntp1 = cntp1 + popcnt64(p3);
                cntp2 = cntp2 + popcnt64(p2);
              }
            }
          }
          
          // Output N_OUTH_OUTW_KN
          y[on * (OH * OW * KN) + oh * (OW * KN) + ow * KN + kn] = cntp2 - cntp1 - cntp1;
        }
      }
    }
  }

  TimePoint end_gemm = std::chrono::high_resolution_clock::now();
  
  PReLU(y.data(), N, KN, OH, OW, tc.relu_alpha, res);

  TimePoint end_prelu = std::chrono::high_resolution_clock::now();

  return get_timing(start_time, end_quantize, end_quantize, end_gemm, end_prelu, tc.timing_type);
}

// needed for single layer tests, uncomment all if multi-layer is used
void register_functions() {
  add_function(&baseline, "Baseline", 1);  // add number of ops instead of 1 for performance measurements

  // add_function(&ly_our_fusing_1, "Our Fusing 1", 1);
  // add_function(&ly_our_fusing_2, "Our Fusing 2", 1);
  // add_function(&ly_our_fusing_3, "Our Fusing 3", 1);

  // add_function(&ly_paper_fusing_1, "Paper Fusing 1", 1);
  // add_function(&ly_paper_fusing_2, "Paper Fusing 2", 1);
  // add_function(&ly_paper_fusing_3, "Paper Fusing 3", 1);

  // add_function(&fused, "Fused 1", 1);
  // add_function(&fused_simd, "Fused 1 SIMD", 1);
  // add_function(&fused_buffer, "Fused 2", 1);
  // add_function(&fused_blocked, "Fused 3", 1);
  // add_function(&specialized_3x3, "3x3 Specialized", 1);

  // add_function(&conv_1x1_v1, "Specialized 1x1 v1", 1);
  // add_function(&conv_1x1_v2, "Specialized 1x1 v2", 1);

  // add_function(&direct_conv, "Direct Conv", 1);  // add number of ops instead of 1 for performance measurements
}

// needed for multi-layer tests, uncomment all if single-layer is used
// FIRST FUNCTION:  N_C_H_W -> N_H_W_C  (will handle first layer)
// SECOND FUNCTION: N_H_W_C -> N_H_W_C  (will handle all other layers)
void register_function_pairs() {
  // addPair(&baseline, &baseline_N_H_W_C, "Baseline");

  // addPair(&ly_paper_fusing_1, &fused, "Fused Code 1");
  // addPair(&ly_paper_fusing_2, &fused_buffer, "Fused Code 2");
  // addPair(&ly_paper_fusing_3, &fused_blocked, "Fused Code 3");

  // addPair(&conv_1x1_v1, &conv_1x1_v1, "Specialized 1x1 v1");
  // addPair(&conv_1x1_v2, &conv_1x1_v2, "Specialized 1x1 v2");
}
