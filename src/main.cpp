#include "TAB/common.h"
#include "TAB/Quantize.h"
#include "TAB/TAB_CPU.h"
#include "TAB/utility.h"
#include <chrono>
#include <numeric>


int Verify() {
    const int Batch_Size = 2;
    const int ReLU_alpha = 1;
    const float Q_Threshold = 0.5;
    
    const int CaseN = 9;
    const int CaseW = 8;
    int TestCases[CaseN][CaseW] = {
        //  c, h,   w, kn, kh, kw, p, s,
           1,  2,   2,  1,  3,  3, 1, 1,
          64,  12, 16, 64,  1,  1, 0, 1,
          32,  12, 16, 52,  1,  1, 0, 2,
          256, 56, 56, 10,  3,  3, 1, 1,
          160, 64, 56, 32,  3,  3, 0, 2, 
          325, 36, 25, 125,  5,  7, 3, 4,
          32,   1,  1, 120, 1,  1, 0, 1,
          512,  1,  1, 1024,1,  1, 0, 1,
          1024, 1,  1, 1640,1,  1, 2, 3,
    };

    // Get the reference matrix
    std::vector<float> TX = generate_array(16 * 64 * 224 * 224, true);  // Ternary X: size = 51,380,224
    std::vector<float> BX = generate_array(16 * 64 * 224 * 224, false); // Binary X : size = 51,380,224
    std::vector<float> TW = generate_array(1024 * 1024 * 3 * 3, true);  // Ternary Weights: size = 51,380,224
    std::vector<float> BW = generate_array(1024 * 1024 * 3 * 3, false); // Binary Weights : size = 51,380,224
    std::vector<int64_t> QW; // Quantized Weights

    // Iterate on layer configurations
    for (int icase = 0; icase < CaseN; icase++) {
        // config the layer shape and relevant sizes
        int c, h, w, kn, kh, kw, p, s;
        c = TestCases[icase][0];
        h = TestCases[icase][1];
        w = TestCases[icase][2];
        kn = TestCases[icase][3];
        kh = TestCases[icase][4];
        kw = TestCases[icase][5];
        p = TestCases[icase][6];
        s = TestCases[icase][7];

        // iterate on conv types
        std::vector< std::string> ConvNames = {"TAB_TNN","TAB_TBN","TAB_BTN","TAB_BNN"};
        for (int iconv = 0; iconv < ConvType::Conv_Types; iconv++) {

            // Get ref input x and weights w 
            float* ref_x = NULL;
            float* ref_w = NULL;
            std::vector<float> y;
            if (iconv == ConvType::TNN) {
                ref_x = TX.data();
                ref_w = TW.data();
                // Ternarize_NCHW_to_NHWCB(float* X, int PaddingH, int PaddingW, float* Q_Threshold, int N, int C, int H, int W)
                QW = Ternarize_NCHW_to_NHWCB(TW.data(), 0, 0, Q_Threshold, kn, c, kh, kw);
                // TAB_Conv(float* X, float* Q_Threshold, int64_t * QWeights, int* BTN_CNT1, ConvType TYPE, int PaddingH, int PaddingW, int StrideH, int StrideW, int Batch_Size, int C, int H, int W, int KN, int KH, int KW, float ReLU_alpha);
                y = TAB_Conv(TX.data(), Q_Threshold, QW.data(), NULL, ConvType::TNN, p, p, s, s, Batch_Size, c, h, w, kn, kh, kw, ReLU_alpha);
            }
            if (iconv == ConvType::TBN) {
                ref_x = TX.data();
                ref_w = BW.data();
                QW = Binarize_NCHW_to_NHWC(BW.data(), 0, 0, kn, c, kh, kw);
                y = TAB_Conv(TX.data(), Q_Threshold, QW.data(), NULL, ConvType::TBN, p, p, s, s, Batch_Size, c, h, w, kn, kh, kw, ReLU_alpha);
            }
            if (iconv == ConvType::BTN) {
                ref_x = BX.data();
                ref_w = TW.data();
                QW = Ternarize_NCHW_to_NHWCB(TW.data(), 0, 0, Q_Threshold, kn, c, kh, kw);
                std::vector<int> BTN_CNT = BTN_CNT_W2(QW.data(), kn, c, kh, kw);
                y = TAB_Conv(BX.data(), Q_Threshold, QW.data(), BTN_CNT.data(), ConvType::BTN, p, p, s, s, Batch_Size, c, h, w, kn, kh, kw, ReLU_alpha);
            }
            if (iconv == ConvType::BNN) {
                ref_x = BX.data();
                ref_w = BW.data();
                QW = Binarize_NCHW_to_NHWC(BW.data(), 0, 0,  kn, c, kh, kw);
                y = TAB_Conv(BX.data(), Q_Threshold, QW.data(), NULL, ConvType::BNN, p, p, s, s, Batch_Size, c, h, w, kn, kh, kw, ReLU_alpha);
            }

            // Get reference conv result: direct conv on ref_x and ref_w

            std::vector<float> px = DirectPad(ref_x, p, p, Batch_Size, c, h, w);
            // std::vector<float> DirectConv2d_FP32(float* x, float* w, int stride1, int stride2, int N, int C, int paddedH, int paddedW, int KN, int KH, int KW)
            int paddedh = h + 2 * p; // height after zero padding
            int paddedw = w + 2 * p; // width  adter zero padding

            int outh = (paddedh - kh) / s + 1; // The output height of y
            int outw = (paddedw - kw) / s + 1; // The output width  of y

            std::vector<float> ref_y = std::vector<float>(Batch_Size * kn * outh * outw);
            DirectConv2d_FP32(px.data(), ref_w, s, s, Batch_Size, c, paddedh, paddedw, kn, kh, kw, ref_y.data());

            // Compare the conv results to ensure the functions are correct

            int cmp;
            if ((p > 0) && ((iconv == ConvType::BTN) || (iconv == ConvType::BNN))) 
                // BTN and BNN regard the padded zeros as 1s because binary quantization only has (+1, -1) no zeros.
                // So we only compare the central part of conv results here, excluding the zero padding part.
                cmp = Compare_Tensor_BNN_Padding(y.data(), ref_y.data(), Batch_Size, kn, outh, outw, p, p);
            else
                cmp = Compare_Tensor_NHWC(y.data(), ref_y.data(), Batch_Size, kn, outh, outw);
            if(cmp>0)
                std::cout << "Test Case " << icase << " kernel: " << kw << "X" << kh << " " << ConvNames[iconv] << " Passed!" << std::endl;    
            else 
                std::cout << "Test Case " << icase << " kernel: " << kw << "X" << kh << " " << ConvNames[iconv] << " Failed!" << std::endl;
        }
    }
    return 0;
}


int Benchmark(int Batch_Size) {
    const float ReLU_alpha = 0.1;
    const float Q_Threshold = 0.5;
    const int CaseN = 20;
    const int CaseW = 8;
    const int RUN_TIMES = 10;
    int TestCases[CaseN][CaseW] = {  // You can define your own test cases to fit your platform
        //  c, h,   w, kn, kh, kw, p, s,

           64, 56, 56,  64, 3, 3, 1, 1, // 3x3 kernels in the same tensor size but diff shape
           64, 56, 56, 128, 3, 3, 1, 1,
          128, 28, 28, 128, 3, 3, 1, 1,
          128, 28, 28, 256, 3, 3, 1, 1,
          256, 14, 14, 256, 3, 3, 1, 1,
          256, 14, 14, 512, 3, 3, 1, 1,

          80, 224, 224, 80, 3, 3, 1, 1, // different conv stride
          80, 224, 224, 80, 3, 3, 1, 2,
          80, 224, 224, 80, 3, 3, 1, 3,
          80, 224, 224, 80, 3, 3, 1, 4,

          512, 56, 56, 256, 1, 1, 0, 1, // 1,3,5,7,9,11 kernels in the same tensor shape
          512, 56, 56, 256, 3, 3, 1, 1,
          512, 56, 56, 256, 5, 5, 2, 1,
          512, 56, 56, 256, 7, 7, 3, 1,
          512, 56, 56, 256, 9, 9, 3, 1,
          512, 56, 56, 256,11,11, 3, 1,

          2000, 1, 1, 4000, 1, 1, 0, 1, // fully connected layers
          4000, 1, 1, 8000, 1, 1, 0, 1,
          8000, 1, 1,16000, 1, 1, 0, 1,
         16000, 1, 1,32000, 1, 1, 0, 1,
    };

    // Get the fake tensors before computation - Batch_Size should be 1~16
    std::vector<float> TX = std::vector<float>(16 * 80 * 224 * 224);  // Ternary X: size = Max(Batch_Size x C x H x W) = 64,225,280
    std::vector<float> BX = std::vector<float>(16 * 80 * 224 * 224); // Binary X : size = Max(Batch_Size x C x H x W) = 64,225,280
    std::vector<int64_t> QW = std::vector<int64_t>(32000*16000/32); // Quantized Weights, size= Max(KN x C x KH x KW x BITS / CNTBITS), 16 * 512 * 11 * 11=991,232
    std::vector<int> BTN_CNT = std::vector<int>(32000, 1); // Used as BTN_Count_Weight_Bit_2, size: Max(KN)

    // Iterate on layer configurations
    for (int icase = 0; icase < CaseN; icase++) {
        // config the layer shape and relevant sizes
        int c, h, w, kn, kh, kw, p, s;
        c = TestCases[icase][0];
        h = TestCases[icase][1];
        w = TestCases[icase][2];
        kn = TestCases[icase][3];
        kh = TestCases[icase][4];
        kw = TestCases[icase][5];
        p = TestCases[icase][6];
        s = TestCases[icase][7];

        // iterate on conv types
        std::vector< std::string> ConvNames = { "TAB_TNN","TAB_TBN","TAB_BTN","TAB_BNN" };
        for (int iconv = 0; iconv < ConvType::Conv_Types; iconv++) {

            // iterate on RUN_TIMES
            int y_size;
            std::vector<int64_t> run_time;
            for (int irun = 0; irun < RUN_TIMES; irun++) {
                std::chrono:: high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
                std::vector<float> y = TAB_Conv(BX.data(), Q_Threshold, QW.data(), BTN_CNT.data(), ConvType::BTN, p, p, s, s, Batch_Size, c, h, w, kn, kh, kw, ReLU_alpha);
                y_size = y.size();
                std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
                std::chrono::nanoseconds duration_ns = end_time - start_time;
                run_time.push_back(duration_ns.count());
            }
            // Compare the conv results to ensure the functions are correct
            int64_t avg_ns = std::accumulate(run_time.begin(),run_time.end(), 0.0)/run_time.size();
            std::cout << "Test Case " << icase <<" " << ConvNames[iconv] << " Input NCHW=" << Batch_Size << "," << c << "," << h << "," << w;
            std::cout << ", kernel: " <<kn << "," <<c << ","<<  kh << "," << kw << ", y_size = " << y_size << " Average execution time " <<avg_ns<<" ns" << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}


int main() {
    Verify();
    Benchmark(1); // batch size = 1~16. 
}

