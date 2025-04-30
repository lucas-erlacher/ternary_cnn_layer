#include "common.h"
#include "Quantize.h"
#include "Img2Row.h"
#include "GEMM.h"
#include "Activation.h"
#include "TAB_CPU.h"

/* Container function of quantization and convolution functions. Can be applied to conv and FC layers.
* Conv: 1X1, 3X3, and larger kernels. FC equals to 1x1 conv.
* type: 
* 0: TAB-TNN
* 1: TAB-TBN
* 2: TAB-BTN
* 3: TAB-BNN
* */
// Ternary and Binary Convolution using N, H, W, C, B format
// Input: 
//   x: input activation in NCHW format 
//   qw: quantized weights in KN_KH_KW_C_Bit format
//   stride: the stride on Height and Width
//   padding: the padding on Height and Width
//   N: batch number, C, channel, H: Height, W: Width
//   KN: number of filters/kernels, KH: Kernel Height, KW, Kernel Width 
// Output:
//   y: convolution result
std::vector<float> TAB_Conv(float * X, float Q_Threshold, int64_t * QWeights, int * BTN_CNT1, ConvType TYPE, int PaddingH, int PaddingW, int StrideH, int StrideW, int Batch_Size, int C, int H, int W,
    int KN, int KH, int KW, float ReLU_alpha) {
    int PackedH, PackedW, OH, OW, PackedC;
    PackedH = H + 2 * PaddingH; // Height after bit-packing
    PackedW = W + 2 * PaddingW; // Width  after bit-packing
    OH = (PackedH - KH) / StrideH + 1; // Output Height
    OW = (PackedW - KW) / StrideW + 1; // Output Width
    PackedC = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // The channel after bit-packing
    
    std::vector<int64_t> qx;
    std::vector<int> yi;

    // Quantize and Img2Row/Img2Col
        
        if ((TYPE == ConvType::TNN) || (TYPE == ConvType::TBN)) {
            qx = Ternarize_NCHW_to_NHWCB(X, PaddingH, PaddingW, Q_Threshold, Batch_Size, C, H, W);
            qx = Img2Row_NHWCB_to_N_OHOW_KHKWC(qx.data(), Batch_Size, PackedC * BITS, PackedH, PackedW, KH, KW, StrideH, StrideW);
        }
        else {
            qx = Binarize_NCHW_to_NHWC(X, PaddingH, PaddingW, Q_Threshold, Batch_Size, C, H, W);
            qx = Img2Row_NHWCB_to_N_OHOW_KHKWC(qx.data(), Batch_Size, PackedC, PackedH, PackedW, KH, KW, StrideH, StrideW);
        }
       
    // Bitwise GEMM
     
        switch (TYPE) {
        case ConvType::TNN: {
            yi = TNNGEMM_baseline(qx.data(), QWeights, Batch_Size * OH * OW, KN, PackedC * KH * KW);
            break;
        }
        case ConvType::TBN: {
            yi = TBNGEMM_baseline(qx.data(), QWeights, Batch_Size * OH * OW, KN, PackedC * KH * KW);
            break;
        }
        case ConvType::BTN: {
            yi = BTNGEMM_baseline(qx.data(), QWeights, BTN_CNT1, Batch_Size * OH * OW, KN, PackedC * KH * KW);
            break;
        }
        case ConvType::BNN: {
            yi = BNNGEMM_baseline(qx.data(), QWeights, Batch_Size * OH * OW, KN, PackedC * KH * KW, C * KH * KW);
            break;
        }
        } // switch
    
    // Activation function: PReLU

    std::vector<float> y = std::vector<float>(Batch_Size * KN * OH * OW);
    PReLU(yi.data(), Batch_Size, KN, OH, OW, ReLU_alpha, y.data());

    return y;
}