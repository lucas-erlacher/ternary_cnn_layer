# HIGH LEVEL IDEA
fuse quantization-packing function into im2row function. 

do it using the im2row loop order in order to make the writebacks of the fused code have good locality. 

compute the needed packed values on the fly. this will lead to recomputations of the same value (e.g. E on the diag) but it might be worth it since (compared to the unfused code) we save 1 memory write and 1 memory read for each unique packed value AND because the computation of a packed value is not particularly expensive. 

there are two alteranative ways to fuse
- "Robins approach": compute packed values with normal_image loop order and then when a packed value is ready write it to all places in im2col matrix where it belongs (see E which gets placed onto entire diagonal in "flow" picture) (DRAWBACK: bad write locality (e.g. the writes of E along the diagonal are not consecutive in memory))
- paper pseudocode (page 13) (DRAWBACK: every packed value is written to memory twice: once in line 15 and then again in line 20)

# IMPLEMENTATION DETAILS
the main trick was to derive formulas for h and w that depend on the loop indices from the im2row loop order s.t. we can compute the required packed value on the fly as we iterate over the im2row matrix in im2row loop order. We drived these formulas by inspecting the indexing of X in the im2col.h file and by correctly accounting for the padding (= the "- PaddingH" and " - PaddingW" terms). Then we inlined the computation of the packed value and used the loop index c (innermost loop) to determine ic from the innermost loop from Quantize.cpp (which loops until priChannel) which allowed to delete this loop. 
The writeback into the im2row matrix uses the same indexing as the im2row.h file (wich is desired/the whole point of this indexing as this indexing has good write locality). In order for the fusion to be correct we need to write back a 0 in case the computed h or w exceed the normal_image bounds.

# Read/Write counts
| Test Case Number | unfused Reads | fused_basic_opts Reads | unfused Writes | fused_basic_opts Writes | c  | h  | w  | kn  | kh | kw | p  | s  |
|------------------|---------------|------------------------|----------------|------------------------|----|----|----|-----|----|----|----|----|
| 1                | 25,344        | 24,576                 | 1,536          | 768                    | 64 | 12 | 16 | 64  | 1  | 1  | 0  | 1  |
| 2                | 12,480        | 3,072                  | 960            | 192                    | 32 | 12 | 16 | 52  | 1  | 1  | 0  | 2  |
| 3                | 1,237,276     | 2,410,560              | 133,404        | 90,396                 | 160| 64 | 56 | 32  | 3  | 3  | 0  | 2  |
| 4                | 68            | 64                     | 8              | 4                      | 32 | 1  | 1  | 120 | 1  | 1  | 0  | 1  |
| 5                | 1,056         | 1,024                  | 64             | 32                     | 512| 1  | 1  | 1024| 1  | 1  | 0  | 1  |
| 6                | 152           | 32                     | 160            | 64                     | 1  | 2  | 2  | 1   | 3  | 3  | 1  | 1  |
| 7                | 2,057,216     | 14,108,672             | 501,760        | 440,896                | 256| 56 | 56 | 10  | 3  | 3  | 1  | 1  |
| 8                | 630,360       | 1,064,700              | 66,960         | 39,312                 | 325| 36 | 25 | 125 | 5  | 7  | 3  | 4  |

# Runtime Comparison to Unfused Code

## Measurements of Runtime:

| Test Case | Original Runtime (ns) | Fused Runtime (ns) | Optimized Runtime (ns) | Im2Row + SIMD (ns) | Im2Row + Removed If (ns) | Normal Image + SIMD (ns) |
|-----------|-----------------------|--------------------|------------------------|---------------------|--------------------------|---------------------------|
| 0         | 10,172,300 (0.0101723 s) | 11,675,800 (0.0116758 s) | 11,265,900 (0.0112659 s) | 3,095,600 (0.0030956 s) | 2,980,300 (0.0029803 s) | 2,486,200 (0.0024862 s) |
| 1         | 17,810,600 (0.0178106 s) | 19,637,100 (0.0196371 s) | 17,894,000 (0.017894 s) | 4,966,900 (0.0049669 s) | 5,119,600 (0.0051196 s) | 4,686,500 (0.0046865 s) |
| 2         | 8,879,600 (0.0088796 s) | 10,166,200 (0.0101662 s) | 9,414,300 (0.0094143 s) | 2,067,300 (0.0020673 s) | 2,056,900 (0.0020569 s) | 1,753,700 (0.0017537 s) |
| 3         | 16,857,700 (0.0168577 s) | 18,009,600 (0.0180096 s) | 17,473,700 (0.0174737 s) | 3,778,800 (0.0037788 s) | 3,915,200 (0.0039152 s) | 3,300,500 (0.0033005 s) |
| 4         | 7,909,800 (0.0079098 s) | 8,410,200 (0.0084102 s) | 8,607,200 (0.0086072 s) | 1,857,700 (0.0018577 s) | 1,832,800 (0.0018328 s) | 1,544,000 (0.0015440 s) |
| 5         | 15,515,600 (0.0155156 s) | 16,245,200 (0.0162452 s) | 16,337,500 (0.0163375 s) | 3,361,300 (0.0033613 s) | 3,307,200 (0.0033072 s) | 3,182,700 (0.0031827 s) |
| 6         | 349,166,300 (0.3491663 s) | 523,638,100 (0.5236381 s) | 427,520,500 (0.4275205 s) | 188,618,200 (0.1886182 s) | 186,402,200 (0.1864022 s) | 84,371,200 (0.0843712 s) |
| 7         | 96,148,800 (0.0961488 s) | 133,020,200 (0.1330202 s) | 105,380,200 (0.1053802 s) | 49,893,100 (0.0498931 s) | 47,432,300 (0.0474323 s) | 33,455,900 (0.0334559 s) |
| 8         | 50,079,800 (0.0500798 s) | 57,961,400 (0.0579614 s) | 46,885,500 (0.0468855 s) | 25,602,100 (0.0256021 s) | 21,402,800 (0.0214028 s) | 23,610,700 (0.0236107 s) |
| 9         | 31,833,100 (0.0318331 s) | 33,722,400 (0.0337224 s) | 28,002,200 (0.0280022 s) | 11,930,500 (0.0119305 s) | 11,694,900 (0.0116949 s) | 18,147,700 (0.0181477 s) |
| 10        | 31,424,700 (0.0314247 s) | 31,994,500 (0.0319945 s) | 32,226,300 (0.0322263 s) | 9,429,300 (0.0094293 s) | 9,434,000 (0.0094340 s) | 9,813,300 (0.0098133 s) |
| 11        | 242,006,800 (0.2420068 s) | 263,232,400 (0.2632324 s) | 256,733,200 (0.2567332 s) | 58,217,700 (0.0582177 s) | 57,301,000 (0.0573010 s) | 52,906,500 (0.0529065 s) |


## Measurement (+ buffer)

| Test Case | Normal Image + SIMD (ns) | Measurement (+ buffer) (ns) |
|-----------|--------------------------|-----------------------------|
| 0         | 2,486,200                | 2,787,200                   |
| 1         | 4,686,500                | 4,922,900                   |
| 2         | 1,753,700                | 1,989,100                   |
| 3         | 3,300,500                | 4,067,000                   |
| 4         | 1,544,000                | 1,733,400                   |
| 5         | 3,182,700                | 3,313,300                   |
| 6         | 84,371,200               | 91,723,800                  |
| 7         | 33,455,900               | 37,733,400                  |
| 8         | 23,610,700               | 25,406,200                  |
| 9         | 18,147,700               | 20,821,700                  |
| 10        | 9,813,300                | 9,694,900                   |
| 11        | 52,906,500               | 51,845,100                  |



## Measurements of bit-packing + img2col (no flags):

| Test Case | Original Runtime (ns)          | Optimized Runtime (ns)         |
|-----------|--------------------------------|--------------------------------|
| 0         | 1491550 (0.00149155 s)         | 6470800 (0.0064708 s)          |
| 1         | 1292950 (0.00129295 s)         | 6151850 (0.00615185 s)         |
| 2         | 844900 (0.0008449 s)           | 3129700 (0.0031297 s)          |
| 3         | 652700 (0.0006527 s)           | 3037250 (0.00303725 s)         |
| 4         | 376950 (0.00037695 s)          | 1308800 (0.0013088 s)          |
| 5         | 341050 (0.00034105 s)          | 1317950 (0.00131795 s)         |
| 6         | 33564450 (0.03356445 s)        | 139559900 (0.1395599 s)        |
| 7         | 22701850 (0.02270185 s)        | 34627850 (0.03462785 s)        |
| 8         | 22755000 (0.022755 s)          | 15873100 (0.0158731 s)         |
| 9         | 22014450 (0.02201445 s)        | 9867550 (0.00986755 s)         |
| 10        | 7678250 (0.00767825 s)         | 5830900 (0.0058309 s)          |
| 11        | 11948000 (0.011948 s)          | 50630600 (0.0506306 s)         |


## Measurements of bit-packing + img2col (-O3 flag):

| Test Case | Original Runtime (ns)       | Optimized Runtime (ns)        |
|-----------|-----------------------------|-------------------------------|
| 0         | 305800 (0.0003058 s)        | 2040400 (0.0020404 s)         |
| 1         | 279050 (0.00027905 s)       | 1742050 (0.00174205 s)        |
| 2         | 132000 (0.000132 s)         | 834400 (0.0008344 s)          |
| 3         | 128700 (0.0001287 s)        | 886900 (0.0008869 s)          |
| 4         | 57850 (0.00005785 s)        | 402950 (0.00040295 s)         |
| 5         | 61050 (0.00006105 s)        | 387150 (0.00038715 s)         |
| 6         | 15925450 (0.01592545 s)     | 92939900 (0.0929399 s)        |
| 7         | 12157500 (0.0121575 s)      | 22958500 (0.0229585 s)        |
| 8         | 11213400 (0.0112134 s)      | 10437000 (0.010437 s)         |
| 9         | 10763850 (0.01076385 s)     | 6261850 (0.00626185 s)        |
| 10        | 2210600 (0.0022106 s)       | 2025850 (0.00202585 s)        |
| 11        | 3365800 (0.0033658 s)       | 14516200 (0.0145162 s)        |



## Test Cases Used:

| c   | h   | w   | kn  | kh  | kw  | p   | s   |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 64  | 56  | 56  | 64  | 3   | 3   | 1   | 1   |
| 64  | 56  | 56  | 128 | 3   | 3   | 1   | 1   |
| 128 | 28  | 28  | 128 | 3   | 3   | 1   | 1   |
| 128 | 28  | 28  | 256 | 3   | 3   | 1   | 1   |
| 256 | 14  | 14  | 256 | 3   | 3   | 1   | 1   |
| 256 | 14  | 14  | 512 | 3   | 3   | 1   | 1   |
| 80  | 224 | 224 | 80  | 3   | 3   | 1   | 1   |
| 80  | 224 | 224 | 80  | 3   | 3   | 1   | 2   |
| 80  | 224 | 224 | 80  | 3   | 3   | 1   | 3   |
| 80  | 224 | 224 | 80  | 3   | 3   | 1   | 4   |
| 512 | 56  | 56  | 256 | 1   | 1   | 0   | 1   |
| 512 | 56  | 56  | 256 | 3   | 3   | 1   | 1   |



# CODE
"HIGH LEVEL IDEA" and "IMPLEMENTATION DETAILS" were written when the following code was the most recent version, other text from this doc might have been written when modified versions of this code were the most recent ones. 

std::vector<int64_t> fused(float * X, float * Q_Threshold, int64_t * QWeights, int * BTN_CNT1, ConvType TYPE, int PaddingH, int PaddingW, int StrideH, int StrideW, int Batch_Size, int C, int H, int W, int KN, int KH, int KW, float ReLU_alpha) {
    int PackedH, PackedW, OH, OW, PackedC;
    std::vector<int64_t> qx_mid;
    std::vector<int> yi;

    //Ternarize_NCHW_to_NHWCB
    const int64_t one = 1;
    int64_t onebit[cntbits];
    // 64-bits, set each bit
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = one << i;
    }

    // The quantized qx, in N_H_W_C_B format
    const int priChannel = C / cntbits;
    // packC: actual packed input channel
    const int packC = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int packH = H + 2 * PaddingH;
    const int packW = W + 2 * PaddingW;

    qx_mid = std::vector<int64_t>(Batch_Size * packH * packW * packC * BITS, 0);
    int64_t* qxptr;
    qxptr = qx_mid.data();

    PackedH = H + 2 * PaddingH; // Height after bit-packing
    PackedW = W + 2 * PaddingW; // Width  after bit-packing
    OH = (PackedH - KH + 1) / StrideH; // Output Height
    OW = (PackedW - KW + 1) / StrideW; // Output Width
    const int H1 = OH * OW;      // Fused Height
    PackedC = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // The channel after bit-packing
    int C_packed_bits = PackedC * BITS;
    const int W1 = KH * KW * C_packed_bits;  // Fused Width

    std::vector<int64_t> qx;
    qx = std::vector<int64_t>(Batch_Size * H1 * W1);

    for (int n = 0; n < Batch_Size; n++) {
        for (int oh = 0; oh < OH; oh++) {
          for (int ow = 0; ow < OW; ow++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        for (int c = 0; c < C_packed_bits; c++) {
                            // compute inds
                            int h = oh * StrideH + kh - PaddingH;
                            int w = ow * StrideW + kw - PaddingW;

                            // compute value
                            int64_t p1 = 0;
                            int64_t p2 = 0;

                            // figure out if c indexes into data produced by loop or by if condition
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                if (c >= priChannel * 2) {  // c indexes into if condition data
                                    for (int bit = 0; bit < (C % cntbits); bit++) {
                                        float currentx = X[((n * C + (priChannel * cntbits + bit)) * H + h) * W + w];
                                        if (currentx > Q_Threshold[n]) {
                                            p2 = p2 | onebit[bit];
                                        }
                                        else if (currentx < (-Q_Threshold[n])) {
                                            p1 = p1 | onebit[bit];
                                            p2 = p2 | onebit[bit];
                                        }
                                    }
                                } else {  // c indexes into loop data so we need to compute ic
                                    int ic = c / 2;
                                    for (int bit = 0; bit < cntbits; bit++) {
                                        float currentx = X[((n * C + (ic * cntbits + bit)) * H + h ) * W + w];
                                        if (currentx > Q_Threshold[n]) {
                                            p2 = p2 | onebit[bit];
                                        }
                                        else if (currentx < (-Q_Threshold[n])) {
                                            p1 = p1 | onebit[bit];
                                            p2 = p2 | onebit[bit];
                                        }
                                    }
                                }
                            }

                            // writeback 
                            qx[(n * H1 + oh * OW + ow) * W1 + kh * KW * C_packed_bits + kw * C_packed_bits + c] = (c % 2 == 0) ? p1 : p2;
                        }
                    }
                }
            }
        }
    } 
    return qx;
}

# HIGH LEVEL IDEA "Robins Approach"
- "Robins approach": compute packed values with normal image loop order and then when a packed value is ready write it to all places in data matrix where it belongs (see E which gets placed onto entire diagonal in "flow" picture). DRAWBACK: bad write locality (e.g. the writes of E along the diagonal are not consecutive in memory). Advantage: no redundant computation is done.
- We need to have the function in 2 data formats. Once for the default (first layer) case NCHW -> NHWC and once for NHWC -> NHWC.
- The fused version of the NHWC -> NHWC case can be found in fused_nhwc.cpp.
- The code iterates through the data matrix in the correct ordering. It performs the quantization along the C dimension.
- whenever the channel has been quantized, it gets written to the (at most) KH * KW locations in the output array.

# NCHW to NHWC (fused)
Done by Lucas and Yuliia after Robin had done his fusing.

Took robins code as a starting point. changed read indexing of X to how its done in original ternarize (because unlike robins code we get tensor in NCHW). changed "(kh * KW + kw) * packC * BITS + ic" to "(kh * KW + kw) * packC * BITS + ic * BITS" (i.e. add the * BITS at the end) in writeback of ic loop. one test case was still failing so we searched for edge cases and came up with sightly improved way of computing OH and OW:
    int OH = (packH - KH) / stride;
    int OW = (packW - KW) / stride;

    if ((packH - KH) % stride == (packW - KW) % stride) {
        OH = (packH - KH) / stride + 1;
        OW = (packW - KW) / stride + 1;
    }