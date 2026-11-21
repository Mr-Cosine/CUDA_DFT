#ifndef DFT_CORE_H
#define DFT_CORE_H

#ifdef __CUDACC__
  #define GLB __global__
  #define RESTR __restrict__
#else
  #define GLB
  #define RESTR
#endif

#define TPB 128                                                 // 1 block = 1 frame = 128 frequencies = 128 threads
#define PI 3.1416f

#include "complexf.cuh"
#include "matrix.cuh"

typedef Complex<float> complexf;
typedef Matrix<float> matrixf;

GLB void dft_kernel(const float* RESTR audio_data,              // input array
                    const float* RESTR frequencies,             // all the frequencies
                    const int sample_rate,                      // sample rate (Hz)
                    const int spf,                              // #sample per frame
                    const int total_sample_count,               // #sample point in total
                    matrixf out                                 // output array
                   ) {

    int f_idx = blockIdx.x;
    int k     = threadIdx.x;

    int start_idx = f_idx * spf;
    int end_idx = (total_sample_count > (f_idx+1) * spf) ? 
                        (f_idx+1) * spf :
                        total_sample_count;

    float f = frequencies[k];

    float angle = 2*PI*f/sample_rate;

    complexf theta(1.0f, 0.0f);                                 // sine wave
    complexf dtheta(cos(-angle), sin(-angle));

    float* out_this = &out[f_idx][k];

    if (end_idx - start_idx < spf) {
        *out_this = 0;
        return;
    }                                                           // EOF - break and just give a bin = 0 back

    complexf acc(0.0f, 0.0f);
    float ncoeff = 0.0f;

    for (int i = 0; i < spf; ++i) {
        float multiplier = 0.5f - 0.5f * cosf(2*PI*i/(spf-1));
        acc += theta * (audio_data[start_idx + i] * multiplier);

        ncoeff += multiplier;

        theta *= dtheta;
    }

    *out_this = acc.abs() / ncoeff;
}

#endif