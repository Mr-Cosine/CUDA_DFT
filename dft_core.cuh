#ifndef DFT_CORE_H
#define DFT_CORE_H

#ifdef __CUDACC__
  #define GLB __global__
  #define RESTR __restrict__
#else
  #define GLB
  #define RESTR
#endif

#define PI 3.14159265359f

#include "complexf.cuh"
#include "matrix.cuh"

typedef Complex<float> complexf;
typedef Matrix<float> matrixf;

GLB void dft_kernel(const float* RESTR audio_data,              // input array
                    const float* RESTR frequencies,             // all the frequencies
                    const int sample_rate,                      // sample rate (Hz)
                    const int spf,                              // #sample per frame
                    const int dsf,                              // distance between frames
                    const int total_sample_count,               // #sample point in total
                    matrixf out                                 // output array
                   ) {

    int f_idx = blockIdx.x;
    int k     = threadIdx.x;

    if (k >= (int)out.col) return;

    float* out_this = &out[f_idx][k];

    int start_idx = f_idx * dsf;
    if (start_idx + spf > total_sample_count) {                 // window would run past the end
        *out_this = 0;
        return;
    }

    float f = frequencies[k];

    float angle = 2*PI*f/sample_rate;

    complexf theta(1.0f, 0.0f);                                 // sine wave
    complexf dtheta(cos(-angle), sin(-angle));

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