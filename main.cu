#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "dft_core.cuh"

#define CUDA_REPORT_ERROR(call)                                                                             \
do {                                                                                                        \
    cudaError_t e = (call);                                                                                 \
    if (e != cudaSuccess) {                                                                                 \
    fprintf(stderr,  "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; }    \
} while (0)

int main() {
    /*
    * Needs for kernel:
    *    1.sample frequency
    *    2.total sample point count
    *    3.frequencies to test
    *    4.sample per frame
    *    5.audio data
    */

    /*--------------READ-------------*/
    int sample_rate, total_sample_count, sample_per_frame, bar_count;
    if (!(std::cin >> sample_rate >> total_sample_count >> sample_per_frame >> bar_count)) {
        fprintf(stderr, "bad config header\n"); return 1;
    }
    const int frame_count = (total_sample_count + sample_per_frame - 1) / sample_per_frame;

    std::string data_path;
    std::getline(std::cin >> std::ws, data_path);      // whole line, spaces included
    if (!std::cin) { fprintf(stderr, "bad config\n"); return 1; }

    std::ifstream audio_file(data_path, std::ios::binary);
    if (!audio_file) { fprintf(stderr, "cannot open %s\n", data_path.c_str()); return 1; }

    float* frequencies = nullptr;
    float* audio_data = nullptr;

    CUDA_REPORT_ERROR(cudaMallocManaged(&frequencies, bar_count*sizeof(float)));
    CUDA_REPORT_ERROR(cudaMallocManaged(&audio_data, size_t(total_sample_count)*sizeof(float)));

    for (int i = 0; i < bar_count; ++i) std::cin >> frequencies[i];
    
    audio_file.read(reinterpret_cast<char*>(audio_data), size_t(total_sample_count) * sizeof(float));

    if (audio_file.gcount() != (std::streamsize)(total_sample_count * sizeof(float))) {
        fprintf(stderr, "file is shorter than expected\n"); 
        return 1;
    }

    // output buffer
    float* out_buf = nullptr;
    CUDA_REPORT_ERROR(cudaMallocManaged(&out_buf, size_t(frame_count) * bar_count * sizeof(float)));
    matrixf out(out_buf, frame_count, bar_count);

    /*--------------PROCESS-------------*/
    dft_kernel<<<frame_count, TPB>>>(audio_data,                 // input array
                                frequencies,                // all the frequencies
                                sample_rate,                // sample rate (Hz)
                                sample_per_frame,           // #sample per frame
                                total_sample_count,         // #sample point in total
                                out);                       // output

    CUDA_REPORT_ERROR(cudaGetLastError());
    CUDA_REPORT_ERROR(cudaDeviceSynchronize());

    /*--------------OUTPUT-------------*/
    fwrite(out_buf, sizeof(float), size_t(frame_count) * bar_count, stdout);
    fflush(stdout);

    cudaFree(frequencies); cudaFree(audio_data); cudaFree(out_buf);

    return 0;
}