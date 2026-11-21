#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "dft_core.cuh"

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
#endif

#define CUDA_REPORT_ERROR(call)                                                                             \
do {                                                                                                        \
    cudaError_t e = (call);                                                                                 \
    if (e != cudaSuccess) {                                                                                 \
    fprintf(stderr,  "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; }    \
} while (0)


int main(int argc, char* argv[]) {
    /* Argument:
    *   dft.exe      
    *   <audio_path>
    *   <sample_rate>
    *   <sample_per_frame>
    *   <total_samples>
    *   <bar_count>
    *   [<f0> <f1>...]
    */

#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    const int arg_types = 7;

    if (argc < arg_types) { 
        fprintf(stderr, "usage: dft <audio> <sample rate> <sample per frame> <total sample> <#of frequencies> <freqencies...>\n"); 
        return 1; 
    }

    const char* audio_path  = argv[1];
    int sample_rate         = std::stoi(argv[2]);
    int sample_per_frame    = std::stoi(argv[3]);
    int distance_frames     = std::stoi(argv[4]);
    int total_sample_count  = std::stoi(argv[5]);
    int bar_count           = std::stoi(argv[6]);
    if (argc != arg_types + bar_count) { fprintf(stderr, "expected %d frequencies\n", bar_count); return 1; }

    const int frame_count = (total_sample_count + distance_frames - 1) / distance_frames;

    float* frequencies = nullptr;
    float* audio_data = nullptr;
    CUDA_REPORT_ERROR(cudaMallocManaged(&frequencies, bar_count * sizeof(float)));
    CUDA_REPORT_ERROR(cudaMallocManaged(&audio_data, size_t(total_sample_count)*sizeof(float)));
    for (int i = 0; i < bar_count; ++i) frequencies[i] = std::stof(argv[arg_types + i]);


    std::ifstream audio_file(audio_path, std::ios::binary);
    if (!audio_file) { fprintf(stderr, "cannot open audio file."); return 1; }

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
    dft_kernel<<<frame_count, bar_count>>>( audio_data,                // input array
                                            frequencies,               // all the frequencies
                                            sample_rate,               // sample rate (Hz)
                                            sample_per_frame,          // #sample per frame
                                            distance_frames,           // the hop between each frame
                                            total_sample_count,        // #sample point in total
                                            out);                      // output

    CUDA_REPORT_ERROR(cudaGetLastError());
    CUDA_REPORT_ERROR(cudaDeviceSynchronize());

    /*--------------OUTPUT-------------*/
    fwrite(out_buf, sizeof(float), size_t(frame_count) * bar_count, stdout);
    fflush(stdout);
    
    cudaFree(frequencies); cudaFree(audio_data); cudaFree(out_buf);
    return 0;
}