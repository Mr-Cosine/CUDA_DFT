#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "dft_core.cuh"
#include "exceptions.h"

#define API(declspec) extern "C" __declspec(declspec)

#define CUDA_ASSERT_SUCCESS(call)                                                               \
    do {                                                                                        \
        cudaError_t e = (call);                                                                 \
        if (e != cudaSuccess) {                                                                 \
            throw cuda_error(e, __FILE__, __LINE__);                                            \
    }} while (0)

static thread_local std::string error = "";
static void report_error(const std::exception& e) {
    error = e.what();
}

// Message of the last failed audio_dft call on this thread.
API(dllexport) const char* error_recall() {
    return error.c_str();
}

/* Returns 0 on success, 1 on failure (message via error_recall()).
*   output      [out] frame_count * bar_count floats, row-major (frames x frequencies),
*                     frame_count = ceil(total_sample_count / distance_frames)
*   samples     [in]  total_sample_count mono float32 samples
*   frequencies [in]  bar_count frequencies (Hz)
*/
API(dllexport) int audio_dft(float*         output,
                             const float*   samples,
                             const int      sample_rate,
                             const int      sample_per_frame,
                             const int      distance_frames,
                             const int      total_sample_count,
                             const int      bar_count,
                             const float*   freqcs) {
    float* frequencies = nullptr;
    float* audio_data = nullptr;
    float* result_arr = nullptr;

    try {
        if (!output || !samples || !freqcs)  throw std::invalid_argument("null pointer argument");
        if (sample_rate <= 0)                throw std::invalid_argument("sample_rate must be > 0");
        if (sample_per_frame < 2)            throw std::invalid_argument("sample_per_frame must be >= 2");
        if (distance_frames <= 0)            throw std::invalid_argument("distance_frames must be > 0");
        if (total_sample_count <= 0)         throw std::invalid_argument("total_sample_count must be > 0");
        if (bar_count <= 0 || bar_count > 1024)
            throw std::invalid_argument("bar_count must be in 1..1024 (one thread per frequency)");

        const int frame_count = (total_sample_count + distance_frames - 1) / distance_frames;
        const size_t audio_bytes = size_t(total_sample_count) * sizeof(float);
        const size_t out_bytes   = size_t(frame_count) * bar_count * sizeof(float);

        // IO data allocation
        CUDA_ASSERT_SUCCESS(cudaMallocManaged(&frequencies, bar_count * sizeof(float)));
        CUDA_ASSERT_SUCCESS(cudaMallocManaged(&audio_data, audio_bytes));
        CUDA_ASSERT_SUCCESS(cudaMallocManaged(&result_arr, out_bytes));

        /*--------------LOAD DATA-------------*/
        CUDA_ASSERT_SUCCESS(cudaMemcpy(frequencies, freqcs, bar_count * sizeof(float), cudaMemcpyDefault));
        CUDA_ASSERT_SUCCESS(cudaMemcpy(audio_data, samples, audio_bytes, cudaMemcpyDefault));

        /*--------------PROCESS-------------*/
        matrixf result(result_arr, frame_count, bar_count);
        dft_kernel<<<frame_count, bar_count>>>( audio_data,                // input array
                                                frequencies,               // all the frequencies
                                                sample_rate,               // sample rate (Hz)
                                                sample_per_frame,          // #sample per frame
                                                distance_frames,           // the hop between each frame
                                                total_sample_count,        // #sample point in total
                                                result);                   // output

        CUDA_ASSERT_SUCCESS(cudaGetLastError());
        CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());

        CUDA_ASSERT_SUCCESS(cudaMemcpy(output, result.data, out_bytes, cudaMemcpyDefault));

        cudaFree(audio_data); cudaFree(frequencies); cudaFree(result_arr);

        return 0;
    }
    catch (const std::exception& e) { report_error(e); cudaFree(audio_data); cudaFree(frequencies); cudaFree(result_arr); return 1; }
}
