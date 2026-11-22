#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// The message is built once and stored by std::runtime_error, so what() stays valid.
class cuda_error : public std::runtime_error {
    public:
        cudaError_t code;

        // file/line come from the call site (see CUDA_ASSERT_SUCCESS in main.cu)
        cuda_error(cudaError_t code, const char* file, int line)
            : std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code)
                                 + " (" + file + ":" + std::to_string(line) + ")"),
              code(code) {}
};

#endif