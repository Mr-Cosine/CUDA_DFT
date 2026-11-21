#ifdef __CUDACC__
  #define HD __host__ __device__
#else
  #define HD
#endif

template<typename T>
struct Matrix {
    std::size_t row, col;
    T* data;

    Matrix(T* data, size_t row, size_t col) : row(row), col(col), data(data) {}
    HD T* operator[](std::size_t r) { return data + r * col; }
};

template<typename T, size_t row, size_t col>
class Matrix_core {
    T data[row*col];
public:
    Matrix_core() {}

    operator Matrix<T>() { return Matrix<T>(data, row, col); }

    HD T*       operator[](std::size_t r)       { return data + r * col; }
    HD const T* operator[](std::size_t r) const { return data + r * col; }
};