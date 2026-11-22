#ifndef COMPLEXF_H
#define COMPLEXF_H

#include <math.h>

// Functions marked HD compile for both CPU and GPU under nvcc,
// and the macro expands to nothing with a normal C++ compiler.
#ifdef __CUDACC__
  #define HD __host__ __device__
#else
  #define HD
#endif

template <typename T>
struct Polar_coord {
    T r;
    T theta;

    HD Polar_coord(T new_r, T new_theta) : r(new_r), theta(new_theta) {}
};

// alignas: lets the GPU load/store both parts in one 8-byte (float) access
template <typename T>
class alignas(2 * sizeof(T)) Complex {
    private:
        T rval;
        T ival;

    public:
        // static member instead of a hidden friend, so member functions can find it;
        // fabs (not fabsf) picks the right overload for float and double
        HD static bool is_equal(T a, T b) { return fabs(a - b) < T(1e-5); }

        // Constructors
        HD Complex(T r = T(0), T i = T(0)) : rval(r), ival(i) {}
        HD Complex(const Polar_coord<T>& c)
            : rval(c.r * cos(c.theta)), ival(c.r * sin(c.theta)) {}
        // Copy constructor and operator= removed: the compiler-generated ones
        // do exactly the same thing and keep the type trivially copyable.

        // Getters
        HD T real() const { return rval; }
        HD T imaginary() const { return ival; }
        HD T norm2() const { return rval * rval + ival * ival; }   // |z|^2, no sqrt
        HD T abs() const { return sqrt(norm2()); }
        HD T arg() const { return atan2(ival, rval); }

        // Conjugate pairs
        HD Complex conjugate() const { return Complex(rval, -ival); }
        HD static Complex conjugate(const Complex& num) { return Complex(num.rval, -num.ival); }

        HD bool is_conjugate(const Complex& num) const {
            return is_equal(rval, num.rval) && is_equal(ival, -num.ival);
        }
        HD static bool is_conjugate(const Complex& n1, const Complex& n2) {
            return is_equal(n1.rval, n2.rval) && is_equal(n1.ival, -n2.ival);
        }

        // Operators
        HD Complex operator + (const T& k) const { return Complex(rval + k, ival); }
        HD Complex operator + (const Complex& num) const { return Complex(rval + num.rval, ival + num.ival); }

        HD Complex operator - (const T& k) const { return Complex(rval - k, ival); }
        HD Complex operator - (const Complex& num) const { return Complex(rval - num.rval, ival - num.ival); }
        HD Complex operator - () const { return Complex(-rval, -ival); }

        HD Complex operator * (const T& k) const { return Complex(rval * k, ival * k); }
        HD Complex operator * (const Complex& num) const {
            return Complex(rval * num.rval - ival * num.ival,
                            rval * num.ival + ival * num.rval);
        }

        HD Complex operator / (const T& k) const { return Complex(rval / k, ival / k); }
        HD Complex operator / (const Complex& num) const {
            return (*this * num.conjugate()) / num.norm2();
        }
        
        HD Complex pow(T p) const {
            if (rval == T(0) && ival == T(0)) return Complex(T(0));
            T mag = exp(p * log(abs()));    // |z|^p
            T ang = p * arg();              // p × angle
            return Complex(mag * cos(ang), mag * sin(ang));
        }

        HD Complex& operator += (const T& k) { rval += k; return *this; }
        HD Complex& operator += (const Complex& num) { rval += num.rval; ival += num.ival; return *this; }

        HD Complex& operator -= (const T& k) { rval -= k; return *this; }
        HD Complex& operator -= (const Complex& num) { rval -= num.rval; ival -= num.ival; return *this; }

        HD Complex& operator *= (const T& k) { rval *= k; ival *= k; return *this; }
        HD Complex& operator *= (const Complex& num) { *this = *this * num; return *this; }

        HD Complex& operator /= (const T& k) { rval /= k; ival /= k; return *this; }
        HD Complex& operator /= (const Complex& num) { *this = *this / num; return *this; }

        HD bool operator == (const Complex& num) const {
            return is_equal(rval, num.rval) && is_equal(ival, num.ival);
        }
        HD bool operator != (const Complex& num) const { return !(*this == num); }
};

template <typename T>
HD inline Complex<T> operator + (const T& k, const Complex<T>& num) { return num + k; }
template <typename T>
HD inline Complex<T> operator - (const T& k, const Complex<T>& num) { return Complex<T>(k) - num; }
template <typename T>
HD inline Complex<T> operator * (const T& k, const Complex<T>& num) { return num * k; }
template <typename T>
HD inline Complex<T> operator / (const T& k, const Complex<T>& num) { return Complex<T>(k) / num; }

template <typename T>
HD inline Complex<T> polar(const T& r, const T& theta) {
    return Complex<T>(Polar_coord<T>(r, theta));
}

#endif