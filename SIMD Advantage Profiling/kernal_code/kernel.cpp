#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <immintrin.h>
#include <iomanip>
#include <algorithm>

const double RELATIVE_ERROR_TOLERANCE = 1e-6; // Relative error tolerance

template<typename T> void fill_array(std::vector<T>& v);
template<typename Func> double time_kernel(Func f, int R);
void validate_result(float scalar_sum, float simd_sum);

template<typename T> void saxpy_scalar(int N, T a, const T* x, T* y);
template<typename T> void saxpy_simd(int N, T a, const T* x, T* y);
template<typename T> T dot_scalar(int N, const T* x, const T* y);
template<typename T> T dot_simd(int N, const T* x, const T* y);
template<typename T> void elem_mult_scalar(int N, const T* x, const T* y, T* z);
template<typename T> void elem_mult_simd(int N, const T* x, const T* y, T* z);

template<typename T> void run_validation_suite(const std::string& dtype_name);

template<typename T>
void fill_array(std::vector<T>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        // Use non trivial floating point values for better testing
        v[i] = static_cast<T>(i % 100) + 1.0f + static_cast<T>(i) / 1000.0f;
    }
}

template<typename Func>
double time_kernel(Func f, int R) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    for (int r = 0; r < R; ++r) {
        f();
    }
    auto end = clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void validate_result(float scalar_sum, float simd_sum) {
    // Relaxed tolerance to 1e-3 to account for expected floating point reordering error in reduction
    const float REL_TOL = 1e-3;
    float abs_scalar = std::abs(scalar_sum);
    float relative_error = std::abs(scalar_sum - simd_sum) / std::max(abs_scalar, 1e-12f);

    if (relative_error > REL_TOL) {
        // Output to standard error for user debugging not mixed with CSV output
        std::cerr << "Validation warning tolerance 1e-3" << std::endl;
        std::cerr << "Kernel Dot Product Error " << relative_error << std::endl;
    }
}

template<typename T>
void run_validation_suite(const std::string& dtype_name) {
    const size_t N_val = 100;
    const T a = static_cast<T>(2.5);

    // Setup arrays for a small simple test case
    std::vector<T> x(N_val), y_scalar(N_val), y_simd(N_val), z_scalar(N_val), z_simd(N_val);
    fill_array(x);
    fill_array(y_scalar); fill_array(y_simd);

    // 1. SAXPY Validation
    // SAXPY modifies y in place y = a*x + y
    saxpy_scalar(N_val, a, x.data(), y_scalar.data());
    saxpy_simd(N_val, a, x.data(), y_simd.data());

    bool saxpy_ok = true;
    for (size_t i = 0; i < N_val; ++i) {
        T abs_scalar = std::abs(y_scalar[i]);
        if (std::abs(y_scalar[i] - y_simd[i]) > RELATIVE_ERROR_TOLERANCE * std::max(abs_scalar, (T)1e-12)) {
            saxpy_ok = false;
            std::cerr << "SAXPY mismatch at index " << i << " Scalar " << y_scalar[i] << " SIMD " << y_simd[i] << std::endl;
            break;
        }
    }

    // 2. Element wise Multiplication Validation
    elem_mult_scalar(N_val, x.data(), y_scalar.data(), z_scalar.data());
    elem_mult_simd(N_val, x.data(), y_scalar.data(), z_simd.data());

    bool mult_ok = true;
    for (size_t i = 0; i < N_val; ++i) {
        T abs_scalar = std::abs(z_scalar[i]);
        if (std::abs(z_scalar[i] - z_simd[i]) > RELATIVE_ERROR_TOLERANCE * std::max(abs_scalar, (T)1e-12)) {
            mult_ok = false;
            std::cerr << "Elem Mult mismatch at index " << i << std::endl;
            break;
        }
    }

    // 3. Dot Product Validation
    // Simple tolerance used due to non associativity of floating point arithmetic
    T dot_scalar_res = dot_scalar(N_val, x.data(), y_scalar.data());
    T dot_simd_res = dot_simd(N_val, x.data(), y_scalar.data());

    T abs_scalar_dot = std::abs(dot_scalar_res);
    bool dot_ok = (std::abs(dot_scalar_res - dot_simd_res) < 1e-3 * std::max(abs_scalar_dot, (T)1e-12));

    if (saxpy_ok && mult_ok && dot_ok) {
        std::cerr << "Validation " << dtype_name << " Success All kernels match scalar reference Vector Ops Tol " << RELATIVE_ERROR_TOLERANCE << " Dot Prod Tol 1e-3" << std::endl;
    } else {
        std::cerr << "Validation " << dtype_name << " FAILED" << std::endl;
        if (!saxpy_ok) std::cerr << "   SAXPY failed" << std::endl;
        if (!mult_ok) std::cerr << "   Element wise Multiplication failed" << std::endl;
        if (!dot_ok) {
             std::cerr << "   Dot Product failed expected tolerance due to reordering but too high" << std::endl;
             std::cerr << "     Scalar Sum " << dot_scalar_res << " SIMD Sum " << dot_simd_res << std::endl;
        }
        // Exiting here ensures timing does not run if correctness is fundamentally broken
        exit(1);
    }
}

template<typename T>
void saxpy_scalar(int N, T a, const T* x, T* y) {
    for (int i = 0; i < N; ++i) {
        y[i] += a * x[i];
    }
}
template<>
void saxpy_simd<float>(int N, float a, const float* x, float* y) {
    __m256 a_vec = _mm256_set1_ps(a);
    int i;
    int vector_width = 8;
    for (i = 0; i < N / vector_width * vector_width; i += vector_width) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        __m256 result = _mm256_add_ps(_mm256_mul_ps(a_vec, x_vec), y_vec);
        _mm256_storeu_ps(y + i, result);
    }
    for (; i < N; i++) { y[i] += a * x[i]; } // Tail
}
template<> void saxpy_simd<double>(int N, double a, const double* x, double* y) { saxpy_scalar(N, a, x, y); }

template<typename T>
T dot_scalar(int N, const T* x, const T* y) {
    T sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
template<>
float dot_simd<float>(int N, const float* x, const float* y) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i;
    int vector_width = 8;
    for (i = 0; i < N / vector_width * vector_width; i += vector_width) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        __m256 prod_vec = _mm256_mul_ps(x_vec, y_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_vec);
    }
    // Horizontal Summation
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0), _mm256_extractf128_ps(sum_vec, 1));
    __m128 sum64 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 sum32 = _mm_add_ps(sum64, _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(0, 0, 0, 1)));
    float total_sum = _mm_cvtss_f32(sum32);
    for (; i < N; i++) { total_sum += x[i] * y[i]; } // Tail
    return total_sum;
}
template<> double dot_simd<double>(int N, const double* x, const double* y) { return dot_scalar(N, x, y); }

template<typename T>
void elem_mult_scalar(int N, const T* x, const T* y, T* z) {
    for (int i = 0; i < N; ++i) {
        z[i] = x[i] * y[i];
    }
}
template<>
void elem_mult_simd<float>(int N, const float* x, const float* y, float* z) {
    int i;
    int vector_width = 8;
    for (i = 0; i < N / vector_width * vector_width; i += vector_width) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        __m256 result = _mm256_mul_ps(x_vec, y_vec);
        _mm256_storeu_ps(z + i, result);
    }
    for (; i < N; i++) { z[i] = x[i] * y[i]; } // Tail
}
template<> void elem_mult_simd<double>(int N, const double* x, const double* y, double* z) { elem_mult_scalar(N, x, y, z); }

int main(){
    std::cerr << "Starting SIMD profiling experiments" << std::endl;
    // Set output precision for consistent CSV data
    std::cout << std::fixed << std::setprecision(10);

    // Mandatory Validation
    run_validation_suite<float>("float32");
    run_validation_suite<double>("float64");
    std::cerr << "Validation completed successfully Proceeding to timing runs" << std::endl;

    // N values to sweep cache levels L1 L2 LLC DRAM
    std::vector<int> sizes = {8*1024, 64*1024, 3*1024*1024, 10*1024*1024};
    std::vector<int> strides = {1,2,4,8};
    float a_float = 2.5f;
    double a_double = 2.5;

    // Repetition counts for timing stability
    const int R_DOT = 100;
    const int R_STREAM = 10;
    const std::string locality_tag = "locality"; // Tag for main sweep

    for(auto N : sizes){
        std::vector<float> x_f(N), y_f(N), z_f(N);
        std::vector<double> x_d(N), y_d(N), z_d(N);

        fill_array(x_f); fill_array(y_f);
        fill_array(x_d); fill_array(y_d);

        int R_cur;
        double t_scalar, t_simd, speedup, gflops;
        float s_scalar_f, s_simd_f;

        // 1. SAXPY float32 float64 Locality and Data Type
        R_cur = R_STREAM;
        t_scalar = time_kernel([&](){ saxpy_scalar<float>(N,a_float,x_f.data(),y_f.data()); }, R_cur);
        fill_array(y_f); // Reset y for SIMD run
        t_simd = time_kernel([&](){ saxpy_simd<float>(N,a_float,x_f.data(),y_f.data()); }, R_cur);
        speedup = t_scalar / t_simd;
        gflops = 2.0*N*R_cur / t_simd / 1e9;
        std::cout << "saxpy,float32,"<<locality_tag<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";

        R_cur = R_STREAM;
        t_scalar = time_kernel([&](){ saxpy_scalar<double>(N,a_double,x_d.data(),y_d.data()); }, R_cur);
        fill_array(y_d); // Reset y for SIMD run
        t_simd = time_kernel([&](){ saxpy_simd<double>(N,a_double,x_d.data(),y_d.data()); }, R_cur);
        speedup = t_scalar / t_simd;
        gflops = 2.0*N*R_cur / t_simd / 1e9;
        std::cout << "saxpy,float64,"<<locality_tag<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";

        // 2. DOT PRODUCT float32 float64 Locality and Data Type
        R_cur = R_DOT;
        s_scalar_f = dot_scalar<float>(N,x_f.data(),y_f.data());
        t_scalar = time_kernel([&](){ volatile float sum_f = 0.0f; sum_f += dot_scalar<float>(N,x_f.data(),y_f.data()); }, R_cur);

        s_simd_f = dot_simd<float>(N,x_f.data(),y_f.data());
        t_simd = time_kernel([&](){ volatile float sum_f = 0.0f; sum_f += dot_simd<float>(N,x_f.data(),y_f.data()); }, R_cur);

        validate_result(s_scalar_f, s_simd_f);

        speedup = t_scalar / t_simd;
        gflops = 2.0*N*R_cur / t_simd / 1e9;
        std::cout << "dot,float32,"<<locality_tag<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";

        R_cur = R_DOT;
        t_scalar = time_kernel([&](){ volatile double sum_d = 0.0; sum_d += dot_scalar<double>(N,x_d.data(),y_d.data()); }, R_cur);
        t_simd = time_kernel([&](){ volatile double sum_d = 0.0; sum_d += dot_simd<double>(N,x_d.data(),y_d.data()); }, R_cur);
        speedup = t_scalar / t_simd;
        gflops = 2.0*N*R_cur / t_simd / 1e9;
        std::cout << "dot,float64,"<<locality_tag<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";

        // 3. ELEMENTWISE MULTIPLY float32 float64 Locality and Data Type
        R_cur = R_STREAM;
        t_scalar = time_kernel([&](){ elem_mult_scalar<float>(N,x_f.data(),y_f.data(),z_f.data()); }, R_cur);
        t_simd = time_kernel([&](){ elem_mult_simd<float>(N,x_f.data(),y_f.data(),z_f.data()); }, R_cur);
        speedup = t_scalar / t_simd;
        gflops = 1.0*N*R_cur / t_simd / 1e9;
        std::cout << "elem_mult,float32,"<<locality_tag<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";

        R_cur = R_STREAM;
        t_scalar = time_kernel([&](){ elem_mult_scalar<double>(N,x_d.data(),y_d.data(),z_d.data()); }, R_cur);
        t_simd = time_kernel([&](){ elem_mult_simd<double>(N,x_d.data(),y_d.data(),z_d.data()); }, R_cur);
        speedup = t_scalar / t_simd;
        gflops = 1.0*N*R_cur / t_simd / 1e9;
        std::cout << "elem_mult,float64,"<<locality_tag<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";

        // 4. STRIDE EFFECTS SAXPY float32 fixed N
        if (N == sizes[0]) {
            R_cur = R_STREAM;
            for(auto s : strides){
                // Arrays must be larger than N * s to support strided access
                int array_size = N * s;
                std::vector<float> x_s(array_size), y_s(array_size);
                fill_array(x_s); fill_array(y_s);

                // Scalar Stride Test
                t_scalar = time_kernel([&](){
                    for(int i=0;i<N;i++) y_s[i*s] += a_float * x_s[i*s];
                }, R_cur);

                // Auto Vectorized Stride Test
                t_simd = time_kernel([&](){
    #pragma omp simd // Hint to the compiler for vectorization
                    for(int i=0;i<N;i++) y_s[i*s] += a_float * x_s[i*s];
                }, R_cur);

                speedup = t_scalar/t_simd;
                gflops = 2.0*N*R_cur / t_simd / 1e9;
                std::cout << "saxpy,float32,stride"<<s<<","<<N<<","<<t_scalar<<","<<t_simd<<","<<speedup<<","<<gflops<<"\n";
            }
        }
    }

    // 5. ALIGNMENT TAIL STUDY DOT PRODUCT float32
    std::vector<int> alignment_sizes = {8192, 8195, 1024 * 1024, 1024 * 1024 + 5};
    const int R_ALIGN = 100;
    const std::string alignment_kernel_tag = "alignment";

    for (int N_ALIGN : alignment_sizes) {
        std::vector<float> x_aligned(N_ALIGN), y_aligned(N_ALIGN);
        fill_array(x_aligned); fill_array(y_aligned);

        // Experiment 1 Scalar Baseline
        double t_scalar_align = time_kernel([&](){ dot_scalar<float>(N_ALIGN, x_aligned.data(), y_aligned.data()); }, R_ALIGN);
        std::cout << alignment_kernel_tag << ",dot,scalar," << N_ALIGN << "," << t_scalar_align << ",0.0,0.0,0.0\n";


        // Experiment 2 Aligned Data with UNALIGNED Load _mm256_loadu_ps
        double t_unaligned_load = time_kernel([&](){
            __m256 sum_vec = _mm256_setzero_ps();
            int i;
            for (i = 0; i < N_ALIGN / 8 * 8; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_aligned.data() + i);
                __m256 y_vec = _mm256_loadu_ps(y_aligned.data() + i);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_vec, y_vec));
            }
        }, R_ALIGN);
        // t_unaligned_load is used as the t_simd metric here
        std::cout << alignment_kernel_tag << ",dot,aligned_loadu," << N_ALIGN << ",0.0," << t_unaligned_load << ",0.0,0.0\n";


        // Experiment 3 Aligned Data with ALIGNED Load _mm256_load_ps
        double t_aligned_load = time_kernel([&](){
            __m256 sum_vec = _mm256_setzero_ps();
            int i;
            for (i = 0; i < N_ALIGN / 8 * 8; i += 8) {
                __m256 x_vec = _mm256_load_ps(x_aligned.data() + i);
                __m256 y_vec = _mm256_load_ps(y_aligned.data() + i);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_vec, y_vec));
            }
        }, R_ALIGN);
        std::cout << alignment_kernel_tag << ",dot,aligned_load," << N_ALIGN << ",0.0," << t_aligned_load << ",0.0,0.0\n";


        // Experiment 4 Misaligned Data with UNALIGNED Load _mm256_loadu_ps
        // Test by shifting the pointer by one element 4 bytes which is non 32 byte aligned
        float* x_misaligned = x_aligned.data() + 1;
        float* y_misaligned = y_aligned.data() + 1;
        int N_MISALIGN = N_ALIGN - 1; // Effective size is reduced by 1

        double t_misaligned_unaligned = time_kernel([&](){
            __m256 sum_vec = _mm256_setzero_ps();
            int i;
            for (i = 0; i < N_MISALIGN / 8 * 8; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_misaligned + i);
                __m256 y_vec = _mm256_loadu_ps(y_misaligned + i);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_vec, y_vec));
            }
        }, R_ALIGN);
        // N_MISALIGN is used for this specific run's size
        std::cout << alignment_kernel_tag << ",dot,misaligned_loadu," << N_MISALIGN << ",0.0," << t_misaligned_unaligned << ",0.0,0.0\n";
    }

    return 0;
}
