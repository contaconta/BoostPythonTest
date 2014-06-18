#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/numpy.hpp>
#include <boost/scoped_array.hpp>

#include <xmmintrin.h>
#include <immintrin.h>

namespace bp = boost::python;
namespace np = boost::numpy;


static void debug_print(const np::ndarray& arr)
{
    std::cout << "\n get_nd:" << arr.get_nd()
              << "\n rows:" << arr.shape(0)
              << "\n cols:" << arr.shape(1)
              << std::endl;
}


static void validateArrayForAdd2d(const np::ndarray& array)
{
    if (array.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        bp::throw_error_already_set();
    }
    if (array.get_nd() != 2) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        bp::throw_error_already_set();
    }
    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous");
        bp::throw_error_already_set();
    }
}


static np::ndarray add2d(const np::ndarray& arr1, const np::ndarray& arr2)
{
    validateArrayForAdd2d(arr1);
    validateArrayForAdd2d(arr2);

    if(arr1.shape(0) != arr2.shape(0) ||
       arr1.shape(1) != arr2.shape(1)) {
        PyErr_SetString(PyExc_TypeError, "Array must be same size: arr1.shape(0) != arr2.shape(0) || arr1.shape(1) != arr2.shape(1)");
        bp::throw_error_already_set();
    }

    const int rows = arr1.shape(0);
    const int cols = arr1.shape(1);

    double* arr1_ptr = reinterpret_cast<double*>(arr1.get_data());
    double* arr2_ptr = reinterpret_cast<double*>(arr2.get_data());

    bp::tuple shape = bp::make_tuple(rows, cols);
    np::dtype dtype = np::dtype::get_builtin<double>();
    np::ndarray result_arr = np::empty(shape, dtype);

    double* result_arr_ptr = reinterpret_cast<double*>(result_arr.get_data());

#if __AVX__
    for(int i=0; i<cols*rows; i+=4){
        __m256d a = _mm256_load_pd(arr1_ptr + i);
        __m256d b = _mm256_load_pd(arr2_ptr + i);
        a = _mm256_add_pd(a, b);
        _mm256_store_pd(result_arr_ptr+i, a);
    }
#elif __SSE3__
    for(int i=0; i<cols*rows; i+=2){
        __m128d a = _mm_load_pd(arr1_ptr + i);
        __m128d b = _mm_load_pd(arr2_ptr + i);
        a = _mm_add_pd(a, b);
        _mm_store_pd(result_arr_ptr+i, a);
    }
#else
    for(int j=0; j<rows; ++j){
        for(int i=0; i<cols; ++i){
            *(result_arr_ptr + j*cols + i)
                    = *(arr1_ptr + j*cols + i) + *(arr2_ptr + j*cols + i);
        }
    }
#endif

    return result_arr;
}


static np::ndarray mul2d(const np::ndarray& arr1, const np::ndarray& arr2)
{
    validateArrayForAdd2d(arr1);
    validateArrayForAdd2d(arr2);

    if(arr1.shape(0) != arr2.shape(0) ||
       arr1.shape(1) != arr2.shape(1)) {
        PyErr_SetString(PyExc_TypeError, "Array must be same size: arr1.shape(0) != arr2.shape(0) || arr1.shape(1) != arr2.shape(1)");
        bp::throw_error_already_set();
    }

    const int rows = arr1.shape(0);
    const int cols = arr1.shape(1);

    double* arr1_ptr = reinterpret_cast<double*>(arr1.get_data());
    double* arr2_ptr = reinterpret_cast<double*>(arr2.get_data());

    bp::tuple shape = bp::make_tuple(rows, cols);
    np::dtype dtype = np::dtype::get_builtin<double>();
    np::ndarray result_arr = np::empty(shape, dtype);

    double* result_arr_ptr = reinterpret_cast<double*>(result_arr.get_data());

#if __AVX__
    for(int i=0; i<cols*rows; i+=4){
        __m256d a = _mm256_load_pd(arr1_ptr + i);
        __m256d b = _mm256_load_pd(arr2_ptr + i);
        a = _mm256_mul_pd(a, b);
        _mm256_store_pd(result_arr_ptr+i, a);
    }
#elif __SSE3__
    for(int i=0; i<cols*rows; i+=2){
        __m128d a = _mm_load_pd(arr1_ptr + i);
        __m128d b = _mm_load_pd(arr2_ptr + i);
        a = _mm_mul_pd(a, b);
        _mm_store_pd(result_arr_ptr+i, a);
    }
#else
    for(int j=0; j<rows; ++j){
        for(int i=0; i<cols; ++i){
            *(result_arr_ptr + j*cols + i)
                    = *(arr1_ptr + j*cols + i) + *(arr2_ptr + j*cols + i);
        }
    }
#endif

    return result_arr;
}



BOOST_PYTHON_MODULE(libnumpytest)
{
    np::initialize();
    boost::python::def("debug_print", debug_print);
    boost::python::def("add2d", add2d);
    boost::python::def("mul2d", mul2d);
}
