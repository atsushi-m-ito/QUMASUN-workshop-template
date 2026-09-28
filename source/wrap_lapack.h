#pragma once

#if defined(__INTEL_LLVM_COMPILER ) || defined(__INTEL_COMPILER )
#include <complex>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#ifdef NO_LAPACKE
#include <mkl_lapacke.h>
#endif
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
#elif defined(_NEC) || defined(__aocc__)
#undef HAVE_LAPACK_CONFIG_H
#pragma message("__aocc__ is defined")
#include <complex>
#include <cblas.h>
//#include <lapack.h>
#ifndef NO_LAPACKE
#include <lapacke.h>
#endif
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>

#else
#pragma message("__aocc__ is not defined")
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_CPP
#include <complex>
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
#define LAPACK_GLOBAL_PATTERN_UC
#define __EMSCRIPTEN__
#include <cblas.h>
//#include <f77blas.h>
#include <lapack.h>
#ifndef NO_LAPACKE
#include <lapacke.h>
#endif
//#define MKL_Complex16 std::complex<double>
#endif

#ifndef LAPACK_ABSTOL
#define LAPACK_ABSTOL (1.0E-13)
#endif
