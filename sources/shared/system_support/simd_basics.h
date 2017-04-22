// Copyright 2015, 2016, 2017 Ingo Steinwart
//
// This file is part of liquidSVM.
//
// liquidSVM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as 
// published by the Free Software Foundation, either version 3 of the 
// License, or (at your option) any later version.
//
// liquidSVM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.


#if !defined (SIMD_BASICS_H) 
	#define SIMD_BASICS_H



#include "sources/shared/system_support/os_specifics.h"


//**********************************************************************************************************************************


#define CACHELINE 64
#define CACHELINE_STEP 8
 
#define PREFETCH_L1 _MM_HINT_T0
#define PREFETCH_L2 _MM_HINT_T1
#define PREFETCH_L3 _MM_HINT_T2
#define PREFETCH_NO _MM_HINT_NTA



//**********************************************************************************************************************************


// In a few situations, inline functions are not inlined and their simd variables are not aligned on the stack.
// The following lines address this issue, by defining a new inline attribute, which should be used for functions
// with simd variables.

#ifdef _MSC_VER
// for MS Visual C++
#define strict_inline__  __forceinline inline
#else
// for GCC and CLANG
#define strict_inline__ __attribute__((always_inline)) inline
#endif


//**********************************************************************************************************************************

// Instructions on simd__ registers.


#ifdef AVX__
	#include <immintrin.h>
	
	#define simdd__ __m256d
	#define SIMD_WORD_SIZE 4
	#if defined(__MACH__) || defined __clang__
		#define cache_prefetch(address, hint) _mm_prefetch(address, hint)
	#elif defined(_WIN32) && !defined(__MINGW32__)
		#define cache_prefetch(address, hint) _mm_prefetch( (char*)((void*)(address)), int(hint))
	#else
		#define cache_prefetch(address, hint) _mm_prefetch(address, _mm_hint(hint))
	#endif
	
	#define zero_simdd _mm256_setzero_pd()
	#define assign_simdd(value) _mm256_set_pd(value, value, value, value)
	#define load_simdd(address) _mm256_load_pd(address)
	#define store_simdd(address, value) _mm256_store_pd(address, value)
	
	#define add_simdd(addend_1, addend_2)  _mm256_add_pd(addend_1, addend_2)
	#define sub_simdd(minuend, subtrahend)  _mm256_sub_pd(minuend, subtrahend)
	#define mult_simdd(factor_1, factor_2)  _mm256_mul_pd(factor_1, factor_2)
	
	#define min_simdd(arg1, arg2) _mm256_min_pd(arg1, arg2)
	#define max_simdd(arg1, arg2) _mm256_max_pd(arg1, arg2)
	#define seq_argmax_simdd(arg1, value1, arg2, value2) _mm256_max_pd(arg1, _mm256_and_pd(arg2,  _mm256_cmp_pd(value2, value1, _CMP_GT_OQ)))
 	
	#define eq_simdd(arg1, arg2) _mm256_cmp_pd(arg1, arg2, _CMP_EQ_OQ)
 	#define neq_simdd(arg1, arg2) _mm256_cmp_pd(arg1, arg2, _CMP_NEQ_OQ)
 	
 	#define eq_cond_val_simdd(arg1, cond1, cond2) _mm256_and_pd(arg1, _mm256_cmp_pd(cond1, cond2, _CMP_EQ_OQ)) 
 	
 	#define abs_simdd(arg) _mm256_andnot_pd(assign_simdd(-0.), arg)
 	#define negate_simdd(arg) _mm256_xor_pd(assign_simdd(-0.), arg)

 	
#elif defined SSE2__
	#include <emmintrin.h>

	#define simdd__ __m128d
	#define SIMD_WORD_SIZE 2
	#if defined(__MACH__) || defined __clang__
		#define cache_prefetch(address, hint) _mm_prefetch(address, hint)
	#elif defined(_WIN32) && !defined(__MINGW32__)
		#define cache_prefetch(address, hint) _mm_prefetch( (char*)((void*)(address)), int(hint))
	#else
		#define cache_prefetch(address, hint) _mm_prefetch(address, _mm_hint(hint))
	#endif
		
	#define zero_simdd _mm_setzero_pd()
	#define assign_simdd(value) _mm_set_pd(value, value)
	#define load_simdd(address) _mm_load_pd(address)
	#define store_simdd(address, value) _mm_store_pd(address, value)
	
	#define add_simdd(addend_1, addend_2)  _mm_add_pd(addend_1, addend_2)
	#define sub_simdd(minuend, subtrahend)  _mm_sub_pd(minuend, subtrahend)
	#define mult_simdd(factor_1, factor_2)  _mm_mul_pd(factor_1, factor_2)
	
	#define min_simdd(arg1, arg2) _mm_min_pd(arg1, arg2)
	#define max_simdd(arg1, arg2) _mm_max_pd(arg1, arg2)
	#define seq_argmax_simdd(arg1, value1, arg2, value2) _mm_max_pd(arg1, _mm_and_pd(arg2, _mm_cmpgt_pd(value2, value1)))
	
	#define eq_simdd(arg1, arg2) _mm_cmpeq_pd(arg1, arg2)
	#define neq_simdd(arg1, arg2) _mm_cmp_pd(arg1, arg2, _CMP_NEQ_OQ)
 	
 	#define eq_cond_val_simdd(arg1, cond1, cond2) _mm_and_pd(arg1, _mm_cmpeq_pd(cond1, cond2)) 
 	
	#define abs_simdd(arg) _mm_andnot_pd(assign_simdd(-0.), arg)
	#define negate_simdd(arg) _mm_xor_pd(assign_simdd(-0.), arg)

#else
	#define simdd__ double
	#define SIMD_WORD_SIZE 1
	#define cache_prefetch(address, hint)
	
	#define zero_simdd 0.0
	#define assign_simdd(value) value
	#define load_simdd(address) *(address)
	#define store_simdd(address, value) *(address) = value
	
	#define add_simdd(addend_1, addend_2)  (addend_1 + addend_2)
	#define sub_simdd(minuend, subtrahend) (minuend - subtrahend)
	#define mult_simdd(factor_1, factor_2)  (factor_1 * factor_2)
	
	#define min_simdd(arg1, arg2) ((arg1 > arg2)? arg2: arg1)
	#define max_simdd(arg1, arg2) ((arg1 > arg2)? arg1: arg2)
	#define seq_argmax_simdd(arg1, value1, arg2, value2) ((value2 > value1)? arg2: arg1)
	
	#define eq_simdd(arg1, arg2) ((arg1 == arg2)? 0xFFFFFFFFFFFFFFFF : 0)
	#define neq_simdd(arg1, arg2) ((arg1 != arg2)? 0xFFFFFFFFFFFFFFFF : 0)
	
	#define eq_cond_val_simdd(arg1, cond1, cond2) ((cond1 == cond2)? arg1 : 0) 
 	
 	#define abs_simdd(arg1) ((arg1 > 0)? arg1: -arg1)
	#define negate_simdd(arg) (-arg)
	
#endif


#define pos_part_simdd(arg) max_simdd(zero_simdd, arg)
#define neg_part_simdd(arg) max_simdd(zero_simdd, negate_simdd(arg))

#define clipp_0max_simdd(arg, top) min_simdd(top, pos_part_simdd(arg))
#define clipp_simdd(arg, bottom, top) max_simdd(bottom, min_simdd(top, arg))

inline simdd__ fuse_mult_add_simdd(simdd__ factor_1_simdd, simdd__ factor_2_simdd, simdd__ addend_simdd);
inline simdd__ fuse_mult_sub_simdd(simdd__ factor_1_simdd, simdd__ factor_2_simdd, simdd__ subtrahend_simdd);
inline simdd__ fuse_mult_mult_simdd(simdd__ factor_1_simdd, simdd__ factor_2_simdd, simdd__ factor_3_simdd);


inline double reduce_sums_simdd(simdd__ sum_simdd);
inline void argmax_simdd(simdd__ arg_simdd, simdd__ value_simdd, unsigned& argmax, double& max);

//**********************************************************************************************************************************

// Instructions on a cacheline

inline void assign_CL(double* vector, double value);
inline void copy_CL(double* target, double* source);

inline void fuse_mult_sum_CL(double* factor_1, double* factor_2, simdd__& sum_simdd);
inline void fuse_mult_sum_CL(simdd__ factor_1_simdd, double* factor_2, simdd__& sum_simdd);

inline void sum_CL(simdd__& sum, double* addend);
inline void add_CL(double* sum, double* addend_1, double* addend_2);
inline void add_CL(double* sum, simdd__ addend_1_simdd, double* addend_2);
inline void mult_CL(double* product, double* factor_1, double* factor_2);
inline void mult_CL(double* product, simdd__ factor_1_simdd, double* factor_2);

inline void fuse_mult_add3_CL(double* factor_1, double* factor_2, double* addend);
inline void fuse_mult_add3_CL(simdd__ factor_1_simdd, double* factor_2, double* addend);
inline void fuse_mult_add4_CL(double* result, simdd__ factor_1_simdd, double* factor_2, simdd__ addend_simdd);
inline void fuse_mult_add5_CL(double* result, simdd__ factor_1_simdd, double* factor_1, simdd__ factor_2_simdd, double* factor_2);


//**********************************************************************************************************************************

// Alignment commands

inline void get_aligned_chunk(unsigned size, unsigned chunks, unsigned chunk_number, unsigned& start_index, unsigned& stop_index);


//**********************************************************************************************************************************

#include "sources/shared/system_support/simd_basics.ins.cpp"



#endif
