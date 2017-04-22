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









#include <algorithm>
using namespace std;

//**********************************************************************************************************************************


inline void assign_CL(double* vector, double value)
{
	unsigned i; 
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(vector+i, assign_simdd(value));
}

//**********************************************************************************************************************************


inline void copy_CL(double* target, double* source)
{
	unsigned i; 
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(target+i, load_simdd(source+i));
}

//**********************************************************************************************************************************

inline void fuse_mult_sum_CL(double* factor_1, double* factor_2, simdd__& sum_simdd)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		sum_simdd = fuse_mult_add_simdd(load_simdd(factor_1+i), load_simdd(factor_2+i), sum_simdd);
}


//**********************************************************************************************************************************

inline void fuse_mult_sum_CL(simdd__ factor_1_simdd, double* factor_2, simdd__& sum_simdd)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		sum_simdd = fuse_mult_add_simdd(factor_1_simdd, load_simdd(factor_2+i), sum_simdd);
}

//**********************************************************************************************************************************


inline void sum_CL(simdd__& sum_simdd, double* addend)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		sum_simdd = add_simdd(sum_simdd, load_simdd(addend+i));
}


//**********************************************************************************************************************************

inline void add_CL(double* sum, double* addend_1, double* addend_2)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(sum+i, add_simdd(load_simdd(addend_1+i), load_simdd(addend_2+i)));
}


//**********************************************************************************************************************************

inline void add_CL(double* sum, simdd__ addend_1_simdd, double* addend_2)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(sum+i, add_simdd(addend_1_simdd, load_simdd(addend_2+i)));
}

//**********************************************************************************************************************************

inline void mult_CL(double* product, double* factor_1, double* factor_2)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(product+i, mult_simdd(load_simdd(factor_1+i), load_simdd(factor_2+i)));
}


//**********************************************************************************************************************************

inline void mult_CL(double* product, simdd__ factor_1_simdd, double* factor_2)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(product+i, mult_simdd(factor_1_simdd, load_simdd(factor_2+i)));
}



//**********************************************************************************************************************************

inline void fuse_mult_add3_CL(double* factor_1, double* factor_2, double* addend)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(addend+i, fuse_mult_add_simdd(load_simdd(factor_1+i), load_simdd(factor_2+i), load_simdd(addend+i)));
}


//**********************************************************************************************************************************

inline void fuse_mult_add3_CL(simdd__ factor_1_simdd, double* factor_2, double* addend)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(addend+i, fuse_mult_add_simdd(factor_1_simdd, load_simdd(factor_2+i), load_simdd(addend+i)));
}


//**********************************************************************************************************************************


inline void fuse_mult_add4_CL(double* result, simdd__ factor_1_simdd, double* factor_2, simdd__ addend_simdd)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(result+i, fuse_mult_add_simdd(factor_1_simdd, load_simdd(factor_2+i), addend_simdd));
}

//**********************************************************************************************************************************


inline void fuse_mult_add5_CL(double* result, simdd__ factor_1_simdd, double* factor_1, simdd__ factor_2_simdd, double* factor_2)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		store_simdd(result+i, fuse_mult_add_simdd(factor_1_simdd, load_simdd(factor_1+i), mult_simdd(factor_2_simdd, load_simdd(factor_2+i))));
}



//**********************************************************************************************************************************

inline double reduce_sums_simdd(simdd__ sum_simdd)
{
	#ifdef AVX__
		double* sum_ptr;
		
		sum_ptr = (double*)&sum_simdd;
		return sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3];
	#elif defined SSE2__
		double* sum_ptr;
		
		sum_ptr = (double*)&sum_simdd;
		return sum_ptr[0] + sum_ptr[1];
	#else
		return sum_simdd;
	#endif
}

//**********************************************************************************************************************************


inline void argmax_simdd(simdd__ arg_simdd, simdd__ value_simdd, unsigned& argmax, double& max)
{
	#ifdef AVX__
		double* arg_ptr;
		double* value_ptr;
		
		arg_ptr = (double*)&arg_simdd;
		value_ptr = (double*)&value_simdd;
		
		argmax = unsigned((value_ptr[1] > value_ptr[0])? arg_ptr[1]:arg_ptr[0]);
		max = (value_ptr[1] > value_ptr[0])? value_ptr[1]:value_ptr[0];
		
		argmax = unsigned((value_ptr[2] > max)? arg_ptr[2]:argmax);
		max = (value_ptr[2] > max)? value_ptr[2]:max;
		
		argmax = unsigned((value_ptr[3] > max)? arg_ptr[3]:argmax);
		max = (value_ptr[3] > max)? value_ptr[3]:max;
	#elif defined SSE2__
		double* arg_ptr;
		double* value_ptr;
		
		arg_ptr = (double*)&arg_simdd;
		value_ptr = (double*)&value_simdd;
		
		argmax = unsigned((value_ptr[1] > value_ptr[0])? arg_ptr[1]:arg_ptr[0]);
		max = (value_ptr[1] > value_ptr[0])? value_ptr[1]:value_ptr[0];
	#else
		max = value_simdd;
		argmax = unsigned(arg_simdd);
	#endif
}
	

//**********************************************************************************************************************************

inline simdd__ fuse_mult_add_simdd(simdd__ factor_1_simdd, simdd__ factor_2_simdd, simdd__ addend_simdd)
{
	#ifdef AVX2__ 
		return _mm256_fmadd_pd(factor_1_simdd, factor_2_simdd, addend_simdd);
	#else
		return add_simdd(addend_simdd, mult_simdd(factor_1_simdd, factor_2_simdd));
	#endif
}

//**********************************************************************************************************************************

inline simdd__ fuse_mult_sub_simdd(simdd__ factor_1_simdd, simdd__ factor_2_simdd, simdd__ subtrahend_simdd)
{
	#ifdef AVX2__ 
		return _mm256_fmsub_pd(factor_1_simdd, factor_2_simdd, subtrahend_simdd);
	#else
		return sub_simdd(mult_simdd(factor_1_simdd, factor_2_simdd), subtrahend_simdd);
	#endif
}

// *********************************************************************************************************************************
inline simdd__ fuse_mult_mult_simdd(simdd__ factor_1_simdd, simdd__ factor_2_simdd, simdd__ factor_3_simdd)
  {
	return mult_simdd(factor_1_simdd, mult_simdd(factor_2_simdd, factor_3_simdd));
  }

  
//**********************************************************************************************************************************


inline void get_aligned_chunk(unsigned size, unsigned chunks, unsigned chunk_number, unsigned& start_index, unsigned& stop_index, bool round_up)
{
	unsigned part;
	unsigned aligned_chunk_size;
	unsigned aligned_size;

	
	if ((size % (CACHELINE_STEP * chunks) == 0) or (round_up == false))
		part = unsigned(size / (CACHELINE_STEP * chunks));
	else
		part = 1 + unsigned(size / (CACHELINE_STEP * chunks));
	aligned_chunk_size = CACHELINE_STEP * part;

	if (size % CACHELINE_STEP == 0)
		part = unsigned(size / CACHELINE_STEP);
	else
		part = 1 + unsigned(size / CACHELINE_STEP);
	aligned_size = CACHELINE_STEP * part;

	start_index = min(size, chunk_number * aligned_chunk_size);
	if (chunk_number+1 != chunks)
		stop_index = min(aligned_size, (chunk_number + 1) * aligned_chunk_size);
	else
		stop_index = aligned_size;
}


//**********************************************************************************************************************************

inline bool can_get_aligned_chunk_round_up(unsigned size, unsigned chunks)
{
	unsigned start_index;
	unsigned stop_index;
	
	get_aligned_chunk(size, chunks, int(chunks) - 1, start_index, stop_index, true);
	
	return (stop_index >= start_index + CACHELINE_STEP);
}



//**********************************************************************************************************************************

  
inline void get_aligned_chunk(unsigned size, unsigned chunks, unsigned chunk_number, unsigned& start_index, unsigned& stop_index)
{
	get_aligned_chunk(size, chunks, chunk_number, start_index, stop_index, can_get_aligned_chunk_round_up(size, chunks));
}
