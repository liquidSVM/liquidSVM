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


#include "sources/shared/system_support/simd_basics.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/memory_constants.h"



#include <limits>
#include <stdlib.h>
#ifndef __MACH__
	#include <malloc.h>
#endif



//**********************************************************************************************************************************

unsigned inline get_aligned_size(unsigned size, unsigned alignment)
{
	return (1 + (size - 1)/alignment) * alignment;
}


//**********************************************************************************************************************************


template <typename Template_type> void my_alloc(Template_type** pointer, size_t size)
{
	if (size > 0)
		*pointer = (Template_type*) malloc(sizeof(Template_type) * size);
	else
		*pointer = NULL;
}


//**********************************************************************************************************************************


template <typename Template_type> void my_realloc(Template_type** pointer, size_t size)
{
	my_dealloc(pointer);
	my_alloc(pointer, size);
}


//**********************************************************************************************************************************


template <typename Template_type> void my_dealloc(Template_type** pointer)
{
	if (*pointer != NULL)
		free(*pointer);
	*pointer = NULL;
}

//**********************************************************************************************************************************


template <typename Template_type> void my_dealloc_return_to_OS(Template_type** pointer)
{
	my_dealloc(pointer);
	
	#if defined __linux__
		malloc_trim(0);
	#endif
}

//**********************************************************************************************************************************


template <typename Template_type> void my_dealloc_ALGD_return_to_OS(Template_type** pointer)
{
	my_dealloc_ALGD(pointer);

	#if defined __linux__
		malloc_trim(0);
	#endif
}



//**********************************************************************************************************************************


template <typename Template_type> size_t allocated_memory_in_bytes_ALGD(Template_type* restrict__* pointer, size_t size)
{
	if (((sizeof(Template_type) * size) % CACHELINE) == 0)
		return size_t(sizeof(Template_type) * size);
	else
		return size_t(sizeof(Template_type) * size + CACHELINE - ((sizeof(Template_type) * size) % CACHELINE));
}


//**********************************************************************************************************************************


template <typename Template_type> size_t allocated_memory_ALGD(Template_type* restrict__* pointer, size_t size)
{
	return allocated_memory_in_bytes_ALGD(pointer, size) / sizeof(Template_type);
}



//**********************************************************************************************************************************



template <typename Template_type> void my_alloc_ALGD(Template_type* restrict__* pointer, size_t size, size_t& used_size)
{
	void* ptr;
	#if defined(SIMD_ACTIVATED) && !defined(_WIN32)
		int alloc_return;
	#endif
	
	if (size == 0)
	{
		*pointer = NULL;
		used_size = 0;
		return;
	}
	
	
// Make sure that size of aligned vector is a multiple of the CACHELINE
// Then allocate memory and make final conversions 

	ptr = NULL;
	used_size = allocated_memory_in_bytes_ALGD(pointer, size);

	#ifdef SIMD_ACTIVATED
		#ifndef _WIN32
			alloc_return = posix_memalign(&ptr, CACHELINE, used_size);
			if (alloc_return != 0)
				flush_exit(ERROR_OUT_OF_MEMORY, "Unsufficient memory while allocating an array of %d MB.", convert_to_MB(used_size));
		#else
			ptr = _aligned_malloc(used_size, size_t(CACHELINE));
		#endif
	#else
		ptr = malloc(used_size);
	#endif
	if (ptr == NULL)
		flush_exit(ERROR_OUT_OF_MEMORY, "Unsufficient memory while allocating an array of %d MB.", convert_to_MB(used_size));

	*pointer = (Template_type*) ptr;
	used_size = used_size / sizeof(Template_type);
}


//**********************************************************************************************************************************



template <typename Template_type> void my_alloc_ALGD(Template_type* restrict__* pointer, size_t size)
{
	size_t used_size;

	
	my_alloc_ALGD(pointer, size, used_size);
}


//**********************************************************************************************************************************


template <typename Template_type> void alloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, size_t& used_size)
{
	size_t i;
	
	
	my_alloc_ALGD(pointer, vec.size(), used_size);
	for (i=0; i<used_size; i++)
		if (i<vec.size())
			(*pointer)[i] = vec[i];
		else
			(*pointer)[i] = 0;
}


//**********************************************************************************************************************************



template <typename Template_type> void alloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec)
{
	size_t used_size;
	
	
	alloc_and_copy_ALGD(pointer, vec, used_size);
}


//**********************************************************************************************************************************


template <typename Template_type> void realloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, size_t& used_size)
{
	my_dealloc_ALGD(pointer);
	alloc_and_copy_ALGD(pointer, vec, used_size);
}

//**********************************************************************************************************************************


template <typename Template_type> void realloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec)
{
	my_dealloc_ALGD(pointer);
	alloc_and_copy_ALGD(pointer, vec);
}


//**********************************************************************************************************************************


template <typename Template_type> void my_realloc_ALGD(Template_type* restrict__* pointer, size_t size, size_t& used_size)
{
	my_dealloc_ALGD(pointer);
	my_alloc_ALGD(pointer, size, used_size);
}


//**********************************************************************************************************************************


template <typename Template_type> void my_realloc_ALGD(Template_type* restrict__* pointer, size_t size)
{
	size_t used_size;
	
	my_realloc_ALGD(pointer, size, used_size);
}




//**********************************************************************************************************************************


template <typename Template_type> void my_dealloc_ALGD(Template_type* restrict__* pointer)
{
	#ifndef _WIN32
		if (*pointer != NULL)
			free(*pointer);
	#else
		_aligned_free(*pointer);
	#endif
	*pointer = NULL;
}


//**********************************************************************************************************************************


#if defined (SYSTEM_WITH_64BIT)



template <typename Template_type> unsigned allocated_memory_in_bytes_ALGD(Template_type* restrict__* pointer, unsigned size)
{
	size_t allocated_mem;
	
	allocated_mem = allocated_memory_in_bytes_ALGD(pointer, size_t(size));
	if (allocated_mem > size_t(std::numeric_limits<unsigned>::max()))
		flush_exit(ERROR_OUT_OF_MEMORY, "Size of memory to be allocated is larger than the largest number of type unsigned.");
	
	return unsigned(allocated_mem);
}

//**********************************************************************************************************************************

template <typename Template_type> unsigned allocated_memory_ALGD(Template_type* restrict__* pointer, unsigned size)
{
	size_t allocated_mem;
	
	allocated_mem = allocated_memory_ALGD(pointer, size_t(size));
	
	if (allocated_mem > size_t(std::numeric_limits<unsigned>::max()))
		flush_exit(ERROR_OUT_OF_MEMORY, "Size of memory to be allocated is larger than the largest number of type unsigned.");
	
	return unsigned(allocated_mem);
}

//**********************************************************************************************************************************


template <typename Template_type> void my_alloc_ALGD(Template_type* restrict__* pointer, unsigned size, unsigned& used_size)
{
	void* ptr;
	#if defined(SIMD_ACTIVATED) && !defined(_WIN32)
		int alloc_return;
	#endif
	
	
	if (size == 0)
	{
		*pointer = NULL;
		used_size = 0;
		return;
	}
	
	
// Make sure that size of aligned vector is a multiple of the CACHELINE
// Then allocate memory and make final conversions 

	ptr = NULL;
	used_size = allocated_memory_in_bytes_ALGD(pointer, size);

	#ifdef SIMD_ACTIVATED
		#ifndef _WIN32
			alloc_return = posix_memalign(&ptr, CACHELINE, used_size);
			if (alloc_return != 0)
				flush_exit(ERROR_OUT_OF_MEMORY, "Unsufficient memory while allocating an array of %d MB.", convert_to_MB(used_size));
		#else
			ptr = _aligned_malloc(size_t(used_size), size_t(CACHELINE));
		#endif
	#else
		ptr = malloc(size_t(used_size));
	#endif
	if (ptr == NULL)
		flush_exit(ERROR_OUT_OF_MEMORY, "Unsufficient memory while allocating an array of %d MB.", convert_to_MB(used_size));

	*pointer = (Template_type*) ptr;
	used_size = used_size / sizeof(Template_type);
}



//**********************************************************************************************************************************


template <typename Template_type> void alloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, unsigned& used_size)
{
	size_t allocated_mem;
	
	
	alloc_and_copy_ALGD(pointer, vec, allocated_mem);
	
	if (allocated_mem > std::numeric_limits<unsigned>::max())
		flush_exit(ERROR_OUT_OF_MEMORY, "Size of memory to be allocated is larger than the largest number of type unsigned.");
	
	used_size = unsigned(allocated_mem);
}


//**********************************************************************************************************************************



template <typename Template_type> void my_realloc_ALGD(Template_type* restrict__* pointer, unsigned size, unsigned& used_size)
{
	my_dealloc_ALGD(pointer);
	my_alloc_ALGD(pointer, size, used_size);
}

//**********************************************************************************************************************************



template <typename Template_type> void realloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, unsigned& used_size)
{
	my_dealloc_ALGD(pointer);
	alloc_and_copy_ALGD(pointer, vec, used_size);
}


//**********************************************************************************************************************************

#endif

