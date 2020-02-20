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


#if !defined (MEMORY_ALLOCATION_H)
	#define MEMORY_ALLOCATION_H



 
#include "sources/shared/system_support/os_specifics.h"

#include <vector>
using namespace std;


//**********************************************************************************************************************************


// Check in Windows
#if _WIN32 || _WIN64
	#if _WIN64
	#define SYSTEM_WITH_64BIT
	#else
	#define SYSTEM_WITH_32BIT
	#endif
#endif

// Check for GCC
#if __GNUC__
	#if __x86_64__ || __ppc64__ || __PPC64__
	#define SYSTEM_WITH_64BIT
	#else
	#define SYSTEM_WITH_32BIT
	#endif
#endif

//**********************************************************************************************************************************

int get_used_memory_in_KB();
inline unsigned get_aligned_size(unsigned size, unsigned alignment);

template <typename Template_type> void my_alloc(Template_type** pointer, size_t size);
template <typename Template_type> void my_realloc(Template_type** pointer, size_t size);
template <typename Template_type> void my_dealloc(Template_type** pointer);

template <typename Template_type> size_t allocated_memory_ALGD(Template_type* restrict__* pointer, size_t size);
template <typename Template_type> size_t allocated_memory_in_bytes_ALGD(Template_type* restrict__* pointer, size_t size);

template <typename Template_type> void my_alloc_ALGD(Template_type* restrict__* pointer, size_t size);
template <typename Template_type> void my_alloc_ALGD(Template_type* restrict__* pointer, size_t size, size_t& used_size);

template <typename Template_type> void my_realloc_ALGD(Template_type* restrict__* pointer, size_t size);
template <typename Template_type> void my_realloc_ALGD(Template_type* restrict__* pointer, size_t size, size_t& used_size);

template <typename Template_type> void my_dealloc_return_to_OS(Template_type** pointer);
template <typename Template_type> void my_dealloc_ALGD(Template_type* restrict__* pointer);
template <typename Template_type> void my_dealloc_ALGD_return_to_OS(Template_type** pointer);

template <typename Template_type> void alloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec);
template <typename Template_type> void alloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, size_t& used_size);

template <typename Template_type> void realloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec);
template <typename Template_type> void realloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, size_t& used_size);


#if defined (SYSTEM_WITH_64BIT)

template <typename Template_type> unsigned allocated_memory_ALGD(Template_type* restrict__* pointer, unsigned size);
template <typename Template_type> unsigned allocated_memory_in_bytes_ALGD(Template_type* restrict__* pointer, unsigned size);
template <typename Template_type> void my_alloc_ALGD(Template_type* restrict__* pointer, unsigned size, unsigned& used_size);
template <typename Template_type> void my_realloc_ALGD(Template_type* restrict__* pointer, unsigned size, unsigned& used_size);
template <typename Template_type> void alloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, unsigned& used_size);
template <typename Template_type> void realloc_and_copy_ALGD(Template_type* restrict__* pointer, vector <Template_type> vec, unsigned& used_size);

#endif


//**********************************************************************************************************************************

#include "sources/shared/system_support/memory_allocation.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/system_support/memory_allocation.cpp"
#endif

#endif

