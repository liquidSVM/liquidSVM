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


#if !defined (OS_SPECIFICS_H) 
	#define OS_SPECIFICS_H

	

	
//**********************************************************************************************************************************
// Linux and Mac
//**********************************************************************************************************************************


#if defined __linux__ || defined __MACH__
	#define POSIX_OS__
	#define THREADING_IMPLEMENTED
	
	#ifdef __SSE2__
		#define SSE2__
	#endif
	
	#ifdef __AVX__
		#define AVX__
	#endif

	#ifdef __AVX2__
		#define AVX2__
	#endif
		
	#define sync_add_and_fetch __sync_add_and_fetch
		
	#define Tmutex pthread_mutex_t
	#define Tthread_handle pthread_t
	#define atomic_unsigned unsigned
	
	#define thread__ __thread
	#define restrict__ __restrict__
#endif


//**********************************************************************************************************************************
// Windows
//**********************************************************************************************************************************


#if defined _WIN32
	#if !defined (NOMINMAX) 
		#define NOMINMAX
	#endif  

	#include <malloc.h> // Has to come before windows.h
	#include <windows.h> // Has to come before emmintrin.h/immintrin.h in simd_basics.h !
	
	#include <iso646.h>
	
	#define THREADING_IMPLEMENTED
	#undef COMPILE_WITH_CUDA__
	
// The next line repeats the default of MS Visual Studio 2012++  
// Basically, it assumes that Windows is not running on a CPU built before 2004 or so.

	#define SSE2__
	
	#ifdef __AVX__
		#define AVX__
	#endif

	#ifdef __AVX2__
		#define AVX2__
	#endif	

	#define Tmutex HANDLE
	#define Tthread_handle HANDLE
	
	#ifdef __MINGW32__
		#define sync_add_and_fetch __sync_add_and_fetch
		#define atomic_unsigned unsigned
		#define thread__ __thread
	#else
		#define sync_add_and_fetch atomic_fetch_add
		#define atomic_unsigned atomic_uint
		#define thread__ __declspec(thread)
	#endif
	
	#define restrict__ __restrict
#endif


//**********************************************************************************************************************************
// Other Operating Systems
//**********************************************************************************************************************************

	
#if !defined(POSIX_OS__) && !defined(_WIN32)
	#define UNKNOWN_OS__
	#undef THREADING_IMPLEMENTED
	
// 	The following definitions are meant as safeguards. Depending on the 
// 	situtation they may or may not be necessary.
	
	#undef SSE2__
	#undef AVX__
	#undef AVX2__
	#undef COMPILE_WITH_CUDA__


	#define Tmutex unsigned
	#define Tthread_handle void*
	#define atomic_unsigned unsigned

	#define thread__
	#define restrict__
#endif






//**********************************************************************************************************************************
// More safeguards
//**********************************************************************************************************************************

// Make sure that simd instructions are only used on systems on which we can 
// safely alloc aligned memory. Currently, this is true for MS Windows and 
// systems supporting POSIX with optional posix_memalign commands.


#if defined _WIN32
	#define SIMD_ACTIVATED
#else
	#include <unistd.h>
	
	#if defined(_POSIX_ADVISORY_INFO) && _POSIX_ADVISORY_INFO > 0
		#define SIMD_ACTIVATED
	#else
		#undef SIMD_ACTIVATED
		#undef AVX2__ 
		#undef AVX__
		#undef SSE2__
	#endif
#endif


#endif
