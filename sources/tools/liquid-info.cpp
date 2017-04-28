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



#include "../shared/system_support/os_specifics.h"
#include "../shared/system_support/compiler_specifics.h"
#include "../shared/basic_functions/flush_print.h"
#include "sources/shared/system_support/thread_manager.h"





//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************


int main(int argc, char **argv)
{
	Tthread_manager_base thread_manager;
	

	#if defined(__GNUC__) 
		flush_info(INFO_1, "\n\nGCC version:     %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
	#endif
		
	#if defined(__clang__) 
		flush_info(INFO_1, "\n\nClang version:   %d", __clang__);
	#endif		
	
	#if defined(_MSC_VER) 
		flush_info(INFO_1, "\n\nMS C++ version:  %d", _MSC_VER);
	#endif
		
	#if defined(__cplusplus)
		#if __cplusplus == 199711L
			flush_info(INFO_1, "\nC++ version:     C++98");
		#elif __cplusplus > 199711L && __cplusplus < 201103L
			flush_info(INFO_1, "\nC++ version:     C++0x");
		#elif __cplusplus == 201103L
			flush_info(INFO_1, "\nC++ version:     C++11");
		#elif __cplusplus == 201402L
			flush_info(INFO_1, "\nC++ version:     C++14");
		#else
			flush_info(INFO_1, "\nC++ version:     unknown");
		#endif
	#endif
			
	#if defined(SIMD_ACTIVATED)
		#ifdef AVX2__ 
			flush_info(INFO_1, "\nSIMD set:        AVX2");
		#elif defined(AVX__) 
			flush_info(INFO_1, "\nSIMD set:        AVX");
		#elif defined(SSE2__) 
			flush_info(INFO_1, "\nSIMD set:        SSE2");
		#endif	
	#else
		flush_info(INFO_1, "\nSIMD set:        ---");
	#endif
		
	#ifdef  COMPILE_WITH_CUDA__
		flush_info(INFO_1, "\nCUDA support:    yes");
	#else
		flush_info(INFO_1, "\nCUDA support:    no");
	#endif
		
	#if defined(THREADING_IMPLEMENTED)
		#if defined(__MACH__)
			flush_info(INFO_1, "\nThread support:  OS X");
		#elif defined(__linux__)
			flush_info(INFO_1, "\nThread support:  POSIX");
		#elif defined(_WIN32)
			flush_info(INFO_1, "\nThread support:  Windows");
		#endif
	#else
		flush_info(INFO_1, "\nThread support:  ---");
	#endif
		
		
	flush_info(INFO_1,"\nPhysical cores:  %d",	thread_manager.get_number_of_physical_cores());
	flush_info(INFO_1,"\nLogical CPUs:    %d",	thread_manager.get_number_of_logical_processors());
		
	flush_info(INFO_1, "\n\n");
}


