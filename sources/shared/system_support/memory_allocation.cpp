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



#include "sources/shared/system_support/memory_allocation.h"

#include "sources/shared/basic_functions/flush_print.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


//**********************************************************************************************************************************
    

unsigned get_used_memory_in_KB()
{
	#if defined(__linux__)
		FILE* fp; 
		int rss;
		int io_return;

		
		fp = fopen( "/proc/self/statm", "r" );
		if (fp == NULL)
			flush_exit(ERROR_UNSPECIFIED, "Could not read process information from /proc/self/statm .");

		io_return = fscanf( fp, "%*s%d", &rss );
		if (io_return == 0)
			flush_exit(ERROR_UNSPECIFIED, "Could not read process information from /proc/self/statm .");
     
		return unsigned((size_t)rss * (size_t)sysconf(_SC_PAGESIZE) / 1024);
	#else
		return 0;
	#endif
}




#endif
