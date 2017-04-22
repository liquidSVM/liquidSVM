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



#include "sources/shared/system_support/os_specifics.h"
#include "sources/shared/basic_functions/flush_print.h"



#include <time.h>
#include <math.h> 

#ifdef __MACH__
	#include <mach/mach_time.h>
#elif defined (_WIN32)
extern "C" {
	#include <windows.h>
}
#endif

// the following is defined on some combinations of system and compiler and on others not - we want to have it always
#ifndef CLOCK_REALTIME
	#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
	#define CLOCK_MONOTONIC 0
#endif
#ifndef CLOCK_THREAD_CPUTIME_ID
	#define CLOCK_THREAD_CPUTIME_ID 0
#endif
#ifndef CLOCK_PROCESS_CPUTIME_ID
	#define CLOCK_PROCESS_CPUTIME_ID 0
#endif

//**********************************************************************************************************************************

inline void my_clock_gettime(int clk_id, struct timespec *time_measured)
{
    #ifdef __MACH__
        uint64_t time;
        mach_timebase_info_data_t timebase;
    
        mach_timebase_info(&timebase);
        time = mach_absolute_time();
    
        time_measured->tv_sec = double(time / 1000000000) * double(timebase.numer)/double(timebase.denom);
        time_measured->tv_nsec = double(time % 1000000000) * double(timebase.numer)/double(timebase.denom);
    #endif
    
    
    #ifdef _WIN32
        double time;
    
        time = double(clock()) / double(CLOCKS_PER_SEC);
    
        time_measured->tv_sec = unsigned(floor(time));
        time_measured->tv_nsec = unsigned((time - floor(time)) * 1000000000.0);
    #endif
    
    #ifdef __linux__
        clock_gettime(clk_id, time_measured);
    #endif
}



	
//**********************************************************************************************************************************


inline double get_wall_time_difference(double time)
{
	timespec time_tmp;
	
	my_clock_gettime(CLOCK_MONOTONIC, &time_tmp);
	
	return double(time_tmp.tv_sec) + double(time_tmp.tv_nsec) / 1000000000.0 - time;
}



//**********************************************************************************************************************************


inline double get_thread_time_difference(double time)
{
	timespec time_tmp;
	
	my_clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time_tmp);
	
	return double(time_tmp.tv_sec) + double(time_tmp.tv_nsec) / 1000000000.0 - time;
}



//**********************************************************************************************************************************


inline double get_process_time_difference(double time)
{
	timespec time_tmp;
	
	my_clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_tmp);
	
	return double(time_tmp.tv_sec) + double(time_tmp.tv_nsec) / 1000000000.0 - time;
}




