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


#include "sources/shared/system_support/timing.h"

#if defined(POSIX_OS__) && !defined(__MINGW32__)
	#include <unistd.h>
#endif


//**********************************************************************************************************************************

// Define some intrinsics to ensure memory consistency. 
// Since NVCC does not support these intrinsics, the macros must only be used in 
// code not compiled by NVCC in a meaningful way.

#ifdef SSE2__
	#include <emmintrin.h>

	#define MM_FENCE _mm_mfence()
	#define MM_CACHELINE_FLUSH(pointer) _mm_clflush(pointer);
#else
	#define MM_FENCE
	#define MM_CACHELINE_FLUSH(pointer)
#endif



//**********************************************************************************************************************************

#ifdef _WIN32
	inline void usleep(int wait_time) 
	{
		__int64 time1;
		__int64 time2;
		__int64 freq;

		time1 = 0;
		time2 = 0;
		freq = 0;

		QueryPerformanceCounter((LARGE_INTEGER*) &time1);
		QueryPerformanceFrequency((LARGE_INTEGER*) &freq);

		do 
			QueryPerformanceCounter((LARGE_INTEGER*) &time2);
		while((time2-time1) < wait_time);
	}
#endif

//**********************************************************************************************************************************


inline bool Tthread_manager_base::hyper_threads_are_pairs() const
{
	return hyper_thread_pairs;
}


//**********************************************************************************************************************************

inline unsigned Tthread_manager_base::get_number_of_physical_cores() const
{
	return number_of_physical_cores;
}

//**********************************************************************************************************************************

inline unsigned Tthread_manager_base::get_number_of_logical_processors() const
{
	return number_of_logical_processors;
}


//**********************************************************************************************************************************

inline unsigned Tthread_manager_base::get_team_size() const
{
	return team_size;
}

//**********************************************************************************************************************************

inline unsigned Tthread_manager_base::get_thread_id() const
{
	return thread_id;
}

//**********************************************************************************************************************************

inline unsigned Tthread_manager_base::get_GPU_id() const
{
	return GPU_id;
}

//**********************************************************************************************************************************

inline bool Tthread_manager_base::is_first_team_member() const
{
	return (thread_id == 0);
}


//**********************************************************************************************************************************

inline bool Tthread_manager_base::is_last_team_member() const
{
	return (thread_id + 1 == team_size);
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::sync_threads()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(SSE2__) && !defined(__MINGW32__)
			sync_threads_without_locks();
		#else
			sync_threads_with_locks();
		#endif
	#endif
}

//**********************************************************************************************************************************


inline void Tthread_manager_base::lazy_sync_threads()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(SSE2__) && !defined(__MINGW32__)
			lazy_sync_threads_without_locks();
		#else
			lazy_sync_threads_with_locks();
		#endif
	#endif
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::sync_threads_without_locks()
{
	#ifdef THREADING_IMPLEMENTED
	// This barrier is essentially taken from Fig. 1(b) of 
	// Fang, Zhang, Carter, Cheng, and Parker: "Fast Synchronization on Shared-Memory Multiprocessors: An Architectural Approach"
	// Journal of Parallel and Distributed Computing 65 (2005), 1158-1170
	// 
	// It is designed for so-called cc-NUMA processors. I have tested it for the following processors:
	// Core2, i5, i7, 
	// IMPORTANT: The barrier does not work, if more threads than cores are used.
		
		if (team_size > 1)
		{
			MM_FENCE;											//		Ensure all memory operations are globally visible. Tests indicate that this is superfluous
			switcher = switcher ^ 1;			//		Bitwise XOR with 1, ie, the value of the first bit is changed. Note: the rest is zero, anyway,
																		//		so that switcher switches between 0 and 1. This ensures that different counters are used for 
																		//		two consequetive runs through the barrier.
			sync_add_and_fetch(&counter[switcher], 1);		//		Atomic increment of the current counter
			if (thread_id == 0)
			{
				while (counter[switcher] < team_size)				//		The first thread spins until every thread increased the counter
					{};																				//
				counter[switcher] = 0;											//		Then it sets the counter back to 0.
				MM_FENCE;																		//		Make double sure that the last operation becomes globally visible.
																										//		Some tests showed that the volatile keyword seems to suffice.
			}
			else
				while (counter[switcher] > 0)								//		All other threads spin until the counter is set back.
					{};																				//
		}
	#endif
}



//**********************************************************************************************************************************


inline void Tthread_manager_base::lazy_sync_threads_without_locks()
{
	#ifdef THREADING_IMPLEMENTED
	// This  barrier is the same as above but with some sleeping during the wait
		
		if (team_size > 1)
		{
			MM_FENCE;
			switcher = switcher ^ 1;
			sync_add_and_fetch(&counter[switcher], 1);
			if (thread_id == 0)
			{
				while (counter[switcher] < team_size)
					usleep(100);
				counter[switcher] = 0;
				MM_FENCE;
			}
			else
				while (counter[switcher] > 0)
					usleep(100);
		}
	#endif
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::sync_threads_with_locks()
{
	#ifdef THREADING_IMPLEMENTED
		int local_switcher;

	// This is a quite similar counting barrier based on locks. 
	// Pros compared to the barrier without locks:
	// 	- Hardware independent, it should work on all processors
	// 	- No restriction on the number of threads
	// Cons
	// 	- Typically slower: Some preliminary experiments suggests between a factor of 2 to 40
	// 	  depending on the number of threads and the particular hardware.

		lock_barrier();
		counter[0]++;
		MM_FENCE;
		local_switcher = global_switcher;

		if (counter[0] == team_size)
		{
			counter[0] = 0;
			global_switcher = local_switcher ^ 1;
			MM_FENCE;
			unlock_barrier();
			return;
		}

		unlock_barrier();
		if (global_switcher == local_switcher)
			while (global_switcher == local_switcher)
				{};
	#endif
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::lazy_sync_threads_with_locks()
{
	#ifdef THREADING_IMPLEMENTED
		int local_switcher;

	// Again the same as above but with some sleeping

		lock_barrier();
		counter[0]++;
		MM_FENCE;
		local_switcher = global_switcher;

		if (counter[0] == team_size)
		{
			counter[0] = 0;
			global_switcher = local_switcher ^ 1;
			MM_FENCE;
			unlock_barrier();
			return;
		}

		unlock_barrier();
		if (global_switcher == local_switcher)
			while (global_switcher == local_switcher)
				usleep(100);
	#endif
}

//**********************************************************************************************************************************


inline void Tthread_manager_base::get_time_difference(double& out_time, double in_time, unsigned thread_id) const
{
	if (Tthread_manager_base::thread_id == thread_id)
		out_time = get_thread_time_difference(in_time);
}

//**********************************************************************************************************************************


inline void Tthread_manager_base::sync_threads_and_get_time_difference(double& out_time, double in_time, unsigned thread_id)
{
	sync_threads();
	if (Tthread_manager_base::thread_id == thread_id)
		out_time = get_thread_time_difference(in_time);
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::lazy_sync_threads_and_get_time_difference(double& out_time, double in_time, unsigned thread_id)
{
	lazy_sync_threads();
	if (Tthread_manager_base::thread_id == thread_id)
		out_time = get_thread_time_difference(in_time);
}


//**********************************************************************************************************************************

inline void Tthread_manager_base::lock_mutex()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			pthread_mutex_lock(&mutex);
		#elif defined(_WIN32)
			WaitForSingleObject(mutex, INFINITE);
		#endif
	#endif
}


//**********************************************************************************************************************************

inline void Tthread_manager_base::unlock_mutex()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			pthread_mutex_unlock(&mutex);
		#elif defined (_WIN32)
			ReleaseMutex(mutex);
		#endif
	#endif
}

//**********************************************************************************************************************************

inline void Tthread_manager_base::lock_barrier()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			pthread_mutex_lock(&barrier_mutex);
		#elif defined(_WIN32)
			WaitForSingleObject(barrier_mutex, INFINITE);
		#endif
	#endif
}


//**********************************************************************************************************************************

inline void Tthread_manager_base::unlock_barrier()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			pthread_mutex_unlock(&barrier_mutex);
		#elif defined (_WIN32)
			ReleaseMutex(barrier_mutex);
		#endif
	#endif
}

//**********************************************************************************************************************************


inline void Tthread_manager_base::thread_safe_set(double* variable, double value)
{
	#ifdef THREADING_IMPLEMENTED
		lock_mutex();
		*(variable) = value;
		MM_CACHELINE_FLUSH(variable);
		unlock_mutex();
	#else
		*(variable) = value;
	#endif
}


//**********************************************************************************************************************************

inline void Tthread_manager_base::thread_safe_add(double* value, double addend)
{
	#ifdef THREADING_IMPLEMENTED
		lock_mutex();
		*(value) = *value + addend;
		MM_CACHELINE_FLUSH(value);
		unlock_mutex();
	#else
		*(value) = *value + addend;
	#endif
}

//**********************************************************************************************************************************

inline double Tthread_manager_base::reduce_sums(double* thread_local_sum)
{
	unsigned t;
	double global_sum;
	
	#ifdef THREADING_IMPLEMENTED
		for (t=0; t<team_size; t=t+CACHELINE_STEP)
			MM_CACHELINE_FLUSH(&thread_local_sum[t]);
		sync_threads();
	#endif
		
	global_sum = 0.0;
	for (t=0; t<team_size; t++)
		global_sum = global_sum + thread_local_sum[t];
	
	return global_sum;
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::increase_counter(int& counter, int addend, unsigned thread_id) const
{
	if (Tthread_manager_base::thread_id == thread_id)
		counter = counter + addend;
}


//**********************************************************************************************************************************


inline void Tthread_manager_base::increase_counter(unsigned& counter, unsigned addend, unsigned thread_id) const
{
	if (Tthread_manager_base::thread_id == thread_id)
		counter = counter + addend;
}
