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


#if !defined (THREAD_MANAGER_H)
	#define THREAD_MANAGER_H


  
 
#include "sources/shared/system_support/parallel_control.h"
#include "sources/shared/system_support/os_specifics.h"

#include <vector>

#if defined (POSIX_OS__) && !defined(__MINGW32__)
	#include <pthread.h>
#elif defined _WIN32
	#include <windows.h>
	#include <process.h>
	#include <atomic>
#endif

//**********************************************************************************************************************************


struct Tthread_chunk
{
	unsigned thread_id;

	unsigned start_index;
	unsigned stop_index;
	unsigned stop_index_aligned;
	unsigned size;
};


//**********************************************************************************************************************************


class Tthread_manager_base: protected Tparallel_control
{
	public:
		Tthread_manager_base();
		virtual ~Tthread_manager_base();

		void clear_threads();
		void reserve_threads(Tparallel_control parallel_ctrl);
		Tparallel_control get_parallel_control() const;
		
		inline bool hyper_threads_are_pairs() const;
		inline unsigned get_number_of_physical_cores() const;
		inline unsigned get_number_of_logical_processors() const;

		
	protected:
		virtual void clear_on_GPU(){};
		virtual void reserve_on_GPU(){};
		
		inline unsigned get_GPU_id() const;
		unsigned get_number_of_CPU_threads_on_used_GPU() const;
		double free_memory_on_GPU() const;
		double available_memory_on_GPU(double allowed_percentage = 1.0) const;
		
		inline unsigned get_team_size() const;
		inline unsigned get_thread_id() const;
		inline bool is_first_team_member() const;
		inline bool is_last_team_member() const;
		Tthread_chunk get_thread_chunk(unsigned size, unsigned alignment = 1) const;
		
		inline void sync_threads();
		inline void lazy_sync_threads();
		
		inline void get_time_difference(double& out_time, double in_time, unsigned thread_id = 0) const;
		inline void sync_threads_and_get_time_difference(double& out_time, double in_time, unsigned thread_id = 0);
		inline void lazy_sync_threads_and_get_time_difference(double& out_time, double in_time, unsigned thread_id = 0);
		
		inline void thread_safe_set(double* variable, double value);
		inline void thread_safe_add(double* value, double addend);
		
		inline double reduce_sums(double* thread_local_sum);
		inline void increase_counter(int& counter, int addend = 1, unsigned thread_id = 0) const;
		inline void increase_counter(unsigned& counter, unsigned addend = 1, unsigned thread_id = 0) const;
		
		inline void lock_mutex();
		inline void unlock_mutex();
		
		
		static vector <Tthread_manager_base*> list_of_thread_managers;

	private:
		void connect_to_GPU();
		void disconnect_from_GPU();
		void assign_thread(unsigned thread_id);
		friend void* call_thread(void* parameter);
		
		vector <unsigned> get_CPU_info_from_os(const char* entry);

		inline void lock_barrier();
		inline void unlock_barrier();
		
		inline void sync_threads_with_locks();
		inline void sync_threads_without_locks();
		inline void lazy_sync_threads_with_locks();
		inline void lazy_sync_threads_without_locks();

		
		unsigned team_size;
		static thread__ unsigned GPU_id;
		static thread__ unsigned thread_id;
		
		Tmutex mutex;
		Tmutex barrier_mutex;
		
		static bool cpu_info_read;
		static bool hyper_thread_pairs;
		static unsigned number_of_logical_processors;
		static unsigned number_of_physical_cores;
		
		volatile int global_switcher;
		static thread__ int switcher;
#ifdef __MINGW32__
		atomic_unsigned counter[2];
#else
		volatile atomic_unsigned counter[2];
#endif
	
		static thread__ bool connected_to_GPU;
		static thread__ unsigned team_size_set_by_Tthread_manager_active;
};

//**********************************************************************************************************************************

#ifndef PRO_VERSION__
	typedef Tthread_manager_base Tthread_manager;
#else
	#include "sources/shared/system_support/thread_manager_full.pro.h"	
#endif

	
//**********************************************************************************************************************************


#include "sources/shared/system_support/thread_manager.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/system_support/thread_manager.cpp"
#endif


#endif

