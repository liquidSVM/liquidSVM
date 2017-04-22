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


#if !defined (THREAD_MANAGER_ACTIVE_CPP)
	#define THREAD_MANAGER_ACTIVE_CPP


#include "sources/shared/system_support/thread_manager_active.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/system_support/binding_specifics.h"

#ifdef __MACH__
	#include <mach/mach_init.h>
	#include <mach/thread_policy.h>
	#include <mach/thread_act.h>
	#include <pthread.h>
#endif

// #define MASTER_AS_FIRST_TEAM_MEMBER 

// #ifndef THREADING_IMPLEMENTED
//   #undef MASTER_AS_FIRST_TEAM_MEMBER
// #endif
#ifndef PRE_THREADS_HOOK
  #define PRE_THREADS_HOOK 
#endif
#ifndef POST_THREADS_HOOK
  #define POST_THREADS_HOOK 
#endif


//**********************************************************************************************************************************


void Tthread_manager_active::reserve_threads(Tparallel_control parallel_ctrl)
{
	Tthread_manager_base::reserve_threads(parallel_ctrl);
	core_number_offset = parallel_ctrl.core_number_offset;
}


//**********************************************************************************************************************************


struct Tthread_parameter
{
	unsigned thread_id;
	unsigned number_of_logical_processors;
	unsigned team_size;
	unsigned core_number_offset;
	bool hyper_thread_pairs;
	Tthread_manager_active* thread_manager;
};

//**********************************************************************************************************************************

#ifdef _WIN32
	void* call_thread(void* parameter);


	unsigned int __stdcall call_thread_windows(void* parameter) 
	{
		call_thread(parameter);
		return 0;
	}

#endif

//**********************************************************************************************************************************

void* call_thread(void* parameter)
{
	unsigned i;
	unsigned thread_id;
	unsigned thread_id_with_offset;
	Tthread_manager_active* calling_thread_manager;
	Tthread_parameter* thread_parameter;
	#ifdef THREADING_IMPLEMENTED
		int hyper_thread_offset;
		int core_number_of_thread;
		#ifdef __linux__
			cpu_set_t cpuset;
		#elif defined(__MACH__)
			thread_port_t threadport;
			thread_affinity_policy policy;
		#endif
	#endif

	try
	{
		thread_parameter = (Tthread_parameter*) parameter;

		thread_id = thread_parameter->thread_id;
		
		calling_thread_manager = thread_parameter->thread_manager;
		calling_thread_manager->team_size_set_by_Tthread_manager_active = thread_parameter->team_size;	
		calling_thread_manager->assign_thread(thread_id);
		
		
		#ifdef THREADING_IMPLEMENTED
			thread_id_with_offset = (thread_id + thread_parameter->core_number_offset) %  thread_parameter->number_of_logical_processors;
			if (thread_parameter->hyper_thread_pairs == true)
			{
				if (thread_id == 0)
					flush_info(INFO_2, "\nLogical cores 0 and 1 reside on physical core 0.");
				hyper_thread_offset = (2 * int(thread_id_with_offset))/int(thread_parameter->number_of_logical_processors);
				if (hyper_thread_offset == 1)
					core_number_of_thread = int(2 * thread_id_with_offset + 1) - int(thread_parameter->number_of_logical_processors);
				else
					core_number_of_thread = 2 * thread_id_with_offset;
			}
			else
			{
				if (thread_id == 0)
					flush_info(INFO_2, "\nLogical cores 0 and 1 reside on physical cores 0 and 1.");
				core_number_of_thread = thread_id_with_offset;
			}
			flush_info(INFO_2, "\nThread %d uses core %d.", thread_id, core_number_of_thread);
			
			#ifdef __linux__
				CPU_ZERO(&cpuset);
				CPU_SET(core_number_of_thread, &cpuset);
				pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
			#elif defined(__MACH__)
				policy.affinity_tag = core_number_of_thread;
				threadport = pthread_mach_thread_np(pthread_self());
				if (thread_policy_set(threadport, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS) 
				{
					flush_exit(ERROR_UNSPECIFIED, "set_realtime() failed while setting thread affinity.\n");
					return NULL;
				}
			#elif defined(_WIN32)
				SetThreadAffinityMask(GetCurrentThread(), 1<<core_number_of_thread);
			#endif
		#endif

				
		calling_thread_manager->connect_to_GPU();
		for (i=0; i<calling_thread_manager->list_of_thread_managers.size(); i++)
		{
			calling_thread_manager->list_of_thread_managers[i]->assign_thread(thread_id);
			calling_thread_manager->list_of_thread_managers[i]->reserve_on_GPU();
		}
	}
	catch(...)
	{
		flush_info(1,"\nException during thread setup.");
		return NULL;
	}

	try
	{
		calling_thread_manager->thread_entry();
	}
	catch (...) 
	{
		flush_info(1,"\nThread entry %d has thrown an exception...", thread_id);
	}
	
	for (i=0; i<calling_thread_manager->list_of_thread_managers.size(); i++)
		calling_thread_manager->list_of_thread_managers[i]->clear_on_GPU();
	calling_thread_manager->disconnect_from_GPU();
	
	calling_thread_manager->lazy_sync_threads();
	calling_thread_manager->team_size_set_by_Tthread_manager_active = 0;
	
	return NULL;
}


//**********************************************************************************************************************************


void Tthread_manager_active::start_threads()
{
	unsigned thread_id;
	Tthread_handle* threads;
	void* (*calling_thread)(void*);
	vector <Tthread_parameter> thread_parameters;
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			void* status;
			int return_value;
		#endif
	#endif
			
		
  PRE_THREADS_HOOK
	my_alloc(&threads, get_team_size());
	thread_parameters.resize(get_team_size());
	
	
// Store thread parameters for all threads of the team and start all except the first thread
	
	for(thread_id=0; thread_id<get_team_size(); thread_id++)
	{
		thread_parameters[thread_id].thread_id = thread_id;
		thread_parameters[thread_id].number_of_logical_processors = get_number_of_logical_processors();
		thread_parameters[thread_id].team_size = get_team_size();
		thread_parameters[thread_id].core_number_offset = core_number_offset;
		thread_parameters[thread_id].hyper_thread_pairs = hyper_threads_are_pairs(); 
		thread_parameters[thread_id].thread_manager = this;

		if(thread_id > 0)
		{
			#ifdef THREADING_IMPLEMENTED
				#if defined(POSIX_OS__) && !defined(__MINGW32__)
					return_value = pthread_create(&(threads[thread_id]), NULL, call_thread, (void*) &(thread_parameters[thread_id]));
					if (return_value > 0)
						flush_exit(ERROR_UNSPECIFIED, "Number of requested threads could not be created.");
				#elif defined(_WIN32)
					threads[thread_id] = Tthread_handle(_beginthreadex(NULL, 0, &call_thread_windows, (void*) &(thread_parameters[thread_id]), 0, NULL));
				#endif
			#endif
		}
	}
	
	
// Finally, call the first thread

	calling_thread = &call_thread;
	calling_thread((void*) &(thread_parameters[0]));


// Now collect all created threads
	
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			for(thread_id=1; thread_id<get_team_size(); thread_id++)
			{
				return_value = pthread_join(threads[thread_id], &status);
				if (return_value > 0)
					flush_exit(ERROR_UNSPECIFIED, "Could not join running threads.");
			}
		#elif defined(_WIN32)
			WaitForMultipleObjects(get_team_size()-1, threads+1, true, INFINITE);

			for(thread_id=1; thread_id<get_team_size(); thread_id++)
				CloseHandle(threads[thread_id]);
		#endif
	#endif
	my_dealloc(&threads);
	POST_THREADS_HOOK
	check_for_user_interrupt();
}




#endif



