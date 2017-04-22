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


#if !defined (THREAD_MANAGER_CPP)
	#define THREAD_MANAGER_CPP


#include "sources/shared/system_support/thread_manager.h"


#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/memory_allocation.h"


#include <cmath> 
#include <vector>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string.h>


#ifdef __MACH__
	#include <pthread.h>
#endif



#ifdef  COMPILE_WITH_CUDA__
	#include <cuda_runtime.h>
#endif


//**********************************************************************************************************************************

vector <Tthread_manager_base*> Tthread_manager_base::list_of_thread_managers;


//**********************************************************************************************************************************


thread__ int Tthread_manager_base::switcher = 0;
thread__ unsigned Tthread_manager_base::GPU_id = 0;
thread__ unsigned Tthread_manager_base::thread_id = 0;
thread__ bool Tthread_manager_base::connected_to_GPU = false;
thread__ unsigned Tthread_manager_base::team_size_set_by_Tthread_manager_active = 0;

bool Tthread_manager_base::cpu_info_read = false;
bool Tthread_manager_base::hyper_thread_pairs = false;
unsigned Tthread_manager_base::number_of_physical_cores = 1;
unsigned Tthread_manager_base::number_of_logical_processors = 1;


//**********************************************************************************************************************************
//**********************************************************************************************************************************

#ifdef _WIN32

	typedef BOOL (WINAPI *LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);

	//**********************************************************************************************************************************

	DWORD count_set_bits(ULONG_PTR bit_mask)
	{
		DWORD i;
		DWORD left_shift;
		DWORD bits_set;
		ULONG_PTR bit_test;    


		bits_set = 0;
		left_shift = sizeof(ULONG_PTR) * 8 - 1;
		bit_test = ULONG_PTR(1) << left_shift;  
			
		for (i = 0; i <= left_shift; i++)
		{
			bits_set = bits_set + ((bit_mask & bit_test)? 1:0);
			bit_test = bit_test / 2;
		}

		return bits_set;
	}

	//**********************************************************************************************************************************


	void get_number_of_cores_and_logical_processors(unsigned& number_of_logical_processors, unsigned& number_of_physical_cores, bool& hyper_thread_pairs)
	{
		bool done;
		LPFN_GLPI glpi;
		DWORD rc;
		DWORD byte_offset;
		DWORD return_length;
		PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr;
		PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer;


		done = false;
		ptr = NULL;
		buffer = NULL;
		byte_offset = 0;
		return_length = 0;
		number_of_physical_cores = 1;
		number_of_logical_processors = 1;

		glpi = (LPFN_GLPI) GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetLogicalProcessorInformation");
		if (NULL == glpi) 
		{
			flush_warn(WARN_ALL, "Could not read CPU information from OS. Continuing with 1 thread.");
			return;
		}

		while (!done)
		{
			rc = glpi(buffer, &return_length);
			if (rc == FALSE) 
			{
				if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
				{
					if (buffer) 
						free(buffer);
					buffer = PSYSTEM_LOGICAL_PROCESSOR_INFORMATION (malloc(return_length));

					if (buffer == NULL) 
					{
						flush_warn(WARN_ALL, "Could not read CPU information from OS. Continuing with 1 thread.");
						return;
					}
				} 
				else 
				{
					flush_warn(WARN_ALL, "Could not read CPU information from OS. Continuing with 1 thread.");
					return;
				}
			} 
			else
				done = true;
		}

		ptr = buffer;
		number_of_physical_cores = 0;
		number_of_logical_processors = 0;
		hyper_thread_pairs = false;

		while (byte_offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= return_length) 
		{
			if (ptr->Relationship == RelationProcessorCore)
			{
				number_of_physical_cores++;
				number_of_logical_processors = number_of_logical_processors + count_set_bits(ptr->ProcessorMask);
				if (ptr->ProcessorMask == 3)
					hyper_thread_pairs = true;
			}

			// The idea of the test for hyper_threaded_pairs is the following: ptr->ProcessorMask 
			// is a bitmask that describes, which logical processors belong to the currently considered
			// core. If for one core the first two bits are set, then this core hosts the logical 
			// processors 0 and 1, and therefore hyper_thread_pairs is true. This is my understanding
			// of the description from microsoft, but due to the lack of having several windows system 
			// I was unable to seriously test it. But at least it works on my laptop. 

			byte_offset = byte_offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
			ptr++;
		}
			
		free(buffer);
	}
#endif


//**********************************************************************************************************************************
//**********************************************************************************************************************************

Tthread_manager_base::Tthread_manager_base()
{
	vector <unsigned> physical_cores;
	vector <unsigned> logical_processors;
	

	#ifndef COMPILE_FOR_R__
		if (team_size_set_by_Tthread_manager_active != 0)
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to construct a Tthread_manager while running with %d threads.", team_size);	
	#endif
	
	team_size = 1;
	GPUs = 0;

	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			pthread_mutex_init(&mutex, NULL);
			pthread_mutex_init(&barrier_mutex, NULL);
		#elif defined(_WIN32)
			mutex = CreateMutex( NULL, FALSE, NULL); 
			barrier_mutex = CreateMutex( NULL, FALSE, NULL); 
		#endif
	#endif

	global_switcher = 0;
	counter[0] = 0;
	counter[1] = 0;
	
	if (cpu_info_read == false)
	{
		#ifdef THREADING_IMPLEMENTED
			#ifdef __MACH__
				physical_cores = get_CPU_info_from_os("core_count");
				logical_processors = get_CPU_info_from_os("thread_count"); 

				if ((physical_cores.size() == 0) or (logical_processors.size() == 0))
				{
					flush_warn(WARN_ALL, "Could not read CPU information from OS. Continuing with 1 thread.");
					number_of_logical_processors = 1;
					number_of_physical_cores = 1;
				}
				else
				{
					number_of_logical_processors = logical_processors[0]; 
					number_of_physical_cores = physical_cores[0];
				}
				
	// 			Currently I do not know how to determine how logical cores are numbered compared to the physical cores in Mac OS,
	// 			so I set the flag to the value, which seems to be the more common one.
				
				hyper_thread_pairs = false;
			#elif defined(__linux__)
				physical_cores = get_CPU_info_from_os("core id");
				logical_processors = get_CPU_info_from_os("processor"); 

				if ((physical_cores.size() == 0) or (logical_processors.size() == 0))
				{
					flush_warn(WARN_ALL, "Could not read CPU information from OS. Continuing with 1 thread.");
					number_of_logical_processors = 1;
					number_of_physical_cores = 1;
				}
				else
				{
					number_of_logical_processors = logical_processors.size();
					number_of_physical_cores = physical_cores[argmax(physical_cores)] + 1;
				}

				if (physical_cores.size() > 1)
					if ((number_of_logical_processors > number_of_physical_cores) and (physical_cores[0] == physical_cores[1]))
						hyper_thread_pairs = true;
			#elif defined(_WIN32)
				get_number_of_cores_and_logical_processors(number_of_logical_processors, number_of_physical_cores, hyper_thread_pairs);
			#endif
		#else
			number_of_logical_processors = 1;
			number_of_physical_cores = 1;
		#endif
		cpu_info_read = true;
	}
}


//**********************************************************************************************************************************

Tthread_manager_base::~Tthread_manager_base()
{
	#ifdef THREADING_IMPLEMENTED
		#if defined(POSIX_OS__) && !defined(__MINGW32__)
			pthread_mutex_destroy(&mutex);
			pthread_mutex_destroy(&barrier_mutex);
		#elif defined(_WIN32)
			CloseHandle(mutex);
			CloseHandle(barrier_mutex);
		#endif
	#endif
}


//**********************************************************************************************************************************

void Tthread_manager_base::reserve_threads(Tparallel_control parallel_ctrl)
{
	vector <unsigned> position;
	
	if ((parallel_ctrl.requested_team_size > 0) and (get_number_of_logical_processors() < unsigned(parallel_ctrl.requested_team_size)))
		flush_exit(ERROR_DATA_MISMATCH, "%d threads requested but the system has only %d cores available.", parallel_ctrl.requested_team_size, get_number_of_logical_processors());
	
	requested_team_size = parallel_ctrl.requested_team_size;
	if (parallel_ctrl.requested_team_size <= 0)
		team_size = unsigned(int(get_number_of_physical_cores()) + parallel_ctrl.requested_team_size);
	else 
		team_size = parallel_ctrl.requested_team_size;

	#ifndef  COMPILE_WITH_CUDA__
		GPUs = 0;
	#else
		if (parallel_ctrl.GPUs > 0)
			GPUs = team_size;
		else 
			GPUs = 0;
		GPU_number_offset = parallel_ctrl.GPU_number_offset;
	#endif
		
	position = find(list_of_thread_managers, this);
	if (position.size() == 0)
		list_of_thread_managers.push_back(this);
}


//**********************************************************************************************************************************

Tparallel_control Tthread_manager_base::get_parallel_control() const
{
	Tparallel_control parallel_ctrl;
	
	parallel_ctrl.requested_team_size = requested_team_size;
	parallel_ctrl.GPUs = GPUs;
	
	return parallel_ctrl;
}




//**********************************************************************************************************************************

void Tthread_manager_base::assign_thread(unsigned thread_id)
{
	if (thread_id >= team_size)
		flush_exit(ERROR_DATA_MISMATCH, "Thread %d does not fit into team of size %d", thread_id, team_size);
	if (team_size_set_by_Tthread_manager_active != team_size)
		flush_exit(ERROR_DATA_MISMATCH, "Current object has %d threads reserved but calling Tthread_manager_active\nobject has %d threads reserved.", team_size, team_size_set_by_Tthread_manager_active);
	
	Tthread_manager_base::thread_id = thread_id;
	switcher = 0;
}


//**********************************************************************************************************************************

void Tthread_manager_base::connect_to_GPU()
{
	#ifdef  COMPILE_WITH_CUDA__
		int available_GPUs;
		cudaError_t error_code;

		if ((connected_to_GPU == false) and (GPUs > 0))
		{
// The following line reserves a huge amount of virtual memory, basically 
// to address CPU and GPU memory together. The memory, however, is nowwhere 
// physically requested. Seee more at 
// http://stackoverflow.com/questions/11631191/why-does-the-cuda-runtime-reserve-80-gib-virtual-memory-upon-initialization

			cudaGetDeviceCount(&available_GPUs); 
			GPU_id = (thread_id + GPU_number_offset) % available_GPUs;

			flush_info(INFO_2, "\nThread %d uses GPU %d.", thread_id, GPU_id);
			
			error_code = cudaSetDevice(GPU_id);
			if (error_code != cudaSuccess)
				flush_exit(ERROR_RUNTIME, "Thread %d cannot connect to GPU %d.", thread_id, GPU_id);
			connected_to_GPU = true;
		}
	#endif
}


//**********************************************************************************************************************************

void Tthread_manager_base::disconnect_from_GPU()
{
	#ifdef  COMPILE_WITH_CUDA__
		if (connected_to_GPU == true)
			cudaThreadExit();
		connected_to_GPU = false;
	#endif
}


//**********************************************************************************************************************************

unsigned Tthread_manager_base::get_number_of_CPU_threads_on_used_GPU() const
{
	#ifdef  COMPILE_WITH_CUDA__
		int available_GPUs;
		unsigned CPU_threads_on_GPU;
	
		cudaGetDeviceCount(&available_GPUs); 

		CPU_threads_on_GPU = get_team_size() / available_GPUs;
		if (thread_id % available_GPUs < get_team_size() % available_GPUs)
			CPU_threads_on_GPU++;
		
		if (team_size_set_by_Tthread_manager_active > 0)
			return CPU_threads_on_GPU;
		else 
			return 0;
	#else
		return 0;
	#endif
}



//**********************************************************************************************************************************

double Tthread_manager_base::free_memory_on_GPU() const
{
	#ifdef  COMPILE_WITH_CUDA__
		size_t free_memory_GPU;
		size_t total_memory_GPU;
		
		cudaMemGetInfo(&free_memory_GPU, &total_memory_GPU);
		
		if ((team_size_set_by_Tthread_manager_active > 0) and (GPUs > 0))
			return double(free_memory_GPU);
		else 
			return 0.0;
	#else
		return 0.0;
	#endif
}

//**********************************************************************************************************************************

double Tthread_manager_base::available_memory_on_GPU(double allowed_percentage) const
{
	if (get_number_of_CPU_threads_on_used_GPU() == 0)
		return 0.0;
	else
		return free_memory_on_GPU() / double(get_number_of_CPU_threads_on_used_GPU()) * allowed_percentage;
}

//**********************************************************************************************************************************

void Tthread_manager_base::clear_threads()
{
	vector <unsigned> position;
	
	#ifndef COMPILE_FOR_R__
		if (team_size_set_by_Tthread_manager_active != 0)
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to clear a Tthread_manager that is currently running with %d threads.", team_size);
	#endif
	
	team_size = 1;
	GPUs = 0;

	position = find(list_of_thread_managers, this);
	if (position.size() > 0)
		list_of_thread_managers.erase(list_of_thread_managers.begin() + position[0]);
	
	global_switcher = 0;
	counter[0] = 0;
	counter[1] = 0;
}


//**********************************************************************************************************************************


Tthread_chunk Tthread_manager_base::get_thread_chunk(unsigned size, unsigned alignment) const
{
	unsigned part;
	unsigned aligned_chunk_size;
	unsigned aligned_size;
	Tthread_chunk thread_chunk;
	
	thread_chunk.thread_id = get_thread_id();

	if (size % (alignment * team_size) == 0)
		part = unsigned(size / (alignment * team_size));
	else
		part = 1 + unsigned(size / (alignment * team_size));
	aligned_chunk_size = alignment * part;
	
	if (size % alignment == 0)
		part = unsigned(size / alignment);
	else
		part = 1 + unsigned(size / alignment);
	aligned_size = alignment * part;
	
	thread_chunk.start_index = min(size, thread_chunk.thread_id * aligned_chunk_size);
	thread_chunk.stop_index = min(size, (thread_chunk.thread_id + 1) * aligned_chunk_size);
	thread_chunk.stop_index_aligned = min(aligned_size, (thread_chunk.thread_id + 1) * aligned_chunk_size);

	thread_chunk.size = thread_chunk.stop_index - thread_chunk.start_index;
	
	return thread_chunk;
}


//**********************************************************************************************************************************


vector <unsigned> Tthread_manager_base::get_CPU_info_from_os(const char* entry)
{
	vector <unsigned> read_entries;

	#if defined(POSIX_OS__) && !defined(__MINGW32__)
		int i;
		char c;
		FILE* fp;
		int io_return;
		char command[128];

		#ifdef __MACH__
			strcpy(command, "sysctl -a | grep machdep.cpu | grep '");
		#else
			strcpy(command, "/bin/cat /proc/cpuinfo | grep '");
		#endif

		strcat(command, entry);
		strcat(command, "'");

		fp = popen(command, "r");
		c = getc(fp);
		while (c != EOF)
		{
			while (c != ':')
				c = getc(fp);
		
			io_return = fscanf(fp, "%d\n", &i);
			if (io_return == 0)
				flush_exit(ERROR_UNSPECIFIED, "Could not read hardware information from /proc/cpuinfo .");
			read_entries.push_back(i);
			
			c = getc(fp);
		}
		pclose(fp);
		
		return read_entries;
	#else
		return read_entries;
	#endif
}


//**********************************************************************************************************************************


#endif



