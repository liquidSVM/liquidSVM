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


#if !defined (KERNEL_CPP)
	#define KERNEL_CPP


#include "sources/shared/kernel/kernel.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/memory_constants.h"
#include "sources/shared/basic_functions/random_subsets.h"
#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/memory_allocation.h"

#ifdef COMPILE_WITH_CUDA__
	#include <cuda_runtime.h>
	#include "sources/shared/system_support/cuda_memory_operations.h"
	#include "sources/shared/kernel/kernel_computation.h"
#endif

//**********************************************************************************************************************************


Tkernel::Tkernel()
{
	assigned = false;
	all_kNNs_assigned = false;
	gamma_factor = 1.0;
	max_kNNs = 0;

	row_labels_ALGD = NULL;
	col_labels_ALGD = NULL;
	kernel_row_ALGD = NULL;

	GPUs = 0;
	
	remainder_is_zero = false;
	row_set_size = 0;
	col_set_size = 0;
	max_aligned_col_set_size = 0;
	
	
// HIERARCHICAL KERNEL DEVELOPMENT
	
	hierarchical_kernel_flag = false;
}

//**********************************************************************************************************************************

Tkernel::~Tkernel()
{
	clear();
}


//**********************************************************************************************************************************

void Tkernel::clear()
{
	clear_matrix(kernel_row, kernel_control.memory_model_kernel);
	clear_matrix(pre_kernel_row, kernel_control.memory_model_pre_kernel);

	assigned = false;
	all_kNNs_assigned = false;
	remainder_is_zero = false;
	max_kNNs = 0;
	
	row_set_size = 0;
	col_set_size = 0;
	max_aligned_col_set_size = 0;

	kernel_control.clear();
	clear_kNN_list();

	my_dealloc_ALGD(&row_labels_ALGD);
	my_dealloc_ALGD(&col_labels_ALGD);
	my_dealloc_ALGD(&kernel_row_ALGD);
	
	cache.clear();
	pre_cache.clear();
	
	clear_threads();
	
	
// HIERARCHICAL KERNEL DEVELOPMENT
	
	hierarchical_kernel_flag = false;
	hierarchical_row_set.clear();
	hierarchical_col_set.clear();
}



//**********************************************************************************************************************************

void Tkernel::reserve(const Tparallel_control& parallel_ctrl, const Tkernel_control& kernel_control)
{
	bool triangular;
	unsigned max_row_set_size;
	unsigned rows_for_cache;
	
	
	// First, prepare Tkernel for multiple threads
	
	Tthread_manager::reserve_threads(parallel_ctrl);

	#ifdef  COMPILE_WITH_CUDA__
		row_labels_GPU.assign(get_team_size(), NULL);
		col_labels_GPU.assign(get_team_size(), NULL);
		kernel_control_GPU.resize(get_team_size());
		matrix_CPU.assign(get_team_size(), NULL);
		pre_matrix_CPU.assign(get_team_size(), NULL);
		hierarchical_coordinate_intervals_GPU.assign(get_team_size(), NULL);
		hierarchical_weights_squared_GPU.assign(get_team_size(), NULL);
	#endif
		

	// Save the kernel_control and deal with cache option. Also change some memory models to 
	// deal with empty data sets and GPUs that require the BLOCK model due to the copying 
	// procedure between mainboard and GPU. Finally, make sure that the memory model is 
	// EMPTY, if the max_row size is 0.

	Tkernel::kernel_control = kernel_control;
	if ((Tkernel::kernel_control.memory_model_pre_kernel == CACHE) and (Tkernel::kernel_control.memory_model_kernel != EMPTY))
		Tkernel::kernel_control.memory_model_kernel = CACHE;
	if ((Tkernel::kernel_control.memory_model_kernel == CACHE) and (Tkernel::kernel_control.memory_model_pre_kernel != EMPTY))
		Tkernel::kernel_control.memory_model_pre_kernel = CACHE;

	if (GPUs > 0)
	{
		if (kernel_control.pre_kernel_store_on_GPU == true)
			Tkernel::kernel_control.memory_model_pre_kernel = EMPTY;
		else
			Tkernel::kernel_control.memory_model_pre_kernel = BLOCK;
		
		if (kernel_control.kernel_store_on_GPU == true)
			Tkernel::kernel_control.memory_model_kernel = EMPTY;
		else
			Tkernel::kernel_control.memory_model_kernel = BLOCK;
		
		if ((kernel_control.kernel_store_on_GPU == false) and (kernel_control.split_matrix_on_GPU_by_rows == false))
			flush_exit(ERROR_DATA_STRUCTURE, "It is impossible to store a kernel matrix on the motherboard's RAM, if it is split by columns on the GPU.");
	}

	if ((Tkernel::kernel_control.max_row_set_size == 0) or (Tkernel::kernel_control.max_col_set_size == 0))
	{
		Tkernel::kernel_control.memory_model_pre_kernel = EMPTY;
		Tkernel::kernel_control.memory_model_kernel = EMPTY;
	}

	
	// With these preparations reserve memory (first for the pre-kernel matrix and then
	// for the kernel matrix). Finally, reserve memory for the kNN_list.

	max_row_set_size = Tkernel::kernel_control.max_row_set_size;
	if (Tkernel::kernel_control.memory_model_pre_kernel == CACHE)
	{
		rows_for_cache = unsigned(double(kernel_control.pre_cache_size * MEGABYTE) / double(kernel_control.max_col_set_size * sizeof(double)));
		Tkernel::kernel_control.max_row_set_size = min(Tkernel::kernel_control.max_row_set_size, rows_for_cache);
		pre_cache.reserve(Tkernel::kernel_control.max_row_set_size);
	}
	triangular = kernel_control.same_data_sets and (GPUs == 0) and (Tkernel::kernel_control.memory_model_pre_kernel != CACHE);
	reserve_matrix(pre_kernel_row, Tkernel::kernel_control.memory_model_pre_kernel, triangular);
	Tkernel::kernel_control.max_row_set_size = max_row_set_size; 
	
	max_row_set_size = Tkernel::kernel_control.max_row_set_size;
	if (Tkernel::kernel_control.memory_model_kernel == CACHE)
	{
		rows_for_cache = unsigned(double(kernel_control.cache_size * MEGABYTE) / double(kernel_control.max_col_set_size * sizeof(double)));
		Tkernel::kernel_control.max_row_set_size = min(Tkernel::kernel_control.max_row_set_size, rows_for_cache);
		cache.reserve(Tkernel::kernel_control.max_row_set_size);
	}
	reserve_matrix(kernel_row, Tkernel::kernel_control.memory_model_kernel, false);
	Tkernel::kernel_control.max_row_set_size = max_row_set_size; 

	reserve_kNN_list();
	assigned = false;
	
	
	
// HIERARCHICAL KERNEL DEVELOPMENT

	if (kernel_control.is_hierarchical_kernel() == true)
	{
		flush_warn(WARN_ALL, "You are currently using an experimental hierarchical kernel.\nIt is only available for completely pre-computed matrices.");
		hierarchical_kernel_flag = true;
		Tkernel::kernel_control.make_consistent();
	}
	else
		hierarchical_kernel_flag = false;
}



//**********************************************************************************************************************************

void Tkernel::reserve_matrix(vector <double*>& rows, unsigned memory_model, bool triangular)
{
	unsigned i;
	unsigned j;
	double* dummy_ptr;
	
	
	if ((kernel_control.max_row_set_size == 0) or (kernel_control.max_col_set_size == 0))
		if (memory_model != EMPTY)
			flush_exit(ERROR_DATA_STRUCTURE, "Memory model for kernel matrix should be EMPTY since:\nrow_size = %d\ncol_size = %d", kernel_control.max_row_set_size, kernel_control.max_col_set_size);
		
	clear_matrix(rows, memory_model);
	rows.resize(kernel_control.max_row_set_size);

	if (triangular == true)
		switch (memory_model)
		{
			case LINE_BY_LINE:
				max_aligned_col_set_size = kernel_control.max_col_set_size;
				for(i=0; i<kernel_control.max_col_set_size; i++)
					my_alloc_ALGD(&rows[i], i);
				flush_info(INFO_DEBUG, "\nTriangular matrix of size %d built.", kernel_control.max_col_set_size);
				break;
			case BLOCK:
				max_aligned_col_set_size = kernel_control.max_col_set_size;
				my_alloc_ALGD(&rows[0], size_t(kernel_control.max_col_set_size) * (size_t(kernel_control.max_col_set_size) - 1) / 2);
				j = 0;
				for(i=1; i<kernel_control.max_col_set_size; i++)
				{
					rows[i] = &(rows[0][j]);
					j = j + i;
				}
				break;
			case CACHE:
				flush_exit(ERROR_DATA_STRUCTURE, "The kernel matrix memory model %d is not available for triangular matrices.", memory_model);
			case EMPTY:
				max_aligned_col_set_size = allocated_memory_ALGD(&dummy_ptr, kernel_control.max_col_set_size);
				rows.clear();
				break;
		}
	else
	{
		max_aligned_col_set_size = allocated_memory_ALGD(&dummy_ptr, kernel_control.max_col_set_size);
		
		switch (memory_model)
		{
			case LINE_BY_LINE:
				for(i=0; i<kernel_control.max_row_set_size; i++)
					my_alloc_ALGD(&rows[i], kernel_control.max_col_set_size);
				flush_info(INFO_DEBUG, "\nRectengular matrix of size %d x %d (aligned %d x %d) built.", kernel_control.max_row_set_size, kernel_control.max_col_set_size, kernel_control.max_row_set_size, max_aligned_col_set_size);
				break;
			case BLOCK:
				my_alloc_ALGD(&rows[0], size_t(max_aligned_col_set_size) * size_t(kernel_control.max_row_set_size));
				for(i=0; i<kernel_control.max_row_set_size; i++)
					rows[i] = &(rows[0][i * max_aligned_col_set_size]);
				flush_info(INFO_DEBUG, "\nBlocked rectengular matrix of size %d x %d (aligned %d x %d) built.", kernel_control.max_row_set_size, kernel_control.max_col_set_size, kernel_control.max_row_set_size, max_aligned_col_set_size);
				break;
			case CACHE:
				for(i=0; i<kernel_control.max_row_set_size; i++)
					my_alloc_ALGD(&rows[i], kernel_control.max_col_set_size);
				flush_info(INFO_DEBUG, "\nRectengular cache matrix of size %d x %d (aligned %d x %d) built.", kernel_control.max_row_set_size, kernel_control.max_col_set_size, kernel_control.max_row_set_size, max_aligned_col_set_size);
				break;
			case EMPTY:
				rows.clear();
				break;
		}
	}
}

//**********************************************************************************************************************************

void Tkernel::clear_matrix(vector <double*>& rows, unsigned memory_model)
{
	unsigned i;

	switch (memory_model)
	{
		case LINE_BY_LINE:
			flush_info(INFO_DEBUG, "\nDeallocating matrix with %d rows.", rows.size());
			for(i=0; i<rows.size(); i++)
				my_dealloc_ALGD(&rows[i]);
			break;
		case BLOCK:
			flush_info(INFO_DEBUG, "\nDeallocating blocked matrix with %d rows.", rows.size());
			if (rows.size() > 0)
				my_dealloc_ALGD_return_to_OS(&rows[0]);
			break;
		case CACHE:
			flush_info(INFO_DEBUG, "\nDeallocating cache matrix with %d rows.", rows.size());
			for(i=0; i<rows.size(); i++)
				my_dealloc(&rows[i]);
			break;
		case EMPTY:
			break;
	}

	rows.clear();
}



//**********************************************************************************************************************************

void Tkernel::reserve_on_GPU()
{
	#ifdef  COMPILE_WITH_CUDA__
		unsigned chunk_row_set_size;
		unsigned chunk_col_set_size;
		double available_memory;
		double required_memory;
		double two_matrices_factor;
		unsigned thread_id;
		Tthread_chunk thread_chunk;
		

		thread_id = get_thread_id();
		if (GPUs > 0)
		{
			available_memory = available_memory_on_GPU(kernel_control.allowed_percentage_of_GPU_RAM);

			if (kernel_control.pre_kernel_store_on_GPU == true)
				two_matrices_factor = 2.0;
			else
				two_matrices_factor = 1.0;
			
			if (kernel_control.split_matrix_on_GPU_by_rows == true)
			{
				thread_chunk = get_thread_chunk(kernel_control.max_row_set_size);
				chunk_col_set_size = max_aligned_col_set_size;
				chunk_row_set_size = thread_chunk.size;
			}
			else
			{
				thread_chunk = get_thread_chunk(max_aligned_col_set_size);
				chunk_col_set_size = thread_chunk.size;
				chunk_row_set_size = kernel_control.max_row_set_size;
			}
			
			kernel_control_GPU[thread_id].size = size_t(chunk_row_set_size) * size_t(chunk_col_set_size);
			required_memory = two_matrices_factor * double(kernel_control_GPU[thread_id].size) * double(sizeof(double));
			if (required_memory > available_memory)
				flush_exit(ERROR_OUT_OF_MEMORY, "The kernel matrix on GPU %d for thread %d requires %d MB but only %d MB\nout of %d MB of free memory are available for it.", get_GPU_id(), thread_id, convert_to_MB(required_memory), convert_to_MB(available_memory), convert_to_MB(free_memory_on_GPU()));

			flush_info(INFO_DEBUG, "\nReserving %d rectengular matrices of size %d x %d on GPU %d by thread %d.", int(two_matrices_factor), chunk_row_set_size,  chunk_col_set_size, get_GPU_id(), thread_id);
				
			my_alloc_GPU(&(kernel_control_GPU[thread_id].kernel_matrix), kernel_control_GPU[thread_id].size);
			if (kernel_control.pre_kernel_store_on_GPU == true)
				my_alloc_GPU(&(kernel_control_GPU[thread_id].pre_kernel_matrix), kernel_control_GPU[thread_id].size);
			else
				kernel_control_GPU[thread_id].pre_kernel_matrix = kernel_control_GPU[thread_id].kernel_matrix;
			
			cudaDeviceSynchronize();
			lazy_sync_threads();
		}
	#endif
}



//**********************************************************************************************************************************

void Tkernel::clear_on_GPU()
{
	#ifdef  COMPILE_WITH_CUDA__
		unsigned thread_id;

		
		if (GPUs > 0)
		{
			thread_id = get_thread_id();
			flush_info(INFO_DEBUG, "\nDeallocating matrix with %d rows on GPU %d by thread %d.", row_set_size, get_GPU_id(), thread_id);

			my_dealloc_GPU(&(kernel_control_GPU[thread_id].row_data_set));
			my_dealloc_GPU(&(kernel_control_GPU[thread_id].col_data_set));
			my_dealloc_GPU(&row_labels_GPU[thread_id]);
			my_dealloc_GPU(&col_labels_GPU[thread_id]);
			my_dealloc_GPU(&(kernel_control_GPU[thread_id].kernel_matrix));
			if (kernel_control.pre_kernel_store_on_GPU == true)
				my_dealloc_GPU(&(kernel_control_GPU[thread_id].pre_kernel_matrix));
			my_dealloc_GPU(&hierarchical_coordinate_intervals_GPU[thread_id]);
			my_dealloc_GPU(&hierarchical_weights_squared_GPU[thread_id]);
			
			lazy_sync_threads();
		}
	#endif
}




//**********************************************************************************************************************************

void Tkernel::clear_kNN_list()
{
	unsigned i;

	if (kNN_list.size() != 0)
		for(i=0;i<kNN_list.size();i++)
			if (kNN_list[i] != NULL)
				delete kNN_list[i];

	kNN_list.clear();
	kNNs_found.clear();
	all_kNNs_assigned = false;
	max_kNNs = 0;
}



//**********************************************************************************************************************************

void Tkernel::reserve_kNN_list()
{
	unsigned i;

	clear_kNN_list();
	if (kernel_control.kNNs > 0)
	{
		kNN_list.resize(kernel_control.max_col_set_size);
		for(i=0; i<kernel_control.max_col_set_size; i++)
			kNN_list[i] = new Tordered_index_set(kernel_control.kNNs);

		kNNs_found.resize(kernel_control.max_col_set_size);
		kNNs_found.assign(kernel_control.max_col_set_size, 0);
	}
}

//**********************************************************************************************************************************

void Tkernel::load(const Tdataset& row_data_set, const Tdataset& col_data_set, double& build_time, double& transfer_time)
{
	unsigned i;
	#ifdef  COMPILE_WITH_CUDA__
		unsigned thread_id;
		unsigned total_number_of_coordinates;
		vector <unsigned> hierarchical_coordinate_intervals;
		Tthread_chunk thread_chunk;
		Tdataset full_coord_dataset;
	#endif

		

	if (is_first_team_member() == true)
	{
		flush_info(INFO_PEDANTIC_DEBUG, "\nLoading datasets of size %d and %d into an object of type Tkernel.", row_data_set.size(), col_data_set.size());
		assigned = false;
		all_kNNs_assigned = false;
		remainder_is_zero = false;
		
		row_set_size = row_data_set.size();
		Tkernel::row_data_set.resize(row_set_size);
		for (i=0; i<row_set_size; i++)
			Tkernel::row_data_set[i] = row_data_set.sample(i);
		realloc_and_copy_ALGD(&row_labels_ALGD, row_data_set.get_labels());

		col_set_size = col_data_set.size();
		Tkernel::col_data_set.resize(col_set_size);
		for (i=0; i<col_set_size; i++)
			Tkernel::col_data_set[i] = col_data_set.sample(i);
		realloc_and_copy_ALGD(&col_labels_ALGD, col_data_set.get_labels(), current_aligned_col_set_size);

		if ((kernel_control.same_data_sets == true) and (kernel_control.include_labels == true))
			kernel_offset = 0.0;
		else
		{
			kernel_offset = 1.0;
			for(i=0; i<row_set_size; i++)
				row_labels_ALGD[i] = 0.0;
			for(i=0; i<col_set_size; i++)
				col_labels_ALGD[i] = 0.0;
		}

		my_realloc_ALGD(&kernel_row_ALGD, max_aligned_col_set_size);

		permutated_indices = random_permutation(row_set_size);
		
		
		// HIERARCHICAL KERNEL DEVELOPMENT

		if (hierarchical_kernel_flag == true)
		{
			kernel_control.convert_to_hierarchical_data_set(row_data_set, hierarchical_row_set);
			kernel_control.convert_to_hierarchical_data_set(col_data_set, hierarchical_col_set);

			weights_square_sum = kernel_control.get_hierarchical_weight_square_sum();
		}

		// HIERARCHICAL KERNEL DEVELOPMENT END		
		
	}
	lazy_sync_threads();

	#ifdef COMPILE_WITH_CUDA__
		if (GPUs > 0)
		{
			thread_id = get_thread_id();
			
			my_dealloc_GPU(&(kernel_control_GPU[thread_id].row_data_set));
			my_dealloc_GPU(&(kernel_control_GPU[thread_id].col_data_set));
			my_dealloc_GPU(&row_labels_GPU[thread_id]);
			my_dealloc_GPU(&col_labels_GPU[thread_id]);
			
			flush_info(INFO_DEBUG, "\nUploading labels of size %d to GPU by thread %d.", kernel_control.max_row_set_size, get_GPU_id(), thread_id);

			my_alloc_GPU(&row_labels_GPU[thread_id], kernel_control.max_row_set_size);
			copy_to_GPU(&row_labels_ALGD[0], row_labels_GPU[thread_id], kernel_control.max_row_set_size);
			
			flush_info(INFO_DEBUG, "\nUploading labels of size %d to GPU by thread %d.", current_aligned_col_set_size, get_GPU_id(), thread_id);

			my_alloc_GPU(&col_labels_GPU[thread_id], current_aligned_col_set_size);
			copy_to_GPU(&col_labels_ALGD[0], col_labels_GPU[thread_id], current_aligned_col_set_size);
			
			if (kernel_control.split_matrix_on_GPU_by_rows == true)
			{
				thread_chunk = get_thread_chunk(kernel_control.max_row_set_size);
				
				kernel_control_GPU[thread_id].row_start = thread_chunk.start_index;
				kernel_control_GPU[thread_id].row_stop = min(row_set_size, thread_chunk.stop_index);
				kernel_control_GPU[thread_id].row_set_size = kernel_control_GPU[thread_id].row_stop - kernel_control_GPU[thread_id].row_start;
				
				kernel_control_GPU[thread_id].col_start = 0;
				kernel_control_GPU[thread_id].col_stop = col_set_size;
				kernel_control_GPU[thread_id].col_set_size = col_set_size;
				kernel_control_GPU[thread_id].col_set_size_aligned = max_aligned_col_set_size;
				
				kernel_control_GPU[thread_id].row_labels = &(row_labels_GPU[thread_id][kernel_control_GPU[thread_id].row_start]);
				kernel_control_GPU[thread_id].col_labels = col_labels_GPU[thread_id];
				
				
				// Finally, set the addresses of the (pre)_kernel_matrix on the motherboard's RAM.

				if (kernel_control.memory_model_pre_kernel != EMPTY)
					pre_matrix_CPU[thread_id] = &(pre_kernel_row[0][thread_chunk.start_index * max_aligned_col_set_size]);
				if (kernel_control.memory_model_kernel != EMPTY)
					matrix_CPU[thread_id] = &(kernel_row[0][thread_chunk.start_index * max_aligned_col_set_size]);
			}
			else
			{
				thread_chunk = get_thread_chunk(max_aligned_col_set_size);
				
				kernel_control_GPU[thread_id].row_start = 0;
				kernel_control_GPU[thread_id].row_stop = row_set_size;
				kernel_control_GPU[thread_id].row_set_size = row_set_size;
				
				kernel_control_GPU[thread_id].col_start = thread_chunk.start_index;
				kernel_control_GPU[thread_id].col_stop = min(col_set_size, thread_chunk.stop_index);
				kernel_control_GPU[thread_id].col_set_size = kernel_control_GPU[thread_id].col_stop - kernel_control_GPU[thread_id].col_start;
				kernel_control_GPU[thread_id].col_set_size_aligned = thread_chunk.size;
				
				kernel_control_GPU[thread_id].row_labels = row_labels_GPU[thread_id];
				kernel_control_GPU[thread_id].col_labels = &(col_labels_GPU[thread_id][kernel_control_GPU[thread_id].col_start]);
				
				
				// Finally, set the addresses of the pre_kernel_matrix on the motherboard's RAM.
				// Note, that it is superfluous to do the same for the kernel_matrix, since this 
				// case is impossible for splits by columns.
				
				if (kernel_control.memory_model_pre_kernel != EMPTY)
					pre_matrix_CPU[thread_id] = &(pre_kernel_row[0][thread_chunk.start_index * kernel_control.max_row_set_size]);
			}
			
			kernel_control_GPU[thread_id].dim = row_data_set.dim();
			kernel_control_GPU[thread_id].kernel_offset = kernel_offset;
			kernel_control_GPU[thread_id].kernel_type = kernel_control.kernel_type;
			kernel_control_GPU[thread_id].full_kernel_type = kernel_control.full_kernel_type;

			
			if (hierarchical_kernel_flag == true)
			{
				// Determine the total number of hierarchical coordinates and the positions of the coordinates

				total_number_of_coordinates = kernel_control.get_hierarchical_coordinate_intervals(hierarchical_coordinate_intervals);
				

				// Allocate memory for weight and coordinates on GPU and copy information onto GPU
				
				flush_info(INFO_DEBUG, "\nUploading hierarchical information of size %d and %d to GPU %d by thread %d.", hierarchical_coordinate_intervals.size(), kernel_control.hierarchical_coordinates.size(), get_GPU_id(), thread_id);
				
				my_dealloc_GPU(&hierarchical_coordinate_intervals_GPU[thread_id]);
				my_dealloc_GPU(&hierarchical_weights_squared_GPU[thread_id]);
				
				my_alloc_GPU(&(hierarchical_coordinate_intervals_GPU[thread_id]), hierarchical_coordinate_intervals.size());
				my_alloc_GPU(&(hierarchical_weights_squared_GPU[thread_id]), kernel_control.hierarchical_coordinates.size());
				
				copy_to_GPU(&hierarchical_coordinate_intervals[0], hierarchical_coordinate_intervals_GPU[thread_id], hierarchical_coordinate_intervals.size());
				copy_to_GPU(&kernel_control.hierarchical_weights_squared[0], hierarchical_weights_squared_GPU[thread_id], kernel_control.hierarchical_coordinates.size());
				
				
				// Pass information to kernel_control_GPU so that code on GPU knows what to do and how.
				
				kernel_control_GPU[thread_id].weights_square_sum = weights_square_sum;
				kernel_control_GPU[thread_id].number_of_nodes = kernel_control.hierarchical_coordinates.size();
				kernel_control_GPU[thread_id].total_number_of_hierarchical_coordinates = total_number_of_coordinates;
				kernel_control_GPU[thread_id].hierarchical_coordinate_intervals = hierarchical_coordinate_intervals_GPU[thread_id];
				kernel_control_GPU[thread_id].hierarchical_weights_squared = hierarchical_weights_squared_GPU[thread_id];
			
				
 				// Now, convert hierarchical data set format into one for GPUs and upload the data.

				kernel_control.convert_to_hierarchical_GPU_data_set(hierarchical_row_set, full_coord_dataset, kernel_control_GPU[thread_id].row_start, kernel_control_GPU[thread_id].row_stop);
				flush_info(INFO_DEBUG, "\nUploading hierarchical row data of size %d and dimension %d to GPU %d by thread %d.", full_coord_dataset.size(), full_coord_dataset.dim(), get_GPU_id(), thread_id);
				kernel_control_GPU[thread_id].row_data_set = full_coord_dataset.upload_to_GPU<double>(0, full_coord_dataset.size());
				
				kernel_control.convert_to_hierarchical_GPU_data_set(hierarchical_col_set, full_coord_dataset, kernel_control_GPU[thread_id].col_start, kernel_control_GPU[thread_id].col_stop);
				flush_info(INFO_DEBUG, "\nUploading hierarchical col data of size %d and dimension %d to GPU %d by thread %d.", full_coord_dataset.size(), full_coord_dataset.dim(), get_GPU_id(), thread_id);
				kernel_control_GPU[thread_id].col_data_set = full_coord_dataset.upload_to_GPU<double>(0, full_coord_dataset.size());
			}
			else
			{
				kernel_control_GPU[thread_id].total_number_of_hierarchical_coordinates = 0;

				flush_info(INFO_DEBUG, "\nUploading row and column data of size %d and %d to GPU %d by thread %d.", kernel_control_GPU[thread_id].row_set_size, kernel_control_GPU[thread_id].col_set_size, get_GPU_id(), thread_id);

				kernel_control_GPU[thread_id].row_data_set = row_data_set.upload_to_GPU<double>(kernel_control_GPU[thread_id].row_start, kernel_control_GPU[thread_id].row_stop);
				kernel_control_GPU[thread_id].col_data_set = col_data_set.upload_to_GPU<double>(kernel_control_GPU[thread_id].col_start, kernel_control_GPU[thread_id].col_stop);
			}
		
			cudaDeviceSynchronize();
			lazy_sync_threads();
		}
	#endif
	
	pre_assign(build_time, transfer_time);
}



//**********************************************************************************************************************************


unsigned Tkernel::get_row_set_size() const
{
	return row_set_size;
}

//**********************************************************************************************************************************


unsigned Tkernel::get_col_set_size() const
{
	return col_set_size;
}

//**********************************************************************************************************************************

double* restrict__ Tkernel::get_row_labels_ALGD()
{
	unsigned i;
	unsigned row_set_size_aligned;
	double* restrict__ row_labels_copy_ALGD;
	
	if (row_set_size > 0)
	{
		my_alloc_ALGD(&row_labels_copy_ALGD, row_set_size, row_set_size_aligned);
		for (i=0; i<row_set_size; i++)
			row_labels_copy_ALGD[i] = row_data_set[i]->label;
		for (i=row_set_size; i<row_set_size_aligned; i++)
			row_labels_copy_ALGD[i] = 0.0;
		
		return row_labels_copy_ALGD;
	}
	else
		return NULL;
}


//**********************************************************************************************************************************

double* restrict__ Tkernel::get_col_labels_ALGD()
{
	unsigned i;
	unsigned col_set_size_aligned;
	double* restrict__ col_labels_copy_ALGD;
	

	if (col_set_size > 0)
	{
		my_alloc_ALGD(&col_labels_copy_ALGD, col_set_size, col_set_size_aligned);
		for (i=0; i<col_set_size; i++)
			col_labels_copy_ALGD[i] = col_data_set[i]->label;
		for (i=col_set_size; i<col_set_size_aligned; i++)
			col_labels_copy_ALGD[i] = 0.0;
		
		return col_labels_copy_ALGD;
	}
	else
		return NULL;
}



//**********************************************************************************************************************************

void Tkernel::pre_assign(double& build_time, double& transfer_time)
{
	unsigned i;
	unsigned ii;
	unsigned j;
	#ifdef  COMPILE_WITH_CUDA__
		unsigned thread_id;
	#endif
	Tthread_chunk thread_chunk;
	

	thread_chunk = get_thread_chunk(kernel_control.max_row_set_size);
	if ((kernel_control.same_data_sets == true) and (kNNs_found.size() > 0))
		for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
			kNNs_found[permutated_indices[i]] = 0;

	if (GPUs > 0)
	{
		#ifdef  COMPILE_WITH_CUDA__
			thread_id = get_thread_id();
			lazy_sync_threads_and_get_time_difference(build_time, build_time);
			compute_pre_kernel_on_GPU(kernel_control_GPU[thread_id]);
			lazy_sync_threads_and_get_time_difference(build_time, build_time);

			lazy_sync_threads_and_get_time_difference(transfer_time, transfer_time);
			if (kernel_control.memory_model_pre_kernel != EMPTY)
				copy_from_GPU(pre_matrix_CPU[thread_id], kernel_control_GPU[thread_id].pre_kernel_matrix, kernel_control_GPU[thread_id].size);
			lazy_sync_threads_and_get_time_difference(transfer_time, transfer_time);
		#endif
	}
	else
	{
		get_time_difference(build_time, build_time);
		if (kernel_control.is_full_matrix_pre_model() == true)
		{
			if (kernel_control.same_data_sets == true)
				for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
				{
					ii = permutated_indices[i];
					if (hierarchical_kernel_flag == true)
						for(j=0;j<ii;j++)
							pre_kernel_row[ii][j] = hierarchical_pre_kernel_function(weights_square_sum, kernel_control.hierarchical_weights_squared, hierarchical_row_set[ii], hierarchical_col_set[j]);
					else
						for(j=0;j<ii;j++)
							pre_kernel_row[ii][j] = pre_kernel_function(kernel_control.kernel_type, row_data_set[ii], col_data_set[j]);
				}
			else
			{
				for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
					if (hierarchical_kernel_flag == true)
						for(j=0;j<col_set_size;j++)
							pre_kernel_row[i][j] = hierarchical_pre_kernel_function(weights_square_sum, kernel_control.hierarchical_weights_squared, hierarchical_row_set[i], hierarchical_col_set[j]);
					else
						for(j=0;j<col_set_size;j++)
							pre_kernel_row[i][j] = pre_kernel_function(kernel_control.kernel_type, row_data_set[i], col_data_set[j]);
			}
		}
		else if ((kernel_control.memory_model_pre_kernel == CACHE) and (is_first_team_member() == true))
			pre_cache.clear();
		lazy_sync_threads();
		get_time_difference(build_time, build_time);
	}
}




//**********************************************************************************************************************************

void Tkernel::assign(double gamma, double& build_time, double& transfer_time, double& kNN_build_time)
{
	unsigned i;
	unsigned ii;
	unsigned j;
	double tmp;
	double* pre_kernel_row_tmp;
	#ifdef  COMPILE_WITH_CUDA__
		unsigned thread_id;
	#endif
	Tthread_chunk thread_chunk;

	
	gamma_factor = compute_gamma_factor(kernel_control.kernel_type, gamma);

	if (GPUs > 0)
	{
		#ifdef  COMPILE_WITH_CUDA__
			thread_id = get_thread_id();
			lazy_sync_threads_and_get_time_difference(transfer_time, transfer_time);
			kernel_control_GPU[thread_id].gamma_factor = gamma_factor;
			if (kernel_control.memory_model_pre_kernel != EMPTY)
				copy_to_GPU(pre_matrix_CPU[thread_id], kernel_control_GPU[thread_id].pre_kernel_matrix, kernel_control_GPU[thread_id].size);
			lazy_sync_threads_and_get_time_difference(transfer_time, transfer_time);

			lazy_sync_threads_and_get_time_difference(build_time, build_time);
			compute_kernel_on_GPU(kernel_control_GPU[thread_id]);
			lazy_sync_threads_and_get_time_difference(build_time, build_time);

			lazy_sync_threads_and_get_time_difference(transfer_time, transfer_time);
			if (kernel_control.memory_model_kernel != EMPTY)
				copy_from_GPU(matrix_CPU[thread_id], kernel_control_GPU[thread_id].kernel_matrix, kernel_control_GPU[thread_id].size);
			lazy_sync_threads_and_get_time_difference(transfer_time, transfer_time);
		#endif
		remainder_is_zero = false;
	}
	else 
	{
		if (kernel_control.is_full_matrix_model() == false)
		{
			if ((kernel_control.memory_model_kernel == CACHE) and (is_first_team_member() == true))
				cache.clear();

			assigned = true;
			sync_threads();
			return;
		}
		
		lazy_sync_threads();
		get_time_difference(build_time, build_time);
		thread_chunk = get_thread_chunk(kernel_control.max_row_set_size);
		
		pre_kernel_row_tmp = NULL;
		if (kernel_control.memory_model_pre_kernel == EMPTY)
			my_alloc_ALGD(&pre_kernel_row_tmp, max_aligned_col_set_size);
			
		if (kernel_control.same_data_sets == true)
			for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
			{
				ii = permutated_indices[i];

				if (kernel_control.memory_model_pre_kernel == EMPTY)
					for(j=0;j<ii;j++)
						pre_kernel_row_tmp[j] = pre_kernel_function(kernel_control.kernel_type, row_data_set[ii], col_data_set[j]);
				else
					pre_kernel_row_tmp = pre_kernel_row[ii];

				for(j=0;j<ii;j++)
				{
					tmp = compute_entry(ii, j, pre_kernel_row_tmp[j]);
					kernel_row[ii][j] = tmp;
					kernel_row[j][ii] = tmp;
				}
				kernel_row[ii][ii] = 1.0;
			}
		else
			for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
			{
				if (kernel_control.memory_model_pre_kernel == EMPTY)
					for(j=0;j<col_set_size;j++)
						pre_kernel_row_tmp[j] = pre_kernel_function(kernel_control.kernel_type, row_data_set[i], col_data_set[j]);
				else
					pre_kernel_row_tmp = pre_kernel_row[i];

				for(j=0;j<col_set_size;j++)
					kernel_row[i][j] = compute_entry(i, j, pre_kernel_row_tmp[j]);
			}
		
		if (kernel_control.memory_model_pre_kernel == EMPTY)
			my_dealloc_ALGD(&pre_kernel_row_tmp);
		lazy_sync_threads();
		get_time_difference(build_time, build_time);
	}
	set_remainder_to_zero();
	assigned = true;

	if (kernel_control.same_data_sets == true)
	{
		get_time_difference(kNN_build_time, kNN_build_time);
		assign_kNN_list();
		get_time_difference(kNN_build_time, kNN_build_time);
	}
};


//**********************************************************************************************************************************

void Tkernel::find_kNNs(unsigned i, unsigned cache_kernel_row_index)
{
	unsigned j;
	unsigned start_index;
	unsigned stop_index;
	unsigned tmp_index;
	unsigned chunk_length;
	unsigned current_chunk;

	if (kernel_control.kNNs != 0)
	{
		if (kernel_control.kNN_number_of_chunks > 1)
		{
			if (kernel_control.kNN_number_of_chunks != get_team_size())
				flush_exit(ERROR_DATA_MISMATCH, "Number of chunks for kNNs does not match team size.");
			
			get_aligned_chunk(col_set_size, 2*kernel_control.kNN_number_of_chunks, 0, start_index, tmp_index);
			get_aligned_chunk(col_set_size, 2*kernel_control.kNN_number_of_chunks, 1, tmp_index, stop_index);

			chunk_length = stop_index - start_index;
			current_chunk = min(i / chunk_length, kernel_control.kNN_number_of_chunks - 1);
			
			start_index = current_chunk * chunk_length;
			if (current_chunk + 1 < kernel_control.kNN_number_of_chunks)
				stop_index = (current_chunk + 1) * chunk_length; 
			else
				stop_index = col_set_size;
		}
		else
		{
			start_index = 0;
			stop_index = col_set_size;
		}

		if (kNNs_found[i] == 0)
		{
			kNN_list[i]->clear();
			for(j=start_index;j<stop_index;j++)
				if (i != j)
					kNN_list[i]->insert(j, abs(kernel_row[cache_kernel_row_index][j]));

			lock_mutex();
			kNNs_found[i] = 1;
			max_kNNs = max(max_kNNs, kNN_list[i]->size());
			unlock_mutex();
		}
	}
}


//**********************************************************************************************************************************

void Tkernel::assign_kNN_list()
{
	unsigned i;
	Tthread_chunk thread_chunk;

	sync_threads();
	if ((kernel_control.kNNs != 0) and ((kernel_control.memory_model_kernel == BLOCK) or (kernel_control.memory_model_kernel == LINE_BY_LINE)))
	{
		thread_chunk = get_thread_chunk(kernel_control.max_row_set_size);
		for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
			find_kNNs(permutated_indices[i], permutated_indices[i]);
	}
	sync_threads();
	all_kNNs_assigned = true;
}

//**********************************************************************************************************************************

void Tkernel::set_remainder_to_zero()
{
	unsigned i;
	unsigned j;
	Tthread_chunk thread_chunk;

	sync_threads();
	if ((remainder_is_zero == false) and ((kernel_control.memory_model_kernel == BLOCK) or (kernel_control.memory_model_kernel == LINE_BY_LINE)))
	{
		thread_chunk = get_thread_chunk(kernel_control.max_row_set_size);
		for(i=thread_chunk.start_index; i<min(row_set_size, thread_chunk.stop_index); i++)
			for(j=col_set_size; j<max_aligned_col_set_size; j++)
				kernel_row[i][j] = 0.0;
	}
	sync_threads();
	remainder_is_zero = true;
}

//**********************************************************************************************************************************

void Tkernel::clear_cache_stats()
{
	cache.clear_stats();
	pre_cache.clear_stats();
}


//**********************************************************************************************************************************

void Tkernel::get_cache_stats(double& pre_cache_hits, double& cache_hits) const
{
	unsigned hits;
	unsigned misses;

	cache.get_stats(hits, misses);
	if ((hits + misses) > 0)
		cache_hits = double(hits)/double(hits + misses);
	else
		cache_hits = 1.0;

	pre_cache.get_stats(hits, misses);
	if ((hits + misses) > 0)
		pre_cache_hits = double(hits)/double(hits + misses);
	else
		cache_hits = 1.0;
}


//**********************************************************************************************************************************


vector <Tsubset_info> Tkernel::get_kNN_list() const
{
	unsigned i;
	unsigned j;
	vector <vector <unsigned> > NN_list_tmp;

	
	if (kernel_control.memory_model_kernel == CACHE)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to get all kNNs of kernel that is only cached.");
	
	if (assigned == false)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to get kNNs of kernel without having assigned values.");

	NN_list_tmp.resize(kNN_list.size());
	for (i=0; i<NN_list_tmp.size(); i++)
	{
		NN_list_tmp[i].resize(min(kNN_list[i]->size(), unsigned(max(int(col_set_size) - 1, 0))));
		for (j=0; j<NN_list_tmp[i].size(); j++)
			NN_list_tmp[i][j] = (*kNN_list[i])[j];
	}

	return NN_list_tmp;
}

//**********************************************************************************************************************************


Tkernel_control_GPU Tkernel::get_kernel_control_GPU() const
{
	return kernel_control_GPU[get_thread_id()];
}



//**********************************************************************************************************************************


#ifndef COMPILE_WITH_CUDA__

void compute_kernel_on_GPU(const Tkernel& kernel){};
void compute_pre_kernel_on_GPU(const Tkernel& kernel){};


#endif

#endif
