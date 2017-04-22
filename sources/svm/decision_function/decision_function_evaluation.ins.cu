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





//**********************************************************************************************************************************

void compute_evaluations_on_GPU(vector<double>& evaluations, Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, const Tdataset& test_set, Tsvm_test_info& test_info)
{
	unsigned i;
	unsigned start_index;
	unsigned stop_index;


	if ((GPU_control.keep_test_data == true) and (GPU_control.number_of_chunks > 1))
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Trying to keep test data on GPU but %d chunks are required.", GPU_control.number_of_chunks);
	
	for (i=0; i<GPU_control.number_of_chunks; i++)
	{
		start_index = GPU_control.start_index + i * GPU_control.max_test_chunk_size;
		stop_index = min(GPU_control.stop_index, GPU_control.start_index + (i + 1) * GPU_control.max_test_chunk_size);

		
		upload_test_set_chunk_onto_GPU(GPU_control, test_set, start_index, stop_index, test_info);
		
		compute_kernel_chunk_on_GPU(GPU_control, test_set, start_index, stop_index, test_info);
		compute_evaluation_chunk_on_GPU(evaluations, GPU_control, start_index, stop_index, test_info);

		clean_test_set_chunk_on_GPU(GPU_control, test_info);
	}
}

//**********************************************************************************************************************************


void compute_kernel_chunk_on_GPU(Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, const Tdataset& test_set, unsigned start_index, unsigned stop_index, Tsvm_test_info& test_info)
{
	dim3 grid_size;
	dim3 block_size;
	unsigned grid_size_x;
	unsigned grid_size_y;
	unsigned test_samples_per_block_kernel;
	unsigned train_samples_per_block_kernel;
	Tcuda_timer cuda_timer;
	

	test_samples_per_block_kernel = 32;
	train_samples_per_block_kernel = 32;

	grid_size_x = (GPU_control.test_set_size - 1)/test_samples_per_block_kernel + 1; 
	grid_size_y = (GPU_control.SV_set_size - 1)/train_samples_per_block_kernel + 1; 
	
	grid_size = dim3(grid_size_x, grid_size_y, 1);
	block_size = dim3(test_samples_per_block_kernel, train_samples_per_block_kernel, 1);

	if (GPU_control.hierarchical_kernel_flag == false)
	{
		cuda_timer.start_timing(test_info.GPU_pre_kernel_time);
		compute_pre_kernels_KERNEL <<< grid_size, block_size >>> (GPU_control);
		cuda_timer.stop_timing(test_info.GPU_pre_kernel_time);

		cuda_timer.start_timing(test_info.GPU_kernel_time);
		if (GPU_control.sparse_evaluation == false)
				compute_kernels_KERNEL <<< grid_size, block_size >>> (GPU_control);
		else 
				compute_sparse_kernels_KERNEL <<< grid_size, block_size >>> (GPU_control);
		cuda_timer.stop_timing(test_info.GPU_kernel_time);
	}
	else
	{
		if (GPU_control.sparse_evaluation == false)
		{
			cuda_timer.start_timing(test_info.GPU_full_kernel_time);
			compute_full_deep_kernels_KERNEL <<< grid_size, block_size >>> (GPU_control);
			cuda_timer.stop_timing(test_info.GPU_full_kernel_time);
		}
		else
		{
			cuda_timer.start_timing(test_info.GPU_pre_kernel_time);
			compute_pre_deep_kernels_KERNEL <<< grid_size, block_size >>> (GPU_control);
			cuda_timer.stop_timing(test_info.GPU_pre_kernel_time);

			cuda_timer.start_timing(test_info.GPU_kernel_time);
			compute_sparse_kernels_KERNEL <<< grid_size, block_size >>> (GPU_control);
			cuda_timer.stop_timing(test_info.GPU_kernel_time);
		}
	}
}

//**********************************************************************************************************************************


void compute_evaluation_chunk_on_GPU(vector<double>& evaluations, Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, unsigned start_index, unsigned stop_index, Tsvm_test_info& test_info)
{
	dim3 grid_size;
	dim3 block_size;
	unsigned grid_size_x;
	unsigned grid_size_y;
	unsigned grid_size_z;
	unsigned dfs_per_block;
	unsigned test_samples_per_block_eval;
	unsigned svs_per_thread;
	unsigned sum_threads_per_block;
	unsigned svs_per_block;
	unsigned required_shared_memory;
	Tcuda_timer cuda_timer;
	
	
	if (GPU_control.sparse_evaluation == false)
	{
		dfs_per_block = 1;
		test_samples_per_block_eval = 32;
		sum_threads_per_block = 1;
		svs_per_block = GPU_control.decision_function_max_size;
	}
	else
	{
		dfs_per_block = 1;
		test_samples_per_block_eval = 32;
		sum_threads_per_block = 8;
		svs_per_thread = 25;
		
		svs_per_block = sum_threads_per_block * svs_per_thread;
	}

	grid_size_x = (GPU_control.test_set_size - 1)/test_samples_per_block_eval + 1; 
	grid_size_y = (GPU_control.number_of_decision_functions - 1)/dfs_per_block + 1; 
	grid_size_z = (GPU_control.decision_function_max_size - 1)/svs_per_block + 1; 

	if (GPU_control.decision_function_chunks == NULL)
		my_alloc_GPU(&GPU_control.decision_function_chunks, sum_threads_per_block * grid_size_z * GPU_control.number_of_decision_functions * GPU_control.max_test_chunk_size);


	grid_size = dim3(grid_size_x, grid_size_y, grid_size_z);
	block_size = dim3(test_samples_per_block_eval, dfs_per_block, sum_threads_per_block);


	if (GPU_control.sparse_evaluation == false)
	{
		required_shared_memory = get_aligned_size(dfs_per_block * GPU_control.decision_function_max_size, test_samples_per_block_eval);
		required_shared_memory = required_shared_memory * sizeof(FLOAT_TYPE);

		GPU_control.decision_function_max_size_aligned = get_aligned_size(GPU_control.decision_function_max_size, test_samples_per_block_eval);

		if (required_shared_memory > 47 * KILOBYTE)
			flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Required shared memory is %d KB", required_shared_memory/KILOBYTE);
		else if (required_shared_memory > 15 * KILOBYTE)
			cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
		else
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

		cuda_timer.start_timing(test_info.GPU_decision_function_time);
		evaluate_decision_functions_KERNEL <<< grid_size, block_size, required_shared_memory >>> (GPU_control);
		cuda_timer.stop_timing(test_info.GPU_decision_function_time);
		
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	}
	else
	{
		cuda_timer.start_timing(test_info.GPU_decision_function_time);
		evaluate_sparse_decision_functions_KERNEL <<< grid_size, block_size >>> (GPU_control, svs_per_thread);
		
		grid_size = dim3(grid_size_x, grid_size_y, 1);
		block_size = dim3(test_samples_per_block_eval, dfs_per_block, 1);

		sum_svs_chunks_decision_functions_KERNEL <<< grid_size, block_size >>> (GPU_control, svs_per_thread, sum_threads_per_block * grid_size_z);
		
		cuda_timer.stop_timing(test_info.GPU_decision_function_time);
	}

	cuda_timer.start_timing(test_info.GPU_download_time);
	copy_from_GPU(&(evaluations[start_index * GPU_control.number_of_decision_functions]), GPU_control.evaluations_GPU, GPU_control.test_set_size * GPU_control.number_of_decision_functions);
	cuda_timer.stop_timing(test_info.GPU_download_time);
}


//**********************************************************************************************************************************


void upload_test_set_chunk_onto_GPU(Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, const Tdataset& test_set, unsigned start_index, unsigned stop_index, Tsvm_test_info& test_info)
{
	Tcuda_timer cuda_timer;
	

	GPU_control.test_set_size = stop_index - start_index;
	if (GPU_control.keep_test_data == false)
	{
		cuda_timer.start_timing(test_info.GPU_data_upload_time);
		GPU_control.test_set_GPU = test_set.upload_to_GPU<FLOAT_TYPE>(start_index, stop_index);
		GPU_control.test_set_mem_size = test_set.required_memory_on_GPU(start_index, stop_index);
		cuda_timer.stop_timing(test_info.GPU_data_upload_time);
	}
}

//**********************************************************************************************************************************


void clean_test_set_chunk_on_GPU(Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, Tsvm_test_info& test_info)
{
	Tcuda_timer cuda_timer;

	cuda_timer.start_timing(test_info.GPU_data_upload_time);
	if (GPU_control.keep_test_data == false)
		my_dealloc_GPU(&GPU_control.test_set_GPU);
	cuda_timer.stop_timing(test_info.GPU_data_upload_time);
}





//**********************************************************************************************************************************
//**********************************************************************************************************************************
// 
// CUDA kernels
// 
//**********************************************************************************************************************************
//**********************************************************************************************************************************



__global__ void compute_pre_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control)
{
	unsigned i;
	unsigned j;
	unsigned pre_kernel_pos;

	
	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 

	if ((i < GPU_control.test_set_size) and (j < GPU_control.SV_set_size))
	{
		pre_kernel_pos = pre_kernel_position_GPU(i, j, &GPU_control);

		GPU_control.pre_kernel_GPU[pre_kernel_pos] = squared_distance(GPU_control.dim, GPU_control.SV_set_size, GPU_control.SV_set_GPU, j, GPU_control.test_set_size, GPU_control.test_set_GPU, i);
	}
}


//**********************************************************************************************************************************

__global__ void compute_pre_deep_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control)
{
	unsigned i;
	unsigned j;
	unsigned pre_kernel_pos;


	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 

	if ((i < GPU_control.test_set_size) and (j < GPU_control.SV_set_size))
	{
		pre_kernel_pos = pre_kernel_position_GPU(i, j, &GPU_control);

		GPU_control.pre_kernel_GPU[pre_kernel_pos] = hierarchical_pre_kernel(GPU_control.full_kernel_type, GPU_control.number_of_nodes, GPU_control.total_number_of_hierarchical_coordinates, GPU_control.hierarchical_coordinate_intervals_GPU, GPU_control.hierarchical_weights_squared_GPU, GPU_control.weights_square_sum, GPU_control.SV_set_size, GPU_control.SV_set_GPU, j, GPU_control.test_set_size, GPU_control.test_set_GPU, i);
	}
}



//**********************************************************************************************************************************

__global__ void compute_full_deep_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control)
{
	unsigned i;
	unsigned j;
	unsigned ig;
	FLOAT_TYPE pre_kernel_value;


	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 

	if ((i < GPU_control.test_set_size) and (j < GPU_control.SV_set_size))
	{
		pre_kernel_value = hierarchical_pre_kernel(GPU_control.full_kernel_type, GPU_control.number_of_nodes, GPU_control.total_number_of_hierarchical_coordinates, GPU_control.hierarchical_coordinate_intervals_GPU, GPU_control.hierarchical_weights_squared_GPU, GPU_control.weights_square_sum, GPU_control.SV_set_size, GPU_control.SV_set_GPU, j, GPU_control.test_set_size, GPU_control.test_set_GPU, i);

		for (ig=0;ig<GPU_control.gamma_list_size;ig++)
			GPU_control.kernel_GPU[kernel_position_GPU(i, j, ig, &GPU_control)] = kernel_function(GPU_control.kernel_type, GPU_control.gamma_list_GPU[ig], pre_kernel_value);
	}
}


//**********************************************************************************************************************************


__global__ void compute_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control)
{
	unsigned i;
	unsigned j;
	unsigned ig;
	FLOAT_TYPE pre_kernel_value;
	
	
	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 
	
	if ((i < GPU_control.test_set_size) and (j < GPU_control.SV_set_size))
		for (ig=0;ig<GPU_control.gamma_list_size;ig++)
		{
			pre_kernel_value = GPU_control.pre_kernel_GPU[pre_kernel_position_GPU(i, j, &GPU_control)];
			GPU_control.kernel_GPU[kernel_position_GPU(i, j, ig, &GPU_control)] = kernel_function(GPU_control.kernel_type, GPU_control.gamma_list_GPU[ig], pre_kernel_value);
		}
}



//**********************************************************************************************************************************


__global__ void evaluate_decision_functions_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control)
{
	unsigned i;
	unsigned j;
	unsigned ig;
	unsigned df;
	unsigned max_size_df;
	FLOAT_TYPE evaluation;
	unsigned coeff_pos_start;
	#if FLOAT_TYPE == double
		extern __shared__ double df_coefficients_SHARED_DP[];
	#elif FLOAT_TYPE == float
		extern __shared__ float df_coefficients_SHARED_SP[];
	#endif


	i = blockIdx.x * blockDim.x + threadIdx.x; 
	df = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (df < GPU_control.number_of_decision_functions)
	{
		max_size_df = GPU_control.decision_function_max_size;
		coeff_pos_start = coefficient_position(df, 0, &GPU_control);
		
		for (j=0; j<GPU_control.decision_function_max_size_aligned; j=j+blockDim.x)
			#if FLOAT_TYPE == double
				df_coefficients_SHARED_DP[j + threadIdx.x] = GPU_control.coefficient_GPU[coeff_pos_start + j + threadIdx.x];
			#elif FLOAT_TYPE == float
				df_coefficients_SHARED_SP[j + threadIdx.x] = GPU_control.coefficient_GPU[coeff_pos_start + j + threadIdx.x];
			#endif

		__syncthreads();
		if (i < GPU_control.test_set_size)
		{
			ig = GPU_control.gamma_indices_GPU[df];
			evaluation = GPU_control.offsets[df];
			for (j=0; j<max_size_df; j++)
				#if FLOAT_TYPE == double
					evaluation = evaluation + df_coefficients_SHARED_DP[j] * GPU_control.kernel_GPU[kernel_position_GPU(i, j, ig, &GPU_control)];
				#elif FLOAT_TYPE == float
					evaluation = evaluation + df_coefficients_SHARED_SP[j] * GPU_control.kernel_GPU[kernel_position_GPU(i, j, ig, &GPU_control)];
				#endif

			if (GPU_control.clipp_values[df] > 0.0)
					evaluation = max(-GPU_control.clipp_values[df], min(GPU_control.clipp_values[df], evaluation));
			GPU_control.evaluations_GPU[evaluation_position(i, df, &GPU_control)] = evaluation;
		}
	}
}




//**********************************************************************************************************************************


__global__ void compute_sparse_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control)
{
	unsigned i;
	unsigned j;
	unsigned ig;
	unsigned SV_no;
	FLOAT_TYPE pre_kernel_value;
	
	
	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 
	
	if ((i < GPU_control.test_set_size) and (j < GPU_control.SVs_with_gamma_max_size))
		for (ig=0;ig<GPU_control.gamma_list_size;ig++)
			if (j<GPU_control.SVs_with_gamma_size_GPU[ig])
			{
				SV_no = GPU_control.SVs_with_gamma_GPU[ig * GPU_control.SVs_with_gamma_max_size + j];
				pre_kernel_value = GPU_control.pre_kernel_GPU[pre_kernel_position_GPU(i, SV_no, &GPU_control)];
				
				GPU_control.kernel_GPU[kernel_position_GPU(i, SV_no, ig, &GPU_control)] = kernel_function(GPU_control.kernel_type, GPU_control.gamma_list_GPU[ig], pre_kernel_value);
			}
}




//**********************************************************************************************************************************


__global__ void evaluate_sparse_decision_functions_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control, unsigned svs_per_thread)
{
	unsigned i;
	unsigned j;
	unsigned coeff_pos;
	unsigned SV_no;
	unsigned ig;
	unsigned df;
	FLOAT_TYPE evaluation;
	unsigned c;
	unsigned svs_start;
	unsigned svs_stop;


	i = blockIdx.x * blockDim.x + threadIdx.x; 
	df = blockIdx.y * blockDim.y + threadIdx.y;
	c = blockIdx.z * blockDim.z + threadIdx.z;
	svs_start = svs_per_thread * c;
	
	
	if ((i < GPU_control.test_set_size) and (df < GPU_control.number_of_decision_functions))
		if (svs_start < GPU_control.decision_function_size_GPU[df])
		{
			ig = GPU_control.gamma_indices_GPU[df];
			evaluation = 0.0;
		
			svs_stop = min(svs_start + svs_per_thread, GPU_control.decision_function_size_GPU[df]);

			
			for (j=svs_start; j<svs_stop; j++)
			{
				coeff_pos = coefficient_position(df, j, &GPU_control);
				SV_no = GPU_control.SVs_of_decision_functions_GPU[coeff_pos];

// 				evaluation = evaluation + GPU_control.coefficient_GPU[coeff_pos] * GPU_control.kernel_GPU[kernel_position_GPU(i, SV_no, ig, &GPU_control)];
				evaluation = evaluation + GPU_control.coefficient_GPU[coeff_pos] * kernel_function(GPU_control.kernel_type, GPU_control.gamma_list_GPU[ig], GPU_control.pre_kernel_GPU[pre_kernel_position_GPU(i, SV_no, &GPU_control)]);
			}
			
			GPU_control.decision_function_chunks[df * blockDim.z * gridDim.z * GPU_control.max_test_chunk_size + c * GPU_control.max_test_chunk_size + i] = evaluation;
		}   
}


//**********************************************************************************************************************************


__global__ void sum_svs_chunks_decision_functions_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control, unsigned svs_per_thread, unsigned number_of_sum_threads)
{
	unsigned c;
	unsigned i;
	unsigned df;
	unsigned effective_svs_chunks;
	FLOAT_TYPE evaluation;

	i = blockIdx.x * blockDim.x + threadIdx.x; 
	df = blockIdx.y * blockDim.y + threadIdx.y;


	if ((i < GPU_control.test_set_size) and (df < GPU_control.number_of_decision_functions))
	{
		if (GPU_control.decision_function_size_GPU[df] > 0)
			effective_svs_chunks = (GPU_control.decision_function_size_GPU[df] - 1) / svs_per_thread + 1;
		else
			effective_svs_chunks = 0;
	
		evaluation = GPU_control.offsets[df];
		for (c=0; c<effective_svs_chunks; c++)
			evaluation = evaluation + GPU_control.decision_function_chunks[df * number_of_sum_threads * GPU_control.max_test_chunk_size + c * GPU_control.max_test_chunk_size + i];
			
		if (GPU_control.clipp_values[df] > 0.0)
			evaluation = max(-GPU_control.clipp_values[df], min(GPU_control.clipp_values[df], evaluation));

		GPU_control.evaluations_GPU[evaluation_position(i, df, &GPU_control)] = evaluation;
	}
}


