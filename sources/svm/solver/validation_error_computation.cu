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


#if !defined (VALIDATION_ERROR_COMPUTATION_CU)
	#define VALIDATION_ERROR_COMPUTATION_CU



#include "sources/svm/solver/validation_error_computation.h"


#include "sources/shared/system_support/cuda_memory_operations.h"

#include "sources/svm/solver/basic_svm.h"



//**********************************************************************************************************************************


const int threads_per_block_val_error_computation = 32;


//**********************************************************************************************************************************


void Tbasic_svm::evaluate_val_predictions_on_GPU()
{
	unsigned thread_id;
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;
	
	
	thread_id = get_thread_id();
	copy_to_GPU(coefficient_delta, coefficient_delta_GPU[thread_id], number_coefficients_changed); 
	copy_to_GPU(coefficient_changed, coefficient_changed_GPU[thread_id], number_coefficients_changed); 
	
	grid_size_y = (kernel_control_GPU[thread_id].col_set_size - 1)/threads_per_block_val_error_computation + 1; 
	grid_size = dim3(1, grid_size_y, 1);
	block_size = dim3(1, threads_per_block_val_error_computation, 1);
	
	evaluate_val_predictions <<< grid_size, block_size >>> (validation_kernel_GPU[thread_id], prediction_GPU[thread_id], coefficient_changed_GPU[thread_id], coefficient_delta_GPU[thread_id], kernel_control_GPU[thread_id].col_set_size, kernel_control_GPU[thread_id].col_set_size_aligned, number_coefficients_changed);
	
	copy_from_GPU(prediction_ALGD + kernel_control_GPU[thread_id].col_start, prediction_GPU[thread_id], kernel_control_GPU[thread_id].col_set_size);

	cudaDeviceSynchronize();
	sync_threads();
}



//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************


__global__ void evaluate_val_predictions(double* validation_kernel_GPU, double* prediction_GPU, unsigned* coefficient_changed_GPU, double* coefficient_delta_GPU, unsigned col_set_size, unsigned col_set_size_aligned, unsigned number_coefficients_changed)
{
	unsigned j; 
	unsigned i;
	double pred;


	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < col_set_size) 
	{ 
		pred = prediction_GPU[j];
		for (i=0; i<number_coefficients_changed; i++)
			pred = pred + coefficient_delta_GPU[i] * validation_kernel_GPU[coefficient_changed_GPU[i] * col_set_size_aligned + j];
		prediction_GPU[j] = pred;
	}
}

#endif

