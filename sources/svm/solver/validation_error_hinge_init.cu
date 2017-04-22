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


#if !defined (VALIDATION_ERROR_COMPUTATION_INIT_CU)
	#define VALIDATION_ERROR_COMPUTATION_INIT_CU





#include "sources/svm/solver/validation_error_hinge_init.h"


#include "sources/shared/system_support/cuda_memory_operations.h"

#include "sources/svm/solver/hinge_svm.h"


//**********************************************************************************************************************************


const int threads_per_block_init_val = 32;


//**********************************************************************************************************************************

void Thinge_svm::init_full_predictions_on_GPU(Tsvm_train_val_info train_val_info)
{
	unsigned thread_id;
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;


	thread_id = get_thread_id();
	grid_size_y = (kernel_control_GPU[thread_id].col_set_size - 1)/threads_per_block_init_val + 1; 
	grid_size = dim3(1, grid_size_y, 1);
	block_size = dim3(1, threads_per_block_init_val, 1);

	init_full_predictions <<< grid_size, block_size >>> (prediction_GPU[thread_id], prediction_init_neg_GPU[thread_id], prediction_init_pos_GPU[thread_id], kernel_control_GPU[thread_id].col_set_size, C_current * train_val_info.neg_weight, C_current * train_val_info.pos_weight);
}


//**********************************************************************************************************************************


void Thinge_svm::init_neg_and_pos_predictions_on_GPU(double* prediction_init_GPU, double sign)
{
	unsigned i;
	unsigned thread_id;
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;
	
	
	sync_threads();
	thread_id = get_thread_id();
	if (is_first_team_member() == true)
	{
		number_coefficients_changed = 0;
		for (i=0;i<training_set_size;i++)
			if (training_label_ALGD[i] * sign > 0.0)
				push_back_update(1.0, i);
	}
	sync_threads();

	copy_to_GPU(coefficient_changed, coefficient_changed_GPU[thread_id], number_coefficients_changed); 
	
	grid_size_y = (kernel_control_GPU[thread_id].col_set_size - 1)/threads_per_block_init_val + 1; 
	grid_size = dim3(1, grid_size_y, 1);	
	block_size = dim3(1, threads_per_block_init_val, 1);
	
	init_neg_and_pos_predictions <<< grid_size, block_size >>> (validation_kernel_GPU[thread_id], prediction_init_GPU, coefficient_changed_GPU[thread_id], kernel_control_GPU[thread_id].col_set_size, kernel_control_GPU[thread_id].col_set_size_aligned, number_coefficients_changed);
}



//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************


__global__ void init_full_predictions(double* prediction_GPU, double* prediction_init_neg_GPU, double* prediction_init_pos_GPU, unsigned col_set_size, double neg_weight, double pos_weight)
{
	unsigned j;
	
	j = blockIdx.y * blockDim.y + threadIdx.y; 
	if (j < col_set_size) 
		prediction_GPU[j] = pos_weight * prediction_init_pos_GPU[j] - neg_weight * prediction_init_neg_GPU[j];
}



//**********************************************************************************************************************************

__global__ void init_neg_and_pos_predictions(double* validation_kernel_GPU, double* prediction_GPU, unsigned* coefficient_changed_GPU, unsigned col_set_size, unsigned col_set_size_aligned, unsigned number_coefficients_changed)
{
	unsigned j; 
	unsigned i;
	double pred;
	
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < col_set_size) 
	{ 
		pred = 0.0;
		for (i=0; i<number_coefficients_changed; i++)
			pred = pred + validation_kernel_GPU[coefficient_changed_GPU[i] * col_set_size_aligned + j];
		prediction_GPU[j] = pred;
	}
}


#endif

