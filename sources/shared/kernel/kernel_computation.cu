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


#if !defined (KERNEL_COMPUTATION_GPU_CU)
	#define KERNEL_COMPUTATION_GPU_CU




#include "sources/shared/kernel/kernel_computation.h"



#include "sources/shared/kernel/kernel.h"
#include "sources/shared/kernel/kernel_functions.h"
#include "sources/shared/basic_functions/flush_print.h"



#include <cuda_runtime.h>


const int threads_per_block_kernel_computation = 8;



//**********************************************************************************************************************************


void compute_pre_kernel_on_GPU(Tkernel_control_GPU control)
{
	unsigned grid_size_x;
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;


	grid_size_x = (control.row_set_size - 1)/threads_per_block_kernel_computation + 1;
	grid_size_y = (control.col_set_size - 1)/threads_per_block_kernel_computation + 1;
	grid_size = dim3(grid_size_x, grid_size_y, 1);
	block_size = dim3(threads_per_block_kernel_computation, threads_per_block_kernel_computation, 1);

	flush_info(INFO_DEBUG, "\nComputing pre-kernel matrix on GPU with rows = %5d, colums = %5d, dim = %d, hier_coords = %d", control.row_set_size, control.col_set_size, control.dim, control.total_number_of_hierarchical_coordinates);

	compute_pre_kernel_matrix <<< grid_size, block_size >>> (control);
	
	cudaDeviceSynchronize();
}



//**********************************************************************************************************************************

__global__ void compute_pre_kernel_matrix(Tkernel_control_GPU control)
{
	unsigned j;
	unsigned i;


	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < control.row_set_size) and (j < control.col_set_size))
	{
		if (control.total_number_of_hierarchical_coordinates == 0)
			control.pre_kernel_matrix[i * control.col_set_size_aligned + j] = squared_distance(control.dim, control.row_set_size, control.row_data_set, i, control.col_set_size, control.col_data_set, j);
		else
			control.pre_kernel_matrix[i * control.col_set_size_aligned + j] = hierarchical_pre_kernel(control.full_kernel_type, control.number_of_nodes, control.total_number_of_hierarchical_coordinates, control.hierarchical_coordinate_intervals, control.hierarchical_weights_squared, control.weights_square_sum, control.row_set_size, control.row_data_set, i, control.col_set_size, control.col_data_set, j);
	}
}




//**********************************************************************************************************************************

void compute_kernel_on_GPU(Tkernel_control_GPU control)
{
	unsigned grid_size_x;
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;


	grid_size_x = (control.row_set_size - 1)/threads_per_block_kernel_computation + 1;
	grid_size_y = (control.col_set_size - 1)/threads_per_block_kernel_computation + 1;
	grid_size = dim3(grid_size_x, grid_size_y, 1);
	block_size = dim3(threads_per_block_kernel_computation, threads_per_block_kernel_computation, 1);


	flush_info(INFO_DEBUG, "\nComputing kernel matrix on GPU with rows = %5d, colums = %5d, dim = %d, hier_coords = %d", control.row_set_size, control.col_set_size, control.dim, control.total_number_of_hierarchical_coordinates);

	compute_kernel_matrix <<< grid_size, block_size >>> (control);
	
	cudaDeviceSynchronize();
}



//**********************************************************************************************************************************

__global__ void compute_kernel_matrix(Tkernel_control_GPU control)
{
	unsigned j;
	unsigned i;
	double tmp;


	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < control.row_set_size) and (j < control.col_set_size))
	{
		tmp = kernel_function(control.kernel_type, control.gamma_factor, control.pre_kernel_matrix[i * control.col_set_size_aligned + j]);
		control.kernel_matrix[i * control.col_set_size_aligned + j] = (control.row_labels[i] * control.col_labels[j] + control.kernel_offset) * tmp;
	}
}


#endif

