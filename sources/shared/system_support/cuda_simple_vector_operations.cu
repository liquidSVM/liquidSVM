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


#if !defined (CUDA_SIMPLE_VECTOR_OPERATIONS_CU)
	#define CUDA_SIMPLE_VECTOR_OPERATIONS_CU
 
#include "sources/shared/system_support/cuda_simple_vector_operations.h"


//**********************************************************************************************************************************


const int threads_per_block_simple_operations = 32;

__global__ void init_vector (double* vector_GPU, unsigned size, double value = 0.0);
__global__ void mult_vector (double coefficient, double* vector_GPU, unsigned size);


//**********************************************************************************************************************************


void init_vector_on_GPU(double* vector_GPU, unsigned size, double value)
{
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;
	
	grid_size_y = (size - 1)/threads_per_block_simple_operations + 1;
	grid_size = dim3(1, grid_size_y, 1); 
	block_size = dim3(1, threads_per_block_simple_operations, 1);

	if (size > 0)
		init_vector <<< grid_size, block_size >>> (vector_GPU, size, value);
}


//**********************************************************************************************************************************

void mult_vector_on_GPU(double coefficient, double* vector_GPU, unsigned size)
{
	unsigned grid_size_y;
	dim3 grid_size;
	dim3 block_size;
	
	grid_size_y = (size - 1)/threads_per_block_simple_operations + 1; 
	grid_size = dim3(1, grid_size_y, 1);
	block_size = dim3(1, threads_per_block_simple_operations, 1);
	
	if (size > 0)
		mult_vector <<< grid_size, block_size >>> (coefficient, vector_GPU, size);
}


//**********************************************************************************************************************************


__global__ void init_vector(double* vector_GPU, unsigned size, double value)
{
	unsigned j;

	j = blockIdx.y*blockDim.y + threadIdx.y; 
	if (j < size) 
		vector_GPU[j] = value;
}




//**********************************************************************************************************************************

__global__ void mult_vector(double coefficient, double* vector_GPU, unsigned size)
{
	unsigned j;

	j = blockIdx.y*blockDim.y + threadIdx.y; 
	if (j < size) 
		vector_GPU[j] = coefficient * vector_GPU[j];
}

#endif


