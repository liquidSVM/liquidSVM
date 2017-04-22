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


#if !defined (KERNEL_COMPUTATION_GPU_H)
	#define KERNEL_COMPUTATION_GPU_H
 

 
#if defined(COMPILE_SEPERATELY__CUDA)
	#include "sources/shared/kernel/kernel_computation_joint_cpugpu.h"
#endif
#include "sources/shared/kernel/kernel_control.h"
#include "sources/shared/system_support/cuda_basics.h"

#if defined(COMPILE_WITH_CUDA__) || defined(__CUDACC__)
	#include "sources/shared/kernel/kernel_control_gpu.h"
#endif

//**********************************************************************************************************************************

#ifdef  COMPILE_WITH_CUDA__
	void compute_pre_kernel_on_GPU(Tkernel_control_GPU control);
	void compute_kernel_on_GPU(Tkernel_control_GPU control);
#endif

#if defined(__CUDACC__)
	__global__ void compute_kernel_matrix(Tkernel_control_GPU control);
	__global__ void compute_pre_kernel_matrix(Tkernel_control_GPU control);

	template <class float_type> __device__ inline float_type squared_distance(unsigned dim, unsigned row_size, float_type* row_data_set, unsigned row_pos, unsigned col_size, float_type* col_data_set, unsigned col_pos);
		
	template <class float_type> __device__ inline float_type hierarchical_pre_kernel(unsigned full_kernel_type, unsigned number_of_nodes, unsigned number_of_coordinates, unsigned* coordinate_starts, float_type* weights, float_type weights_square_sum, unsigned row_size, float_type* row_data_set, unsigned row_pos, unsigned col_size, float_type* col_data_set, unsigned col_pos);
#endif

template <class float_type> __device__ inline float_type pre_kernel_init_value(unsigned full_kernel_type, float_type weights_square_sum);
template <class float_type> __device__ inline float_type pre_kernel_value_summand(unsigned full_kernel_type, float_type weight, float_type pre_kernel_l_value);
template <class float_type> __device__ inline float_type pre_kernel_l_value_conversion(unsigned full_kernel_type, float_type squared_distance);
template <class float_type> __device__ inline float_type pre_kernel_update_l_value_conversion(unsigned full_kernel_type, float_type squared_distance, float_type old_pre_kernel_l_value);


#if !defined(COMPILE_SEPERATELY__CUDA)
	__target_device__ inline unsigned feature_pos_on_GPU(unsigned sample_size, unsigned sample_no, unsigned feature_no);
	__target_device__ inline unsigned next_feature_pos_on_GPU(unsigned sample_size, unsigned current_feature_pos);
#endif
	
//**********************************************************************************************************************************



#if defined(__CUDACC__)
	#include "sources/shared/kernel/kernel_computation.ins.cu"
#endif

#if !defined(COMPILE_SEPERATELY__CUDA)
	#include "sources/shared/kernel/kernel_computation.ins.cpp"
#endif



#if !defined(COMPILE_SEPERATELY__) && !defined(COMPILE_SEPERATELY__CUDA)
	#include "sources/shared/kernel/kernel_computation.cu"
#endif



#endif
