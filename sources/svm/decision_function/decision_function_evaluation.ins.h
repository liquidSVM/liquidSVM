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


void compute_evaluations_on_GPU(vector<double>& evaluations, Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, const Tdataset& test_set, Tsvm_test_info& test_info);
void compute_kernel_chunk_on_GPU(Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, const Tdataset& test_set, unsigned start_index, unsigned stop_index, Tsvm_test_info& test_info);
void compute_evaluation_chunk_on_GPU(vector<double>& evaluations, Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, unsigned start_index, unsigned stop_index, Tsvm_test_info& test_info);

void upload_test_set_chunk_onto_GPU(Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, const Tdataset& test_set, unsigned start_index, unsigned stop_index, Tsvm_test_info& test_info);
void clean_test_set_chunk_on_GPU(Tsvm_decision_function_GPU_control<FLOAT_TYPE>& GPU_control, Tsvm_test_info& test_info);

//**********************************************************************************************************************************


#ifdef  COMPILE_WITH_CUDA__

__global__ void compute_pre_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control);
__global__ void compute_pre_deep_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control);

__global__ void compute_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control);
__global__ void compute_sparse_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control);

__global__ void compute_full_deep_kernels_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control);

__global__ void evaluate_decision_functions_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control);
__global__ void evaluate_sparse_decision_functions_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control, unsigned svs_per_thread);
__global__ void sum_svs_chunks_decision_functions_KERNEL(Tsvm_decision_function_GPU_control<FLOAT_TYPE> GPU_control, unsigned svs_per_thread, unsigned number_of_sum_threads);

#endif

//**********************************************************************************************************************************
