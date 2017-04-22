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


#if !defined (SVM_DECISION_FUNCTION_GPU_CONTROL_H)
	#define SVM_DECISION_FUNCTION_GPU_CONTROL_H

	
	
#include "sources/shared/basic_types/dataset.h"

#include "sources/svm/decision_function/svm_test_info.h"

//**********************************************************************************************************************************


template <class float_type> struct Tsvm_decision_function_GPU_control
{
	unsigned dim;
	float_type* SV_set_GPU;
	float_type* SV_set_GPU_copy;
	float_type* SV_set_GPU_safe;
	unsigned SV_set_size;
	unsigned SV_set_mem_size;
	unsigned training_set_size;
	
	float_type* test_set_GPU;
	float_type* test_set_GPU_copy;
	float_type* test_set_GPU_safe;
	unsigned test_set_size;
	unsigned test_set_mem_size;
	bool keep_test_data;
	
	float_type* gamma_list_GPU;
	unsigned gamma_list_size;
	unsigned* gamma_indices_GPU;
	
	unsigned* SVs_GPU;
	unsigned* SVs_of_decision_functions_GPU;
	unsigned* SVs_with_gamma_GPU;
	unsigned SVs_with_gamma_max_size;
	unsigned* SVs_with_gamma_size_GPU;

	unsigned number_of_decision_functions;
	unsigned decision_function_max_size;
	unsigned decision_function_max_size_aligned;
	unsigned* decision_function_size_GPU;
	float_type* coefficient_GPU;
	float_type* decision_function_chunks;

	unsigned kernel_type;
	unsigned full_kernel_type;
	float_type* pre_kernel_GPU;
	float_type* kernel_GPU;
	float_type* evaluations_GPU;
	
	float_type* offsets;
	float_type* clipp_values;
	
	unsigned start_index;
	unsigned stop_index;
	unsigned number_of_chunks;
	unsigned max_test_chunk_size;
	
	float_type weights_square_sum;
	unsigned number_of_nodes;
	unsigned total_number_of_hierarchical_coordinates;
	unsigned* hierarchical_coordinate_intervals_GPU;
	float_type* hierarchical_weights_squared_GPU;
	
	bool hierarchical_kernel_flag;
	bool sparse_evaluation;
};




//**********************************************************************************************************************************




#endif
