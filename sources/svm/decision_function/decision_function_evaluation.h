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


#if !defined (SVM_DECISION_FUNCTION_EVALUATION_H)
	#define SVM_DECISION_FUNCTION_EVALUATION_H


#include "sources/shared/system_support/cuda_basics.h"

#include "sources/svm/decision_function/svm_decision_function_GPU_control.h"


//**********************************************************************************************************************************


template <class float_type> __target_device__ inline unsigned pre_kernel_position(unsigned test_sample_no, unsigned SV_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control);
template <class float_type> __target_device__ inline unsigned pre_kernel_position_GPU(unsigned test_sample_no, unsigned SV_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control);
template <class float_type> __target_device__ inline unsigned kernel_position(unsigned test_sample_no, unsigned SV_no, unsigned gamma_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control);
template <class float_type> __target_device__ inline unsigned kernel_position_GPU(unsigned test_sample_no, unsigned SV_no, unsigned gamma_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control);


template <class float_type> __target_device__ inline unsigned evaluation_position(unsigned test_sample_no, unsigned df, Tsvm_decision_function_GPU_control<float_type>* GPU_control);
template <class float_type> __target_device__ inline unsigned coefficient_position(unsigned df, unsigned coefficient_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control);


//**********************************************************************************************************************************



#define FLOAT_TYPE float
#include "sources/svm/decision_function/decision_function_evaluation.ins.h"
#undef FLOAT_TYPE

#define FLOAT_TYPE double
#include "sources/svm/decision_function/decision_function_evaluation.ins.h"
#undef FLOAT_TYPE



//**********************************************************************************************************************************


#include "sources/svm/decision_function/decision_function_evaluation.ins.cpp"

#if !defined(COMPILE_SEPERATELY__) && !defined(COMPILE_SEPERATELY__CUDA)
	#include "sources/svm/decision_function/decision_function_evaluation.cu"
#endif


#endif
