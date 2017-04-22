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


#ifdef COMPILE_WITH_CUDA__


//**********************************************************************************************************************************



template <class float_type> __target_device__ inline unsigned pre_kernel_position(unsigned test_sample_no, unsigned SV_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{
	return test_sample_no * GPU_control->training_set_size + SV_no;
}



//**********************************************************************************************************************************



template <class float_type> __target_device__ inline unsigned pre_kernel_position_GPU(unsigned test_sample_no, unsigned SV_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{
	return SV_no * GPU_control->test_set_size + test_sample_no;
}


//**********************************************************************************************************************************


template <class float_type> __target_device__ inline unsigned kernel_position(unsigned test_sample_no, unsigned SV_no, unsigned gamma_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{
	unsigned thread_position;
	
	
	thread_position = test_sample_no * GPU_control->training_set_size * GPU_control->gamma_list_size;
	
	return thread_position + gamma_no * GPU_control->training_set_size + SV_no;
}

//**********************************************************************************************************************************


template <class float_type> __target_device__ inline unsigned kernel_position_GPU(unsigned test_sample_no, unsigned SV_no, unsigned gamma_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{
	unsigned SV_no_position;
	
	
	SV_no_position = GPU_control->test_set_size * GPU_control->gamma_list_size * SV_no;
	
	return SV_no_position + GPU_control->test_set_size * gamma_no + test_sample_no;
}



//**********************************************************************************************************************************


template <class float_type> __target_device__ inline unsigned evaluation_position(unsigned test_sample_no, unsigned df, Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{
	return test_sample_no * GPU_control->number_of_decision_functions + df;
	
// 	The following line would be better but it is currently not matched by the access to the evaluations by decision_function_manager.
// 	Also, the improvement is marginal!
	
// 	return df * GPU_control->test_set_size + test_sample_no;
}


//**********************************************************************************************************************************



template <class float_type> __target_device__ inline unsigned coefficient_position(unsigned df, unsigned coefficient_no, Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{	
	return df * GPU_control->decision_function_max_size + coefficient_no;
	
//	The following line should be more efficient, but it does not work, since SVs_of_decision_functions_GPU and coefficient_GPU
// 	need to be ordered and uploaded according to this order, too. In particular for the SVS_... this may lead to wrong values
// 	for SVs which in turn may lead to undefined array accesses!
	
// 	return coefficient_no * GPU_control->number_of_decision_functions + df;
}

#endif

