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


#if !defined (SVM_TEST_INFO_CPP)
	#define SVM_TEST_INFO_CPP


#include "sources/shared/basic_functions/flush_print.h"
	
#include "sources/svm/decision_function/svm_test_info.h"


//**********************************************************************************************************************************



Tsvm_test_info::Tsvm_test_info()
{
	clear();
}

//**********************************************************************************************************************************


void Tsvm_test_info::clear()
{
	Ttest_info::clear();

	data_convert_time = 0.0;
	SVs_determine_time = 0.0;
	
	init_kernels_time = 0.0;
	pre_kernel_time = 0.0;
	kernel_time = 0.0;
	
	pre_kernel_candidates = 0;
	pre_kernel_evaluations = 0;
	kernel_candidates = 0;
	kernel_evaluations = 0;
	
	GPU_data_upload_time = 0.0;
	GPU_misc_upload_time = 0.0;
	GPU_download_time = 0.0;

	GPU_pre_kernel_time = 0.0;
	GPU_full_kernel_time = 0.0;
	GPU_kernel_time = 0.0;
	GPU_gradient_time = 0.0;
	
	unit_for_kernel_evaluations = 1000;
}


//**********************************************************************************************************************************



void Tsvm_test_info::display(unsigned display_mode, unsigned info_level) const
{
	double prep_time;
	double pure_test_time;
	
	flush_info(info_level, "\n\nFull test time              %3.4f", full_test_time);
	flush_info(info_level, "\nTest time                   %3.4f", test_time);
	flush_info(info_level, "\nGPU full time               %3.4f", get_mem_GPU_time() + get_comp_GPU_time());
	
	flush_info(info_level, "\n\nThread overhead time        %3.4f", thread_overhead_time);
	flush_info(info_level, "\nData cell assign time       %3.4f", data_cell_assign_time);
	flush_info(info_level, "\nSVs determine time          %3.4f", SVs_determine_time);
	flush_info(info_level, "\nMisc data prep time         %3.4f", misc_preparation_time);
	flush_info(info_level, "\nData convert time           %3.4f", data_convert_time);
	flush_info(info_level, "\nPrediction combination time %3.4f", predict_convert_time);
	flush_info(info_level, "\nError computation time      %3.4f", final_predict_time);
	
	prep_time = thread_overhead_time + data_cell_assign_time + SVs_determine_time + misc_preparation_time + data_convert_time + predict_convert_time + final_predict_time;

	
	if (get_comp_GPU_time() == 0.0)
	{
		flush_info(info_level, "\n\nInit kernels time           %3.4f", init_kernels_time);
		flush_info(info_level, "\nPre_kernel time             %3.4f", pre_kernel_time);
		flush_info(info_level, "\nKernel time                 %3.4f", kernel_time);
		flush_info(info_level, "\nDecision function time      %3.4f", decision_function_time);

		pure_test_time = init_kernels_time + pre_kernel_time + kernel_time + decision_function_time;
		
		flush_info(info_level, "\n\nUnaccounted time            %3.4f", test_time - prep_time - pure_test_time);
		
		flush_info(info_level, "\n\nPre_kernel candidates       %d K", pre_kernel_candidates);
		flush_info(info_level, "\nPre_kernel evaluations      %d K", pre_kernel_evaluations);
		flush_info(info_level, "\nKernel candidates           %d K", kernel_candidates);
		flush_info(info_level, "\nKernel evaluations          %d K\n", kernel_evaluations);
	}
	else
	{
		flush_info(info_level, "\n\nGPU data upload time        %3.4f", GPU_data_upload_time);
		flush_info(info_level, "\nGPU misc upload time        %3.4f", GPU_misc_upload_time);
		flush_info(info_level, "\nGPU download time           %3.4f", GPU_download_time);
		
		flush_info(info_level, "\n\nGPU pre kernel time         %3.4f", GPU_pre_kernel_time);
		flush_info(info_level, "\nGPU full kernel time        %3.4f", GPU_full_kernel_time);
		flush_info(info_level, "\nGPU kernel time             %3.4f", GPU_kernel_time);
		flush_info(info_level, "\nGPU decision function time  %3.4f\n", GPU_decision_function_time);
		
		if (GPU_gradient_time > 0.0)
			flush_info(info_level, "GPU gradient descent time   %3.4f\n", GPU_gradient_time);
	}
}


//**********************************************************************************************************************************

double Tsvm_test_info::get_mem_GPU_time() const
{
	double full_time;
	
	full_time = Ttest_info::get_mem_GPU_time();
	
	return full_time;
}


//**********************************************************************************************************************************

double Tsvm_test_info::get_comp_GPU_time() const
{
	double full_time;
	
	full_time = Ttest_info::get_comp_GPU_time();
	full_time = full_time + GPU_pre_kernel_time;
	full_time = full_time + GPU_full_kernel_time;
	full_time = full_time + GPU_kernel_time;
	full_time = full_time + GPU_gradient_time;

	return full_time;
}

//**********************************************************************************************************************************


#endif
