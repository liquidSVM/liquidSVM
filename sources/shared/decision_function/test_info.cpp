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


#if !defined (TEST_INFO_CPP)
	#define TEST_INFO_CPP


	
#include "sources/shared/decision_function/test_info.h"

	
#include "sources/shared/basic_functions/flush_print.h"



//**********************************************************************************************************************************



Ttest_info::Ttest_info()
{
	clear();
}

//**********************************************************************************************************************************


void Ttest_info::clear()
{
	test_errors.clear();
	
	test_time = 0.0;
	full_test_time = 0.0;
	
	thread_overhead_time = 0.0;
	misc_preparation_time = 0.0;
	data_cell_assign_time = 0.0;
	predict_convert_time = 0.0;
	final_predict_time = 0.0;

	decision_function_time = 0.0;
	
	GPU_data_upload_time = 0.0;
	GPU_misc_upload_time = 0.0;
	GPU_download_time = 0.0;

	GPU_decision_function_time = 0.0;
}


//**********************************************************************************************************************************



void Ttest_info::display(unsigned display_mode, unsigned info_level) const
{
// 	flush_info(info_level, "\nTest error              %3.4f", test_error);
	
	flush_info(info_level, "\n\nFull test time          %3.4f", full_test_time);
	flush_info(info_level, "\nTest time               %3.4f", test_time);
	
	flush_info(info_level, "\n\nDecision function time  %3.4f", decision_function_time);
	flush_info(info_level, "\nError computation time  %3.4f", predict_convert_time * final_predict_time);
	flush_info(info_level, "\nGPU full time           %3.4f", get_mem_GPU_time() + get_comp_GPU_time());
	
	
	flush_info(info_level, "\n\nGPU data upload time    %3.4f", GPU_data_upload_time);
	flush_info(info_level, "\nGPU misc upload time    %3.4f", GPU_misc_upload_time);
	flush_info(info_level, "\nGPU download time       %3.4f", GPU_download_time);
	
	flush_info(info_level, "\nGPU evaluation time     %3.4f", GPU_decision_function_time);
}


//**********************************************************************************************************************************

double Ttest_info::get_mem_GPU_time() const
{
	double full_time;
	
	full_time = GPU_data_upload_time;
	full_time = full_time + GPU_misc_upload_time;
	full_time = full_time + GPU_download_time;
	
	return full_time;
}


//**********************************************************************************************************************************

double Ttest_info::get_comp_GPU_time() const
{
	double full_time;
	
	full_time = GPU_decision_function_time;
	
	return full_time;
}

//**********************************************************************************************************************************


#endif
