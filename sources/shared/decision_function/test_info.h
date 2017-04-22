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


#if !defined (TEST_INFO_H)
	#define TEST_INFO_H


#include <vector>
using namespace std;

//**********************************************************************************************************************************


class Ttest_info
{
	public:
		Ttest_info();
		
		void clear();
		void display(unsigned display_mode, unsigned info_level) const;
		double get_mem_GPU_time() const;
		double get_comp_GPU_time() const;
		
		vector <double> test_errors;
		double test_time;
		double full_test_time;
		
		double thread_overhead_time;
		double misc_preparation_time;
		double data_cell_assign_time;
		double predict_convert_time;
		double final_predict_time;
		
		double decision_function_time;
		
		double GPU_data_upload_time;
		double GPU_misc_upload_time;
		double GPU_download_time;
		
		double GPU_decision_function_time;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/decision_function/test_info.cpp"
#endif


#endif
