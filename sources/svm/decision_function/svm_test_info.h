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


#if !defined (SVM_TEST_INFO_H)
	#define SVM_TEST_INFO_H


#include "sources/shared/decision_function/test_info.h"

//**********************************************************************************************************************************


class Tsvm_test_info: public Ttest_info
{
	public:
		Tsvm_test_info();
		
		void clear();
		void display(unsigned display_mode, unsigned info_level) const;
		double get_mem_GPU_time() const;
		double get_comp_GPU_time() const;

		
		double data_convert_time;
		double SVs_determine_time;

		double init_kernels_time;
		double pre_kernel_time;
		double kernel_time;
		
		unsigned pre_kernel_candidates;
		unsigned pre_kernel_evaluations;
		unsigned kernel_candidates;
		unsigned kernel_evaluations;
		
		double GPU_data_upload_time;
		double GPU_misc_upload_time;
		double GPU_download_time;
		
		double GPU_pre_kernel_time;
		double GPU_full_kernel_time;
		double GPU_kernel_time;
		double GPU_gradient_time;
		
		unsigned unit_for_kernel_evaluations;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/decision_function/svm_test_info.cpp"
#endif


#endif
