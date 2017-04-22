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


#if !defined (FULL_RUN_INFO_H)
	#define FULL_RUN_INFO_H


	
	
#include <string>
#include <vector>
using namespace std;

//**********************************************************************************************************************************


class Tfull_run_info
{
	public:
		Tfull_run_info();
		
		void clear();
		
		void display(unsigned display_mode, unsigned info_level) const;
		
		string displaystring_pre_train() const;
		string displaystring_post_train() const;
		string displaystring_post_select() const;
		string displaystring_post_test() const;

		
		unsigned data_dim;
		unsigned number_of_labels;
		unsigned training_set_size;
		unsigned test_set_size;
		
		unsigned number_of_tasks;
		unsigned total_number_of_cells;
		vector <unsigned> number_of_cells_for_task;

		
		unsigned hit_smallest_gamma;
		unsigned hit_largest_gamma;
		unsigned hit_smallest_lambda;
		unsigned hit_largest_lambda;
		unsigned hit_smallest_weight;
		unsigned hit_largest_weight;
		
		double train_time;
		double train_full_time;
		double train_partition_time;
		double train_cell_assign_time;
		double train_kernel_time;
		double train_solver_time;
		double train_validation_time;
		
		double select_time;
		double select_full_time;
		double select_cell_assign_time;
		double select_kernel_time;
		double select_solver_time;

		
		vector <double> test_errors;
		
		double test_time;
		double test_full_time;
		double test_cell_assign_time;
		double test_kernel_time;
		double test_eval_time;
		double test_misc_time;
		

};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/training_validation/full_run_info.cpp"
#endif


#endif
