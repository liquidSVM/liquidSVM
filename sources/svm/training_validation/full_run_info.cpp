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


#if !defined (FULL_RUN_INFO_CPP)
	#define FULL_RUN_INFO_CPP


#include "sources/svm/training_validation/full_run_info.h"



#include "sources/shared/basic_functions/extra_string_functions.h"


//**********************************************************************************************************************************



Tfull_run_info::Tfull_run_info()
{
	clear();
}



//**********************************************************************************************************************************


void Tfull_run_info::clear()
{
	data_dim = 0;
	number_of_labels = 0;
	training_set_size = 0;
	
	number_of_tasks = 0;
	total_number_of_cells = 0;
	number_of_cells_for_task.clear();	
	
	train_time = 0.0;
	train_full_time = 0.0;
	train_partition_time = 0.0;
	train_cell_assign_time = 0.0;
	train_kernel_time = 0.0;
	train_solver_time = 0.0;
	train_validation_time = 0.0;
	
	
	select_time = 0.0;
	select_full_time = 0.0;
	select_cell_assign_time = 0.0;
	select_kernel_time = 0.0;
	select_solver_time = 0.0;

	
	hit_smallest_gamma = 0;
	hit_largest_gamma = 0;
	hit_smallest_lambda = 0;
	hit_largest_lambda = 0;
	hit_smallest_weight = 0;
	hit_largest_weight = 0;
	
	
	test_errors.clear();
	
	test_set_size = 0;
	test_time = 0.0;
	test_full_time = 0.0;
	test_cell_assign_time = 0.0;
	test_kernel_time = 0.0;
	test_eval_time = 0.0;
	test_misc_time = 0.0;
}



//**********************************************************************************************************************************


string Tfull_run_info::displaystring_pre_train() const
{
	unsigned task;
	string output;
	
	output = output + "data dim                 = " + number_to_string(data_dim, 0) + "\n";
	output = output + "training set size        = " + number_to_string(training_set_size, 0) + "\n";
	output = output + "number of labels         = " + number_to_string(number_of_labels, 0) + "\n";
	
	output = output + "number of tasks          = " + number_to_string(number_of_tasks, 0) + "\n";
	output = output + "total number of cells    = " + number_to_string(total_number_of_cells, 0) + "\n";
	output = output + "number of cells per task =";
	for (task=0; task<number_of_tasks; task++)
		output = output + " " + number_to_string(number_of_cells_for_task[task], 0);
	output = output + "\n";
	
	
	output = output + "\n";
	
	return output;
}




//**********************************************************************************************************************************


string Tfull_run_info::displaystring_post_train() const
{
	string output;

	
	output = output + "train full time        = " + number_to_string(train_full_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "train time             = " + number_to_string(train_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "train partition time   = " + number_to_string(train_partition_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "train cell_assign time = " + number_to_string(train_cell_assign_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "train kernel time      = " + number_to_string(train_kernel_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "train solver time      = " + number_to_string(train_solver_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "train validation time  = " + number_to_string(train_validation_time, 7, DISPLAY_FLOAT) + "\n";
	
	output = output + "\n";
	
	return output;
}


//**********************************************************************************************************************************


string Tfull_run_info::displaystring_post_select() const
{
	string output;
	

	output = output + "hit largest gamma   = " + number_to_string(hit_largest_gamma, 0) + "\n";
	output = output + "hit smallest gamma  = " + number_to_string(hit_smallest_gamma, 0) + "\n";
	
	output = output + "hit largest lambda  = " + number_to_string(hit_largest_lambda, 0) + "\n";
	output = output + "hit smallest lambda = " + number_to_string(hit_smallest_lambda, 0) + "\n";

	output = output + "hit largest weight  = " + number_to_string(hit_largest_weight, 0) + "\n";
	output = output + "hit smallest weight = " + number_to_string(hit_smallest_weight, 0) + "\n";
	
	output = output + "\nselect full time        = " + number_to_string(select_full_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "select time             = " + number_to_string(select_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "select cell_assign time = " + number_to_string(select_cell_assign_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "select kernel time      = " + number_to_string(select_kernel_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "select solver time      = " + number_to_string(select_solver_time, 7, DISPLAY_FLOAT) + "\n";
	
	output = output + "\n";
	
	return output;
}



//**********************************************************************************************************************************


string Tfull_run_info::displaystring_post_test() const
{
	string output;
	
	
	output = output + "test set size         = " + number_to_string(test_set_size, 0) + "\n";
	
	output = output + "test full time        = " + number_to_string(test_full_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "test time             = " + number_to_string(test_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "test cell assign time = " + number_to_string(test_cell_assign_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "test kernel time      = " + number_to_string(test_kernel_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "test eval time        = " + number_to_string(test_eval_time, 7, DISPLAY_FLOAT) + "\n";
	output = output + "test misc time        = " + number_to_string(test_misc_time, 7, DISPLAY_FLOAT) + "\n";

	output = output + "test errors           =";
	for (unsigned i=0; i<test_errors.size(); i++)
		output = output + " " + number_to_string(test_errors[i], 7, DISPLAY_FLOAT);
	output = output + "\n";
	
	output = output + "\n";
	
	return output;
}


//**********************************************************************************************************************************

#endif
