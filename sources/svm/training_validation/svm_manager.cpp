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


#if !defined (SVM_MANAGER_CPP)
	#define SVM_MANAGER_CPP
	

#include "sources/svm/training_validation/svm_manager.h"


#include "sources/svm/decision_function/svm_test_info.h"
#include "sources/shared/training_validation/fold_manager.h"

#include "sources/svm/training_validation/svm_train_val_info.h"




//**********************************************************************************************************************************


Tsvm_manager::Tsvm_manager()
{
	use_current_grid = false;
	fp_log_train_read = NULL;
	fp_aux_train_read = NULL;
	fp_sol_train_read = NULL;
}

//**********************************************************************************************************************************


Tsvm_manager::~Tsvm_manager()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tsvm_manager ...");
	clear();
	flush_info(INFO_PEDANTIC_DEBUG, "\nTsvm_manager destroyed.");
}



//**********************************************************************************************************************************



void Tsvm_manager::load(const Tdataset& data_set)
{
	Tdataset_info data_set_info;
	
	
	clear();
	Tsvm_manager::data_set = data_set;
	Tsvm_manager::data_set.enforce_ownership();
	
	full_run_info.clear();
	full_run_info.data_dim = data_set.dim();
	full_run_info.training_set_size = data_set.size();
	
	if (data_set.is_classification_data() == true)
	{
		data_set_info = Tdataset_info(data_set, true);
		full_run_info.number_of_labels = data_set_info.label_list.size();
	}
}

//**********************************************************************************************************************************

	
void Tsvm_manager::clear()
{	
	clear_flags();
	
	working_set_manager.clear();
	decision_function_manager.clear();
	
	if (use_current_grid == false)
		current_grids.clear();
	list_of_grids.clear();
	list_of_fold_managers.clear();
}

//**********************************************************************************************************************************

	
void Tsvm_manager::clear_flags()
{
	fp_aux_train_read = NULL;
	fp_log_train_read = NULL;
	fp_sol_train_read = NULL;
	
	read_train_log_from_file_flag = false;
	read_train_aux_from_file_flag = false;
	read_train_sol_from_file_flag = false;

	write_train_log_to_file_flag = false;
	write_train_aux_to_file_flag = false;

	write_select_log_to_file_flag = false;
	write_select_sol_to_file_flag = false;
	
	use_stored_logs = false;
	use_stored_solution = false;
	store_logs_internally = false;
	
	append_decision_functions = false;
	store_decision_functions_internally = false;
}



//**********************************************************************************************************************************

unsigned Tsvm_manager::dim()
{ 
	return data_set.dim(); 
}


//**********************************************************************************************************************************

unsigned Tsvm_manager::size()
{ 
	return data_set.size(); 
}


//**********************************************************************************************************************************

unsigned Tsvm_manager::decision_functions_size()
{ 
	return decision_function_manager.size(); 
}


//**********************************************************************************************************************************

unsigned Tsvm_manager::number_of_all_tasks()
{
	unsigned all_tasks;
	
	all_tasks = working_set_manager.number_of_tasks();
	
	if (working_set_manager.get_working_set_control().working_set_selection_method > FULL_SET)
		all_tasks++;

	return all_tasks;
}


//**********************************************************************************************************************************

Tworking_set_manager Tsvm_manager::get_working_set_manager() const
{
	return working_set_manager;
}


//**********************************************************************************************************************************

Tsvm_decision_function_manager Tsvm_manager::get_decision_function_manager() const
{
	return decision_function_manager;
}



//**********************************************************************************************************************************

void Tsvm_manager::read_decision_function_manager_from_file(FILE* fpsolread, bool& data_loaded_from_sol_file)
{
	read_decision_function_manager_from_file(decision_function_manager, fpsolread, data_loaded_from_sol_file);
}


//**********************************************************************************************************************************

void Tsvm_manager::write_decision_function_manager_to_file(FILE* fpsolwrite)
{
	write_decision_function_manager_to_file(decision_function_manager, fpsolwrite);
}


//**********************************************************************************************************************************

void Tsvm_manager::read_decision_function_manager_from_file(Tsvm_decision_function_manager& decision_function_manager, const string& filename, double& file_time)
{
	bool data_loaded_from_sol_file;
	FILE* fpsolread;
	
	
	file_time = get_process_time_difference(file_time);
	fpsolread = open_file(filename, "r");
	read_decision_function_manager_from_file(decision_function_manager, fpsolread, data_loaded_from_sol_file);
	close_file(fpsolread);
	file_time = get_process_time_difference(file_time);
}


//**********************************************************************************************************************************

void Tsvm_manager::write_decision_function_manager_to_file(const Tsvm_decision_function_manager& decision_function_manager, const string& filename, double& file_time)
{
	FILE* fpsolwrite;
	
	
	file_time = get_process_time_difference(file_time);
	fpsolwrite = open_file(filename, "w");
	write_decision_function_manager_to_file(decision_function_manager, fpsolwrite);
	close_file(fpsolwrite);
	file_time = get_process_time_difference(file_time);
}




//**********************************************************************************************************************************

void Tsvm_manager::read_decision_function_manager_from_file(Tsvm_decision_function_manager& decision_function_manager, FILE* fpsolread, bool& data_loaded_from_sol_file)
{
	string filename;
	unsigned filetype;
	unsigned dim;
	unsigned size;
	
	
	filename = get_filename_of_fp(fpsolread);
	filetype = get_filetype(filename);
	check_solution_filename(filename);
	

	if (filetype == FSOL)
	{
		file_read(fpsolread, size);
		file_read(fpsolread, dim);
		data_set.read_from_file(fpsolread, CSV, size, dim);
		data_loaded_from_sol_file = true;
	}
	else
		data_loaded_from_sol_file = false;

	
	file_read(fpsolread, scale_data);
	if (scale_data == true)
	{
		file_read(fpsolread, scaling);
		file_read(fpsolread, translate);
		
		if (filetype != FSOL)
			data_set.apply_scaling(scaling, translate);
	}
	decision_function_manager.read_from_file(fpsolread, data_set);
}
		



//**********************************************************************************************************************************

void Tsvm_manager::write_decision_function_manager_to_file(const Tsvm_decision_function_manager& decision_function_manager, FILE* fpsolwrite)
{
	string filename;
	unsigned filetype;
	
	filename = get_filename_of_fp(fpsolwrite);
	filetype = get_filetype(filename);
	check_solution_filename(filename);
	
	if (filetype == FSOL)
	{
		file_write(fpsolwrite, data_set.size());
		file_write(fpsolwrite, data_set.dim());
		file_write_eol(fpsolwrite);
		
		data_set.write_to_file(fpsolwrite, CSV);
	}
	
	file_write(fpsolwrite, scale_data);
	file_write_eol(fpsolwrite);
	if (scale_data == true)
	{
		file_write(fpsolwrite, scaling);
		file_write(fpsolwrite, translate);
	}
	decision_function_manager.write_to_file(fpsolwrite);
}





//**********************************************************************************************************************************

void::Tsvm_manager::display_run_statistics()
{
	string tmp;
	
	tmp = full_run_info.displaystring_pre_train();
	flush_info(INFO_1, "\n\n%s", tmp.c_str());
	
	tmp = full_run_info.displaystring_post_train();
	flush_info(INFO_1, "\n%s\n", tmp.c_str());
	
	tmp = full_run_info.displaystring_post_select();
	flush_info(INFO_1, "\n%s\n", tmp.c_str());
	
	tmp = full_run_info.displaystring_post_test();
	flush_info(INFO_1, "\n%s\n", tmp.c_str());
}


//**********************************************************************************************************************************


vector <vector <vector < Ttrain_info_grid> > > Tsvm_manager::get_list_of_train_info() const
{
	unsigned task;
	unsigned cell;
	unsigned fold;
	vector <vector <vector < Ttrain_info_grid> > > list_of_train_info;
	
	list_of_train_info.resize(list_of_grids.size());	
	for (task=0; task<list_of_grids.size(); task++)
	{
		list_of_train_info[task].resize(list_of_grids[task].size());
		for (cell=0; cell<list_of_grids[task].size(); cell++)
		{
			list_of_train_info[task][cell].resize(list_of_grids[task][cell].size());
			for (fold=0; fold<list_of_grids[task][cell].size(); fold++)
				list_of_train_info[task][cell][fold] = list_of_grids[task][cell][fold].train_val_info;
		}
	}
	
	return list_of_train_info;
}


//**********************************************************************************************************************************


vector <vector <vector < Tsvm_train_val_info> > > Tsvm_manager::get_list_of_select_info() const
{
	return list_of_select_info;
};
	


//**********************************************************************************************************************************

	
void Tsvm_manager::train(const Ttrain_control& train_control, Tsvm_full_train_info& svm_full_train_info)
{
	string display_string;
	unsigned task;
	FILE* fpauxwrite;


// Elementary preparations
	
	svm_full_train_info.full_time = get_wall_time_difference();
	svm_full_train_info.file_time = 0.0;
	svm_full_train_info.train_time = 0.0;

	clear();
	Tsvm_manager::train_control = train_control;
	scale_data = train_control.scale_data;
	if (scale_data == true)
	{
		data_set.compute_scaling(scaling, translate, 0.0, QUANTILE, false, true);
		data_set.apply_scaling(scaling, translate);
	}
	working_set_manager = Tworking_set_manager(train_control.working_set_control, data_set);

	
// Store some elementary statistics
	
	if (clear_previous_train_info == true)
	{
		full_run_info.clear();
		svm_full_train_info.train_val_info_log.clear();
	}
	
	full_run_info.number_of_tasks = working_set_manager.number_of_tasks();
	full_run_info.total_number_of_cells = working_set_manager.total_number_of_working_sets();
	full_run_info.number_of_cells_for_task.resize(full_run_info.number_of_tasks);
	for (task=0; task<working_set_manager.number_of_tasks(); task++)
		full_run_info.number_of_cells_for_task[task] = working_set_manager.number_of_cells(task);
	
	if (train_control.summary_log_filename.size() > 0)
	{
		fpauxwrite = open_file(train_control.summary_log_filename, "w");
		
		display_string = full_run_info.displaystring_pre_train();
		fprintf(fpauxwrite, "%s", display_string.c_str());

		close_file(fpauxwrite);
	}

	
// Prepare communication with filesystem and the internal storage
	
	write_train_log_to_file_flag = (train_control.write_log_train_filename.size() > 0);
	write_train_aux_to_file_flag = (train_control.write_aux_train_filename.size() > 0);

	store_logs_internally = train_control.store_logs_internally;
	

// Make sure all information can be stored in the way it is specified. Notice, that 
// storing internally and to the file system is simultaneously possible.
	
	if (write_train_aux_to_file_flag == true)
	{
		if (write_train_log_to_file_flag == false)
			flush_exit(ERROR_DATA_MISMATCH, "Missing log_train filename in Ttrain_control.");	
		
		svm_full_train_info.file_time = get_process_time_difference(svm_full_train_info.file_time);
		
		remove(train_control.write_log_train_filename.c_str());
		remove(train_control.write_sol_train_filename.c_str());
		
		fpauxwrite = open_file(train_control.write_aux_train_filename, "w");
		write_train_aux_to_file(fpauxwrite);
		close_file(fpauxwrite);
		
		svm_full_train_info.file_time = get_process_time_difference(svm_full_train_info.file_time);
	}

	if (store_logs_internally == true)
	{
		list_of_grids.resize(working_set_manager.number_of_tasks());
		list_of_fold_managers.resize(working_set_manager.number_of_tasks());
		
		for (task=0; task<working_set_manager.number_of_tasks(); task++)
		{
			list_of_grids[task].resize(working_set_manager.number_of_cells(task));
			list_of_fold_managers[task].resize(working_set_manager.number_of_cells(task));
		}
		
		Tsvm_manager::train_control.solver_control.save_solution = train_control.store_solutions_internally;
	}


// 	Call common train routine
	
	Tsvm_manager::train_control.grid_control.ignore_resize = use_current_grid;
	train_common(svm_full_train_info, false);
	
	
// 	Final duties
	
	working_set_manager.get_timings(full_run_info.train_partition_time, full_run_info.train_cell_assign_time);
	
	full_run_info.train_kernel_time = svm_full_train_info.train_val_info_log.full_kernel_time();
	full_run_info.train_solver_time = svm_full_train_info.train_val_info_log.init_time + svm_full_train_info.train_val_info_log.train_time;
	full_run_info.train_validation_time = svm_full_train_info.train_val_info_log.val_time;
	full_run_info.train_time = svm_full_train_info.train_time;
	
	svm_full_train_info.full_time = get_wall_time_difference(svm_full_train_info.full_time);
	full_run_info.train_full_time = svm_full_train_info.full_time;
	
	if (train_control.summary_log_filename.size() > 0)
	{
		fpauxwrite = open_file(train_control.summary_log_filename, "a");

		display_string = full_run_info.displaystring_post_train();
		fprintf(fpauxwrite, "%s", display_string.c_str());
		
		close_file(fpauxwrite);
	}
}



//**********************************************************************************************************************************

	
void Tsvm_manager::select(const Tselect_control& select_control, Tsvm_full_train_info& svm_full_train_info)
{
	string display_string;
	FILE* fpauxwrite;
	
	
// Elementary preparations
	
	svm_full_train_info.full_time = get_wall_time_difference();
	svm_full_train_info.file_time = 0.0;
	svm_full_train_info.train_time = 0.0;
	
	if (select_control.use_stored_logs == false)
		clear();
	else if (select_control.use_stored_solution == false)
	{
		clear_flags();
		decision_function_manager.clear();
	}
	
	Tsvm_manager::select_control = select_control;
	
	
// Prepare communication with filesystem and the internal storage

	clear_flags();
	
	if ((select_control.use_stored_solution == true) and (select_control.use_stored_logs == false))
		flush_exit(ERROR_DATA_MISMATCH, "Cannot use stored solution without using stored logs.");
	
	read_train_aux_from_file_flag = (select_control.read_aux_train_filename.size() > 0) and (not select_control.use_stored_logs);
	read_train_log_from_file_flag = (select_control.read_log_train_filename.size() > 0) and (not select_control.use_stored_logs);
	read_train_sol_from_file_flag = (select_control.read_sol_train_filename.size() > 0) and (not select_control.use_stored_logs);
	
	write_select_log_to_file_flag = (select_control.write_log_select_filename.size() > 0);
	write_select_sol_to_file_flag = (select_control.write_sol_select_filename.size() > 0);
	
	store_logs_internally = false;
	use_stored_logs = select_control.use_stored_logs;
	use_stored_solution = select_control.use_stored_solution;
	append_decision_functions = select_control.append_decision_functions;
	store_decision_functions_internally = select_control.store_decision_functions_internally;
	
	
// Make sure all required information is available.

	if (read_train_aux_from_file_flag == true)
	{
		if (read_train_log_from_file_flag == false)
			flush_exit(ERROR_DATA_MISMATCH, "Missing log_train filename in Tselect_control.");

		svm_full_train_info.file_time = get_process_time_difference(svm_full_train_info.file_time);
		
		fp_aux_train_read = open_file(select_control.read_aux_train_filename, "r");
		read_train_aux_from_file(fp_aux_train_read);
		
		fp_log_train_read = open_file(select_control.read_log_train_filename, "r");
		if (read_train_sol_from_file_flag == true)
			fp_sol_train_read = open_file(select_control.read_sol_train_filename, "r");
		
		svm_full_train_info.file_time = get_process_time_difference(svm_full_train_info.file_time);
		
		current_grids.resize(train_control.fold_control.number);
	}
	else 
		if (select_control.use_stored_logs == false)
			flush_exit(ERROR_DATA_MISMATCH, "Tselect_control requires either filename information or stored logs.");

	
// 	Call common train routine

	train_common(svm_full_train_info, true);
	

// 	Final duties

	svm_full_train_info.file_time = get_process_time_difference(svm_full_train_info.file_time);

	close_file(fp_log_train_read);
	close_file(fp_aux_train_read);
	close_file(fp_sol_train_read);
	
	full_run_info.select_kernel_time = svm_full_train_info.train_val_info_log.full_kernel_time();
	full_run_info.select_solver_time = svm_full_train_info.train_val_info_log.init_time + svm_full_train_info.train_val_info_log.train_time;
	full_run_info.select_time = svm_full_train_info.train_time;
	
	svm_full_train_info.file_time = get_process_time_difference(svm_full_train_info.file_time);
	svm_full_train_info.full_time = get_wall_time_difference(svm_full_train_info.full_time);
	
	full_run_info.select_full_time = svm_full_train_info.full_time;
	
	if (select_control.summary_log_filename.size() > 0)
	{
		fpauxwrite = open_file(select_control.summary_log_filename, "a");

		display_string = full_run_info.displaystring_post_select();
		fprintf(fpauxwrite, "%s", display_string.c_str());
		
		close_file(fpauxwrite);
	}
}




//**********************************************************************************************************************************

	
void Tsvm_manager::train_common(Tsvm_full_train_info& svm_full_train_info, bool select_mode)
{
	unsigned f;
	unsigned cell;
	unsigned task;
	Tdataset working_set;
	Tdataset_info working_set_info;
	Tcv_control cv_control;
	Tcv_manager<Tsvm_solution, Tsvm_train_val_info, Tsvm_solver_control, Tbasic_svm> cv_manager;
	vector <Tsvm_solution> solutions;
	Tsvm_decision_function new_decision_function;
	Tsvm_decision_function_manager decision_function_manager_current;
	Tsvm_decision_function_manager decision_function_manager_file;
	vector <Tsvm_train_val_info> select_val_info;
	Tparallel_control parallel_control;
	Tsvm_solver_control solver_control;


//---------------------------------------------------------------------------------------------------------------------

// Initialize local controls

	list_of_select_info.resize(working_set_manager.number_of_tasks());
	
	select_control.copy_to_cv_control(cv_control);
	cv_control.grid_control = train_control.grid_control;
	
	cv_control.full_search = train_control.full_search;
	cv_control.max_number_of_increases = train_control.max_number_of_increases;
	cv_control.max_number_of_worse_gammas = train_control.max_number_of_worse_gammas;
	
	solver_control = train_control.solver_control;
	if (cv_control.select_method == SELECT_ON_ENTIRE_TRAIN_SET)
		decision_function_manager_current = Tsvm_decision_function_manager(working_set_manager, data_set, 1);
	else
		decision_function_manager_current = Tsvm_decision_function_manager(working_set_manager, data_set, train_control.fold_control.number);

	
// Train for each task and cell

	for (task=0; task<working_set_manager.number_of_tasks(); task++)
	{
		list_of_select_info[task].resize(working_set_manager.number_of_cells(task));
		for (cell=0; cell<working_set_manager.number_of_cells(task); cell++)
		{
		//-------------- Create next working set and its folds, prepare strucures --------------------------
			
			flush_info(INFO_1, "\n\nConsidering cell %u out of %u for task %d out of %d.", cell+1, working_set_manager.number_of_cells(task), task+1, working_set_manager.number_of_tasks());
			
			full_run_info.train_cell_assign_time = get_wall_time_difference(full_run_info.train_cell_assign_time);
			working_set_manager.build_working_set(working_set, task, cell);
			working_set_info = Tdataset_info(working_set, true);
			full_run_info.train_cell_assign_time = get_wall_time_difference(full_run_info.train_cell_assign_time);

			if (working_set_manager.get_squared_radius_of_cell(task, cell) > 0.0)
			{
				flush_info(INFO_1, "\nCell contains %d samples and has a radius of %1.4f", working_set_info.size, sqrt(working_set_manager.get_squared_radius_of_cell(task, cell)));
				
				cv_control.grid_control.spatial_scale_factor = working_set_manager.get_squared_radius_of_cell(task, cell) *  pow(double(working_set_info.dim), -1.0);
			}
			else
				cv_control.grid_control.spatial_scale_factor = 1.0;
	
			cv_control.fold_manager = Tfold_manager(train_control.fold_control, working_set);
			cv_control.grid_control.scale_endpoints(train_control.fold_control, working_set.size(), working_set_manager.average_working_set_size(), working_set.dim());

			get_train_controls(cv_control, working_set, task, cell, svm_full_train_info.file_time);

			
		//----------------- Train on folds -------------------------------------------------------------------

			svm_full_train_info.train_time = get_wall_time_difference(svm_full_train_info.train_time);

			solver_control.set_clipping(working_set_info.max_abs_label);
			
			// Set some flags optimizing parallel and GPU execution
			
			parallel_control = train_control.parallel_control.set_to_single_threaded(working_set.size() < 500);
			parallel_control.keep_GPU_alive_after_disconnection = true;
			if ((select_mode == true) and (cv_control.use_stored_solution == true))
				parallel_control.GPUs = 0;
			

			cv_manager.reserve_threads(parallel_control);
			try
			{
				if (select_mode == false)
					cv_manager.train_all_folds(cv_control, solver_control, current_grids);
				else
				{
					cv_manager.select_all_folds(cv_control, solver_control, current_grids, solutions, select_val_info);
				
					if ((cv_manager.hit_smallest_gamma + cv_manager.hit_largest_gamma > 0) and (working_set_manager.total_number_of_working_sets() > 1))
						flush_info(INFO_2, "\n\nWarning: The best gamma was %d times at the lower boundary and %d times at the\n"
						"upper boundary of your gamma grid.", cv_manager.hit_smallest_gamma, cv_manager.hit_largest_gamma);
					
					if ((cv_manager.hit_smallest_weight + cv_manager.hit_largest_weight > 0) and (working_set_manager.total_number_of_working_sets() > 1))
						flush_info(INFO_2, "\n\nWarning: The best weight was %d times at the lower boundary and %d times at the\n"
						"upper boundary of your weight grid.", cv_manager.hit_smallest_weight, cv_manager.hit_largest_weight);

					if ((cv_manager.hit_smallest_lambda + cv_manager.hit_largest_lambda > 0) and (working_set_manager.total_number_of_working_sets() > 1))
						flush_info(INFO_2, "\n\nWarning: The best lambda was %d times at the lower boundary and %d times at the\n"
						"upper boundary of your lambda grid.", cv_manager.hit_smallest_lambda, cv_manager.hit_largest_lambda);
					
					full_run_info.hit_smallest_gamma = full_run_info.hit_smallest_gamma + cv_manager.hit_smallest_gamma;
					full_run_info.hit_largest_gamma = full_run_info.hit_largest_gamma + cv_manager.hit_largest_gamma;
					
					full_run_info.hit_smallest_weight = full_run_info.hit_smallest_weight + cv_manager.hit_smallest_weight;
					full_run_info.hit_largest_weight = full_run_info.hit_largest_weight + cv_manager.hit_largest_weight;
					
					full_run_info.hit_smallest_lambda = full_run_info.hit_smallest_lambda + cv_manager.hit_smallest_lambda;
					full_run_info.hit_largest_lambda = full_run_info.hit_largest_lambda + cv_manager.hit_largest_lambda;
				}
			}
			catch(...)
			{
				cv_manager.clear_threads();
				throw;
			}

			cv_manager.clear_threads();


			svm_full_train_info.train_time = get_wall_time_difference(svm_full_train_info.train_time);

			
		//----------------- Create decision_function_manager and store select info ----------------------------------------------------------

			if ((write_select_sol_to_file_flag == true) or (store_decision_functions_internally == true))
			{
				list_of_select_info[task][cell] = select_val_info;
				for(f=0;f<select_val_info.size();f++)
				{
					new_decision_function = Tsvm_decision_function(&(solutions[f]), solver_control.kernel_control_train, select_val_info[f], compose(working_set_manager.working_set_of_cell(task,cell), cv_manager.get_train_set_info(f)));

					new_decision_function.set_labels(working_set.get_original_labels());
					new_decision_function.set_error(select_val_info[f]); 
					
					decision_function_manager_current.replace_decision_function(task, cell, f, new_decision_function);
				}
			}
				
		//-------------- Perform some post-train file duties-------------------------------- --------------------------

			store_train_controls(cv_control, select_val_info, task, cell, svm_full_train_info.file_time);

			for (f=0;f<train_control.fold_control.number;f++)
				svm_full_train_info.train_val_info_log = svm_full_train_info.train_val_info_log + current_grids[f].summarize();
		}
	}
	
	if (full_run_info.hit_smallest_gamma + full_run_info.hit_largest_gamma > 0)
		flush_info(INFO_1, "\n\nWarning: The best gamma was %d times at the lower boundary and %d times at the\n"
		"upper boundary of your gamma grid. %d times a gamma value was selected.", full_run_info.hit_smallest_gamma, full_run_info.hit_largest_gamma, working_set_manager.total_number_of_working_sets() * cv_control.fold_manager.folds());
	
	if (full_run_info.hit_smallest_weight + full_run_info.hit_largest_weight > 0)
		flush_info(INFO_1, "\n\nWarning: The best weight was %d times at the lower boundary and %d times at the\n"
		"upper boundary of your weight grid. %d times a weight value was selected.", full_run_info.hit_smallest_weight, full_run_info.hit_largest_weight, working_set_manager.total_number_of_working_sets() * cv_control.fold_manager.folds());

	if (full_run_info.hit_smallest_lambda + full_run_info.hit_largest_lambda > 0)
		flush_info(INFO_1, "\n\nWarning: The best lambda was %d times at the lower boundary and %d times at the\n"
		"upper boundary of your lambda grid. %d times a lambda value was selected.", full_run_info.hit_smallest_lambda, full_run_info.hit_largest_lambda, working_set_manager.total_number_of_working_sets() * cv_control.fold_manager.folds());
	
	
// Store decision_function_manager

	if (write_select_sol_to_file_flag == true)
	{
		if ((file_exists(select_control.write_sol_select_filename) == true) and (append_decision_functions == true))
		{
			read_decision_function_manager_from_file(decision_function_manager_file, select_control.write_sol_select_filename, svm_full_train_info.file_time);

			decision_function_manager_file.push_back(decision_function_manager_current);
		}
		else
			decision_function_manager_file = decision_function_manager_current;

		write_decision_function_manager_to_file(decision_function_manager_file, select_control.write_sol_select_filename, svm_full_train_info.file_time);
	}

	if (store_decision_functions_internally == true)
	{
		if (append_decision_functions == false)
			decision_function_manager = decision_function_manager_current;
		else
			decision_function_manager.push_back(decision_function_manager_current);
	}

	cv_control.fold_manager.clear();
}	
	

	
//**********************************************************************************************************************************

	
void Tsvm_manager::test(const Tdataset& test_set, const Ttest_control& test_control, Tsvm_full_test_info& svm_full_test_info)
{
	unsigned i;
	unsigned j;
	unsigned t;
	unsigned task;
	unsigned test_chunk_size;
	unsigned test_chunk_size_current;
	unsigned number_of_test_chunks;
	Tdataset test_set_chunk;
	vector <Tsvm_train_val_info> train_val_info_chunk;
	vector <Tsvm_train_val_info> train_val_info_full;
	FILE* fpauxwrite;
	string display_string;
	Tsvm_test_info svm_test_info;
	
	
	svm_full_test_info.test_time = get_wall_time_difference();
	
	if (test_set.is_unsupervised_data() == false)
		if ((test_set.is_classification_data() == false) and (test_control.vote_control.scenario != VOTE_REGRESSION))
			flush_exit(ERROR_DATA_STRUCTURE, "Non-classification data requires vote_scenario = %d.", VOTE_REGRESSION);
	
	predictions.resize(test_set.size());
	
	if (test_control.read_sol_select_filename.size() > 0)
	{
		svm_full_test_info.test_time = get_wall_time_difference(svm_full_test_info.test_time);
		read_decision_function_manager_from_file(decision_function_manager, test_control.read_sol_select_filename, svm_full_test_info.file_time);
		svm_full_test_info.test_time = get_wall_time_difference(svm_full_test_info.test_time);
	}
	else 
		decision_function_manager.replace_kernel_control(train_control.solver_control.kernel_control_train);
	
	
	number_of_test_chunks = 1 + test_set.size() / decision_function_manager.get_max_test_set_size(test_control.max_used_RAM_in_MB);
	test_chunk_size = 1 + test_set.size() / number_of_test_chunks;

	for (i=0; i<number_of_test_chunks; i++)
	{
		test_set_chunk.clear();
		test_chunk_size_current = min(test_set.size(), (i+1) * test_chunk_size) - i * test_chunk_size;
		for (j = i * test_chunk_size; j < min(test_set.size(), (i+1) * test_chunk_size); j++)
			test_set_chunk.push_back(test_set.sample(j));
		
		if (scale_data == true)
			test_set_chunk.apply_scaling(scaling, translate);
	
		if (number_of_test_chunks > 1)
			flush_info(INFO_1, "\nComputing predictions for test chunk %d/%d of size %d.", i+1, number_of_test_chunks, test_set_chunk.size());
		decision_function_manager.make_predictions(test_set_chunk, test_control.vote_control, test_control.parallel_control, svm_test_info);
		
		if (test_set.is_unsupervised_data() == false)
			train_val_info_chunk = decision_function_manager.compute_errors(test_control.loss_control, test_control.vote_control.loss_weights_are_set);
		
		if (i == 0)
		{
			train_val_info_full.resize(train_val_info_chunk.size());
			for (t=0; t<train_val_info_chunk.size(); t++)
				train_val_info_full[t] = double(test_set_chunk.size()) * train_val_info_chunk[t];
		}
		else
			for (t=0; t<train_val_info_chunk.size(); t++)
				train_val_info_full[t] = train_val_info_full[t] + double(test_set_chunk.size()) * train_val_info_chunk[t];
		
		for (j=0; j<test_chunk_size_current; j++)
			predictions[i * test_chunk_size + j] = decision_function_manager.get_predictions_for_test_sample(j);
	}
	
	for (t=0; t<train_val_info_full.size(); t++)
		train_val_info_full[t] = (1.0 / double(test_set.size())) * train_val_info_full[t];
	svm_full_test_info.train_val_info = train_val_info_full;

	
	working_set_manager = decision_function_manager.get_working_set_manager();
	svm_full_test_info.number_of_tasks = working_set_manager.number_of_tasks();
	svm_full_test_info.number_of_all_tasks = decision_function_manager.number_of_all_tasks();
	
	if (number_of_test_chunks > 1)
		flush_info(INFO_1, "\n");
	
	
	
// Final duties
	
	full_run_info.test_set_size = test_set.size();
	full_run_info.test_cell_assign_time = svm_test_info.data_cell_assign_time;
	full_run_info.test_eval_time = svm_test_info.decision_function_time;
	full_run_info.test_kernel_time = svm_test_info.init_kernels_time + svm_test_info.pre_kernel_time + svm_test_info.kernel_time;
	full_run_info.test_misc_time = svm_test_info.misc_preparation_time + svm_test_info.predict_convert_time + svm_test_info.data_convert_time + svm_test_info.SVs_determine_time;

	full_run_info.test_time = svm_test_info.test_time;
	
	svm_full_test_info.test_time = get_wall_time_difference(svm_full_test_info.test_time);
	full_run_info.test_full_time = svm_full_test_info.test_time;
	svm_test_info.full_test_time = svm_full_test_info.test_time;
	
	for(task=0; task<svm_full_test_info.train_val_info.size(); task++)
	  full_run_info.test_errors.push_back(svm_full_test_info.train_val_info[task].val_error);

	
	svm_test_info.display(TRAIN_INFO_DISPLAY_FORMAT_REGULAR, INFO_2);

	
	if (test_control.summary_log_filename.size() > 0)
	{
		fpauxwrite = open_file(test_control.summary_log_filename, "a");

		display_string = full_run_info.displaystring_post_test();
		fprintf(fpauxwrite, "%s", display_string.c_str());
		
		close_file(fpauxwrite);
	}
}


//**********************************************************************************************************************************

vector <double> Tsvm_manager::get_predictions_for_task(unsigned task) const
{
	unsigned i;
	vector <double> tmp_predictions;
	
	if (task >= decision_function_manager.number_of_all_tasks())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to get predictions for task %d but only %d tasks have been considered.", task, decision_function_manager.number_of_all_tasks());
	
	if (predictions.size() == 0)
		flush_exit(ERROR_DATA_MISMATCH, "Trying to get predictions, but no predictions have been computed, yet.");
	
	tmp_predictions.resize(predictions.size());
	for (i=0; i<predictions.size(); i++)
		tmp_predictions[i] = predictions[i][task];
	
	return tmp_predictions;
}

//**********************************************************************************************************************************

vector <double> Tsvm_manager::get_predictions_for_test_sample(unsigned i) const
{
	if (i >= predictions.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to access prediction for sample %d, but there are only %d predictions\navailable.", i, unsigned(predictions.size()));	
	
	return predictions[i];
}


//**********************************************************************************************************************************

	
void Tsvm_manager::replace_kernel_control(const Tkernel_control& new_kernel_control)
{
	train_control.solver_control.kernel_control_val = new_kernel_control;
	train_control.solver_control.kernel_control_train = new_kernel_control;

	decision_function_manager.replace_kernel_control(new_kernel_control);
}


//**********************************************************************************************************************************

	
void Tsvm_manager::get_train_controls(Tcv_control& cv_control, const Tdataset working_set, unsigned task, unsigned cell, double& file_time)
{
	unsigned fold;
	

	if (read_train_aux_from_file_flag == true)
	{

		file_time = get_process_time_difference(file_time);
		current_grids.resize(train_control.fold_control.number);
		for(fold=0; fold<train_control.fold_control.number; fold++)
		{
			current_grids[fold].resize(cv_control.grid_control);
			current_grids[fold].read_from_file(fp_log_train_read, fp_sol_train_read);
		}
		cv_control.fold_manager.read_from_file(fp_aux_train_read, working_set);
		file_time = get_process_time_difference(file_time);
		
		cv_control.use_stored_solution = (fp_sol_train_read != NULL);
	}
	
	if (use_stored_logs == true)
	{
		current_grids = list_of_grids[task][cell];

		cv_control.use_stored_solution = use_stored_solution;
		cv_control.fold_manager.clear();
		cv_control.fold_manager = list_of_fold_managers[task][cell];
	}
}


//**********************************************************************************************************************************

	
void Tsvm_manager::store_train_controls(const Tcv_control& cv_control, const vector <Tsvm_train_val_info>& select_val_info, unsigned task, unsigned cell, double& file_time)
{
	unsigned f;
	
	FILE* fplogwrite;
	FILE* fpauxwrite;
	FILE* fpsolwrite;
	

	if (write_train_aux_to_file_flag == true)
	{
		file_time = get_process_time_difference(file_time);
		
		fpauxwrite = open_file(train_control.write_aux_train_filename, "a");
		fplogwrite = open_file(train_control.write_log_train_filename, "a");
		fpsolwrite = open_file(train_control.write_sol_train_filename, "a");
		
		cv_control.fold_manager.write_to_file(fpauxwrite);

		file_write(fpsolwrite, train_control.fold_control.number * current_grids[0].size());
		file_write_eol(fpsolwrite);

		for (f=0;f<train_control.fold_control.number;f++)
			current_grids[f].write_to_file(fplogwrite, fpsolwrite);
		
		close_file(fpauxwrite);
		close_file(fplogwrite);
		close_file(fpsolwrite);
		
		file_time = get_process_time_difference(file_time);
	}
	
	if (store_logs_internally == true)
	{
		list_of_grids[task][cell] = current_grids;
		list_of_fold_managers[task][cell] = cv_control.fold_manager;
	}

	if (write_select_log_to_file_flag == true)
	{
		file_time = get_process_time_difference(file_time);
		
		fplogwrite = open_file(select_control.write_log_select_filename, "a");
		for(f=0;f<select_val_info.size();f++)
			select_val_info[f].write_to_file(fplogwrite);
		close_file(fplogwrite);
		
		file_time = get_process_time_difference(file_time);
	}
}


//**********************************************************************************************************************************

	
void Tsvm_manager::write_train_aux_to_file(FILE* fpauxwrite)
{
	train_control.parallel_control.write_to_file(fpauxwrite);
	train_control.grid_control.write_to_file(fpauxwrite);
	train_control.fold_control.write_to_file(fpauxwrite);
	train_control.solver_control.write_to_file(fpauxwrite);
	
	file_write(fpauxwrite, scale_data);
	if (scale_data == true)
	{
		file_write(fpauxwrite, scaling);
		file_write(fpauxwrite, translate);
	}
	
	working_set_manager.write_to_file(fpauxwrite);
}

//**********************************************************************************************************************************

	
void Tsvm_manager::read_train_aux_from_file(FILE* fpauxread)
{
	train_control.parallel_control.read_from_file(fpauxread);
	train_control.grid_control.read_from_file(fpauxread);
	train_control.fold_control.read_from_file(fpauxread);
	train_control.solver_control.read_from_file(fpauxread);
	
	file_read(fpauxread, scale_data);
	if (scale_data == true)
	{
		file_read(fpauxread, scaling);
		file_read(fpauxread, translate);
		
		data_set.apply_scaling(scaling, translate);
	}
	
	working_set_manager.read_from_file(fpauxread, data_set);
}


#endif
