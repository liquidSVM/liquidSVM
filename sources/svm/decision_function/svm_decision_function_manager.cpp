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


#if !defined (SVM_DECISION_FUNCTION_MANAGER_CPP)
	#define SVM_DECISION_FUNCTION_MANAGER_CPP


#include "sources/svm/decision_function/svm_decision_function_manager.h"


#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/system_support/cuda_memory_operations.h"
#include "sources/shared/system_support/binding_specifics.h"

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/kernel/kernel_functions.h"
#include "sources/shared/training_validation/working_set_control.h"





#ifdef COMPILE_WITH_CUDA__
	#include "sources/svm/decision_function/decision_function_evaluation.h"
	#include <cuda_runtime.h>
#endif


#include <math.h>
#include <vector>



//*********************************************************************************************************************************



Tsvm_decision_function_manager::Tsvm_decision_function_manager()
{
	this->init_internal();
}


//*********************************************************************************************************************************



Tsvm_decision_function_manager::Tsvm_decision_function_manager(const Tworking_set_manager& working_set_manager, const Tdataset& training_set, unsigned folds)
{
	this->init_internal();
	construct(working_set_manager, training_set, folds);
}



//*********************************************************************************************************************************


Tsvm_decision_function_manager::Tsvm_decision_function_manager(const Tsvm_decision_function_manager& svm_decision_function_manager)
{
	this->init_internal();
	copy(svm_decision_function_manager);
}



//*********************************************************************************************************************************



Tsvm_decision_function_manager::~Tsvm_decision_function_manager()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tsvm_decision_function_manager of size %d ...", size());
	my_dealloc(&kernel_eval);
	my_dealloc(&pre_kernel_eval);
	flush_info(INFO_PEDANTIC_DEBUG, "\nTsvm_decision_function_manager destroyed.", size());
}


//**********************************************************************************************************************************


Tsvm_decision_function_manager& Tsvm_decision_function_manager::operator = (const Tsvm_decision_function_manager& svm_decision_function_manager)
{
	copy(svm_decision_function_manager);
	return *this;
}



//*********************************************************************************************************************************



void Tsvm_decision_function_manager::clear_internal()
{
	my_dealloc(&kernel_eval);
	my_dealloc(&pre_kernel_eval);
		
	gamma_list.clear();
	gamma_indices.clear();
	
	SVs.clear();
	SVs_with_gamma.clear();
	SVs_in_working_set.clear();
	SVs_with_gamma_in_working_set.clear();
	
	hierarchical_training_set.clear();
	hierarchical_test_set.clear();
}

//*********************************************************************************************************************************



void Tsvm_decision_function_manager::init_internal()
{
	kernel_eval = NULL;
	pre_kernel_eval = NULL;
	
	kernel_type = GAUSS_RBF;
	full_kernel_type = GAUSS_RBF;
}


//*********************************************************************************************************************************


void Tsvm_decision_function_manager::clear()
{
	Tsvm_decision_function_manager::clear_internal();
	Tdecision_function_manager<Tsvm_decision_function, Tsvm_train_val_info, Tsvm_test_info>::clear();
	
	
// 	HIERARCHICAL KERNEL DEVELOPMENT
	
	hierarchical_kernel_flag = false;
}



//*********************************************************************************************************************************


void Tsvm_decision_function_manager::copy(const Tsvm_decision_function_manager& svm_decision_function_manager)
{
	clear();
	
	Tdecision_function_manager<Tsvm_decision_function, Tsvm_train_val_info, Tsvm_test_info>::copy(svm_decision_function_manager);
	
	gamma_list = svm_decision_function_manager.gamma_list;
	gamma_indices = svm_decision_function_manager.gamma_indices;
	
	SVs = svm_decision_function_manager.SVs;
	SVs_with_gamma = svm_decision_function_manager.SVs_with_gamma;
	SVs_in_working_set = svm_decision_function_manager.SVs_in_working_set;
	SVs_with_gamma_in_working_set = svm_decision_function_manager.SVs_with_gamma_in_working_set;


	if (training_set.size() * gamma_list.size() * get_team_size() > 0)
	{
		my_realloc(&pre_kernel_eval, training_set.size() * get_team_size());
		my_realloc(&kernel_eval, training_set.size() * gamma_list.size() * get_team_size());
	}
	
	
// 	HIERARCHICAL KERNEL DEVELOPMENT

	kernel_control = svm_decision_function_manager.kernel_control;
	copy_internal_kernel_parameters_from_kernel_control();
}



//*********************************************************************************************************************************


void Tsvm_decision_function_manager::replace_kernel_control(const Tkernel_control& new_kernel_control)
{
	unsigned i;
	
	kernel_control = new_kernel_control;
	copy_internal_kernel_parameters_from_kernel_control();

	for (i=0; i<size(); i++)
	{
		decision_functions[i].kernel_type = new_kernel_control.kernel_type;
		decision_functions[i].hierarchical_kernel_control_read_filename = new_kernel_control.hierarchical_kernel_control_read_filename;
	}
}



//*********************************************************************************************************************************


void Tsvm_decision_function_manager::copy_internal_kernel_parameters_from_kernel_control()
{
	kernel_control.make_consistent();
	
	kernel_type = kernel_control.kernel_type;
	
	full_kernel_type = kernel_control.full_kernel_type;
	hierarchical_kernel_flag = kernel_control.is_hierarchical_kernel();
	weights_square_sum = kernel_control.get_hierarchical_weight_square_sum();
}


//*********************************************************************************************************************************


void Tsvm_decision_function_manager::read_hierarchical_kernel_info_from_df_file_if_possible(unsigned task, unsigned cell)
{
	if (size() > 0)
	{
		check_cell(task, cell); 
		if (decision_functions[decision_function_number(task, cell, 0)].hierarchical_kernel_control_read_filename.size() > 0)
		{
			kernel_control.hierarchical_kernel_control_read_filename = decision_functions[decision_function_number(task, cell, 0)].hierarchical_kernel_control_read_filename;
			kernel_control.read_hierarchical_kernel_info_from_file();
			copy_internal_kernel_parameters_from_kernel_control();
		}
	}
}




//*********************************************************************************************************************************


void Tsvm_decision_function_manager::setup_internal(const Tvote_control& vote_control, const Tparallel_control& parallel_control)
{
	unsigned is;
	unsigned task;
	unsigned cell;

	
	if (size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to setup an empty decision_function number.");
	
	
// Find the support vectors for each working set
		
	test_info.SVs_determine_time = get_wall_time_difference(test_info.SVs_determine_time);
	find_SVs(SVs, SVs_in_working_set);
	flush_info(INFO_2,"\nUsing %d samples out of %d samples as support vectors", SVs.size(), training_set.size());
	for (task=0; task<working_set_manager.number_of_tasks(); task++)
		for (cell=0; cell<working_set_manager.number_of_cells(task); cell++)
			flush_info(INFO_2, "\nUsing %d samples as support vectors in cell %d of task %d.", SVs_in_working_set[working_set_manager.working_set_number(task, cell)].size(), cell+1, task+1);
	test_info.SVs_determine_time = get_wall_time_difference(test_info.SVs_determine_time);
		

// 	HIERARCHICAL KERNEL DEVELOPMENT
	
	read_hierarchical_kernel_info_from_df_file_if_possible();

	
// Find all gammas and then find all support vectors for each gamma and working set
	
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);
	find_gammas();
	for (is=0; is<gamma_list.size(); is++)
		find_SVs(SVs_with_gamma[is], SVs_with_gamma_in_working_set[is], gamma_list[is]);
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);

	
// Finally, reserve some memory.

	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);
	
	kernel_init_times.assign(get_team_size(), 0.0);
	pre_kernel_times.assign(get_team_size(), 0.0);
	kernel_times.assign(get_team_size(), 0.0);
	
	pre_kernel_eval_counter_small.assign(get_team_size(), 0);
	pre_kernel_tries_counter_small.assign(get_team_size(), 0);
	
	pre_kernel_eval_counter_large.assign(get_team_size(), 0);
	pre_kernel_tries_counter_large.assign(get_team_size(), 0);
	
	kernel_eval_counter_small.assign(get_team_size(), 0);
	kernel_tries_counter_small.assign(get_team_size(), 0);
	
	kernel_eval_counter_large.assign(get_team_size(), 0);
	kernel_tries_counter_large.assign(get_team_size(), 0);
	
	my_realloc(&pre_kernel_eval, training_set.size() * get_team_size());
	my_realloc(&kernel_eval, training_set.size() * gamma_list.size() * get_team_size());
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);
}



//*********************************************************************************************************************************


void inline Tsvm_decision_function_manager::adjust_counters(unsigned& small_counter, unsigned& large_counter, unsigned unit)
{
	large_counter = large_counter + (small_counter / unit);
	small_counter = small_counter % unit;
}


//*********************************************************************************************************************************


void Tsvm_decision_function_manager::find_gammas()
{
	unsigned i;
	unsigned j;
	bool already_in_list;
	vector <bool> SV_flags; 

	
	gamma_list.clear();
	gamma_indices.resize(size());
	
	for (j=0;j<size();j++)
		if (weights[j] > 0.0)
		{
// 		This loop is not terribly efficient, but with typical numbers of decision functions, it should not matter.
// 		Compared to a set approach, however, the advantage, is that we have direct [i] access to the gamma_list

			already_in_list = false;
			for (i=0;(i<gamma_list.size() and (already_in_list == false));i++)
				already_in_list = (decision_functions[j].gamma == gamma_list[i]);

			if (already_in_list == false)
			{
				gamma_indices[j] = unsigned(gamma_list.size());
				gamma_list.push_back(decision_functions[j].gamma);
			}
			else
				gamma_indices[j] = i-1;
		}
	
	SVs_with_gamma.clear();
	SVs_with_gamma.resize(gamma_list.size());
	
	SVs_with_gamma_in_working_set.clear();
	SVs_with_gamma_in_working_set.resize(gamma_list.size());
}



//*********************************************************************************************************************************


void Tsvm_decision_function_manager::find_SVs(vector <unsigned>& SVs_list, vector <vector <unsigned> >& SVs_in_ws_list, double gamma)
{
	unsigned i;
	unsigned j;
	unsigned task;
	unsigned cell;
	unsigned fold;
	unsigned ws_number;
	vector <bool> SVs_flags; 
	vector <bool> SVs_in_ws_flags; 
	
	
	SVs_list.clear();
	SVs_flags.assign(training_set.size(), false);
	
	SVs_in_ws_list.clear();
	SVs_in_ws_flags.assign(training_set.size(), false);
	SVs_in_ws_list.resize(working_set_manager.total_number_of_working_sets());

	for (task=0; task<working_set_manager.number_of_tasks(); task++)
	{
		SVs_in_ws_flags.assign(training_set.size(), false);
		for (cell=0; cell<working_set_manager.number_of_cells(task); cell++)
		{
			ws_number = working_set_manager.working_set_number(task, cell);
			for (fold=0; fold<folds(); fold++)
			{
				j = fold + ws_number * folds();
				if ((weights[j] > 0.0) and ((gamma == -1.0) or (gamma == decision_functions[j].gamma)))
					for (i=0; i<decision_functions[j].size(); i++)
					{
						if (SVs_flags[decision_functions[j].sample_number[i]] == false)
						{
							SVs_flags[decision_functions[j].sample_number[i]] = true;
							SVs_list.push_back(decision_functions[j].sample_number[i]);
						}
						
						SVs_in_ws_list[ws_number].push_back(decision_functions[j].sample_number[i]);
					}
			}
		}
	}
}

//*********************************************************************************************************************************


void Tsvm_decision_function_manager::compute_pre_kernel_row(unsigned test_sample_number, unsigned ws_number, vector <bool>& pre_SVs_computed)
{
	unsigned i;
	unsigned thread_id;
	unsigned SV_number;
	unsigned pre_thread_position;
	
	thread_id = get_thread_id();
	get_time_difference(pre_kernel_times[thread_id], pre_kernel_times[thread_id], thread_id); 
// 	get_time_difference(test_info.pre_kernel_time, test_info.pre_kernel_time); 
	
	if (hierarchical_kernel_flag == true)
	{	
		pre_thread_position = get_pre_thread_position();
		for (i=0; i<SVs_in_working_set[ws_number].size(); i++)
		{
			SV_number = SVs_in_working_set[ws_number][i];
			if (pre_SVs_computed[SV_number] == false)
			{
				pre_kernel_eval[pre_thread_position + SV_number] = hierarchical_pre_kernel_function(weights_square_sum, kernel_control.hierarchical_weights_squared, hierarchical_test_set[test_sample_number], hierarchical_training_set[SV_number]);
				
				pre_SVs_computed[SV_number] = true;
				pre_kernel_eval_counter_small[get_thread_id()]++;
			}
		}
	}
	else
	{
		pre_thread_position = get_pre_thread_position();
		for (i=0; i<SVs_in_working_set[ws_number].size(); i++)
		{
			SV_number = SVs_in_working_set[ws_number][i];
			if (pre_SVs_computed[SV_number] == false)
			{
				pre_kernel_eval[pre_thread_position + SV_number] = pre_kernel_function(kernel_type, test_set.sample(test_sample_number), training_set.sample(SV_number));
				
				pre_SVs_computed[SV_number] = true;
				pre_kernel_eval_counter_small[get_thread_id()]++;
			}
		}
	}

	adjust_counters(pre_kernel_eval_counter_small[get_thread_id()], pre_kernel_eval_counter_large[get_thread_id()], test_info.unit_for_kernel_evaluations);

	pre_kernel_tries_counter_small[get_thread_id()] = pre_kernel_tries_counter_small[get_thread_id()] + SVs_in_working_set[ws_number].size();
	adjust_counters(pre_kernel_tries_counter_small[get_thread_id()], pre_kernel_tries_counter_large[get_thread_id()], test_info.unit_for_kernel_evaluations);
	
	get_time_difference(pre_kernel_times[thread_id], pre_kernel_times[thread_id], thread_id); 
// 	get_time_difference(test_info.pre_kernel_time, test_info.pre_kernel_time); 
}


//*********************************************************************************************************************************

void Tsvm_decision_function_manager::compute_kernel_row(unsigned test_sample_number, unsigned ws_number, vector <bool>& SVs_computed)
{
	unsigned is;
	unsigned j;
	unsigned thread_id;
	unsigned SV_number;
	unsigned SV_ig_position;
	unsigned thread_position;
	unsigned pre_thread_position;
	double gamma_factor;

	
	thread_id = get_thread_id();
	get_time_difference(kernel_times[thread_id], kernel_times[thread_id], thread_id); 
// 	get_time_difference(test_info.kernel_time, test_info.kernel_time); 

	thread_position = get_thread_position();
	pre_thread_position = get_pre_thread_position();

	for (is=0;is<gamma_list.size();is++)
	{
		gamma_factor = compute_gamma_factor(kernel_type, gamma_list[is]); 
		for (j=0;j<SVs_with_gamma_in_working_set[is][ws_number].size();j++)
		{
			SV_number = SVs_with_gamma_in_working_set[is][ws_number][j];
			SV_ig_position = is * training_set.size() + SV_number;
			if (SVs_computed[SV_ig_position] == false)
			{
				kernel_eval[thread_position + SV_ig_position] =  kernel_function(kernel_type, gamma_factor, pre_kernel_eval[pre_thread_position + SV_number]);
				
				SVs_computed[SV_ig_position] = true;
				kernel_eval_counter_small[get_thread_id()]++;
			}
		}
		
		kernel_tries_counter_small[get_thread_id()] = kernel_tries_counter_small[get_thread_id()] + SVs_with_gamma_in_working_set[is][ws_number].size();
		adjust_counters(kernel_tries_counter_small[get_thread_id()], kernel_tries_counter_large[get_thread_id()], test_info.unit_for_kernel_evaluations);
	}
	
	adjust_counters(kernel_eval_counter_small[get_thread_id()], kernel_eval_counter_large[get_thread_id()], test_info.unit_for_kernel_evaluations);
	
	thread_id = get_thread_id();
	get_time_difference(kernel_times[thread_id], kernel_times[thread_id], thread_id); 
// 	get_time_difference(test_info.kernel_time, test_info.kernel_time); 
}

//*********************************************************************************************************************************

void Tsvm_decision_function_manager::clear_kernel_row_flags(unsigned test_sample_number, vector <bool>& SVs_computed, vector <bool>& pre_SVs_computed)
{
	unsigned i;
	unsigned j;
	unsigned is;
	unsigned task;
	unsigned cell;
	unsigned SV_number;
	unsigned ws_number;
	unsigned SV_ig_position;
	unsigned thread_id;
	unsigned number_of_cells;
	
	
	thread_id = get_thread_id();
	get_time_difference(kernel_init_times[thread_id], kernel_init_times[thread_id], thread_id); 
	
	number_of_cells = 0;
	for (task=0; task<working_set_manager.number_of_tasks(); task++)
		number_of_cells = number_of_cells + cell_number_test[task][test_sample_number].size();
	

	if (training_set.size() < min(unsigned(1000000), 10 * number_of_cells * unsigned(working_set_manager.average_working_set_size() * gamma_list.size())))
	{
		pre_SVs_computed.assign(training_set.size(), false);
		SVs_computed.assign(gamma_list.size() * training_set.size(), false);
	}
	else
	{
		for (task=0; task<working_set_manager.number_of_tasks(); task++)
			for (cell=0; cell<cell_number_test[task][test_sample_number].size(); cell++)
			{
				ws_number = working_set_manager.working_set_number(task, cell_number_test[task][test_sample_number][cell]);
				
				for (i=0; i<SVs_in_working_set[ws_number].size(); i++)
				{
					SV_number = SVs_in_working_set[ws_number][i];
					pre_SVs_computed[SV_number] = false;
				}
				
				for (is=0;is<gamma_list.size();is++)
					for (j=0;j<SVs_with_gamma_in_working_set[is][ws_number].size();j++)
					{
						SV_number = SVs_with_gamma_in_working_set[is][ws_number][j];
						SV_ig_position = is * training_set.size() + SV_number;
						SVs_computed[SV_ig_position] = false;
					}
			}
	}
	
	get_time_difference(kernel_init_times[thread_id], kernel_init_times[thread_id], thread_id); 
}


//*********************************************************************************************************************************

unsigned Tsvm_decision_function_manager::get_thread_position()
{
	return unsigned(training_set.size() * gamma_list.size()) * get_thread_id();
}

//*********************************************************************************************************************************

unsigned Tsvm_decision_function_manager::get_pre_thread_position()
{
	return training_set.size() * get_thread_id();
}



//*********************************************************************************************************************************

void Tsvm_decision_function_manager::convert_to_hierarchical_data_sets()
{
	get_time_difference(test_info.data_convert_time, test_info.data_convert_time);
	if (is_first_team_member() == true)
	{
		kernel_control.convert_to_hierarchical_data_set(test_set, hierarchical_test_set);
		weights_square_sum = kernel_control.get_hierarchical_weight_square_sum();
	}
	sync_threads();
	if (is_last_team_member() == true)
		kernel_control.convert_to_hierarchical_data_set(training_set, hierarchical_training_set);
	sync_threads();
	get_time_difference(test_info.data_convert_time, test_info.data_convert_time);
}

//*********************************************************************************************************************************

void Tsvm_decision_function_manager::make_evaluations() 
{
	unsigned i;
	unsigned f;
	unsigned df;
	unsigned task;
	unsigned cell;
	unsigned ws_number;
	unsigned thread_id;
	unsigned thread_position;
	vector <bool> pre_SVs_computed;
	vector <bool> SVs_computed;
	Tthread_chunk thread_chunk;
	#ifdef  COMPILE_WITH_CUDA__
		Tsvm_decision_function_GPU_control<double> GPU_control;
	#endif
	Tdataset data_set_dummy;
	

	thread_id = get_thread_id();
	thread_position = get_thread_position();
	thread_chunk = get_thread_chunk(test_set.size());

	
// 	HIERARCHICAL KERNEL DEVELOPMENT

	if (hierarchical_kernel_flag == true)
		convert_to_hierarchical_data_sets();

	if (GPUs == 0)
	{
		flush_info(INFO_2, "\nThread %d is computing decision functions on the test data chunk of size %d.", thread_id, int(thread_chunk.stop_index) - int(thread_chunk.start_index)); 

		get_time_difference(kernel_init_times[thread_id], kernel_init_times[thread_id], thread_id); 
		pre_SVs_computed.assign(training_set.size(), false);
		SVs_computed.assign(gamma_list.size() * training_set.size(), false);
		get_time_difference(kernel_init_times[thread_id], kernel_init_times[thread_id], thread_id); 
		
		for(i=thread_chunk.start_index; i<thread_chunk.stop_index; i++)
		{
			clear_kernel_row_flags(i, SVs_computed, pre_SVs_computed);
			
			if (i % 1000 == 0)
				check_for_user_interrupt();
			
			for (task=0; task<working_set_manager.number_of_tasks(); task++)
			{
				for (cell=0; cell<cell_number_test[task][i].size(); cell++)
				{
					ws_number = working_set_manager.working_set_number(task, cell_number_test[task][i][cell]);
					
					compute_pre_kernel_row(i, ws_number, pre_SVs_computed);
					compute_kernel_row(i, ws_number, SVs_computed);

					get_time_difference(test_info.decision_function_time, test_info.decision_function_time); 

					for (f=0;f<folds();f++)
					{
						df = decision_function_number(task, cell_number_test[task][i][cell], f);
						evaluations[evaluation_position(i, df)] = decision_functions[df].evaluate(kernel_eval, training_set.size(), gamma_indices[df], thread_position);
					}
					
					get_time_difference(test_info.decision_function_time, test_info.decision_function_time); 
				}
			}
		}
	}
	else
	{
		#ifdef  COMPILE_WITH_CUDA__
			GPU_control.keep_test_data = false;
			setup_GPU(&GPU_control, data_set_dummy, true);
			compute_evaluations_on_GPU(evaluations, GPU_control, data_set_dummy, test_info);
			clean_GPU(&GPU_control);
		#endif
	}
	
	sync_threads();
	
	if (is_first_team_member() == true)
	{
		for(i=1; i<get_team_size(); i++)
		{
			pre_kernel_eval_counter_small[0] = pre_kernel_eval_counter_small[0] + pre_kernel_eval_counter_small[i]; 
			pre_kernel_eval_counter_large[0] = pre_kernel_eval_counter_large[0] + pre_kernel_eval_counter_large[i]; 

			kernel_eval_counter_small[0] = kernel_eval_counter_small[0] + kernel_eval_counter_small[i]; 
			kernel_eval_counter_large[0] = kernel_eval_counter_large[0] + kernel_eval_counter_large[i]; 
			
			
			pre_kernel_tries_counter_small[0] = pre_kernel_tries_counter_small[0] + pre_kernel_tries_counter_small[i]; 
			pre_kernel_tries_counter_large[0] = pre_kernel_tries_counter_large[0] + pre_kernel_tries_counter_large[i]; 

			kernel_tries_counter_small[0] = kernel_tries_counter_small[0] + kernel_tries_counter_small[i]; 
			kernel_tries_counter_large[0] = kernel_tries_counter_large[0] + kernel_tries_counter_large[i]; 
			
			kernel_init_times[0] = max(kernel_init_times[0], kernel_init_times[i]); 
			pre_kernel_times[0] = max(pre_kernel_times[0], pre_kernel_times[i]); 
			kernel_times[0] = max(kernel_times[0], kernel_times[i]); 
		}
		
		adjust_counters(pre_kernel_eval_counter_small[0], pre_kernel_eval_counter_large[0], test_info.unit_for_kernel_evaluations);
		adjust_counters(kernel_eval_counter_small[0], kernel_eval_counter_large[0], test_info.unit_for_kernel_evaluations);

		test_info.pre_kernel_evaluations = test_info.pre_kernel_evaluations + pre_kernel_eval_counter_large[0];
		test_info.kernel_evaluations = test_info.kernel_evaluations + kernel_eval_counter_large[0];
		
		adjust_counters(pre_kernel_tries_counter_small[0], pre_kernel_tries_counter_large[0], test_info.unit_for_kernel_evaluations);
		adjust_counters(kernel_tries_counter_small[0], kernel_tries_counter_large[0], test_info.unit_for_kernel_evaluations);

		test_info.pre_kernel_candidates = test_info.pre_kernel_candidates + pre_kernel_tries_counter_large[0];
		test_info.kernel_candidates = test_info.kernel_candidates + kernel_tries_counter_large[0];
		
		test_info.init_kernels_time = test_info.init_kernels_time + kernel_init_times[0];
		test_info.pre_kernel_time = test_info.pre_kernel_time + pre_kernel_times[0];
		test_info.kernel_time = test_info.kernel_time + kernel_times[0];
	}
}


//*********************************************************************************************************************************

Tsvm_decision_function Tsvm_decision_function_manager::get_decision_function(unsigned task, unsigned cell, unsigned fold)
{
	unsigned df;
	
	
	check_cell(task,cell);
	if (fold >= folds())
		flush_exit(ERROR_DATA_STRUCTURE, "Tried to access fold %d in a decision_function_manager that only has %d folds.", fold, folds());
	
	df = decision_function_number(task, cell, fold);
	
	return decision_functions[df];
}



//*********************************************************************************************************************************

unsigned Tsvm_decision_function_manager::size_of_largest_decision_function()
{
	unsigned df;
	unsigned largest_size;
	
	largest_size = 0;
	for (df=0; df<size(); df++)
		largest_size = max(largest_size, decision_functions[df].size());
	
	return largest_size;
}

//*********************************************************************************************************************************

unsigned Tsvm_decision_function_manager::size_of_largest_SV_with_gamma()
{
	unsigned i;
	unsigned largest_size;
	
	largest_size = 0;
	for (i=0; i<SVs_with_gamma.size(); i++)
		largest_size = max(largest_size, unsigned(SVs_with_gamma[i].size()));
	
	return largest_size;
}

#endif

