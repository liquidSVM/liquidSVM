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



#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_functions/memory_constants.h"
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/training_validation/working_set_control.h"


#include <math.h>

//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::Tdecision_function_manager()
{
	untouched = true;
	new_team_size = true;
	new_training_set = true;
	new_decision_functions = true;
	old_team_size = 0;
	
	init();
}


//*********************************************************************************************************************************



template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::Tdecision_function_manager(const Tworking_set_manager& working_set_manager, const Tdataset& training_set, unsigned folds)
{
	construct(working_set_manager, training_set, folds);
}



//**********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>& Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::operator = (const Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>& decision_function_manager)
{
	copy(decision_function_manager);
	return *this;
}

//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::copy(const Tdecision_function_manager& decision_function_manager)
{
	clear();
	
	weights = decision_function_manager.weights;
	evaluations = decision_function_manager.evaluations;
		
	cell_number_test = decision_function_manager.cell_number_test;
	cell_number_train = decision_function_manager.cell_number_train;
	decision_functions = decision_function_manager.decision_functions;
		
	test_set = decision_function_manager.test_set;
	test_set.enforce_ownership();
	training_set = decision_function_manager.training_set;
	training_set.enforce_ownership();
	
	test_set_info = decision_function_manager.test_set_info;
	training_set_info = decision_function_manager.training_set_info;
	working_set_manager = decision_function_manager.working_set_manager;
	
	untouched = decision_function_manager.untouched;
	new_team_size = decision_function_manager.new_team_size;
	new_training_set = decision_function_manager.new_training_set;
	new_decision_functions = decision_function_manager.new_decision_functions;
	old_team_size = decision_function_manager.old_team_size;
	
	all_tasks = decision_function_manager.all_tasks;
	number_of_folds = decision_function_manager.number_of_folds;
	
	cover_dataset = decision_function_manager.cover_dataset;
	cover_dataset.enforce_ownership();
	
	vote_control = decision_function_manager.vote_control;
	
	predictions = decision_function_manager.predictions;
	default_labels = decision_function_manager.default_labels;
	ties = decision_function_manager.ties;
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::construct(const Tworking_set_manager& working_set_manager, const Tdataset& training_set, unsigned folds)
{
	untouched = false;
	new_team_size = true;
	new_training_set = true;
	new_decision_functions = true;
	old_team_size = 0;
	
	init();
	
	Tdecision_function_manager::number_of_folds = folds;
	Tdecision_function_manager::working_set_manager = working_set_manager;
	
	decision_functions.resize(working_set_manager.total_number_of_working_sets() * folds);
	
	reserve(training_set, folds);
}

//*********************************************************************************************************************************



template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::~Tdecision_function_manager()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tdecision_function_manager of size %d ...", size());
	clear();
	flush_info(INFO_PEDANTIC_DEBUG, "\nTdecision_function_manager destroyed.", size());
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::write_to_file(FILE* fp) const
{
	unsigned j;
	
	if ((untouched == true) or (size() == 0))
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to write empty decision function manager to file.");
	
	file_write(fp, number_of_folds);
	working_set_manager.write_to_file(fp);
	
	for (j=0;j<size();j++)
		decision_functions[j].write_to_file(fp);
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::read_from_file(FILE* fp, const Tdataset& training_set)
{
	unsigned j;


	clear();
	untouched = false;
	
	file_read(fp, number_of_folds);
	working_set_manager.read_from_file(fp, training_set);
	
	decision_functions.resize(working_set_manager.total_number_of_working_sets() * number_of_folds);
	for (j=0;j<size();j++)
		decision_functions[j].read_from_file(fp);
	
	reserve(training_set, number_of_folds);
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::replace_decision_function(unsigned task, unsigned cell, unsigned fold, const Tdecision_function_type& new_decision_function)
{
	if (size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to replace a decision function in an empty decision function manager.");
	
	if (fold >= number_of_folds)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to replace a decision function of fold %d in decision_function_manager, which only has %d% folds.", fold, number_of_folds);

	new_decision_functions = true;
	decision_functions[decision_function_number(task, cell, fold)] = new_decision_function;
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline Tworking_set_manager Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::get_working_set_manager() const
{
	return working_set_manager;
}

 

//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::push_back(const Tdecision_function_manager& new_decision_function_manager)
{
	unsigned folds;
	unsigned new_number_of_working_sets;
	Tworking_set_manager new_working_set_manager;

	
	if (size() == 0)
		copy(new_decision_function_manager);
	else
	{
		untouched = false;
		new_team_size = true;
		new_training_set = true;
		new_decision_functions = true;
		old_team_size = 0;
		
		new_working_set_manager = new_decision_function_manager.working_set_manager;
		new_number_of_working_sets = new_working_set_manager.total_number_of_working_sets();
		folds = new_decision_function_manager.size() / new_number_of_working_sets;

		if (new_decision_function_manager.size() % new_number_of_working_sets != 0)
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to push %d decision functions for %d working sets to decision_function_manager.", new_decision_function_manager.size(), new_number_of_working_sets);

		if ((new_decision_function_manager.untouched == true) or (new_decision_function_manager.size() == 0))
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to push empty decision_function_manager into a decision_function_manager.");
		else if (folds != Tdecision_function_manager::number_of_folds)
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to push decision functions with %d folds to decision_function_manager having %d folds.", folds, Tdecision_function_manager::number_of_folds);

		all_tasks = all_tasks + new_working_set_manager.number_of_tasks();
		decision_functions.insert(decision_functions.end(), new_decision_function_manager.decision_functions.begin(), new_decision_function_manager.decision_functions.end());
		working_set_manager.push_back(new_working_set_manager);

		check_integrity();
	}
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::init()
{
	all_tasks = 0;
	number_of_folds = 0;

	test_set.clear();
	training_set.clear();
	default_labels.clear();
}

//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::clear()
{
	this->clear_internal();
	
	test_set.clear();
	training_set.clear();
	cover_dataset.clear();
	
	weights.clear();
	evaluations.clear();
	predictions.clear();
	
	cell_number_test.clear();
	cell_number_train.clear();
	decision_functions.clear();
	working_set_manager.clear();
	
	default_labels.clear();
	ties.clear();
	
	untouched = true;
	new_team_size = true;
	new_training_set = true;
	new_decision_functions = true;
	old_team_size = 0;
	
	init();
	this->init_internal();
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::reserve(const Tdataset& training_set, unsigned folds)
{
	training_set_info = Tdataset_info(training_set, true);
	Tdecision_function_manager::training_set = training_set;
	Tdecision_function_manager::number_of_folds = folds;

	working_set_manager.determine_cell_numbers_for_data_set(training_set, cell_number_train);
	
	new_team_size = true;
	new_training_set = true;
	new_decision_functions = true;
	old_team_size = 0;
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::setup(const Tvote_control& vote_control, const Tparallel_control& parallel_control)
{
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);
	
	GPUs = parallel_control.GPUs;
	reserve_threads(parallel_control);
	new_team_size = (old_team_size != get_team_size());

	all_tasks = working_set_manager.number_of_tasks();
	if (working_set_manager.get_working_set_control().working_set_selection_method > FULL_SET)
		all_tasks++;

	Tdecision_function_manager::vote_control = vote_control;
	
	if ((working_set_manager.get_working_set_control().working_set_selection_method == MULTI_CLASS_ONE_VS_ALL) and (vote_control.scenario != VOTE_REGRESSION))
	{
		Tdecision_function_manager::vote_control.scenario = VOTE_REGRESSION;
		flush_warn(WARN_ALL, "Vote method changed to regression since learning scenario is OvA.\n");
	}
	
	if (new_decision_functions == true)
		compute_weights();
	
	thread_start_times.assign(get_team_size(), 0.0);
	thread_stop_times.assign(get_team_size(), 0.0);
	
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);

	this->setup_internal(vote_control, parallel_control);
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::reduce_number_of_decision_functions(unsigned new_size)
{
	if ((working_set_manager.number_of_tasks() != 1) or (working_set_manager.number_of_cells(0) != 1))
		flush_exit(ERROR_DATA_STRUCTURE, "Cannot reduce number of decision functions for more than one cell or task.");
	
	if ((weights.size() == 0) or (decision_functions.size() == 0))
		flush_exit(ERROR_DATA_STRUCTURE, "Cannot reduce number of decision functions without weights or decision functions.");
		
	weights.resize(new_size);
	decision_functions.resize(new_size);
	number_of_folds = new_size;
}




//*********************************************************************************************************************************



template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::size() const
{
	return decision_functions.size();
}


//*********************************************************************************************************************************



template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::folds() const
{
	return number_of_folds;
}


//*********************************************************************************************************************************



template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::number_of_all_tasks() const
{
	return all_tasks;
}




//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_weights()
{
	unsigned f;
	unsigned df_nr;
	unsigned task;
	unsigned cell;
	double val_size;
	vector <double> errors;
	double max_error;
	double weight_sum;
	bool single_weight_flag;
	

	weights.resize(size());
	errors.resize(size());
	
	for (task=0; task<working_set_manager.number_of_tasks(); task++)
		for (cell=0; cell<working_set_manager.number_of_cells(task); cell++)
		{
			if (vote_control.weighted_folds == false)
				for(f=0;f<number_of_folds;f++)
					weights[decision_function_number(task, cell, f)] = 1.0 / double(number_of_folds);
			else
			{
// 				Determine the errors used for computing the weights
				
				for(f=0;f<number_of_folds;f++)
				{
					df_nr = decision_function_number(task, cell, f);
					if (vote_control.scenario == VOTE_NPL)
					{
						if (vote_control.npl_class == -1)
							errors[df_nr] = decision_functions[df_nr].pos_error;
						else
							errors[df_nr] = decision_functions[df_nr].neg_error;
					}
					else
						errors[df_nr] = decision_functions[df_nr].error;
				}
				
				
// 				Compute a first approximation of the weights using normalized errors
				
				weight_sum = 0.0;
				max_error = errors[argmax(errors, decision_function_number(task, cell, 0), number_of_folds)];
				if (max_error == 0.0)
					max_error = 1.0;
				val_size = min(double(working_set_manager.size_of_working_set_of_cell(task, cell))/double(number_of_folds), 100000.0) / 100.0;

				for(f=0;f<number_of_folds;f++)
				{
					df_nr = decision_function_number(task, cell, f);

					weights[df_nr] = exp( - val_size * errors[df_nr] / max_error);
					weight_sum = weight_sum + weights[df_nr];
				}
				for(f=0;f<number_of_folds;f++)
					weights[decision_function_number(task, cell, f)] = weights[decision_function_number(task, cell, f)] / weight_sum;

				
// 			This statement prunes the weights. The pruning is quite conservative 
					
				weight_sum = 0.0;
				for(f=0;f<number_of_folds;f++)
					if (weights[decision_function_number(task, cell, f)] < 0.01/double(number_of_folds))
						weights[decision_function_number(task, cell, f)] = 0.0;
					else
						weight_sum = weight_sum + weights[decision_function_number(task, cell, f)]; 
					
				for(f=0;f<number_of_folds;f++)
					weights[decision_function_number(task, cell, f)] = weights[decision_function_number(task, cell, f)] / weight_sum;
			
				
// 			In classification, a weight > 0.5 means absolute majority, so other decision functions are irrelevant
		
				if ((vote_control.scenario == VOTE_CLASSIFICATION) or (vote_control.scenario == VOTE_NPL))
				{
					single_weight_flag = false;
					for(f=0;f<number_of_folds;f++)
						if (weights[decision_function_number(task, cell, f)] > 0.5)
							single_weight_flag = true;

					if (single_weight_flag == true)
					{
						for(f=0;f<number_of_folds;f++)
							if (weights[decision_function_number(task, cell, f)] > 0.5)
								weights[decision_function_number(task, cell, f)] = 1.0;
							else
								weights[decision_function_number(task, cell, f)] = 0.0;
					}
				}
				
				for(f=0;f<number_of_folds;f++)
					if (weights[decision_function_number(task, cell, f)] == 0.0)
						 decision_functions[decision_function_number(task, cell, f)].set_to_zero();
			
				
// 			Report computed weights
				
				flush_info(INFO_2,"\n\nConsidering cell %u out of %u for task %d out of %d.", cell+1, working_set_manager.number_of_cells(task), task+1, working_set_manager.number_of_tasks());
				for(f=0;f<number_of_folds;f++)
					flush_info(INFO_2,"\nDecision function %d has validation error %1.4f which results in weight %1.4f.", f + 1, errors[decision_function_number(task, cell, f)], weights[decision_function_number(task, cell, f)]);
				flush_info(INFO_2,"\n");
			}
		}
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
vector <double> Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::get_predictions_for_task(unsigned task) const
{
	unsigned i;
	vector <double> tmp_predictions;
	
	
	if (task >= number_of_all_tasks())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to get predictions for task %d of decision function manager that only contains %d tasks.", task, number_of_all_tasks());
	
	tmp_predictions.resize(test_set.size());
	for (i=0; i<test_set.size(); i++)
		tmp_predictions[i] = predictions[prediction_position(i, task)];
	
	return tmp_predictions;
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
vector <double> Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::get_predictions_for_test_sample(unsigned i) const
{
	unsigned t;
	vector <double> tmp_predictions;
	
	
	if (i >= test_set.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to get predictions for sample %d of a dataset that only contains %d samples.", i, test_set.size());
	
	tmp_predictions.resize(number_of_all_tasks());
	for (t=0; t<number_of_all_tasks(); t++)
		tmp_predictions[t] = predictions[prediction_position(i, t)];
	
	return tmp_predictions;
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::prepare_for_making_predictions(const Tdataset& test_set, const Tvote_control& vote_control, const Tparallel_control& parallel_control)
{
	unsigned task;
	unsigned old_test_set_size;
	Tdataset subset;
	Tdataset_info dataset_info;

	
	// Initial checks
	
	if (decision_functions.size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to make predictions with an empty decision function manager.");
	
	
	// Prepare data structures for prediction 

	Tdecision_function_manager::setup(vote_control, parallel_control);
	
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);
	test_set_info = Tdataset_info(test_set, true);
	old_test_set_size = Tdecision_function_manager::test_set.size();
	Tdecision_function_manager::test_set = test_set;
	
	if (old_test_set_size != test_set.size())
	{
		evaluations.assign(size_type_double_vector__(test_set.size()) * size(), 0.0);
		predictions.assign(test_set.size() * all_tasks, 0.0);
	}
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);

	test_info.data_cell_assign_time = get_wall_time_difference(test_info.data_cell_assign_time);
	working_set_manager.determine_cell_numbers_for_data_set(test_set, cell_number_test);
	test_info.data_cell_assign_time = get_wall_time_difference(test_info.data_cell_assign_time);

	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);
	if (new_training_set == true)
	{
		default_labels.resize(working_set_manager.number_of_tasks());
		for (task=0; task<working_set_manager.number_of_tasks(); task++)
		{
			if (training_set.is_classification_data() == true)
			{
				training_set.create_subset(subset, working_set_manager.working_set_of_task(task));
				dataset_info = Tdataset_info(subset, true);

				default_labels[task] = dataset_info.label_list[dataset_info.most_frequent_label_number];
			}
			else
				default_labels[task] = 0.0;
		}
	}
	test_info.misc_preparation_time = get_wall_time_difference(test_info.misc_preparation_time);

	new_training_set = false;
	new_decision_functions = false;
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::get_max_test_set_size(unsigned allowed_RAM_in_MB) const
{
	return unsigned(double(MEGABYTE) * double(allowed_RAM_in_MB) / double(sizeof(double) * (size() + working_set_manager.number_of_tasks())));
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::make_predictions(const Tdataset& test_set, const Tvote_control& vote_control, const Tparallel_control& parallel_control, Ttest_info_type& test_info)
{
	unsigned t;
	double test_time;
	double misc_prep_time;
	double thread_stop_time;
	double thread_overhead_time;
	unsigned task;
	
	// Prepare data structures for prediction 
	
	test_time = get_wall_time_difference(0.0);
	Tdecision_function_manager::test_info = test_info;
	
	try
	{
		prepare_for_making_predictions(test_set, vote_control, parallel_control);
	
		misc_prep_time = get_wall_time_difference(0.0);
		ties.resize(number_of_all_tasks());
		for (task=0; task<number_of_all_tasks(); task++)
			ties[task].assign(test_set.size(), 0);
		misc_prep_time = get_wall_time_difference(misc_prep_time);

		// Start parallel execution for making the predictions and clean up the mess afterwards

		thread_start_time = get_wall_time_difference(0.0);
		start_threads();
	}
	catch(...)
	{
		clear_threads();
		throw;
	}
	
	thread_stop_time = get_wall_time_difference(0.0);
	for (t=0; t<get_team_size(); t++)
		thread_stop_times[t] = thread_stop_time - thread_stop_times[t];
	thread_overhead_time = thread_start_times[argmax(thread_start_times)] + thread_stop_times[argmax(thread_stop_times)];
	
	old_team_size = get_team_size();
	clear_threads();
	
	
	
	test_info = Tdecision_function_manager::test_info;
	test_time = get_wall_time_difference(test_time);
	
	test_info.test_time = test_info.test_time + test_time;
	test_info.thread_overhead_time = test_info.thread_overhead_time + thread_overhead_time;
	test_info.misc_preparation_time = test_info.misc_preparation_time + misc_prep_time;
	
// 	test_info.display(TRAIN_INFO_DISPLAY_FORMAT_REGULAR, INFO_1);
}



//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
double Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::vote(unsigned task, unsigned test_sample_number)
{
	unsigned f;
	unsigned df_nr;
	unsigned c;
	unsigned cell;
	double decision_on_cell;
	double decision;
	double tie_decision;
	double default_decision;
	double default_decision_on_cell;
	
	decision = 0.0;
	decision_on_cell = 0.0;
	
	default_decision = 0.0;
	default_decision_on_cell = 0.0;
	
	tie_decision = 0.0;
	if (vote_control.scenario == VOTE_REGRESSION)
	{
		for (c=0; c<cell_number_test[task][test_sample_number].size(); c++)
		{
			decision_on_cell = 0.0;
			cell = cell_number_test[task][test_sample_number][c];
			for (f=0;f<number_of_folds;f++)
			{
				df_nr = decision_function_number(task, cell, f);
				decision_on_cell = decision_on_cell + weights[df_nr] * evaluations[evaluation_position(test_sample_number, df_nr)];
			}
			decision = decision + decision_on_cell;
		}
		return decision / double(cell_number_test[task][test_sample_number].size());
	}
	else
	{
		for (c=0; c<cell_number_test[task][test_sample_number].size(); c++)
		{
// 			Compute weighted decision on the cell
			
			decision_on_cell = 0.0;
			cell = cell_number_test[task][test_sample_number][c];
			for (f=0;f<number_of_folds;f++)
			{
				df_nr = decision_function_number(task, cell, f);
				decision_on_cell = decision_on_cell + weights[df_nr] * sign(evaluations[evaluation_position(test_sample_number, df_nr)]);
			}
			

			
// 			Compute default decision on cell
			
			df_nr = decision_function_number(task, cell, 0);
			if (decision_functions[df_nr].default_label == decision_functions[df_nr].label2)
				default_decision_on_cell = 1.0;
			else
				default_decision_on_cell = -1.0;

			
// 			If weighted decision is tied, use default decision 
			
			if (decision_on_cell == 0.0)
				decision_on_cell = default_decision_on_cell;
			else 
				decision_on_cell = sign(decision_on_cell);
			
			if (c == 0)
				tie_decision = decision_on_cell;

			decision = decision + decision_on_cell;
			default_decision = default_decision + default_decision_on_cell;
		}
		
		
	// Deal with ties occuring over multiple cells. Currently,
	// the global default label is only overwritten
	// if there seems to be enough evidence for doing so.

		if (decision == 0.0)
		{
			if ((decision - decision_on_cell == default_decision - default_decision_on_cell) and (decision - decision_on_cell == tie_decision))
				decision = decision - decision_on_cell;
			ties[task][test_sample_number] = 1;
		}

		return convert_class_probability_to_class(task, decision);
	}
}



//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
double Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::convert_class_probability_to_class(unsigned task, double class_probability)
{
	double decision;
	vector <int> labels_of_task;
	

	labels_of_task = working_set_manager.get_labels_of_task(task);

	if (class_probability == 0.0)
		decision = default_labels[task];
	else if (class_probability < 0.0)
		decision = labels_of_task[0];
	else 
		decision = labels_of_task[1];

	return decision;
}


//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::thread_entry()
{
	unsigned task_offset;

	
	thread_start_times[get_thread_id()] = get_wall_time_difference(thread_start_time);

	this->make_evaluations();
	
	task_offset = get_task_offset();
	convert_evaluations_to_predictions();

	sync_threads();

	get_time_difference(test_info.final_predict_time, test_info.final_predict_time);
	switch (working_set_manager.get_working_set_control().working_set_selection_method)
	{
		case MULTI_CLASS_ALL_VS_ALL:
			make_final_predictions_most(task_offset);
			break;
		case MULTI_CLASS_ONE_VS_ALL:
			make_final_predictions_best(task_offset);
			break;
		case BOOT_STRAP:
			make_final_predictions_bootstrap(task_offset);
			break;
	}
	get_time_difference(test_info.final_predict_time, test_info.final_predict_time);
	
	thread_stop_times[get_thread_id()] = get_wall_time_difference(0.0);
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::get_task_offset() const
{
	if ((working_set_manager.get_working_set_control().working_set_selection_method == MULTI_CLASS_ALL_VS_ALL) or (working_set_manager.get_working_set_control().working_set_selection_method == MULTI_CLASS_ONE_VS_ALL) or 
		(working_set_manager.get_working_set_control().working_set_selection_method == BOOT_STRAP))
		return 1;
	else 
		return 0;
}

//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::convert_evaluations_to_predictions()
{
	unsigned i;
	unsigned t;
	unsigned task_offset;
	Tthread_chunk thread_chunk;
	
	sync_threads();
	get_time_difference(test_info.predict_convert_time, test_info.predict_convert_time);

	task_offset = get_task_offset();
	thread_chunk = get_thread_chunk(test_set.size());

	for(i=thread_chunk.start_index; i<thread_chunk.stop_index;i++)
		for(t=0;t<working_set_manager.number_of_tasks();t++)
			predictions[prediction_position(i, t + task_offset)] = vote(t, i);

	sync_threads();
	get_time_difference(test_info.predict_convert_time, test_info.predict_convert_time);
}


//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
double Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_error_for_task(unsigned task, Tloss_control loss_control, bool use_weights_from_training)
{
	unsigned i;
	unsigned task_tmp;
	double error;
	Tloss_function loss_function;
	
	
	if (use_weights_from_training == true)
	{
		if ((all_tasks > working_set_manager.number_of_tasks()) and (task > 0))
			task_tmp = task - 1;
		else
			task_tmp = task;
	
		loss_control.pos_weight = decision_functions[decision_function_number(task_tmp, 0, 0)].pos_weight;
		loss_control.neg_weight = decision_functions[decision_function_number(task_tmp, 0, 0)].neg_weight;
	}
	loss_function = Tloss_function(loss_control);

	error = 0.0;
	for (i=0; i<test_set.size(); i++)
		if (test_set.sample(i)->labeled == true)
			error = error + loss_function.evaluate(test_set.sample(i)->label, predictions[prediction_position(i, task)]);

	return error/double(test_set.size());
}


//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
double Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_AvA_error_for_task(unsigned task)
{
	unsigned i;
	unsigned df_nr;
	double label1;
	double label2;
	double current_prediction;
	double error;
	unsigned part_test_set_size;
	Tloss_control loss_control;
	Tloss_function loss_function;


	loss_control.type = MULTI_CLASS_LOSS;
	loss_function = Tloss_function(loss_control);
	
	df_nr = decision_function_number(task, 0, 0);
	label1 = decision_functions[df_nr].label1;
	label2 = decision_functions[df_nr].label2;
	
	error = 0.0;
	part_test_set_size = 0;
	for(i=0; i<test_set.size(); i++)
		if (test_set.sample(i)->labeled == true)
			if ((test_set.sample(i)->label == label1) or (test_set.sample(i)->label == label2))
			{
				part_test_set_size++;
				
				if (vote_control.scenario == VOTE_CLASSIFICATION)
					current_prediction = predictions[prediction_position(i, task + 1)];
				else
					current_prediction = convert_class_probability_to_class(task, predictions[prediction_position(i, task + 1)]);
				
				error = error + loss_function.evaluate(test_set.sample(i)->label, current_prediction);
			}

	if (part_test_set_size > 0)
		error = error/double(part_test_set_size);
		
	return error;
}


//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
double Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_OvA_error_for_task(unsigned task)
{
	unsigned i;
	unsigned error;

	
	error = 0;
	for(i=0; i<test_set.size(); i++)
		if (test_set.sample(i)->labeled == true)
		{
			if ((task == training_set_info.get_label_number(test_set.sample(i)->label)) and (predictions[prediction_position(i, task + 1)] <= 0.0))
				error++;
			
			if ((task != training_set_info.get_label_number(test_set.sample(i)->label)) and (predictions[prediction_position(i, task + 1)] > 0.0))
				error++;
		}
	return error/double(test_set.size());
}


//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type> Ttrain_val_info_type Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_two_class_error_for_task(Tloss_control loss_control, unsigned task)
{
	unsigned i;
	double tie_error;
	double number_of_ties;
	Tloss_function loss_function;
	Ttrain_val_info_type test_info;
	
	
	loss_control.type = CLASSIFICATION_LOSS;
	loss_function = Tloss_function(loss_control);
	
	tie_error = 0.0;
	test_info.pos_val_error = 0.0;
	test_info.neg_val_error = 0.0;

	for(i=0; i<test_set.size(); i++)
		if (test_set.sample(i)->labeled == true)
		{
			if (test_set.sample(i)->label == loss_control.yp)
				test_info.pos_val_error = test_info.pos_val_error + loss_function.evaluate(loss_control.yp, predictions[prediction_position(i, task)]);
			else
				test_info.neg_val_error = test_info.neg_val_error + loss_function.evaluate(loss_control.ym, predictions[prediction_position(i, task)]);

			if (ties[task][i] == 1)
			{
				if (test_set.sample(i)->label == loss_control.yp)
					tie_error = tie_error + loss_function.evaluate(loss_control.yp, predictions[prediction_position(i, task)]);
				else
					tie_error = tie_error + loss_function.evaluate(loss_control.ym, predictions[prediction_position(i, task)]);
			}
		}
		
	if (test_set_info.label_count[1] > 0)
		test_info.pos_val_error = test_info.pos_val_error / double(test_set_info.label_count[1]);
		
	if (test_set_info.label_count[0] > 0) 
		test_info.neg_val_error = test_info.neg_val_error / double(test_set_info.label_count[0]);
	
	number_of_ties = sum(ties[task]);
	if (number_of_ties > 0)
	{
		tie_error = tie_error / number_of_ties;
		flush_info(INFO_1, "There are %d ties (%2.2f%) for task %d. The error rate on the ties is %1.4f.", unsigned(number_of_ties), 100.0 * number_of_ties / double(test_set.size()), task, tie_error);
	}

	return test_info;
}

//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
Ttrain_val_info_type Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_NPL_error_for_task(Tloss_control loss_control, unsigned task, int npl_class)
{
	Ttrain_val_info_type test_info;
	
	
	test_info = compute_two_class_error_for_task(loss_control, task);
	
	if (npl_class == -1)
		test_info.val_error = test_info.pos_val_error;
	else
		test_info.val_error = test_info.neg_val_error;

	return test_info;
}



//*********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
vector <Ttrain_val_info_type> Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::compute_errors(Tloss_control loss_control, bool use_weights_from_training)
{
	bool labels_match;
	unsigned t;
	vector <Ttrain_val_info_type> train_val_info;


	if (predictions.size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to compute errors without having made predictions");
	
	train_val_info.resize(all_tasks);
	
	if ((working_set_manager.get_working_set_control().working_set_selection_method == MULTI_CLASS_ALL_VS_ALL) or (working_set_manager.get_working_set_control().working_set_selection_method == MULTI_CLASS_ONE_VS_ALL))
	{
		if (test_set_info.kind != CLASSIFICATION)
			flush_exit(ERROR_DATA_MISMATCH, "Trying to do multi-class classification for test data that is not of classification type.");
		
		loss_control.type = MULTI_CLASS_LOSS;
		train_val_info[0].val_error = compute_error_for_task(0, loss_control, use_weights_from_training);
		
		if (working_set_manager.get_working_set_control().working_set_selection_method == MULTI_CLASS_ALL_VS_ALL)
			for(t=0; t<working_set_manager.number_of_tasks(); t++)
				train_val_info[t + 1].val_error = compute_AvA_error_for_task(t);
		else
			for(t=0; t<working_set_manager.number_of_tasks(); t++)
				train_val_info[t + 1].val_error = compute_OvA_error_for_task(t);
	}
	else
	{
		if (vote_control.scenario == VOTE_NPL)
		{
			if (test_set_info.kind != CLASSIFICATION)
				flush_exit(ERROR_DATA_MISMATCH, "Trying to do NPL classification for test data that is not of classification type.");
			
			if ((test_set_info.label_list[0] != -1) or ((test_set_info.label_list.size() == 2) and (test_set_info.label_list[1] != 1))
				or (test_set_info.label_list.size() > 2))
				flush_exit(ERROR_DATA_MISMATCH, "Trying to do NPL classification for test data that does not have labels equal to +-1.");
			
			for(t=0; t<all_tasks; t++)
				train_val_info[t] = compute_NPL_error_for_task(loss_control, t, vote_control.npl_class);
		}
		else 
		{
			if (loss_control.type == CLASSIFICATION_LOSS)
			{
				if (test_set_info.kind != CLASSIFICATION)
					flush_exit(ERROR_DATA_MISMATCH, "Trying to do binary classification for test data that is not of classification type.");
				
				if (test_set_info.label_list.size() > 2)
					flush_exit(ERROR_DATA_MISMATCH, "Trying to do binary classification for test data that is of multi-class type.");
				
				labels_match = true;
				if (test_set_info.label_list.size() == 2)
					labels_match = (training_set_info.label_list[0] == test_set_info.label_list[0]) and (training_set_info.label_list[1] == test_set_info.label_list[1]);
				else if (test_set_info.label_list.size() == 1)
					labels_match = (training_set_info.label_list[0] == test_set_info.label_list[0]) or (training_set_info.label_list[1] == test_set_info.label_list[0]);
					
				if (labels_match == false)
					flush_exit(ERROR_DATA_MISMATCH, "Binary classification labels of train and test file do not match.");

				loss_control.ym = training_set_info.label_list[0];
				loss_control.yp = training_set_info.label_list[1];

				for(t=0; t<all_tasks; t++)
					train_val_info[t] = compute_two_class_error_for_task(loss_control, t);
			}
			
			for(t=0; t<all_tasks; t++)
				train_val_info[t].val_error = compute_error_for_task(t, loss_control, use_weights_from_training);
		}
	}
	
	return train_val_info;
}




//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::check_integrity()
{
	if (working_set_manager.total_number_of_working_sets() * number_of_folds != size())
		flush_exit(ERROR_DATA_STRUCTURE, "Tdecision_function_manager lost its integrity:\nIt has %d decision functions, but it should have %d.", size(), working_set_manager.total_number_of_working_sets() * number_of_folds);
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::decision_function_number(unsigned task, unsigned cell, unsigned fold) const
{
	return working_set_manager.working_set_number(task, cell) * number_of_folds + fold;
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline size_type_double_vector__ Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::evaluation_position(unsigned test_sample_number, unsigned decision_function_number) const
{
	return size_type_double_vector__(test_sample_number) * size() + decision_function_number;
}

//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
inline unsigned Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::prediction_position(unsigned test_sample_number, unsigned task) const
{
	return test_sample_number * all_tasks + task;
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::make_final_predictions_bootstrap(unsigned task_offset)
{
	if (vote_control.scenario == VOTE_REGRESSION)
		make_final_predictions_average(task_offset);
	else
		make_final_predictions_most(task_offset);
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::make_final_predictions_average(unsigned task_offset)
{
	unsigned i;
	Tthread_chunk thread_chunk;

	
	thread_chunk = get_thread_chunk(test_set.size());
	for(i=thread_chunk.start_index; i<thread_chunk.stop_index; i++)
		predictions[prediction_position(i, 0)] = mean(predictions, prediction_position(i, task_offset), working_set_manager.number_of_tasks());
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::make_final_predictions_most(unsigned task_offset)
{
	unsigned i;
	unsigned t;
	unsigned best_t;
	double current_prediction;
	vector <unsigned> predictions_count; 
	Tthread_chunk thread_chunk;


	thread_chunk = get_thread_chunk(test_set.size());
	for(i=thread_chunk.start_index; i<thread_chunk.stop_index; i++)
	{
		predictions_count.assign(training_set_info.label_count.size(), 0);
		for(t=0;t<working_set_manager.number_of_tasks();t++)
		{
			if (vote_control.scenario == VOTE_CLASSIFICATION)
				current_prediction = predictions[prediction_position(i, t + task_offset)];
			else
				current_prediction = convert_class_probability_to_class(t, predictions[prediction_position(i, t + task_offset)]);

			predictions_count[training_set_info.get_label_number(current_prediction)]++;
		}

		best_t = argmax(predictions_count);
		predictions[prediction_position(i, 0)] = training_set_info.label_list[best_t];
	}
}


//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::make_final_predictions_best(unsigned task_offset)
{
	unsigned i;
	unsigned best_t;
	Tthread_chunk thread_chunk;

	
	thread_chunk = get_thread_chunk(test_set.size());
	for(i=thread_chunk.start_index; i<thread_chunk.stop_index; i++)
	{
		best_t = argmax(predictions, prediction_position(i, task_offset), working_set_manager.number_of_tasks()) - (prediction_position(i, task_offset)); 
		predictions[prediction_position(i, 0)] = training_set_info.label_list[best_t];
	}
}



//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::check_task(unsigned task) const
{
	if (task >= working_set_manager.number_of_tasks())
		flush_exit(ERROR_DATA_STRUCTURE, "Tried to access task %d in a decision_function_manager that only has %d tasks.", task, working_set_manager.number_of_tasks());
}





//*********************************************************************************************************************************


template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type>
void Tdecision_function_manager<Tdecision_function_type, Ttrain_val_info_type, Ttest_info_type>::check_cell(unsigned task, unsigned cell) const
{
	check_task(task);
	
	if (cell >= working_set_manager.number_of_cells(task))
		flush_exit(ERROR_DATA_STRUCTURE, "Tried to access cell %d in task %d in a decision_function_manager that only has %d cells.", cell, task, working_set_manager.number_of_cells(task));
}
