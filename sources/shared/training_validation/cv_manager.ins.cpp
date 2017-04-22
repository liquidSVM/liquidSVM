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







#include "sources/shared/basic_functions/random_subsets.h"
#include "sources/shared/basic_types/vector.h"
#include "sources/shared/training_validation/fold_manager.h"


#include <algorithm>

//**********************************************************************************************************************************


enum GRID_TRAIN_METHODS {FULL_GRID, TRAIN_LINE, TRAIN_SINGLE, GRID_TRAIN_METHODS_MAX}; 


//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::~Tcv_manager()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tcv_manager.");
}


//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::clear_threads()
{
	Tthread_manager_active::clear_threads();
	train_kernel.clear_threads();
	val_kernel.clear_threads();
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::train_all_folds(Tcv_control cv_control, const Tsolver_control_type& solver_ctrl, vector < Tgrid<Tsolution_type, Ttrain_val_info_type> >& grids)
{
	unsigned f;
	
	solver_control = solver_ctrl;
	
	Tcv_manager::cv_control = cv_control;
	Tcv_manager::grids.clear();
	Tcv_manager::grids.resize(cv_control.fold_manager.folds());
	for(f=0;f<cv_control.fold_manager.folds();f++)
		Tcv_manager::grids[f].resize(cv_control.grid_control);
	
	
	create_solver();
	solver->reserve(solver_control, get_parallel_control());
	
	grid_train_method = FULL_GRID;
	
	solver_control.kernel_control_train.same_data_sets = true;
	solver_control.kernel_control_train.max_col_set_size = cv_control.fold_manager.max_train_size();
	solver_control.kernel_control_train.max_row_set_size = cv_control.fold_manager.max_train_size();
	solver_control.kernel_control_train.kernel_store_on_GPU = false;
	solver_control.kernel_control_train.pre_kernel_store_on_GPU = true;
	solver_control.kernel_control_train.split_matrix_on_GPU_by_rows = true;
	solver_control.kernel_control_train.allowed_percentage_of_GPU_RAM = 0.95 * double(cv_control.fold_manager.max_train_size()) / double(cv_control.fold_manager.max_train_size() + cv_control.fold_manager.max_val_size());
	solver_control.kernel_control_train.read_hierarchical_kernel_info_from_file();
	train_kernel.reserve(get_parallel_control(), solver_control.kernel_control_train);

	
	solver_control.kernel_control_val.kNNs = 0;
	solver_control.kernel_control_val.same_data_sets = false;
	solver_control.kernel_control_val.max_col_set_size = cv_control.fold_manager.max_val_size();
	solver_control.kernel_control_val.max_row_set_size = cv_control.fold_manager.max_train_size();
	solver_control.kernel_control_val.kernel_store_on_GPU = true;
	solver_control.kernel_control_val.pre_kernel_store_on_GPU = true;
	solver_control.kernel_control_val.split_matrix_on_GPU_by_rows = false;
	solver_control.kernel_control_val.allowed_percentage_of_GPU_RAM = 0.95 * double(cv_control.fold_manager.max_val_size()) / double(cv_control.fold_manager.max_train_size() + cv_control.fold_manager.max_val_size());
	solver_control.kernel_control_val.read_hierarchical_kernel_info_from_file();
	val_kernel.reserve(get_parallel_control(), solver_control.kernel_control_val);


	start_threads();


	train_kernel.clear();
	val_kernel.clear();

	delete solver;
	
	grids = Tcv_manager::grids;
}




//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::select_all_folds(Tcv_control& cv_control, const Tsolver_control_type& solver_ctrl, vector < Tgrid<Tsolution_type, Ttrain_val_info_type> >& grids, vector <Tsolution_type>& solutions, vector <Ttrain_val_info_type>& select_val_info)
{
	unsigned f;
	vector <unsigned> weight_index_list;
		

	if ((cv_control.grid_control.weight_size > 1) and (cv_control.weight_number == 0) and (cv_control.npl == false))
		flush_exit(ERROR_DATA_MISMATCH, "A weight number needs to be specified since the logfile contains %d weights.",    cv_control.grid_control.weight_size);

	if (cv_control.weight_number > cv_control.grid_control.weight_size)
		flush_exit(ERROR_DATA_MISMATCH, "Weight number %d is larger than the number %d of weigths in the logfile.", cv_control.weight_number, cv_control.grid_control.weight_size);

	if (grids.size() != cv_control.fold_manager.folds())
		flush_exit(ERROR_DATA_STRUCTURE, "Number of grids %d does not match number of folds %d when calling\nTcv_manager::select_all_folds(...).", grids.size(), cv_control.fold_manager.folds());
	
	if (cv_control.weight_number > 0)
	{
		weight_index_list.assign(1, cv_control.weight_number-1);
		for(f=0; f<cv_control.fold_manager.folds(); f++)
			grids[f].reduce_weights(weight_index_list);
	}

// 	Take average grid if select method requires this
	
	if (cv_control.select_method == SELECT_ON_ENTIRE_TRAIN_SET)
	{
		for (f=1; f<cv_control.fold_manager.folds(); f++)
			grids[0] = grids[0] + grids[f];
		grids[0] = (1.0 / double(cv_control.fold_manager.folds())) * grids[0]; 
		grids.resize(1);

		cv_control.fold_manager.trivialize();
		grid_train_method = TRAIN_SINGLE;
	}
	else
		grid_train_method = TRAIN_LINE;

	
//	Set internal counters to zero

	hit_smallest_gamma = 0;
	hit_largest_gamma = 0;
	
	hit_smallest_weight = 0;
	hit_largest_weight = 0;

	hit_smallest_lambda = 0;
	hit_largest_lambda = 0;
	
	
// 	Copy control objects into Tcv_manager
	
	solver_control = solver_ctrl;
	solver_control.save_solution = true;

	Tcv_manager::grids = grids;
	Tcv_manager::cv_control = cv_control;
	Tcv_manager::solutions.resize(cv_control.fold_manager.folds());
	Tcv_manager::select_val_info.resize(cv_control.fold_manager.folds());

	create_solver();
	solver->reserve(solver_control, get_parallel_control());

	for (f=0; f<cv_control.fold_manager.folds(); f++)
		resize_grid_for_select(f);

	solver_control.kernel_control_train.same_data_sets = true;
	solver_control.kernel_control_train.max_col_set_size = cv_control.fold_manager.max_train_size();
	solver_control.kernel_control_train.max_row_set_size = cv_control.fold_manager.max_train_size();
	solver_control.kernel_control_train.kernel_store_on_GPU = false;
	solver_control.kernel_control_train.pre_kernel_store_on_GPU = true;
	solver_control.kernel_control_train.split_matrix_on_GPU_by_rows = true;
	solver_control.kernel_control_train.allowed_percentage_of_GPU_RAM = 0.95;
	solver_control.kernel_control_train.read_hierarchical_kernel_info_from_file();
	train_kernel.reserve(get_parallel_control(), solver_control.kernel_control_train);
	
	solver_control.kernel_control_val.kNNs = 0;
	solver_control.kernel_control_val.same_data_sets = false;
	solver_control.kernel_control_val.max_col_set_size = 0;
	solver_control.kernel_control_val.max_row_set_size = 0;
	solver_control.kernel_control_val.kernel_store_on_GPU = true;
	solver_control.kernel_control_val.pre_kernel_store_on_GPU = true;
	solver_control.kernel_control_val.split_matrix_on_GPU_by_rows = false;
	solver_control.kernel_control_val.allowed_percentage_of_GPU_RAM = 0.0;
	val_kernel.reserve(get_parallel_control(), solver_control.kernel_control_val);


	start_threads();
	
	train_kernel.clear();
	val_kernel.clear();
	delete solver;
	
	grids = Tcv_manager::grids;
	solutions = Tcv_manager::solutions;
	select_val_info = Tcv_manager::select_val_info;
}


//*********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
Tsubset_info Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::get_train_set_info(unsigned fold) const
{
	Tsubset_info train_set_info;
	
	
	if (permutations.size() < fold) 
		flush_exit(ERROR_DATA_STRUCTURE, "Cannot access ordering information for fold %d since only %d folds are stored in cv_manager", fold, permutations.size());
	
	train_set_info = cv_control.fold_manager.get_train_set_info(fold + 1);
	apply_permutation(train_set_info, permutations[fold]);
	
	return train_set_info;
}

//*********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::thread_entry()
{
	unsigned f;
	unsigned start_index_1;
	unsigned start_index_2;
	unsigned stop_index_1;	
	unsigned stop_index_2;

	
	if (is_first_team_member() == true)
	{
		best_ig_count.assign(grids[0].train_val_info.size(), 0);
		permutations.resize(cv_control.fold_manager.folds());
	}
	assumed_best_ig = grids[0].train_val_info.size()/2;
	assumed_ig_search_direction = 1;
	
	for(f=0;f<cv_control.fold_manager.folds();f++)
	{
		if (is_first_team_member() == true)
		{
			cv_control.fold_manager.build_train_and_val_set(f+1, training_set, validation_set);
			flush_info(INFO_3,"\n");
			if (grid_train_method != FULL_GRID)
				validation_set.clear();
		
			if (solver_control.order_data == SOLVER_ODER_DATA_SPATIALLY)
			{
				get_aligned_chunk(training_set.size(), 2*get_team_size(), 0, start_index_1, stop_index_1);
				get_aligned_chunk(training_set.size(), 2*get_team_size(), 1, start_index_2, stop_index_2);
				training_set.group_spatially(stop_index_2 - start_index_1, get_team_size(), permutations[f]);
			}
			else
				permutations[f] = id_permutation(training_set.size());
		}
		lazy_sync_threads();

		if (grid_train_method == FULL_GRID)
		{
			if (is_first_team_member() == true)
				flush_info(INFO_1,"\nFold %d: training set size %d,   validation set size %d.", f+1, training_set.size(), validation_set.size());
			train_on_grid(grids[f]);
		}
		else 
			select_on_grid(f);
	}
}




//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::train_on_grid(Tgrid<Tsolution_type, Ttrain_val_info_type>& grid)
{
	unsigned k;
	unsigned ig_split_position;
	unsigned ig;
	unsigned iig;
	unsigned iw;
	unsigned il;
	unsigned iil;
	unsigned number_of_lower_best_igs;
	unsigned number_of_higher_best_igs;
	vector <unsigned> igs;
	vector <unsigned> ils;
	unsigned max_number_of_weights;
	double dummy_time;
	bool continue_searching_gamma;
	bool search_this_weight;
	bool one_weight_searched;
	bool continue_searching_lambda;
	
	double previous_val_error;
	unsigned conseq_val_error_increases;
	double best_val_error_in_lambda_row;
	vector <double> best_val_errors_of_weight;
	vector <unsigned> best_val_errors_of_weight_position;

	
	val_kernel.load(training_set, validation_set, grid.train_val_info[0][0][0].val_pre_build_time, grid.train_val_info[0][0][0].val_build_transfer_time);
	train_kernel.load(training_set, training_set, grid.train_val_info[0][0][0].train_pre_build_time, grid.train_val_info[0][0][0].train_build_transfer_time);

	solver->load(&train_kernel, &val_kernel);

	
// 	Here the gamma values are arranged according to the assumed best start position and search direction
	
	k = 0;
	ig_split_position = 0;
	continue_searching_gamma = true;
	if (cv_control.full_search == true)
		igs = id_permutation(grid.train_val_info.size());
	else
	{
		igs.resize(grid.train_val_info.size());
		for (iig = assumed_best_ig; ((iig < grid.train_val_info.size()) and (iig >= 0)); iig=iig + assumed_ig_search_direction)
		{
			igs[k] = iig;
			k++;
		}
		ig_split_position = k;
		
		for (iig = assumed_best_ig - assumed_ig_search_direction; ((iig < grid.train_val_info.size()) and (iig >= 0)); iig=iig - assumed_ig_search_direction)
		{
			igs[k] = iig;
			k++;
		}
	}

	max_number_of_weights = 0;
	for (ig=0; ig<grid.train_val_info.size(); ig++)
		max_number_of_weights = max(max_number_of_weights, unsigned(grid.train_val_info[ig].size()));
	
	best_val_errors_of_weight_position.assign(max_number_of_weights, igs[0]);
	best_val_errors_of_weight.assign(max_number_of_weights, std::numeric_limits<double>::max());


// WARNING
// The order in which the gamma values are considered may have a small impact on SVM solvers:
// For runtime efficiency, the kNNs are computed for the first considered gamma value using
// the kernel matrix. Therefore, small numerical inaccuracies for the kernel matrix may lead
// to slightly different kNNs, which in turn may result in a different solver behavior.
	
	for (iig=0; ((iig<igs.size()) and (continue_searching_gamma == true)); iig++)
	{
		ig = igs[iig];

		train_kernel.assign(grid.train_val_info[ig][0][0].gamma, grid.train_val_info[ig][0][0].train_build_time, grid.train_val_info[ig][0][0].train_build_transfer_time, grid.train_val_info[ig][0][0].train_kNN_build_time);

		val_kernel.assign(grid.train_val_info[ig][0][0].gamma, grid.train_val_info[ig][0][0].val_build_time, grid.train_val_info[ig][0][0].val_build_transfer_time, dummy_time);
	
		search_this_weight = true;
		one_weight_searched = false;
		solver->initialize_new_weight_and_lambda_line(grid.train_val_info[ig][0][0]);
		
		for (iw=0; iw<grid.train_val_info[ig].size(); iw++)
		{
			search_this_weight = (abs(int(ig) - int(best_val_errors_of_weight_position[iw])) <= int(cv_control.max_number_of_worse_gammas)) or (cv_control.full_search);
			if (search_this_weight == true)
			{
				one_weight_searched = true;
				continue_searching_lambda = true;
				conseq_val_error_increases = 0;
				previous_val_error =  std::numeric_limits<double>::max();
				best_val_error_in_lambda_row =  std::numeric_limits<double>::max();
				
				solver->initialize_new_lambda_line(grid.train_val_info[ig][iw][0]);

				ils = id_permutation(grid.train_val_info[ig][iw].size());
				if (solver_control.init_direction == SOLVER_INIT_BACKWARD)
					std::reverse(ils.begin(), ils.end());
				
				for (iil=0; ((iil<ils.size()) and (continue_searching_lambda == true)); iil++)
				{
					il = ils[iil];

					solver->run_solver(grid.train_val_info[ig][iw][il], grid.solution[ig][iw][il]);
					train_kernel.get_cache_stats(grid.train_val_info[ig][iw][il].train_pre_cache_hits, grid.train_val_info[ig][iw][il].train_cache_hits);

					train_kernel.clear_cache_stats();
					
					if (grid.train_val_info[ig][iw][il].val_error > previous_val_error)
						conseq_val_error_increases++;
					else
						conseq_val_error_increases = 0;
					continue_searching_lambda = (conseq_val_error_increases < cv_control.max_number_of_increases) or (cv_control.full_search);
					
					previous_val_error = grid.train_val_info[ig][iw][il].val_error;
					best_val_error_in_lambda_row = min(best_val_error_in_lambda_row, grid.train_val_info[ig][iw][il].val_error);
				}
				if (is_first_team_member() == true)
					flush_info(INFO_2, "\nBest validation error in this lambda row is  %1.5f.\n", best_val_error_in_lambda_row);
				
				best_val_errors_of_weight[iw] = min(best_val_errors_of_weight[iw], best_val_error_in_lambda_row);
				if (best_val_errors_of_weight[iw] == best_val_error_in_lambda_row)
					best_val_errors_of_weight_position[iw] = ig;

				if (is_first_team_member() == true)
					flush_info(INFO_2, "\n");

				lazy_sync_threads();
			}
			else if (is_first_team_member() == true)
				flush_info(INFO_2, "\nSkipping search over lambdas for weight %d and gamma position %d.\n", iw, ig);
			
		}
		
// 	If search in one directions has been abondoned, jump to split position and proceed
		
		continue_searching_gamma = one_weight_searched or cv_control.full_search;
		if ((iig < ig_split_position) and (continue_searching_gamma == false))
		{
			iig = ig_split_position - 1;
			continue_searching_gamma = true;
			if (is_first_team_member() == true)
				flush_info(INFO_2, "\nJumping to gamma position %d.\n", igs[min(int(ig_split_position), int(igs.size() - 1))]);
		}
		 
		if ((is_first_team_member() == true) and (continue_searching_gamma == false))
			flush_info(INFO_2, "\nTerminating search over gammas.\n");
		
		if (is_first_team_member() == true)
			flush_info(INFO_2, "\n");
	}
	
	
// 	Determine best guesses for next fold
	
	if (is_first_team_member() == true)
	{
		for (iw=0; iw<best_val_errors_of_weight_position.size(); iw++)
			best_ig_count[best_val_errors_of_weight_position[iw]]++;
		
		assumed_best_ig = argmax(best_ig_count);
		
		number_of_lower_best_igs = 0;
		for (ig=0; ig < assumed_best_ig; ig++)
			number_of_lower_best_igs = number_of_lower_best_igs + best_ig_count[ig];
		
		number_of_higher_best_igs = 0;
		for (ig=assumed_best_ig+1; ig < best_ig_count.size(); ig++)
			number_of_higher_best_igs = number_of_higher_best_igs + best_ig_count[ig];
		
		if (number_of_lower_best_igs > number_of_higher_best_igs)
			assumed_ig_search_direction = -1;
		else
			assumed_ig_search_direction = 1;
	}
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::select_on_grid(unsigned fold)
{
	unsigned best_il;
	Ttrain_val_info_type train_val_info_tmp;


	// ------------- Get best lambda position and the corresponding train_val_info --------------------------------

	if (solver_control.init_direction == SOLVER_INIT_BACKWARD)
		best_il = 0;
	else
		best_il = grids[fold].train_val_info[0][0].size()-1;
	train_val_info_tmp = grids[fold].train_val_info[0][0][best_il];

	
	// ------------- Display best hyper-parameters and their error ---------------------------------------------------
	
	if (is_first_team_member() == true)
	{
		if (cv_control.npl == false)
			flush_info(INFO_1,"\nFold %d: best validation error %1.4f.", fold+1, train_val_info_tmp.val_error);
		else
		{
			if (cv_control.npl_class == 1)
			{
				if (train_val_info_tmp.pos_val_error <= cv_control.npl_constraint)
					flush_info(INFO_1,"\nFold %d: best DR %1.4f. Constraint %1.4f on class %d satisfied by FAR %1.4f.", fold+1, 1.0 - train_val_info_tmp.neg_val_error, cv_control.npl_constraint, cv_control.npl_class, train_val_info_tmp.pos_val_error);
				else 
					flush_info(INFO_1,"\nFold %d: best DR %1.4f. Adjusted constraint %1.4f on class %d satisfied by FAR %1.4f.", fold+1, 1.0 - train_val_info_tmp.neg_val_error, cv_control.npl_constraint, cv_control.npl_class, train_val_info_tmp.pos_val_error);
			}
			else
			{
				if (train_val_info_tmp.neg_val_error <= cv_control.npl_constraint)
					flush_info(INFO_1,"\nFold %d: best DR %1.4f. Constraint %1.4f on class %d satisfied by FAR %1.4f.", fold+1, 1.0 - train_val_info_tmp.pos_val_error, cv_control.npl_constraint, cv_control.npl_class, train_val_info_tmp.neg_val_error);
				else
					flush_info(INFO_1,"\nFold %d: best DR %1.4f. Adjusted constraint %1.4f on class %d satisfied by FAR %1.4f.", fold+1, 1.0 - train_val_info_tmp.pos_val_error, cv_control.npl_constraint, cv_control.npl_class, train_val_info_tmp.neg_val_error);
			}
		}
		
		train_val_info_tmp.display(TRAIN_INFO_DISPLAY_FORMAT_REGULAR, INFO_2);
		deactivate_display();
	}
	lazy_sync_threads();
	
	
	// ------------- Retrain and copy information to output structures --------------------------------------------------------------------

	if (cv_control.use_stored_solution == false)
		train_on_grid(grids[fold]);

	if (is_first_team_member() == true)
	{
		solutions[fold] = grids[fold].solution[0][0][best_il];
	
		select_val_info[fold] = grids[fold].train_val_info[0][0][best_il];
		select_val_info[fold].val_error = train_val_info_tmp.val_error;
		select_val_info[fold].pos_val_error = train_val_info_tmp.pos_val_error;
		select_val_info[fold].neg_val_error = train_val_info_tmp.neg_val_error;
	}
	
	// ------------- Display logs for best hyper-parameters --------------------------------------------------------------------
	
	if (is_first_team_member() == true)
	{
		reactivate_display();
		select_val_info[fold].display(TRAIN_INFO_DISPLAY_FORMAT_REGULAR, INFO_3);
		flush_info(INFO_2,"\n");
	}
	lazy_sync_threads();
}



//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type>
void Tcv_manager<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type, Tsolver_type>::resize_grid_for_select(unsigned fold)
{
	unsigned il;
	unsigned best_ig;
	unsigned best_iw;
	unsigned best_il;
	vector <unsigned> best_param_list;

	
// Get the best parameters
	
	if (cv_control.npl == false)
		grids[fold].get_entry_with_best_val_error(best_ig, best_iw, best_il);
	else
		grids[fold].get_entry_with_best_npl_error(cv_control.npl_class, cv_control.npl_constraint, best_ig, best_iw, best_il);
	
	
// 	Check if best parameters are on the boundary.
	
	if (grids[fold].train_val_info.size() > 0)
	{
		if (best_ig == 0)
			hit_largest_gamma++;
		if (best_ig + 1 == grids[fold].train_val_info.size())
			hit_smallest_gamma++;
	}
	if (grids[fold].train_val_info[best_ig].size() > 1)
	{
		if (best_iw == 0)
			hit_smallest_weight++;
		if (best_iw + 1 == grids[fold].train_val_info[best_ig].size())
			hit_largest_weight++;
	}
	if (grids[fold].train_val_info[best_ig][best_iw].size() > 1)
	{
		if (best_il == 0)
			hit_largest_lambda++;
		if (best_il + 1 == grids[fold].train_val_info[best_ig][best_iw].size())
			hit_smallest_lambda++;
	}
	
	
	best_param_list.push_back(best_ig);
	grids[fold].reduce_gammas(best_param_list);
	
	best_param_list[0] = best_iw;
	grids[fold].reduce_weights(best_param_list);
	
	best_param_list.clear();
	if ((solver_control.init_direction == SOLVER_INIT_NO_DIRECTION) or (grid_train_method == TRAIN_SINGLE))
		best_param_list.push_back(best_il);
	else
	{
		if (solver_control.init_direction == SOLVER_INIT_BACKWARD)
			for (il=best_il; il<grids[fold].train_val_info[0][0].size(); il++)
				best_param_list.push_back(il);
		else
			for (il=0; il<=best_il; il++)
				best_param_list.push_back(il);
	}

	grids[fold].reduce_lambdas(best_param_list);
}



