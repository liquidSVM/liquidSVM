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


#if !defined (BASIC_SVM_CPP)
	#define BASIC_SVM_CPP


#include "sources/svm/solver/basic_svm.h"


#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_types/vector.h"

#include "sources/shared/system_support/binding_specifics.h"




#ifdef  COMPILE_WITH_CUDA__
	#include "sources/shared/system_support/cuda_memory_operations.h"
	#include "sources/shared/system_support/cuda_simple_vector_operations.h"
#endif



//**********************************************************************************************************************************

Tbasic_svm::Tbasic_svm()
{
	training_kernel = NULL;
	validation_kernel = NULL;

	alpha_ALGD = NULL;
	index_ALGD = NULL;
	gradient_ALGD = NULL;
	weight_ALGD = NULL;
	
	training_label_ALGD = NULL;
	validation_label_ALGD = NULL;

	coefficient_delta = NULL;
	coefficient_changed = NULL;
	number_coefficients_changed = 0;
	
	prediction_ALGD = NULL;
	
	C_current  = 1.0;
	C_old = NO_PREVIOUS_LAMBDA;
	solver_clipp_value = 0.0;
	validation_clipp_value = 0.0;
	
	offset = 0.0;
}



//**********************************************************************************************************************************

Tbasic_svm::~Tbasic_svm()
{
	this->clear();
}



//**********************************************************************************************************************************

void Tbasic_svm::clear()
{
	my_dealloc_ALGD(&alpha_ALGD);
	my_dealloc_ALGD(&index_ALGD);
	my_dealloc_ALGD(&gradient_ALGD);
	my_dealloc_ALGD(&weight_ALGD);
	
	my_dealloc_ALGD(&training_label_ALGD);
	my_dealloc_ALGD(&validation_label_ALGD);
	
	my_dealloc(&coefficient_delta);
	my_dealloc(&coefficient_changed);
	
	my_dealloc_ALGD(&prediction_ALGD);
	
	Tsolver<Tsvm_solution, Tsvm_train_val_info, Tsvm_solver_control>::clear();
}


//**********************************************************************************************************************************

void Tbasic_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	#ifdef  COMPILE_WITH_CUDA__
		validation_kernel_GPU.resize(parallel_control.GPUs);
		kernel_control_GPU.resize(parallel_control.GPUs);
		
		coefficient_delta_GPU.assign(parallel_control.GPUs, NULL);
		coefficient_changed_GPU.assign(parallel_control.GPUs, NULL);
		prediction_GPU.assign(parallel_control.GPUs, NULL);
	#endif
		
	if ((solver_control.warm_start == SOLVER_INIT_SHRINK_UNIFORMLY) or (solver_control.warm_start == SOLVER_INIT_SHRINK))
		solver_control.init_direction = SOLVER_INIT_BACKWARD;
	else if ((solver_control.warm_start == SOLVER_INIT_ZERO) or (solver_control.warm_start == SOLVER_INIT_FULL))
		solver_control.init_direction = SOLVER_INIT_NO_DIRECTION;
	else
		solver_control.init_direction = SOLVER_INIT_FORWARD;

	Tsolver<Tsvm_solution, Tsvm_train_val_info, Tsvm_solver_control>::reserve(solver_control, parallel_control);
	
	
	primal_dual_gap.resize(2 * get_team_size());
	norm_etc_local.resize(2 * get_team_size());
	norm_etc_global.resize(2 * get_team_size());

	slack_sum_local.resize(2 * get_team_size());
	slack_sum_global.resize(2 * get_team_size());
}


//**********************************************************************************************************************************

void Tbasic_svm::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	unsigned i;
	unsigned used_size;
	vector <double> labels_tmp;
	#ifdef  COMPILE_WITH_CUDA__
		unsigned thread_id;
	#endif


	Tsolver<Tsvm_solution, Tsvm_train_val_info, Tsvm_solver_control>::load(training_kernel, validation_kernel);

	if (GPUs > 0)
	{
		#ifdef  COMPILE_WITH_CUDA__
			thread_id = get_thread_id();
			
			validation_kernel_GPU[thread_id] = validation_kernel->get_kernel_control_GPU().kernel_matrix;
			kernel_control_GPU[thread_id] = validation_kernel->get_kernel_control_GPU();

			my_realloc_GPU(&(coefficient_delta_GPU[thread_id]), training_set_size);
			my_realloc_GPU(&(coefficient_changed_GPU[thread_id]), training_set_size);
			my_realloc_GPU(&(prediction_GPU[thread_id]), kernel_control_GPU[thread_id].col_set_size);
		#endif
	}

	if (is_first_team_member() == true)
	{
		my_realloc_ALGD(&alpha_ALGD, training_set_size, used_size);
		my_realloc_ALGD(&gradient_ALGD, 2 * used_size);
		my_realloc_ALGD(&index_ALGD, training_set_size);
		for (i=0;i<training_set_size_aligned;i++)
			index_ALGD[i] = double(i);
		my_realloc_ALGD(&weight_ALGD, training_set_size);

		
		my_dealloc_ALGD(&training_label_ALGD);
		training_label_ALGD = training_kernel->get_row_labels_ALGD();
		
		if (training_set_size > 0)
		{
			labels_tmp.resize(training_set_size);
			for (i=0; i<training_set_size; i++)
				labels_tmp[i] = training_label_ALGD[i];

			min_label = labels_tmp[argmin(labels_tmp)];
			max_label = labels_tmp[argmax(labels_tmp)];
			if (max_label > min_label)
				label_spread = 0.5 * (max_label - min_label);
			else
				label_spread = 1.0;
		}
		else
			label_spread = 1.0;
		
		classification_data = true;
		for (i=0; i<training_set_size; i++)
			if (abs(training_label_ALGD[i]) != 1.0)
				classification_data = false;
			

		my_dealloc_ALGD(&validation_label_ALGD);
		validation_label_ALGD = validation_kernel->get_col_labels_ALGD();

		my_realloc(&coefficient_delta, training_set_size);
		my_realloc(&coefficient_changed, training_set_size);
		my_realloc_ALGD(&prediction_ALGD, validation_set_size);

		solution_current.reserve(training_set_size);
		solution_old.reserve(training_set_size);
		SV_list.reserve(training_set_size);

// 		primal_dual_gap.resize(2 * get_team_size());
// 		norm_etc_local.resize(2 * get_team_size());
// 		norm_etc_global.resize(2 * get_team_size());
// 
// 		slack_sum_local.resize(2 * get_team_size());
// 		slack_sum_global.resize(2 * get_team_size());
	}
}




//**********************************************************************************************************************************


void Tbasic_svm::clear_on_GPU()
{
	#ifdef  COMPILE_WITH_CUDA__
		if ((GPUs > 0) and (validation_set_size > 0))
		{
			my_dealloc_GPU(&(coefficient_delta_GPU[get_thread_id()]));
			my_dealloc_GPU(&(coefficient_changed_GPU[get_thread_id()]));
			
			my_dealloc_GPU(&(prediction_GPU[get_thread_id()]));
		}
	#endif
}



//**********************************************************************************************************************************

void Tbasic_svm::run_solver(Tsvm_train_val_info& train_val_info, Tsvm_solution& solution)
{
	if (solver_ctrl.fixed_loss == false)
		loss_function.set_weights(train_val_info.neg_weight, train_val_info.pos_weight);
	
	if (solver_ctrl.loss_control.clipp_value == -1.0)
		validation_clipp_value = max(abs(min_label), abs(max_label));
	else 
		validation_clipp_value = solver_ctrl.loss_control.clipp_value;	
	loss_function.set_clipp_value(validation_clipp_value);
	
	if (training_set_size > 0)
	{
		C_current = 1.0/(2.0 * train_val_info.lambda * double(training_set_size));
		stop_eps = solver_ctrl.stop_eps /(2.0 * train_val_info.lambda);
		
		if (is_first_team_member() == true)
			flush_info(INFO_DEBUG, "\nType %d, C = %f, stop_eps = %f, lambda = %f, Suggested solver clipping at %f, Suggested validation clipping at %f, Train size %d", solver_ctrl.solver_type, C_current, stop_eps, train_val_info.lambda, solver_ctrl.global_clipp_value, solver_ctrl.loss_control.clipp_value, training_set_size);

		if (C_old == NO_PREVIOUS_LAMBDA)
			this->initialize_solver(solver_ctrl.cold_start, train_val_info);
		else
			this->initialize_solver(solver_ctrl.warm_start, train_val_info);
	}

	get_time_difference(train_val_info.train_time, 0.0);
	if (training_set_size > 0)
		this->core_solver(train_val_info);
	get_time_difference(train_val_info.train_time, train_val_info.train_time);

	this->build_solution(train_val_info);
	solution_current.set_prediction_modifiers(offset, validation_clipp_value);
	solution_current.set_weights(train_val_info.neg_weight, train_val_info.pos_weight);
	if ((solver_ctrl.save_solution == true) and (is_first_team_member() == true))
		solution = solution_current;

	lazy_sync_threads_and_get_time_difference(train_val_info.val_time, train_val_info.val_time);
	this->get_train_error(train_val_info);
	get_val_error(train_val_info);
	lazy_sync_threads_and_get_time_difference(train_val_info.val_time, train_val_info.val_time);

	if (is_first_team_member() == true)
		train_val_info.display(TRAIN_INFO_DISPLAY_FORMAT_REGULAR, INFO_2);
	
	check_for_user_interrupt();

	C_old = C_current;
	sync_threads();

}


//**********************************************************************************************************************************

void Tbasic_svm::build_SV_list(Tsvm_train_val_info& train_val_info)
{
	unsigned i;

	if (is_first_team_member() == true)
	{
		SV_list.clear();
		for (i=0;i<training_set_size;i++)
			if (alpha_ALGD[i] != 0.0)
				SV_list.push_back(i);
		train_val_info.SVs = unsigned(SV_list.size());
	}
}

//**********************************************************************************************************************************

void Tbasic_svm::compute_val_predictions(unsigned& val_iterations)
{
	unsigned i;
	unsigned ii;
	unsigned iv;
	unsigned j;
	simdd__ delta_simdd;
	Tthread_chunk thread_chunk;
	double* restrict__ kernel_row_ALGD;


	sync_threads();
	if ((validation_set_size == 0) or (solution_current.size() == 0) or (prediction_ALGD == NULL))
		return;

	if (is_first_team_member() == true)
	{
		number_coefficients_changed = 0;
		if (solution_old.size() == 0)
		{
			for (j=0; j<validation_set_size; j++)
				prediction_ALGD[j] = 0.0;
			for (i=0; i<solution_current.size(); i++)
				push_back_update(solution_current.coefficient[i], solution_current.index[i]);
		}
		else
		{
			i = 0;
			j = 0;
			do
			{
				if (solution_current.index[i] == solution_old.index[j])
				{
					if (solution_current.coefficient[i] != solution_old.coefficient[j])
						push_back_update(solution_current.coefficient[i] - solution_old.coefficient[j], solution_current.index[i]);
					i++;
					j++;
				}
				else
				{
					if (solution_current.index[i] > solution_old.index[j])
						push_back_update(-solution_old.coefficient[j], solution_old.index[j], &j);
					else
						push_back_update(solution_current.coefficient[i], solution_current.index[i], &i);
				}
			}
			while ((i < solution_current.size()) and (j < solution_old.size()));

			if (i == solution_current.size())
				for (ii=j; ii<solution_old.size(); ii++)
					push_back_update(-solution_old.coefficient[ii], solution_old.index[ii]);
			else
				for (ii=i; ii<solution_current.size(); ii++)
					push_back_update(solution_current.coefficient[ii], solution_current.index[ii]);
		}
		val_iterations = number_coefficients_changed;
	}
	sync_threads();

	if (GPUs > 0)
		evaluate_val_predictions_on_GPU();
	else
	{
		thread_chunk = get_thread_chunk(validation_set_size, CACHELINE_STEP);
		for (i=0;i<number_coefficients_changed;i++)
		{
			iv = coefficient_changed[i];
			kernel_row_ALGD = validation_kernel->row(iv, thread_chunk.start_index, thread_chunk.stop_index);
			delta_simdd = assign_simdd(coefficient_delta[i]);

			for (j=thread_chunk.start_index; j+CACHELINE_STEP<=thread_chunk.stop_index_aligned; j+=CACHELINE_STEP)
			{
				cache_prefetch(kernel_row_ALGD+j+32, PREFETCH_L1);
				cache_prefetch(prediction_ALGD+j+32, PREFETCH_NO);
				fuse_mult_add3_CL(delta_simdd, kernel_row_ALGD+j, prediction_ALGD+j);
			}
		}
	}
	sync_threads();
}




//**********************************************************************************************************************************

void Tbasic_svm::get_val_error(Tsvm_train_val_info& train_val_info)
{
	unsigned i;

	compute_val_predictions(train_val_info.val_iterations);
	if (is_first_team_member() == true)
	{
		train_val_info.val_error = 0.0;
		solution_old = solution_current;

		for (i=0; i<validation_set_size; i++)
			train_val_info.val_error = train_val_info.val_error + loss_function.evaluate(validation_label_ALGD[i], prediction_ALGD[i] + offset);

		train_val_info.val_error = ( (validation_set_size > 0)? train_val_info.val_error / double(validation_set_size) : train_val_info.train_error);
	}
}


//**********************************************************************************************************************************

void Tbasic_svm::initialize_new_weight_and_lambda_line(Tsvm_train_val_info& train_val_info)
{
	unsigned j;
	Tthread_chunk thread_chunk;
	

	sync_threads_and_get_time_difference(train_val_info.val_time, train_val_info.val_time);
	
	if (is_first_team_member() == true)
	{
		if (training_kernel->all_kNN_assigned() == true)
			kNN_list = training_kernel->get_kNN_list();
		else
		{
			kNN_list.clear();
			kNN_list.resize(training_set_size);
		}

		solution_old.resize(0);
		solution_current.resize(0);
	}

	if (validation_set_size != 0)
	{
		if (GPUs > 0)
		{
			#ifdef  COMPILE_WITH_CUDA__
				init_vector_on_GPU(prediction_GPU[get_thread_id()], kernel_control_GPU[get_thread_id()].col_set_size);
			#endif
		}
		else
		{
			thread_chunk = get_thread_chunk(validation_set_size, CACHELINE_STEP);
			for (j=thread_chunk.start_index; j+CACHELINE_STEP <= thread_chunk.stop_index_aligned; j+=CACHELINE_STEP)
				assign_CL(prediction_ALGD+j, 0.0);
		}
	}
	sync_threads_and_get_time_difference(train_val_info.val_time, train_val_info.val_time);
}


//**********************************************************************************************************************************

void Tbasic_svm::initialize_new_lambda_line(Tsvm_train_val_info& train_val_info)
{
	C_old = NO_PREVIOUS_LAMBDA;
}


//**********************************************************************************************************************************


#ifndef COMPILE_WITH_CUDA__

void Tbasic_svm::evaluate_val_predictions_on_GPU(){}

#endif


#endif

