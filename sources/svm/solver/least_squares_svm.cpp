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


#if !defined (LEAST_SQUARES_SVM_CPP) 
	#define LEAST_SQUARES_SVM_CPP


#include "sources/svm/solver/least_squares_svm.h"


#include "sources/shared/system_support/timing.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_types/vector.h"




//**********************************************************************************************************************************

Tleast_squares_svm::Tleast_squares_svm()
{
}


//**********************************************************************************************************************************

Tleast_squares_svm::~Tleast_squares_svm()
{
}

//**********************************************************************************************************************************

void Tleast_squares_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	solver_control.kernel_control_train.include_labels = false;

	if (solver_control.cold_start == SOLVER_INIT_DEFAULT)
		solver_control.cold_start = SOLVER_INIT_ZERO;
	else if (solver_control.cold_start != SOLVER_INIT_ZERO)
		flush_exit(1, "\nLeast squares solver must not be cold started by method %d.\n" 
			"Allowed methods are %d.", solver_control.cold_start, SOLVER_INIT_ZERO);
		
	if (solver_control.warm_start == SOLVER_INIT_DEFAULT)
		solver_control.warm_start = SOLVER_INIT_RECYCLE;
	else if ((solver_control.warm_start != SOLVER_INIT_ZERO) and (solver_control.warm_start != SOLVER_INIT_RECYCLE))
		flush_exit(1, "\nLeast squares solver must not be warm started by method %d.\n" 
			"Allowed methods are %d and %d.", solver_control.warm_start, SOLVER_INIT_ZERO, SOLVER_INIT_RECYCLE);
		
	Tsvm_2D_solver_generic_base_name::reserve(solver_control, parallel_control);
	Tbasic_svm::reserve(solver_control, parallel_control);
	
	primal_dual_gap.resize(get_team_size());
	norm_etc_local.resize(get_team_size());
	norm_etc_global.resize(get_team_size());
	slack_sum_local.resize(get_team_size());
	slack_sum_global.resize(get_team_size());
	
	alpha_squared_sum_local.resize(get_team_size());
	alpha_squared_sum_global.resize(get_team_size());
}


//**********************************************************************************************************************************

void Tleast_squares_svm::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	unsigned i;

	Tbasic_svm::load(training_kernel, validation_kernel);
	if (is_first_team_member() == true)
	{
		if (training_set_size > 0)
			label_offset = mean(convert_to_vector(training_label_ALGD, training_set_size));
		else
			label_offset = 0.0;
		
		for (i=0; i<training_set_size; i++)
			training_label_ALGD[i] = transform_label(training_label_ALGD[i]);
	}
}

//**********************************************************************************************************************************

void Tleast_squares_svm::get_train_error(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	double prediction;

	train_val_info.train_error = 0.0;
	for (i=0; i<training_set_size; i++)
	{
		prediction = training_label_ALGD[i] - gradient_ALGD[i] - half_over_C * alpha_ALGD[i];
		train_val_info.train_error = train_val_info.train_error + loss_function.evaluate(inverse_transform_label(training_label_ALGD[i]), inverse_transform_label(prediction));
	}
	train_val_info.train_error = train_val_info.train_error / double(training_set_size);
}

//**********************************************************************************************************************************

void Tleast_squares_svm::initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);

	half_over_C = 0.5 / C_current;
	C_magic_factor_1 = 1.0 + half_over_C; 
	C_magic_factor_2 = 0.5 * C_magic_factor_1;
	C_magic_factor_3 = C_magic_factor_1 * C_magic_factor_1;
	C_magic_factor_4 = 4.0 * C_current / (1.0 + 4.0 * C_current);

	for (i=training_set_size;i<training_set_size_aligned;i++)
	{
		alpha_ALGD[i] = 0.0;
		gradient_ALGD[i] = 0.0;
		training_label_ALGD[i] = 0.0;
	}

	switch (init_method)
	{
		case SOLVER_INIT_ZERO:
			init_zero();
			break;
		case SOLVER_INIT_RECYCLE:
			init_keep();
			break;
		default:
			flush_exit(1, "Unknown solver initialization method %d for LS-SVM solver.", init_method);
			break;
	}
	train_val_info.init_iterations = 1;

	if (solver_ctrl.global_clipp_value == ADAPTIVE_CLIPPING)
	{
		if (classification_data == true) 
			solver_clipp_value = 1.0;
		else
			solver_clipp_value = 0.0;
	}
	else 
		solver_clipp_value = solver_ctrl.clipp_value;


	
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);
	
	if (is_first_team_member() == true)
		flush_info(INFO_DEBUG, "\nInit method %d. norm_etc = %f, slack_sum = %f, pd_gap = %f, Solver clipping at %f, Validation clipping at %f", init_method, norm_etc_global[0], slack_sum_global[0], primal_dual_gap[0], solver_clipp_value, validation_clipp_value);
}


//**********************************************************************************************************************************

void Tleast_squares_svm::init_zero()
{
	unsigned i;
	unsigned j;
	simdd__ slack_sum_simdd;
	unsigned thread_id;
	Tthread_chunk thread_chunk;

	
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	slack_sum_simdd = assign_simdd(0.0);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
		{
			store_simdd(alpha_ALGD+i+j, assign_simdd(0.0));
			store_simdd(gradient_ALGD+i+j, load_simdd(training_label_ALGD+i+j));
			slack_sum_simdd = fuse_mult_add_simdd(load_simdd(gradient_ALGD+i+j), load_simdd(gradient_ALGD+i+j), slack_sum_simdd);
		}
	}	
	slack_sum_local[thread_id] = reduce_sums_simdd(slack_sum_simdd);
	slack_sum_global[thread_id] = C_current * reduce_sums(&slack_sum_local[0]);
	
	norm_etc_global[thread_id] = 0.0;
	primal_dual_gap[thread_id] = slack_sum_global[thread_id];
};


//**********************************************************************************************************************************

void Tleast_squares_svm::init_keep()
{
	unsigned i;
	unsigned j;
	simdd__ magic_factor_simdd;
	simdd__ alpha_squared_sum_simdd;
	unsigned thread_id;
	Tthread_chunk thread_chunk;

// flush_info(INFO_DEBUG, "\nGetting Thread_id");
	thread_id = get_thread_id();
// flush_info(INFO_DEBUG, "\nThread_id is %d", thread_id);
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
// flush_info(INFO_DEBUG, "\nInit on %d %d %d %d", thread_id, thread_chunk.start_index, thread_chunk.stop_index_aligned, training_set_size_aligned);
	alpha_squared_sum_simdd = assign_simdd(0.0);
	magic_factor_simdd = assign_simdd(0.5 / C_old - half_over_C);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(alpha_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
		{
			alpha_squared_sum_simdd = fuse_mult_add_simdd(load_simdd(alpha_ALGD+i+j), load_simdd(alpha_ALGD+i+j), alpha_squared_sum_simdd);
			store_simdd(gradient_ALGD+i+j, fuse_mult_add_simdd(magic_factor_simdd, load_simdd(alpha_ALGD+i+j), load_simdd(gradient_ALGD+i+j)));
		}
	}

// flush_info(INFO_DEBUG, "\nalpha_squared_sum_local");
	alpha_squared_sum_local[thread_id] = reduce_sums_simdd(alpha_squared_sum_simdd);
// flush_info(INFO_DEBUG, "\nalpha_squared_sum_global");
	alpha_squared_sum_global[thread_id] = reduce_sums(&alpha_squared_sum_local[0]);

// flush_info(INFO_DEBUG, "\nslack_sum_local");
	slack_sum_local[thread_id] = compute_slack_sum(thread_chunk.start_index, thread_chunk.stop_index_aligned);
// flush_info(INFO_DEBUG, "\nslack_sum_global");
	slack_sum_global[thread_id] = reduce_sums(&slack_sum_local[0]);

// flush_info(INFO_DEBUG, "\nsnorm_etc_local");
	norm_etc_global[thread_id] = norm_etc_global[thread_id] + (0.25 / C_old - 0.5 * half_over_C) * alpha_squared_sum_global[thread_id];
// flush_info(INFO_DEBUG, "\nprimal_dual_gap");
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
};


//**********************************************************************************************************************************

void Tleast_squares_svm::build_solution(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	unsigned iv;
	unsigned size;

	sync_threads();
	if (is_first_team_member() == true)
	{
		this->build_SV_list(train_val_info);
		size = unsigned(SV_list.size());
		solution_current.resize(size);
		
		for (i=0; i<size; i++)
		{
			iv = SV_list[i];
			solution_current.coefficient[i] = label_spread * alpha_ALGD[iv];
			solution_current.index[i] = iv;
		}
		
		offset = label_offset;
	}
}




//**********************************************************************************************************************************

void Tleast_squares_svm::core_solver(Tsvm_train_val_info& train_val_info)
{
	core_solver_generic_part(train_val_info);
	if (is_first_team_member() == true)
	{	
		MM_CACHELINE_FLUSH(&slack_sum_local[0]);
	}
	
	sync_threads();
	slack_sum_global[get_thread_id()] = slack_sum_local[0];
}



#endif







