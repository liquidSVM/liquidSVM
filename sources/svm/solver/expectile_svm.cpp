// Copyright 2015, 2016, 2017 Muhammad Farooq and Ingo Steinwart
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


#if !defined (EXPECTILE_SVM_CPP) 
	#define EXPECTILE_SVM_CPP



#include "sources/svm/solver/expectile_svm.h"


#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/simd_basics.h"
#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_types/vector.h"




//**********************************************************************************************************************************

Texpectile_svm::Texpectile_svm()
{
	beta_ALGD = NULL;
	gamma_ALGD = NULL;
	gradient_beta_ALGD = NULL;
	gradient_gamma_ALGD = NULL;
	
	training_label_transformed_ALGD = NULL;
	
	tau_initialized = false;
} 


//**********************************************************************************************************************************

Texpectile_svm::~Texpectile_svm()
{
 	my_dealloc_ALGD(&beta_ALGD);
 	my_dealloc_ALGD(&gamma_ALGD);
 	my_dealloc_ALGD(&gradient_beta_ALGD);
 	my_dealloc_ALGD(&gradient_gamma_ALGD);
	my_dealloc_ALGD(&training_label_transformed_ALGD);
}

//**********************************************************************************************************************************

void Texpectile_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	weight_display_mode = DISPLAY_WEIGHTS_NO_ERROR;
	
	solver_control.kernel_control_train.include_labels = false;

	if (solver_control.cold_start == SOLVER_INIT_DEFAULT)
		solver_control.cold_start = SOLVER_INIT_ZERO;
	else if (solver_control.cold_start != SOLVER_INIT_ZERO)
		flush_exit(1, "\nExpectile solver must not be cold started by method %d.\n" 
			"Allowed methods are %d.", solver_control.cold_start, SOLVER_INIT_ZERO);
		
	if (solver_control.warm_start == SOLVER_INIT_DEFAULT)
		solver_control.warm_start = SOLVER_INIT_RECYCLE;
	else if ((solver_control.warm_start != SOLVER_INIT_ZERO) and (solver_control.warm_start != SOLVER_INIT_RECYCLE))
		flush_exit(1, "\nExpectile solver must not be warm started by method %d.\n" 
			"Allowed methods are %d and %d.", solver_control.warm_start, SOLVER_INIT_ZERO, SOLVER_INIT_RECYCLE);
		
	Tbasic_2D_svm::reserve(solver_control, parallel_control);
	Tbasic_svm::reserve(solver_control, parallel_control);
	
	beta_gamma_squared_sum_local.resize(get_team_size());
	beta_gamma_squared_sum_global.resize(get_team_size());
}


//**********************************************************************************************************************************

void Texpectile_svm::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	Tbasic_svm::load(training_kernel, validation_kernel);
	if (is_first_team_member() == true)
	{
		my_realloc_ALGD(&beta_ALGD, training_set_size);
		my_realloc_ALGD(&gamma_ALGD, training_set_size);
		my_realloc_ALGD(&gradient_beta_ALGD, training_set_size);
		my_realloc_ALGD(&gradient_gamma_ALGD, training_set_size);
		my_realloc_ALGD(&training_label_transformed_ALGD, training_set_size);
	}
}


//**********************************************************************************************************************************

void Texpectile_svm::initialize_new_lambda_line(Tsvm_train_val_info& train_val_info)
{
	tau_initialized = false; 
	Tbasic_svm::initialize_new_lambda_line(train_val_info);
}

// *************************************************************************************************************************************

void Texpectile_svm::initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);

	if ((tau_initialized == false) and (is_first_team_member() == true))
	{
		tau = train_val_info.pos_weight / (train_val_info.neg_weight + train_val_info.pos_weight);
		tau_magic_factor = 1.0 / (tau * (1.0 - tau));
		
		if (training_set_size > 0)
			label_offset = expectile(convert_to_vector(training_label_ALGD, training_set_size), tau);
		else
			label_offset = 0.0;
		
		for (i=0; i<training_set_size; i++)
			training_label_transformed_ALGD[i] = transform_label(training_label_ALGD[i]);
		
		tau_initialized = true;
	}
	sync_threads();
	
	b1 = (2.0 * C_current * tau + 1.0)/(2.0 * C_current * tau);
	b2 = (2.0 * C_current * (1.0 - tau) + 1.0)/(2.0 * C_current * (1.0 - tau));
	reciprocal_b1 = 1.0 / b1;
	reciprocal_b2 = 1.0 / b2;
	
	half_over_C_tau_1 = 0.5 / (C_current * tau);
	half_over_C_tau_2 = 0.5 / (C_current * (1.0 - tau));
	
	C_tau_magic_factor_1 = 0.5 *(1.0 + half_over_C_tau_1);
	C_tau_magic_factor_2 = 0.5 *(1.0 + half_over_C_tau_2);
	C_tau_magic_factor_3 = 1.0 + 0.5 * half_over_C_tau_1;
	C_tau_magic_factor_4 = 1.0 + 0.5 * half_over_C_tau_2;

		
	for (i=training_set_size;i<training_set_size_aligned;i++)
	{
		beta_ALGD[i] = 0.0;
		gamma_ALGD[i]= 0.0;
		gradient_beta_ALGD[i] = 0.0;
		gradient_gamma_ALGD[i] = 0.0;
		training_label_transformed_ALGD[i] = 0.0;	
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

	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);

	if (solver_ctrl.global_clipp_value == ADAPTIVE_CLIPPING)
	{
		if (classification_data == true) 
			solver_clipp_value = 1.0;
		else
			solver_clipp_value = 0.0;
	}
	else 
		solver_clipp_value = solver_ctrl.clipp_value;


	if (is_first_team_member() == true)
		flush_info(INFO_DEBUG, "\nInit method %d. norm_etc = %f, slack_sum = %f, pd_gap = %f, Solver clipping at %f, Validation clipping at %f", init_method, norm_etc_global[0], slack_sum_global[0], primal_dual_gap[0], solver_clipp_value, validation_clipp_value);
}


//**********************************************************************************************************************************

void Texpectile_svm::init_zero()
{
	unsigned i;
	unsigned j;
	unsigned thread_id;
	Tthread_chunk thread_chunk;

	
	simdd__ slack_pos_simdd;
	simdd__ slack_neg_simdd;
	simdd__ slack_sum_simdd;
	simdd__ tau_1_simdd;
	simdd__ tau_2_simdd;
	

	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	slack_sum_simdd = assign_simdd(0.0);
	tau_1_simdd = assign_simdd(tau);
	tau_2_simdd = assign_simdd(1.0 - tau);
	
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
		for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
		{
			store_simdd(beta_ALGD+i+j, assign_simdd(0.0));
			store_simdd(gamma_ALGD+i+j, assign_simdd(0.0));

			store_simdd(gradient_beta_ALGD+i+j, load_simdd(training_label_transformed_ALGD+i+j));
			store_simdd(gradient_gamma_ALGD+i+j, mult_simdd(load_simdd(training_label_transformed_ALGD+i+j),assign_simdd(-1.0)));
			
			slack_pos_simdd = max_simdd(assign_simdd(0.0), load_simdd(training_label_transformed_ALGD+i+j));
			slack_neg_simdd = max_simdd(assign_simdd(0.0), mult_simdd(assign_simdd(-1.0), load_simdd(training_label_transformed_ALGD+i+j)));
			
			slack_sum_simdd = add_simdd(slack_sum_simdd, add_simdd(fuse_mult_mult_simdd(tau_1_simdd, slack_pos_simdd, slack_pos_simdd), fuse_mult_mult_simdd(tau_2_simdd, slack_neg_simdd, slack_neg_simdd))); 
		}
  
	slack_sum_local[thread_id] = reduce_sums_simdd(slack_sum_simdd);
	slack_sum_global[thread_id] = C_current * reduce_sums(&slack_sum_local[0]);
	
	norm_etc_global[thread_id] = 0.0;
	primal_dual_gap[thread_id] = slack_sum_global[thread_id];
};



//**********************************************************************************************************************************

void Texpectile_svm::init_keep()
{

	unsigned i;
	unsigned j;
	unsigned thread_id;
	Tthread_chunk thread_chunk;

	double half_over_tau_1;
	double half_over_tau_2;

	simdd__ slack_pos_simdd;
	simdd__ slack_neg_simdd;
	simdd__ slack_sum_simdd;
	simdd__ tau_1_simdd;
	simdd__ tau_2_simdd;
	simdd__ half_over_tau_1_simdd;
	simdd__ half_over_tau_2_simdd;
	simdd__ half_over_C_tau_1_simdd;
	simdd__ magic_factor_1_simdd;
	simdd__ magic_factor_2_simdd;
	simdd__ beta_gamma_squared_sum_simdd;
	simdd__ gradient_beta_simdd;
	
	
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	half_over_tau_1 = 0.5 / tau;
	half_over_tau_2 = 0.5 / (1.0 - tau);
	
	slack_sum_simdd = assign_simdd(0.0);
	beta_gamma_squared_sum_simdd = assign_simdd(0.0);
	  
	tau_1_simdd = assign_simdd(tau);
	tau_2_simdd = assign_simdd(1.0 - tau);
	  
	half_over_tau_1_simdd = assign_simdd(half_over_tau_1);
	half_over_tau_2_simdd = assign_simdd(half_over_tau_2);
	  
	half_over_C_tau_1_simdd = assign_simdd(half_over_C_tau_1);
	  
	magic_factor_1_simdd = assign_simdd(tau_magic_factor);
	magic_factor_2_simdd = assign_simdd(1.0 / C_old - 1.0 / C_current);
	   
	
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(beta_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gamma_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_beta_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_gamma_ALGD+i+32, PREFETCH_L1);
		for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
		{
			beta_gamma_squared_sum_simdd = add_simdd(beta_gamma_squared_sum_simdd, mult_simdd(magic_factor_1_simdd,add_simdd(mult_simdd(tau_2_simdd, mult_simdd(load_simdd(beta_ALGD+i+j), load_simdd(beta_ALGD+i+j))), mult_simdd(tau_1_simdd, mult_simdd(load_simdd(gamma_ALGD+i+j), load_simdd(gamma_ALGD+i+j))))));
			
			gradient_beta_simdd = add_simdd(load_simdd(gradient_beta_ALGD+i+j), mult_simdd(magic_factor_2_simdd, mult_simdd(load_simdd(beta_ALGD+i+j), half_over_tau_1_simdd)));
			store_simdd(gradient_beta_ALGD+i+j, gradient_beta_simdd);
			store_simdd(gradient_gamma_ALGD+i+j, add_simdd(load_simdd(gradient_gamma_ALGD+i+j), mult_simdd(magic_factor_2_simdd, mult_simdd(load_simdd(gamma_ALGD+i+j), half_over_tau_2_simdd))));
			  
			slack_pos_simdd = max_simdd(assign_simdd(0.0), fuse_mult_add_simdd(load_simdd(beta_ALGD+i+j), half_over_C_tau_1_simdd, gradient_beta_simdd));
			slack_neg_simdd = max_simdd(assign_simdd(0.0), mult_simdd(assign_simdd(-1.0), fuse_mult_add_simdd(load_simdd(beta_ALGD+i+j), half_over_C_tau_1_simdd, gradient_beta_simdd)));
			slack_sum_simdd = add_simdd(slack_sum_simdd, add_simdd(fuse_mult_mult_simdd(tau_1_simdd, slack_pos_simdd, slack_pos_simdd), fuse_mult_mult_simdd(tau_2_simdd, slack_neg_simdd, slack_neg_simdd)));  	
		}
	    
	}
	beta_gamma_squared_sum_local[thread_id] = reduce_sums_simdd(beta_gamma_squared_sum_simdd);
	beta_gamma_squared_sum_global[thread_id] = reduce_sums(&beta_gamma_squared_sum_local[0]);
	  
	slack_sum_local[thread_id] = reduce_sums_simdd(slack_sum_simdd);
	slack_sum_global[thread_id] = C_current * reduce_sums(&slack_sum_local[0]);
	  
	norm_etc_global[thread_id] = norm_etc_global[thread_id] -0.25 * (1.0 / C_old - 1.0 / C_current) * beta_gamma_squared_sum_global[thread_id];
	primal_dual_gap[thread_id] = norm_etc_global[thread_id] + slack_sum_global[thread_id];

};


//**********************************************************************************************************************************

void Texpectile_svm::build_solution(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	unsigned iv;
	unsigned size;

	if (is_first_team_member() == true)
	{
		this->build_SV_list(train_val_info);
		size = unsigned(SV_list.size());
		solution_current.resize(size);

		for (i=0; i<size; i++)
		{
			iv = SV_list[i];
			solution_current.coefficient[i] = label_spread * (beta_ALGD[iv] - gamma_ALGD[iv]);
			solution_current.index[i] = iv;
		}
		
		offset = label_offset;
	}
}


//**********************************************************************************************************************************

void Texpectile_svm::build_SV_list(Tsvm_train_val_info& train_val_info)
{
	unsigned i;

	if (is_first_team_member() == true)
	{	
		SV_list.clear();
		for (i=0;i<training_set_size;i++)
			if ((beta_ALGD[i] - gamma_ALGD[i]) != 0.0)
				SV_list.push_back(i);
		train_val_info.SVs = unsigned(SV_list.size());
	}
}


//**********************************************************************************************************************************

void Texpectile_svm::core_solver(Tsvm_train_val_info& train_val_info)
{
	
	unsigned start_index;
	unsigned stop_index;
	unsigned best_index_1;
	unsigned best_index_2;
	unsigned new_best_index_1;
	unsigned new_best_index_2;
	
	
	double new_beta_1;
	double new_beta_2;
	double new_gamma_1;
	double new_gamma_2;
		
	double best_gain;
	double best_gain_1;
	double best_gain_2;
	double new_best_gain_1;
	double new_best_gain_2;
	
	double delta_1;
	double delta_2;
	double eta_1;
	double eta_2;
	unsigned i;
	unsigned k;
	bool changed;
	
	unsigned thread_id;

	unsigned temp_index_1;
	unsigned temp_index_2;

	
	thread_id = get_thread_id();
	slack_sum_local[thread_id] = slack_sum_global[thread_id];
	
	if (is_first_team_member() == true)
	{
		train_val_info.train_iterations = 0;
		train_val_info.gradient_updates = 0;
		
		best_index_1 = 0;
		best_index_2 = 0;
		new_best_index_1 = 0;
		new_best_index_2 = 0;
		
		new_beta_1 = 0.0;
		new_beta_2 = 0.0;
		new_gamma_1 = 0.0;
		new_gamma_2 = 0.0;

		if (training_set_size <= CACHELINE_STEP)
		{
			for (i=0; i<training_set_size; i++)
			{
				beta_ALGD[i] = 0.0;
				gamma_ALGD[i] = 0.0;
			}
			primal_dual_gap[thread_id] = 0.0;
			flush_info(INFO_DEBUG, "Switching to average vote since the training set size is not greater than %d.", CACHELINE_STEP);
		}		
		
		if (primal_dual_gap[thread_id] > stop_eps)
		{
			get_aligned_chunk(training_set_size, 2, 0, start_index, stop_index);
			get_optimal_1D_direction(start_index, stop_index, best_index_1, best_gain_1);
			get_aligned_chunk(training_set_size, 2, 1, start_index, stop_index);
			get_optimal_1D_direction(start_index, stop_index, best_index_2, best_gain_2);

			order_indices(best_index_1, best_gain_1, best_index_2, best_gain_2);

			optimize_2D(best_index_1, best_index_2, new_beta_1, new_beta_2, new_gamma_1, new_gamma_2);
			kernel_row1_ALGD = training_kernel->row(best_index_1);
			kernel_row2_ALGD = training_kernel->row(best_index_2);
		}
		
		while (primal_dual_gap[thread_id] > stop_eps)
		{
			delta_1 = new_beta_1 - beta_ALGD[best_index_1];
			delta_2 = new_beta_2 - beta_ALGD[best_index_2];
			eta_1   = new_gamma_1 - gamma_ALGD[best_index_1];
			eta_2   = new_gamma_2 - gamma_ALGD[best_index_2];
	
		  
			beta_ALGD[best_index_1] = beta_ALGD[best_index_1] + delta_1;
			beta_ALGD[best_index_2] = beta_ALGD[best_index_2] + delta_2;
			gamma_ALGD[best_index_1] = gamma_ALGD[best_index_1] + eta_1;
			gamma_ALGD[best_index_2] = gamma_ALGD[best_index_2] + eta_2;
		  
			norm_etc_global[thread_id] = norm_etc_global[thread_id] - (delta_1 * (2.0 * gradient_beta_ALGD[best_index_1] - training_label_transformed_ALGD[best_index_1] + ((beta_ALGD[best_index_1] - delta_1) * half_over_C_tau_1) - (C_tau_magic_factor_3 * delta_1)));
			norm_etc_global[thread_id] = norm_etc_global[thread_id] - (delta_2 * (2.0 * gradient_beta_ALGD[best_index_2] - training_label_transformed_ALGD[best_index_2] + ((beta_ALGD[best_index_2] - delta_2) * half_over_C_tau_1) - (C_tau_magic_factor_3 * delta_2)));
			norm_etc_global[thread_id] = norm_etc_global[thread_id] - (eta_1 * (2.0 * gradient_gamma_ALGD[best_index_1] + training_label_transformed_ALGD[best_index_1] + ((gamma_ALGD[best_index_1] - eta_1) * half_over_C_tau_2) - (C_tau_magic_factor_4 * eta_1)));
			norm_etc_global[thread_id] = norm_etc_global[thread_id] - (eta_2 * (2.0 * gradient_gamma_ALGD[best_index_2] + training_label_transformed_ALGD[best_index_2] + ((gamma_ALGD[best_index_2] - eta_2) * half_over_C_tau_2) - (C_tau_magic_factor_4 * eta_2)));
			norm_etc_global[thread_id] = norm_etc_global[thread_id] + (2.0 * (delta_1 - eta_1) * (delta_2 - eta_2) * kernel_row1_ALGD[best_index_2] - 2.0 * ((delta_1 * eta_1) + (delta_2 * eta_2)));

			gradient_beta_ALGD[best_index_1] = gradient_beta_ALGD[best_index_1] - half_over_C_tau_1 * delta_1; 
			gradient_beta_ALGD[best_index_2] = gradient_beta_ALGD[best_index_2] - half_over_C_tau_1 * delta_2; 
			gradient_gamma_ALGD[best_index_1]= gradient_gamma_ALGD[best_index_1]- half_over_C_tau_2 * eta_1; 
			gradient_gamma_ALGD[best_index_2]= gradient_gamma_ALGD[best_index_2]- half_over_C_tau_2 * eta_2; 

			
			slack_sum_local[thread_id] = 0.0;
			get_aligned_chunk(training_set_size, 2, 0, start_index, stop_index);
			inner_loop(start_index, stop_index, best_index_1, best_index_2, delta_1, delta_2, eta_1, eta_2, slack_sum_local[thread_id], new_best_index_1, new_best_gain_1); 			
			get_aligned_chunk(training_set_size, 2, 1, start_index, stop_index);
			inner_loop(start_index, stop_index, best_index_1, best_index_2, delta_1, delta_2, eta_1, eta_2, slack_sum_local[thread_id], new_best_index_2, new_best_gain_2); 			
		  
			primal_dual_gap[thread_id] = norm_etc_global[thread_id] + C_current * slack_sum_local[thread_id];
		  

			order_indices(new_best_index_1, new_best_gain_1, new_best_index_2, new_best_gain_2);

			best_gain = optimize_2D(new_best_index_1, new_best_index_2, new_beta_1, new_beta_2, new_gamma_1, new_gamma_2);
		 
			temp_index_1 = new_best_index_1;
			temp_index_2 = new_best_index_2; 
			
			changed = false;
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_beta_1, new_gamma_1, new_beta_2, new_gamma_2, best_gain, temp_index_1, best_index_1, changed);
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_beta_1, new_gamma_1, new_beta_2, new_gamma_2, best_gain, temp_index_1, best_index_2, changed); 
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_beta_1, new_gamma_1, new_beta_2, new_gamma_2, best_gain, temp_index_2, best_index_1, changed);
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_beta_1, new_gamma_1, new_beta_2, new_gamma_2, best_gain, temp_index_2, best_index_2, changed);
		 
			if (solver_ctrl.wss_method == USE_NNs)
			{
				train_val_info.tries_2D++;
				changed = false;
				
				if (kNN_list[new_best_index_1].size() == 0)
					kNN_list[new_best_index_1] = training_kernel->get_kNNs(new_best_index_1);

				for (k=0; k<kNN_list[new_best_index_1].size(); k++)
					compare_pair_of_indices(new_best_index_1, new_best_index_2, new_beta_1, new_gamma_1, new_beta_2, new_gamma_2, best_gain, new_best_index_1, kNN_list[new_best_index_1][k], changed);
				
				if (changed == true)
					train_val_info.hits_2D++;
			}

			best_index_1 = new_best_index_1;
			best_index_2 = new_best_index_2;
		  
			kernel_row1_ALGD = training_kernel->row(best_index_1);
			kernel_row2_ALGD = training_kernel->row(best_index_2);
	  
			train_val_info.train_iterations++;
			train_val_info.gradient_updates = train_val_info.gradient_updates + 2;
		}

		build_SV_list(train_val_info);
		
	}
}

//**********************************************************************************************************************************

void Texpectile_svm::get_train_error(Tsvm_train_val_info& train_val_info)
{
  unsigned i;
  double prediction;

  
  train_val_info.train_error = 0.0;
	for (i=0; i<training_set_size; i++)
	{
		prediction = training_label_transformed_ALGD[i] - gradient_beta_ALGD[i] - (beta_ALGD[i]/(2.0 * C_current * tau));
		train_val_info.train_error = train_val_info.train_error + loss_function.evaluate(training_label_ALGD[i], inverse_transform_label(prediction));
	}
	train_val_info.train_error = train_val_info.train_error/ double (training_set_size);
   
}

//**********************************************************************************************************************************



#endif



