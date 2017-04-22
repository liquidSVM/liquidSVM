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







//**********************************************************************************************************************************

inline double Tleast_squares_svm_generic_ancestor::compute_slack_sum(unsigned start_index, unsigned stop_index)
{
	unsigned i;
	unsigned j;
	simdd__ C_simdd;
	simdd__ slack_sum_simdd;
	simdd__ neg_clipp_value_simdd;
	simdd__ pos_clipp_value_simdd;

	
	slack_sum_simdd = assign_simdd(0.0);
	C_simdd = assign_simdd(half_over_C);
	if (solver_clipp_value == 0.0)
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
		{
			cache_prefetch(alpha_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
			for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
				add_to_slack_sum_simdd(slack_sum_simdd, load_simdd(gradient_ALGD+i+j), load_simdd(alpha_ALGD+i+j), C_simdd);
		}
	else
	{
		pos_clipp_value_simdd = assign_simdd(transform_label(solver_clipp_value));
		neg_clipp_value_simdd = assign_simdd(transform_label(-solver_clipp_value));
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
		{
			cache_prefetch(alpha_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(training_label_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
			for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
				add_to_clipped_slack_sum_simdd(slack_sum_simdd, load_simdd(gradient_ALGD+i+j), load_simdd(training_label_ALGD+i), neg_clipp_value_simdd, pos_clipp_value_simdd, load_simdd(alpha_ALGD+i+j), C_simdd);
		}
	}
	return C_current * reduce_sums_simdd(slack_sum_simdd);
}

//**********************************************************************************************************************************


inline void Tleast_squares_svm_generic_ancestor::add_to_slack_sum_simdd(simdd__& slack_sum_simdd, simdd__ gradient_simdd, simdd__ alpha_simdd, simdd__ C_simdd)
{
	simdd__ tmp_simdd;

	tmp_simdd = fuse_mult_add_simdd(alpha_simdd, C_simdd, gradient_simdd);
	slack_sum_simdd = fuse_mult_add_simdd(tmp_simdd, tmp_simdd, slack_sum_simdd);
}

//**********************************************************************************************************************************


inline void Tleast_squares_svm_generic_ancestor::add_to_clipped_slack_sum_simdd(simdd__& slack_sum_simdd, simdd__ gradient_simdd, simdd__ label_simdd, simdd__ neg_clipp_simdd, simdd__ pos_clipp_simdd, simdd__ alpha_simdd, simdd__ C_simdd)
{
	simdd__ tmp_simdd;

	tmp_simdd = sub_simdd(label_simdd, fuse_mult_add_simdd(alpha_simdd, C_simdd, gradient_simdd));
	tmp_simdd = sub_simdd(label_simdd, clipp_simdd(tmp_simdd, neg_clipp_simdd, pos_clipp_simdd));
	slack_sum_simdd = fuse_mult_add_simdd(tmp_simdd, tmp_simdd, slack_sum_simdd);
}

//**********************************************************************************************************************************



inline void Tleast_squares_svm_generic_ancestor::loop_without_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ C_simdd, simdd__& slack_sum_simdd, simdd__& best_gain_simdd, simdd__& best_index_simdd, double* restrict__ kernel_row1_ALGD, double* restrict__ kernel_row2_ALGD)
{
	simdd__ grad_simdd;
	simdd__ gain_simdd;
	unsigned j;
	
	
	cache_prefetch(alpha_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(index_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row1_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row2_ALGD+32+index, PREFETCH_L2);
		
	for(j=index; j<index+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		grad_simdd = update_2gradients_simdd(load_simdd(gradient_ALGD+j), delta_1_simdd, delta_2_simdd, load_simdd(kernel_row1_ALGD+j), load_simdd(kernel_row2_ALGD+j));
		add_to_slack_sum_simdd(slack_sum_simdd, grad_simdd, load_simdd(alpha_ALGD+j), C_simdd);
		store_simdd(gradient_ALGD+j, grad_simdd);
		gain_simdd = mult_simdd(grad_simdd, grad_simdd);
		get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+j), gain_simdd);
	}
}

//**********************************************************************************************************************************



inline void Tleast_squares_svm_generic_ancestor::loop_with_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ neg_clipp_simdd, simdd__ pos_clipp_simdd, simdd__ C_simdd, simdd__& slack_sum_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd, double* restrict__ kernel_row1_ALGD, double* restrict__ kernel_row2_ALGD)
{
	simdd__ grad_simdd;
	simdd__ gain_simdd;
	unsigned j;

	
	cache_prefetch(alpha_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(index_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(training_label_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row1_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row2_ALGD+32+index, PREFETCH_L2);
	
	for(j=index; j<index+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		grad_simdd = update_2gradients_simdd(load_simdd(gradient_ALGD+j), delta_1_simdd, delta_2_simdd, load_simdd(kernel_row1_ALGD+j), load_simdd(kernel_row2_ALGD+j));
		
		add_to_clipped_slack_sum_simdd(slack_sum_simdd, grad_simdd, load_simdd(training_label_ALGD+j), neg_clipp_simdd, pos_clipp_simdd, load_simdd(alpha_ALGD+j), C_simdd);
		
		store_simdd(gradient_ALGD+j, grad_simdd);
		
		gain_simdd = mult_simdd(grad_simdd, grad_simdd);
		get_index_with_better_gain_simdd(new_best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+j), gain_simdd);
	}
}





//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::prepare_core_solver(Tsvm_train_val_info& train_val_info)
{
}


//**********************************************************************************************************************************


inline void Tsvm_2D_solver_generic_base_name::trivial_solution()
{
	unsigned i;
	
	for (i=0; i<training_set_size; i++)
		alpha_ALGD[i] = 0.0;
	primal_dual_gap[get_thread_id()] = 0.0;
	flush_info(INFO_DEBUG, "Switching to average vote since the training set size is not greater than %d.", CACHELINE_STEP);
}


//**********************************************************************************************************************************



inline void Tsvm_2D_solver_generic_base_name::get_optimal_1D_CL(unsigned index, simdd__& best_gain_simdd, simdd__& best_index_simdd)
{
	unsigned j;
	simdd__ grad_simdd;
	
	
	cache_prefetch(index_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+32+index, PREFETCH_L1);
	for(j=index; j<index+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		grad_simdd = load_simdd(gradient_ALGD+j);
		get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+j), mult_simdd(grad_simdd, grad_simdd));
	}
}





//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::inner_loop(unsigned start_index, unsigned stop_index, double delta_1, double delta_2, unsigned best_index_1, unsigned best_index_2, unsigned& new_best_index, double& best_gain, double& slack_sum)
{
	unsigned i;
	simdd__ C_simdd;
	simdd__ delta_1_simdd;
	simdd__ delta_2_simdd;
	simdd__ neg_clipp_value_simdd;
	simdd__ pos_clipp_value_simdd;
	simdd__ slack_sum_simdd;
	simdd__ best_gain_simdd;
	simdd__ best_index_simdd;
	double* restrict__ kernel_row1_ALGD;
	double* restrict__ kernel_row2_ALGD;
	
	
	if ((best_index_1 >= start_index) and (best_index_1 < stop_index))
		gradient_ALGD[best_index_1] = gradient_ALGD[best_index_1] - delta_1 * half_over_C;
	if ((best_index_2 >= start_index) and (best_index_2 < stop_index))
		gradient_ALGD[best_index_2] = gradient_ALGD[best_index_2] - delta_2 * half_over_C;
	
	delta_1_simdd = assign_simdd(-delta_1);
	delta_2_simdd = assign_simdd(-delta_2);
	C_simdd = assign_simdd(half_over_C);
	slack_sum_simdd = assign_simdd(0.0);
	best_gain_simdd = assign_simdd(-1.0);
	best_index_simdd = assign_simdd(0.0);

	kernel_row1_ALGD = training_kernel->row(best_index_1);
	kernel_row2_ALGD = training_kernel->row(best_index_2);
	
	if (solver_clipp_value == 0.0)
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
			loop_without_clipping_CL(i, delta_1_simdd, delta_2_simdd, C_simdd, slack_sum_simdd, best_gain_simdd, best_index_simdd, kernel_row1_ALGD, kernel_row2_ALGD);
	else
	{
		pos_clipp_value_simdd = assign_simdd(transform_label(solver_clipp_value));
		neg_clipp_value_simdd = assign_simdd(transform_label(-solver_clipp_value));
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
			loop_with_clipping_CL(i, delta_1_simdd, delta_2_simdd, neg_clipp_value_simdd, pos_clipp_value_simdd, C_simdd, slack_sum_simdd, best_gain_simdd, best_index_simdd, kernel_row1_ALGD, kernel_row2_ALGD);
	}
	slack_sum = slack_sum + C_current * reduce_sums_simdd(slack_sum_simdd);
	argmax_simdd(best_index_simdd, best_gain_simdd, new_best_index, best_gain);
}




//**********************************************************************************************************************************

inline double Tsvm_2D_solver_generic_base_name::optimize_2D(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double weight_1, double weight_2, double label_1, double label_2, double& new_alpha_1, double& new_alpha_2, double K_ij, bool same_indices)
{
	double delta_1;
	double delta_2;
	double delta_factor;


	delta_factor = 1.0/(C_magic_factor_3 - K_ij * K_ij);

	delta_1 = delta_factor * (C_magic_factor_1 * gradient_1 - K_ij * gradient_2);
	delta_2 = delta_factor * (C_magic_factor_1 * gradient_2 - K_ij * gradient_1);

	new_alpha_1 = current_alpha_1 + delta_1;
	new_alpha_2 = current_alpha_2 + delta_2;

	return delta_2 * (gradient_2 - delta_1 * K_ij - delta_2 * C_magic_factor_2) + delta_1 * (gradient_1 - delta_1 * C_magic_factor_2);
}


//**********************************************************************************************************************************


inline void Tsvm_2D_solver_generic_base_name::update_norm_etc(double delta_1, double delta_2, unsigned index_1, unsigned index_2, double& norm_etc)
{
	norm_etc = norm_etc - delta_1 * (training_label_ALGD[index_1] - 2.0 * gradient_ALGD[index_1] - half_over_C * (alpha_ALGD[index_1] - delta_1));
	norm_etc = norm_etc - delta_2 * (training_label_ALGD[index_2] - 2.0 * gradient_ALGD[index_2] - half_over_C * (alpha_ALGD[index_2] - delta_2));
	norm_etc = norm_etc - 2.0 * delta_1 * delta_2 * training_kernel->entry(index_1, index_2) - (1.0 / C_magic_factor_4) * (delta_1 * delta_1 + delta_2 * delta_2);
}


//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::set_NNs_search(unsigned& index_1, unsigned& index_2, double& alpha_1, double& alpha_2, unsigned iterations)
{
	NNs_search = true;
}






