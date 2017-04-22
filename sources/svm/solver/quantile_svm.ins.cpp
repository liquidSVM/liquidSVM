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


inline void Tquantile_svm_generic_ancestor::loop_without_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__& neg_slack_sum_simdd, simdd__& pos_slack_sum_simdd, simdd__ low_weight_simdd, simdd__ up_weight_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd)
{
	unsigned j;
	simdd__ grad_simdd;
	simdd__ gain_simdd;
	
	cache_prefetch(index_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(alpha_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row1_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row2_ALGD+32+index, PREFETCH_L1);

	for(j=index; j<index+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		grad_simdd = load_simdd(gradient_ALGD+j);
		grad_simdd = fuse_mult_add_simdd(delta_1_simdd, load_simdd(kernel_row1_ALGD+j), grad_simdd);
		grad_simdd = fuse_mult_add_simdd(delta_2_simdd, load_simdd(kernel_row2_ALGD+j), grad_simdd);

		store_simdd(gradient_ALGD+j, grad_simdd);
		
		pos_slack_sum_simdd = add_simdd(pos_slack_sum_simdd, max_simdd(zero_simdd, grad_simdd));
		neg_slack_sum_simdd = add_simdd(neg_slack_sum_simdd, min_simdd(zero_simdd, grad_simdd));
		
		gain_simdd = get_gain_simdd(grad_simdd, low_weight_simdd, up_weight_simdd, load_simdd(alpha_ALGD+j));
		get_index_with_better_gain_simdd(new_best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+j), gain_simdd);
	}
}


//**********************************************************************************************************************************


inline void Tquantile_svm_generic_ancestor::loop_with_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__& neg_slack_sum_simdd, simdd__& pos_slack_sum_simdd, simdd__  neg_clipp_value_simdd, simdd__ pos_clipp_value_simdd, simdd__ low_weight_simdd, simdd__ up_weight_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd)
{
	unsigned j;
	simdd__ grad_simdd;
	simdd__ gain_simdd;
	simdd__ label_simdd;
	simdd__ clipped_pred_diff_simdd;
	
	cache_prefetch(index_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(alpha_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(training_label_transformed_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row1_ALGD+32+index, PREFETCH_L1);
	cache_prefetch(kernel_row2_ALGD+32+index, PREFETCH_L1);
	
	for(j=index; j<index+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		grad_simdd = load_simdd(gradient_ALGD+j);
		grad_simdd = fuse_mult_add_simdd(delta_1_simdd, load_simdd(kernel_row1_ALGD+j), grad_simdd);
		grad_simdd = fuse_mult_add_simdd(delta_2_simdd, load_simdd(kernel_row2_ALGD+j), grad_simdd);

		store_simdd(gradient_ALGD+j, grad_simdd);
		
		label_simdd = load_simdd(training_label_transformed_ALGD+j);
		clipped_pred_diff_simdd = sub_simdd(label_simdd, clipp_simdd(sub_simdd(label_simdd, grad_simdd), neg_clipp_value_simdd, pos_clipp_value_simdd));
		
		pos_slack_sum_simdd = add_simdd(pos_slack_sum_simdd, max_simdd(zero_simdd, clipped_pred_diff_simdd));
		neg_slack_sum_simdd = add_simdd(neg_slack_sum_simdd, min_simdd(zero_simdd, clipped_pred_diff_simdd));
		
		gain_simdd = get_gain_simdd(grad_simdd, low_weight_simdd, up_weight_simdd, load_simdd(alpha_ALGD+j));
		get_index_with_better_gain_simdd(new_best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+j), gain_simdd);
	}
}




//**********************************************************************************************************************************



inline simdd__ Tquantile_svm_generic_ancestor::get_gain_simdd(simdd__ gradient_simdd, simdd__ low_weight_simdd, simdd__ up_weight_simdd, simdd__ alpha_simdd)
{
	simdd__ new_delta_simdd;
	
	new_delta_simdd = sub_simdd(clipp_simdd(add_simdd(gradient_simdd, alpha_simdd), low_weight_simdd, up_weight_simdd), alpha_simdd);
	return pos_part_simdd(mult_simdd(new_delta_simdd, fuse_mult_add_simdd(assign_simdd(-0.5), new_delta_simdd, gradient_simdd)));
}





//**********************************************************************************************************************************

inline unsigned Tquantile_svm_generic_ancestor::constraint_segment(double alpha)
{
	return ((alpha > up_weight)? 2:((alpha < low_weight)? 0:1));
}


//**********************************************************************************************************************************


inline double Tquantile_svm_generic_ancestor::optimize_2D_corner(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double border_1, double border_2, double& new_alpha_1, double& new_alpha_2, double K_ij)
{
	double gain_1;
	double gain_2;


	gain_1 = gain_2D(gradient_1, gradient_2, border_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
	gain_2 = gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, border_2 - current_alpha_2, K_ij);

	if (gain_1 > gain_2)
	{
		new_alpha_1 = border_1;
		return gain_1;
	}
	else
	{
		new_alpha_2 = border_2;
		return gain_2;
	}
}


//**********************************************************************************************************************************

inline double Tquantile_svm_generic_ancestor::gain_2D(double gradient_1, double gradient_2, double delta_1, double delta_2, double K_ij)
{
	return delta_1 * (gradient_1 - 0.5 * delta_1) + delta_2 * (gradient_2 - 0.5 * delta_2 - delta_1 * K_ij);
}



//**********************************************************************************************************************************


inline double Tquantile_svm_generic_ancestor::clipp_to_box(double alpha)
{
	return min(max(alpha, low_weight), up_weight);
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
// 	unsigned i;
// 	
// 	for (i=0; i<training_set_size; i++)
// 		alpha_ALGD[i] = 0.0;
// 	primal_dual_gap[get_thread_id()] = 0.0;
	flush_exit(1, "Trivial solution not implemented, yet");
}



//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::set_NNs_search(unsigned& index_1, unsigned& index_2, double& alpha_1, double& alpha_2, unsigned iterations)
{
	NNs_search = true;
}



//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::update_norm_etc(double delta_1, double delta_2, unsigned index_1, unsigned index_2, double& norm_etc)
{
	norm_etc = norm_etc + delta_1 * (2.0 * gradient_ALGD[index_1] - training_label_transformed_ALGD[index_1] - delta_1);
	norm_etc = norm_etc + delta_2 * (2.0 * gradient_ALGD[index_2] - training_label_transformed_ALGD[index_2] - delta_2);
	norm_etc = norm_etc - 2.0 * delta_1 * delta_2 * training_kernel->entry(index_1, index_2);
}



//**********************************************************************************************************************************


inline void Tsvm_2D_solver_generic_base_name::inner_loop(unsigned start_index, unsigned stop_index, double delta_1, double delta_2, unsigned index_1, unsigned index_2, unsigned& best_index, double& best_gain, double& slack_sum)
{
	unsigned i;
	double pos_slack_sum;
	double neg_slack_sum;
	
	simdd__ delta_1_simdd;
	simdd__ delta_2_simdd;
	simdd__ pos_slack_sum_simdd;
	simdd__ neg_slack_sum_simdd;
	simdd__ low_weight_simdd;
	simdd__ up_weight_simdd;
	simdd__ best_gain_simdd;
	simdd__ new_best_index_simdd;
	simdd__ neg_clipp_value_simdd;
	simdd__ pos_clipp_value_simdd;


	delta_1_simdd = assign_simdd(-delta_1);
	delta_2_simdd = assign_simdd(-delta_2);
	pos_slack_sum_simdd = assign_simdd(0.0);
	neg_slack_sum_simdd = assign_simdd(0.0);
	low_weight_simdd = assign_simdd(low_weight);
	up_weight_simdd = assign_simdd(up_weight);
	best_gain_simdd = assign_simdd(-1.0);
	new_best_index_simdd = assign_simdd(0.0);
	
	kernel_row1_ALGD = training_kernel->row(index_1);
	kernel_row2_ALGD = training_kernel->row(index_2);
	
	if (solver_clipp_value == 0.0)
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
			loop_without_clipping_CL(i, delta_1_simdd, delta_2_simdd, neg_slack_sum_simdd, pos_slack_sum_simdd, low_weight_simdd, up_weight_simdd, best_gain_simdd, new_best_index_simdd);
	else
	{
		pos_clipp_value_simdd = assign_simdd(transform_label(solver_clipp_value));
		neg_clipp_value_simdd = assign_simdd(transform_label(-solver_clipp_value));
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
			loop_with_clipping_CL(i, delta_1_simdd, delta_2_simdd, neg_slack_sum_simdd, pos_slack_sum_simdd, neg_clipp_value_simdd, pos_clipp_value_simdd, low_weight_simdd, up_weight_simdd, best_gain_simdd, new_best_index_simdd);
	}
	
	pos_slack_sum = reduce_sums_simdd(pos_slack_sum_simdd);
	neg_slack_sum = reduce_sums_simdd(neg_slack_sum_simdd);
	
	slack_sum = slack_sum + up_weight * pos_slack_sum + low_weight * neg_slack_sum; 
	argmax_simdd(new_best_index_simdd, best_gain_simdd, best_index, best_gain);
}






//**********************************************************************************************************************************

inline double Tsvm_2D_solver_generic_base_name::optimize_2D(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double weight_1, double weight_2, double label_1, double label_2, double& new_alpha_1,double& new_alpha_2, double K_ij, bool same_indices)
{
	double gamma_1;
	double gamma_2;
	double delta;
	double delta_sum;
	double denominator;
	

	if (same_indices == true)
	{
		new_alpha_1 = current_alpha_1;
		new_alpha_2 = clipp_to_box(gradient_2 + current_alpha_2);
		delta = new_alpha_2 - current_alpha_2;
		return delta * (gradient_2 - 0.5 * delta);
	}

	if (K_ij == 1.0)
	{
		if (label_1 == label_2)
		{
			new_alpha_1 = clipp_to_box(0.5 * (gradient_1 + current_alpha_1 + current_alpha_2));
			new_alpha_2 = new_alpha_1;
			delta_sum = new_alpha_1 - current_alpha_1 + new_alpha_2 - current_alpha_2;
			return delta_sum * (gradient_1 - 0.5 * delta_sum); 
		}
		else if (label_1 > label_2)
		{
			new_alpha_1 = clipp_to_box(gradient_1 + current_alpha_1 + current_alpha_2 - low_weight);
			new_alpha_2 = clipp_to_box(gradient_2 + current_alpha_1 + current_alpha_2 - up_weight);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, up_weight, low_weight, new_alpha_1, new_alpha_2, K_ij);
		}
		else
		{
			new_alpha_1 = clipp_to_box(gradient_1 + current_alpha_1 + current_alpha_2 - up_weight);
			new_alpha_2 = clipp_to_box(gradient_2 + current_alpha_1 + current_alpha_2 - low_weight);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, low_weight, up_weight, new_alpha_1, new_alpha_2, K_ij);
		}
	}

	
	denominator = 1.0/(1.0 - K_ij * K_ij);
	gamma_1 = gradient_1 + current_alpha_1 + current_alpha_2 * K_ij;
	gamma_2 = gradient_2 + current_alpha_2 + current_alpha_1 * K_ij;

	new_alpha_1 = (gamma_1 - gamma_2 * K_ij) * denominator;
	new_alpha_2 = (gamma_2 - gamma_1 * K_ij) * denominator;


	
	switch (3 * constraint_segment(new_alpha_1) + constraint_segment(new_alpha_2))
	{
		case 0:
			new_alpha_1 = clipp_to_box(gamma_1 - low_weight * K_ij);
			new_alpha_2 = clipp_to_box(gamma_2 - low_weight * K_ij);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, low_weight, low_weight, new_alpha_1, new_alpha_2, K_ij);
		case 1:
			new_alpha_1 = low_weight;
			new_alpha_2 = clipp_to_box(gamma_2 - low_weight * K_ij);
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 2:
			new_alpha_1 = clipp_to_box(gamma_1 - up_weight * K_ij);
			new_alpha_2 = clipp_to_box(gamma_2 - low_weight * K_ij);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, low_weight, up_weight, new_alpha_1, new_alpha_2, K_ij);
		case 3:
			new_alpha_1 = clipp_to_box(gamma_1 - low_weight * K_ij);
			new_alpha_2 = low_weight;
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 4:
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 5:
			new_alpha_1 = clipp_to_box(gamma_1 - up_weight * K_ij);
			new_alpha_2 = up_weight;
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 6:
			new_alpha_1 = clipp_to_box(gamma_1 - low_weight * K_ij);
			new_alpha_2 = clipp_to_box(gamma_2 - up_weight * K_ij);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, up_weight, low_weight, new_alpha_1, new_alpha_2, K_ij);
		case 7:
			new_alpha_1 = up_weight;
			new_alpha_2 = clipp_to_box(gamma_2 - up_weight * K_ij);
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 8:
			new_alpha_1 = clipp_to_box(gamma_1 - up_weight * K_ij);
			new_alpha_2 = clipp_to_box(gamma_2 - up_weight * K_ij);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, up_weight, up_weight, new_alpha_1, new_alpha_2, K_ij);
		default:
			return -1.0;
	}
}




//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::get_optimal_1D_CL(unsigned index, simdd__& best_gain_simdd, simdd__& best_index_simdd)
{
	unsigned j;
	simdd__ gain_simdd;
	simdd__ low_weight_simdd;
	simdd__ up_weight_simdd;
	
	
	low_weight_simdd = assign_simdd(low_weight);
	up_weight_simdd = assign_simdd(up_weight);
	
	cache_prefetch(alpha_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(index_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+index+32, PREFETCH_L1);
	for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		gain_simdd = get_gain_simdd(load_simdd(gradient_ALGD+index+j), low_weight_simdd, up_weight_simdd, load_simdd(alpha_ALGD+index+j));
		get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+index+j), gain_simdd);
	}
}







//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************















//**********************************************************************************************************************************


inline void Tquantile_svm::add_to_gradient(simdd__ factor_simdd, double* restrict__ kernel_row_ALGD)
{
	unsigned i;
	Tthread_chunk thread_chunk;

	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(kernel_row_ALGD+i+32, PREFETCH_NO);
		fuse_mult_add3_CL(factor_simdd, kernel_row_ALGD+i, gradient_ALGD+i);
	}
}








