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



inline void Thinge_svm_generic_ancestor::loop_with_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__& slack_sum_simdd, simdd__& best_gain_simdd, simdd__& best_index_simdd, double* restrict__ kernel_row1_ALGD, double* restrict__ kernel_row2_ALGD)
{
	unsigned i;
	simdd__ grad_simdd;
	simdd__ gain_simdd;
	simdd__ weight_simdd;

	
	cache_prefetch(alpha_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(index_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(weight_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(kernel_row1_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(kernel_row2_ALGD+index+32, PREFETCH_L2);
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
	{
		grad_simdd = update_2gradients_simdd(load_simdd(gradient_ALGD+index+i), delta_1_simdd, delta_2_simdd, load_simdd(kernel_row1_ALGD+index+i), load_simdd(kernel_row2_ALGD+index+i));
		weight_simdd = load_simdd(weight_ALGD+index+i);

		slack_sum_simdd = fuse_mult_add_simdd(weight_simdd, clipp_02_simdd(grad_simdd), slack_sum_simdd);
		store_simdd(gradient_ALGD+index+i, grad_simdd);
		
		gain_simdd = get_gain_simdd(grad_simdd, weight_simdd, load_simdd(alpha_ALGD+index+i));
		get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+index+i), gain_simdd);
	}
}



//**********************************************************************************************************************************



inline simdd__ Thinge_svm_generic_ancestor::get_gain_simdd(simdd__ gradient_simdd, simdd__ weight_simdd, simdd__ alpha_simdd)
{
	simdd__ new_delta_simdd;
	
	new_delta_simdd = sub_simdd(clipp_0max_simdd(add_simdd(gradient_simdd, alpha_simdd), weight_simdd), alpha_simdd);
	return mult_simdd(new_delta_simdd, sub_simdd(gradient_simdd, mult_simdd(assign_simdd(0.5), new_delta_simdd)));
}





//**********************************************************************************************************************************

inline unsigned Thinge_svm_generic_ancestor::constraint_segment(double weight, double alpha)
{
	return ((alpha > weight)? 2:((alpha < 0.0)? 0:1));
}



//**********************************************************************************************************************************

inline double Thinge_svm_generic_ancestor::gain_2D(double gradient_1, double gradient_2, double delta_1, double delta_2, double K_ij)
{
	return delta_1 * (gradient_1 - 0.5 * delta_1) + delta_2 * (gradient_2 - 0.5 * delta_2 - delta_1 * K_ij);
}





//**********************************************************************************************************************************


inline double Thinge_svm_generic_ancestor::optimize_2D_corner(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double border_1, double border_2, double& new_alpha_1, double& new_alpha_2, double K_ij)
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
	if (neg_train_size > pos_train_size)
		offset = -1.0;
	else
		offset = 1.0;
	flush_info(INFO_DEBUG, "Switching to average vote since the training set size is not greater than %d.", CACHELINE_STEP);
}


//**********************************************************************************************************************************



inline void Tsvm_2D_solver_generic_base_name::set_NNs_search(unsigned& index_1, unsigned& index_2, double& alpha_1, double& alpha_2, unsigned iterations)
{
	if ((alpha_1 > 0.0) and (alpha_1 < weight_ALGD[index_1]))
		inner_optimizations++;
	if ((alpha_2 > 0.0) and (alpha_2 < weight_ALGD[index_2]))
		inner_optimizations++;
		
	if ((iterations % 10) == 0)
	{
		NNs_search = (inner_optimizations >= 15);
		inner_optimizations = 0;
	}
}



//**********************************************************************************************************************************


inline void Tsvm_2D_solver_generic_base_name::update_norm_etc(double delta_1, double delta_2, unsigned index_1, unsigned index_2, double& norm_etc)
{
	double tmp;
	
	norm_etc = norm_etc + delta_1 * (2.0 * gradient_ALGD[index_1] - 1.0 - delta_1);
	tmp = gradient_ALGD[index_2] - delta_1 * kernel_row1_ALGD[index_2];
	norm_etc = norm_etc + delta_2 * (2.0 * tmp - 1.0 - delta_2);
}




//**********************************************************************************************************************************


inline void Tsvm_2D_solver_generic_base_name::inner_loop(unsigned start_index, unsigned stop_index, double delta_1, double delta_2, unsigned index_1, unsigned index_2, unsigned& new_best_index, double& best_gain, double& slack_sum)
{
	unsigned i;
	simdd__ delta_1_simdd;
	simdd__ delta_2_simdd;
	simdd__ slack_sum_simdd;
	simdd__ best_gain_simdd;
	simdd__ best_index_simdd;
	double* restrict__ kernel_row1_ALGD;
	double* restrict__ kernel_row2_ALGD;
	

	delta_1_simdd = assign_simdd(-delta_1);
	delta_2_simdd = assign_simdd(-delta_2);
	slack_sum_simdd = assign_simdd(0.0);
	best_gain_simdd = assign_simdd(-1.0);
	best_index_simdd = assign_simdd(0.0);

	kernel_row1_ALGD = training_kernel->row(index_1);
	kernel_row2_ALGD = training_kernel->row(index_2);

	for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
		loop_with_clipping_CL(i, delta_1_simdd, delta_2_simdd, slack_sum_simdd, best_gain_simdd, best_index_simdd, kernel_row1_ALGD, kernel_row2_ALGD);

	slack_sum = slack_sum + reduce_sums_simdd(slack_sum_simdd);
	argmax_simdd(best_index_simdd, best_gain_simdd, new_best_index, best_gain);
}


//**********************************************************************************************************************************

inline double Tsvm_2D_solver_generic_base_name::optimize_2D(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double weight_1, double weight_2, double label_1, double label_2, double& new_alpha_1, double& new_alpha_2, double K_ij, bool same_indices)
{
	double gamma_1;
	double gamma_2;
	double delta;
	double delta_sum;
	double denominator;
	
	
	if (same_indices == true)
	{
		new_alpha_1 = current_alpha_1;
		new_alpha_2 = clipp_0max(gradient_2 + current_alpha_2, weight_2);
		delta = new_alpha_2 - current_alpha_2;
		return delta * (gradient_2 - 0.5 * delta);
	}

	if (K_ij == 1.0)
	{
		new_alpha_1 = clipp_0max(0.5 * (gradient_1 + current_alpha_1 + current_alpha_2), weight_1);
		new_alpha_2 = new_alpha_1;
		delta_sum = new_alpha_1 - current_alpha_1 + new_alpha_2 - current_alpha_2;
		return delta_sum * (gradient_1 - 0.5 * delta_sum); 
	}

	if (K_ij == -1.0)
	{
		new_alpha_1 = clipp_0max(gradient_1 + current_alpha_1 - current_alpha_2 + weight_2, weight_1);
		new_alpha_2 = clipp_0max(gradient_2 + current_alpha_2 - current_alpha_1 + weight_1, weight_2);
		return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, weight_1, weight_2, new_alpha_1, new_alpha_2, K_ij);
	}
	
	denominator = 1.0/(1.0 - K_ij * K_ij);
	gamma_1 = gradient_1 + current_alpha_1 + current_alpha_2 * K_ij;
	gamma_2 = gradient_2 + current_alpha_2 + current_alpha_1 * K_ij;

	new_alpha_1 = (gamma_1 - gamma_2 * K_ij) * denominator;
	new_alpha_2 = (gamma_2 - gamma_1 * K_ij) * denominator;

	switch (3 * constraint_segment(weight_1, new_alpha_1) + constraint_segment(weight_2, new_alpha_2))
	{
		case 0:
			new_alpha_1 = clipp_0max(gamma_1, weight_1);
			new_alpha_2 = clipp_0max(gamma_2, weight_2);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, 0.0, 0.0, new_alpha_1, new_alpha_2, K_ij);
		case 1:
			new_alpha_1 = 0.0;
			new_alpha_2 = clipp_0max(gamma_2, weight_2);
			return gain_2D(gradient_1, gradient_2, -current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 2:
			new_alpha_1 = clipp_0max(gamma_1 - weight_2 * K_ij, weight_1);
			new_alpha_2 = clipp_0max(gamma_2, weight_2);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, 0.0, weight_2, new_alpha_1, new_alpha_2, K_ij);
		case 3:
			new_alpha_1 = clipp_0max(gamma_1, weight_1);
			new_alpha_2 = 0.0;
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, -current_alpha_2, K_ij);
		case 4:
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 5:
			new_alpha_1 = clipp_0max(gamma_1 - weight_2 * K_ij, weight_1);
			new_alpha_2 = weight_2;
			return gain_2D(gradient_1, gradient_2, new_alpha_1 - current_alpha_1, weight_2 - current_alpha_2, K_ij);
		case 6:
			new_alpha_1 = clipp_0max(gamma_1, weight_1);
			new_alpha_2 = clipp_0max(gamma_2 - weight_1 * K_ij, weight_2);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, weight_1, 0.0, new_alpha_1, new_alpha_2, K_ij);
		case 7:
			new_alpha_1 = weight_1;
			new_alpha_2 = clipp_0max(gamma_2 - weight_1 * K_ij, weight_2);
			return gain_2D(gradient_1, gradient_2, weight_1 - current_alpha_1, new_alpha_2 - current_alpha_2, K_ij);
		case 8:
			new_alpha_1 = clipp_0max(gamma_1 - weight_2 * K_ij, weight_1);
			new_alpha_2 = clipp_0max(gamma_2 - weight_1 * K_ij, weight_2);
			return optimize_2D_corner(current_alpha_1, current_alpha_2, gradient_1, gradient_2, weight_1, weight_2, new_alpha_1, new_alpha_2, K_ij);
		default:
			return -1.0;
	}
};



//**********************************************************************************************************************************



inline void Tsvm_2D_solver_generic_base_name::get_optimal_1D_CL(unsigned index, simdd__& best_gain_simdd, simdd__& best_index_simdd)
{
	unsigned j;
	simdd__ gain_simdd;
	
	cache_prefetch(alpha_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(index_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(weight_ALGD+index+32, PREFETCH_L1);
	cache_prefetch(gradient_ALGD+index+32, PREFETCH_L1);
	for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
	{
		gain_simdd = get_gain_simdd(load_simdd(gradient_ALGD+index+j), load_simdd(weight_ALGD+index+j), load_simdd(alpha_ALGD+index+j));
		get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+index+j), gain_simdd);
	}
}





























