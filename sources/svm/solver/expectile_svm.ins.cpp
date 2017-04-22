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



//**************************************************************************************************************************************
 
inline void Texpectile_svm::get_optimal_1D_direction(unsigned start_index, unsigned stop_index, unsigned& best_index, double& best_gain)
{

	unsigned i;
	unsigned j;
	simdd__ b_1_simdd;
	simdd__ b_1_reciprocal_simdd;
	simdd__ b_2_reciprocal_simdd;
	simdd__ c_value_simdd;
	simdd__ gradient_beta_simdd;
	simdd__ gradient_gamma_simdd;
	simdd__ beta_simdd;
	simdd__ gamma_simdd;
	simdd__ delta_simdd;
	simdd__ eta_simdd;
	simdd__ gain_simdd;
	simdd__ gain_factor_1_simdd;
	simdd__ gain_factor_2_simdd;
	simdd__ best_gain_simdd;
	simdd__ best_index_simdd;
	simdd__ C_tau_magic_factor_1_simdd;
	simdd__ C_tau_magic_factor_2_simdd;
	
	
	b_1_simdd = assign_simdd(-b1);
	b_1_reciprocal_simdd = assign_simdd(reciprocal_b1);
	b_2_reciprocal_simdd = assign_simdd(-reciprocal_b2);
	C_tau_magic_factor_1_simdd = assign_simdd(-C_tau_magic_factor_1);
	C_tau_magic_factor_2_simdd = assign_simdd(-C_tau_magic_factor_2);
	best_gain_simdd = assign_simdd(-1.0);
	best_index_simdd = zero_simdd;
	
	for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
 	{
		cache_prefetch(index_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(beta_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gamma_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_beta_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_gamma_ALGD+i+32, PREFETCH_L1);
	
		for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
		{
			gradient_beta_simdd = load_simdd(gradient_beta_ALGD+i+j);
			gradient_gamma_simdd = load_simdd(gradient_gamma_ALGD+i+j);
			
			beta_simdd = load_simdd(beta_ALGD+i+j);
			gamma_simdd = load_simdd(gamma_ALGD+i+j);
			c_value_simdd = sub_simdd(gradient_beta_simdd, fuse_mult_add_simdd(b_1_simdd, beta_simdd, gamma_simdd)); 

			delta_simdd = fuse_mult_sub_simdd(b_1_reciprocal_simdd, pos_part_simdd(c_value_simdd), beta_simdd);
			gain_factor_1_simdd = fuse_mult_add_simdd(C_tau_magic_factor_1_simdd, delta_simdd, gradient_beta_simdd);

			eta_simdd = fuse_mult_sub_simdd(b_2_reciprocal_simdd, min_simdd(zero_simdd, c_value_simdd), gamma_simdd);
			gain_factor_2_simdd = fuse_mult_add_simdd(C_tau_magic_factor_2_simdd, eta_simdd, gradient_gamma_simdd);

			gain_simdd = fuse_mult_add_simdd(delta_simdd, gain_factor_1_simdd, fuse_mult_add_simdd(delta_simdd, eta_simdd, mult_simdd(eta_simdd, gain_factor_2_simdd)));

			get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+i+j), gain_simdd);
 		}  
	} 	
 	argmax_simdd(best_index_simdd, best_gain_simdd, best_index, best_gain);

}



//*********************************************************************************************************************************************	


inline double Texpectile_svm::get_1D_gain(unsigned best_index, double delta, double eta)
{
	double magic_factor1;
	double magic_factor2;
	
	magic_factor1 = 2.0 * C_current * tau;
	magic_factor2 = 2.0 * C_current * (1-tau);
	
	return delta * (gradient_beta_ALGD[best_index] - ((delta + magic_factor1 * delta)/(2.0 * magic_factor1))) + eta * (gradient_gamma_ALGD[best_index] - ((eta + magic_factor2 * eta)/(2.0 * magic_factor2))) + delta * eta;

};



//**********************************************************************************************************************************


inline double Texpectile_svm::optimize_2D(unsigned index_1, unsigned index_2, double& new_beta_1,double& new_beta_2, double& new_gamma_1, double& new_gamma_2)
{
	double K_ij;
	double c_i;
	double c_j;
	
	double indicator_1;
	double indicator_2;
	double indicator_3;
	double indicator_4;
	
	double Det;
	
	double delta_1;
	double delta_2;
	double eta_1;
	double eta_2;
	double gain;
	
	kernel_row1_ALGD = training_kernel->row(index_1);
	kernel_row2_ALGD = training_kernel->row(index_2);
	K_ij = kernel_row1_ALGD[index_2];
	
	c_i = gradient_beta_ALGD[index_1] + b1 * beta_ALGD[index_1] - gamma_ALGD[index_1] + (beta_ALGD[index_2] - gamma_ALGD[index_2]) * K_ij;
	c_j = gradient_beta_ALGD[index_2] + b1 * beta_ALGD[index_2] - gamma_ALGD[index_2] + (beta_ALGD[index_1] - gamma_ALGD[index_1]) * K_ij;
	
	indicator_1 = b1 * c_i - K_ij * c_j;
	indicator_2 = b1 * c_j - K_ij * c_i;
	indicator_3 = K_ij * c_j - b2 * c_i;
	indicator_4 = K_ij * c_i - b2 * c_j;
	
	if ((indicator_3 >= 0.0) && (indicator_4 >= 0.0))
	{
		Det = b2 * b2 - K_ij * K_ij;
		new_beta_1 = 0.0;
		new_beta_2 = 0.0;
		new_gamma_1 = indicator_3/Det;
		new_gamma_2 = indicator_4/Det;
	
	}
	
	else if ((indicator_1 >= 0.0) && (indicator_2 >= 0.0))
	{
		Det = b1 * b1 - K_ij * K_ij;
		new_beta_1 = indicator_1/Det;
		new_beta_2 = indicator_2/Det;
		new_gamma_1 = 0.0;
		new_gamma_2 = 0.0;
	}
	
	else if ((indicator_1 < 0.0) && (indicator_4 < 0.0))
	{
		Det = K_ij * K_ij - b1 * b2;
		new_beta_1 = 0.0;
		new_beta_2 = indicator_4/Det;
		new_gamma_1 = indicator_1/Det;
		new_gamma_2 = 0.0;
	}
	else 
	{
		Det = K_ij * K_ij - b1 * b2;
		new_beta_1 = indicator_3/Det;
		new_beta_2 = 0.0 ;
		new_gamma_1 = 0.0;
		new_gamma_2 = indicator_2/Det;
	}
	
	delta_1 = new_beta_1 - beta_ALGD[index_1];
	delta_2 = new_beta_2 - beta_ALGD[index_2];
	eta_1 = new_gamma_1 - gamma_ALGD[index_1];
	eta_2 = new_gamma_2 - gamma_ALGD[index_2];
	gain = get_2D_gain(index_1, index_2, delta_1, delta_2, eta_1, eta_2, K_ij);
	return gain;

};


//**********************************************************************************************************************************************



inline double Texpectile_svm::get_2D_gain(unsigned index_1, unsigned index_2, double delta_1, double delta_2, double eta_1, double eta_2, double k)
{
	double gain_1;
	double gain_2;
  
	gain_1 = get_1D_gain(index_1, delta_1, eta_1);
	gain_2 = get_1D_gain(index_2, delta_2, eta_2);
  
	return (gain_1 + gain_2 - (delta_1 - eta_1) * (delta_2 - eta_2) * k);
  
};



// **********************************************************************************************************************************

inline void Texpectile_svm::compare_pair_of_indices(unsigned& best_index_1, unsigned& best_index_2, double& new_beta_1, double& new_gamma_1, double& new_beta_2, double& new_gamma_2, double& best_gain, unsigned index_1, unsigned index_2, bool& changed)
{
	double gain;
	double beta_temp_1;
	double beta_temp_2;
	double gamma_temp_1;
	double gamma_temp_2;
	
	gain = optimize_2D(index_1, index_2, beta_temp_1, beta_temp_2, gamma_temp_1, gamma_temp_2);

	if(gain > best_gain)
	{
		best_index_1 = index_1;
		best_index_2 = index_2;
		new_beta_1 = beta_temp_1;
		new_beta_2 = beta_temp_2;
		new_gamma_1= gamma_temp_1;
		new_gamma_2= gamma_temp_2;
		best_gain = gain;
		changed = true;
		
	}
	else
		changed = (false or changed);
	
};
// *******************************************************************************************************************************************************************************

inline void Texpectile_svm::inner_loop(unsigned start_index, unsigned stop_index, unsigned index_1, unsigned index_2, double delta_1, double delta_2, double eta_1, double eta_2, double& slack_sum_local, unsigned& best_index, double& best_gain)
{
	unsigned i;
	unsigned j;
	simdd__ delta_eta_1_simdd;
 	simdd__ delta_eta_2_simdd;
 	simdd__ delta_eta_kernel_sum_simdd;
	simdd__ tau_1_simdd;
	simdd__ tau_2_simdd;
	
	simdd__ label_simdd;
	simdd__ pos_clipp_value_simdd;
	simdd__ neg_clipp_value_simdd;
	simdd__ slack_simdd;
	simdd__ slack_pos_simdd;
	simdd__ slack_neg_simdd;
	simdd__ slack_sum_simdd;
	simdd__ slack_sum_pos_simdd;
	simdd__ slack_sum_neg_simdd;
	simdd__ half_over_C_tau_1_simdd;
	simdd__ gradient_beta_simdd;
	simdd__ gradient_gamma_simdd;
	
	simdd__ b_1_simdd;
	simdd__ b_1_reciprocal_simdd;
	simdd__ b_2_reciprocal_simdd;
	simdd__ c_value_simdd;
	simdd__ beta_simdd;
	simdd__ gamma_simdd;
	simdd__ delta_simdd;
	simdd__ eta_simdd;
	simdd__ gain_simdd;
	simdd__ gain_factor_1_simdd;
	simdd__ gain_factor_2_simdd;
	simdd__ best_gain_simdd;
	simdd__ best_index_simdd;
	simdd__ C_tau_magic_factor_1_simdd;
	simdd__ C_tau_magic_factor_2_simdd;
	
	
	delta_eta_1_simdd = assign_simdd(delta_1 - eta_1);
 	delta_eta_2_simdd = assign_simdd(delta_2 - eta_2);

	half_over_C_tau_1_simdd = assign_simdd(half_over_C_tau_1);
	slack_sum_pos_simdd = zero_simdd;
	slack_sum_neg_simdd = zero_simdd;
	
	b_1_simdd = assign_simdd(-b1);
	b_1_reciprocal_simdd = assign_simdd(reciprocal_b1);
	b_2_reciprocal_simdd = assign_simdd(-reciprocal_b2);
	C_tau_magic_factor_1_simdd = assign_simdd(-C_tau_magic_factor_1);
	C_tau_magic_factor_2_simdd = assign_simdd(-C_tau_magic_factor_2);
  	best_gain_simdd = assign_simdd(-1.0);
	best_index_simdd = zero_simdd;
	
	if (solver_clipp_value == 0.0)
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
		{
			cache_prefetch(index_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(beta_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gamma_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gradient_beta_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gradient_gamma_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(kernel_row1_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(kernel_row2_ALGD+i+32, PREFETCH_L1);
		
			for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
			{
				delta_eta_kernel_sum_simdd = fuse_mult_add_simdd(delta_eta_2_simdd, load_simdd(kernel_row2_ALGD+i+j), mult_simdd(delta_eta_1_simdd, load_simdd(kernel_row1_ALGD+i+j)));

				gradient_beta_simdd = sub_simdd(load_simdd(gradient_beta_ALGD+i+j), delta_eta_kernel_sum_simdd);
				gradient_gamma_simdd = add_simdd(load_simdd(gradient_gamma_ALGD+i+j), delta_eta_kernel_sum_simdd);

				store_simdd(gradient_beta_ALGD+i+j, gradient_beta_simdd);
				store_simdd(gradient_gamma_ALGD+i+j, gradient_gamma_simdd);

				beta_simdd = load_simdd(beta_ALGD+i+j);
				slack_simdd = fuse_mult_add_simdd(beta_simdd, half_over_C_tau_1_simdd, gradient_beta_simdd);
				slack_pos_simdd = pos_part_simdd(slack_simdd);
				slack_neg_simdd = sub_simdd(slack_pos_simdd, slack_simdd);

				slack_sum_pos_simdd = fuse_mult_add_simdd(slack_pos_simdd, slack_pos_simdd, slack_sum_pos_simdd);
				slack_sum_neg_simdd = fuse_mult_add_simdd(slack_neg_simdd, slack_neg_simdd, slack_sum_neg_simdd);

				gamma_simdd = load_simdd(gamma_ALGD+i+j);
				c_value_simdd = sub_simdd(gradient_beta_simdd, fuse_mult_add_simdd(b_1_simdd, beta_simdd, gamma_simdd)); 

				delta_simdd = fuse_mult_sub_simdd(b_1_reciprocal_simdd, pos_part_simdd(c_value_simdd), beta_simdd);
				gain_factor_1_simdd = fuse_mult_add_simdd(C_tau_magic_factor_1_simdd, delta_simdd, gradient_beta_simdd);

				eta_simdd = fuse_mult_sub_simdd(b_2_reciprocal_simdd, min_simdd(zero_simdd, c_value_simdd), gamma_simdd);
				gain_factor_2_simdd = fuse_mult_add_simdd(C_tau_magic_factor_2_simdd, eta_simdd, gradient_gamma_simdd);

				gain_simdd = fuse_mult_add_simdd(delta_simdd, gain_factor_1_simdd, fuse_mult_add_simdd(delta_simdd, eta_simdd, mult_simdd(eta_simdd, gain_factor_2_simdd)));

				get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+i+j), gain_simdd);
			}  
		} 
	else
	{
		pos_clipp_value_simdd = assign_simdd(transform_label(solver_clipp_value));
		neg_clipp_value_simdd = assign_simdd(transform_label(-solver_clipp_value));
		for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
		{
			cache_prefetch(index_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(beta_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gamma_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(training_label_transformed_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gradient_beta_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(gradient_gamma_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(kernel_row1_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(kernel_row2_ALGD+i+32, PREFETCH_L1);
		
			for(j=0; j<CACHELINE_STEP; j+=SIMD_WORD_SIZE)
			{
				delta_eta_kernel_sum_simdd = fuse_mult_add_simdd(delta_eta_2_simdd, load_simdd(kernel_row2_ALGD+i+j), mult_simdd(delta_eta_1_simdd, load_simdd(kernel_row1_ALGD+i+j)));

				gradient_beta_simdd = sub_simdd(load_simdd(gradient_beta_ALGD+i+j), delta_eta_kernel_sum_simdd);
				gradient_gamma_simdd = add_simdd(load_simdd(gradient_gamma_ALGD+i+j), delta_eta_kernel_sum_simdd);

				store_simdd(gradient_beta_ALGD+i+j, gradient_beta_simdd);
				store_simdd(gradient_gamma_ALGD+i+j, gradient_gamma_simdd);

				beta_simdd = load_simdd(beta_ALGD+i+j);
				label_simdd = load_simdd(training_label_transformed_ALGD+i+j);
				
				slack_simdd = sub_simdd(label_simdd, clipp_simdd(sub_simdd(label_simdd, fuse_mult_add_simdd(beta_simdd, half_over_C_tau_1_simdd, gradient_beta_simdd)), neg_clipp_value_simdd, pos_clipp_value_simdd));
				slack_pos_simdd = pos_part_simdd(slack_simdd);
				slack_neg_simdd = sub_simdd(slack_pos_simdd, slack_simdd);

				slack_sum_pos_simdd = fuse_mult_add_simdd(slack_pos_simdd, slack_pos_simdd, slack_sum_pos_simdd);
				slack_sum_neg_simdd = fuse_mult_add_simdd(slack_neg_simdd, slack_neg_simdd, slack_sum_neg_simdd);

				gamma_simdd = load_simdd(gamma_ALGD+i+j);
				c_value_simdd = sub_simdd(gradient_beta_simdd, fuse_mult_add_simdd(b_1_simdd, beta_simdd, gamma_simdd)); 

				delta_simdd = fuse_mult_sub_simdd(b_1_reciprocal_simdd, pos_part_simdd(c_value_simdd), beta_simdd);
				gain_factor_1_simdd = fuse_mult_add_simdd(C_tau_magic_factor_1_simdd, delta_simdd, gradient_beta_simdd);

				eta_simdd = fuse_mult_sub_simdd(b_2_reciprocal_simdd, min_simdd(zero_simdd, c_value_simdd), gamma_simdd);
				gain_factor_2_simdd = fuse_mult_add_simdd(C_tau_magic_factor_2_simdd, eta_simdd, gradient_gamma_simdd);

				gain_simdd = fuse_mult_add_simdd(delta_simdd, gain_factor_1_simdd, fuse_mult_add_simdd(delta_simdd, eta_simdd, mult_simdd(eta_simdd, gain_factor_2_simdd)));

				get_index_with_better_gain_simdd(best_index_simdd, best_gain_simdd, load_simdd(index_ALGD+i+j), gain_simdd);
			}  
		} 
	}
	
	
	tau_1_simdd = assign_simdd(tau);
	tau_2_simdd = assign_simdd(1.0 - tau);
	
	slack_sum_simdd = mult_simdd(tau_1_simdd, slack_sum_pos_simdd);
	slack_sum_simdd = fuse_mult_add_simdd(tau_2_simdd, slack_sum_neg_simdd, slack_sum_simdd);
 	slack_sum_local = slack_sum_local + reduce_sums_simdd(slack_sum_simdd);
	
 	argmax_simdd(best_index_simdd, best_gain_simdd, best_index, best_gain);
};
