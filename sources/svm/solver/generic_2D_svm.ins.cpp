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


inline void Tsvm_2D_solver_generic_base_name::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{	
	if (solver_control.wss_method == DEFAULT_WSS_METHOD)
		solver_control.wss_method = USE_NNs;
	if (solver_control.kernel_control_train.kNNs == DEFAULT_NN)
	{
		if (solver_control.wss_method == USE_NNs)
			solver_control.kernel_control_train.kNNs = 10;
		else
			solver_control.kernel_control_train.kNNs = 0;
	}
}




//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::core_solver_generic_part(Tsvm_train_val_info& train_val_info)
{
	unsigned start_index;
	unsigned stop_index;
	unsigned best_index_1;
	unsigned best_index_2;
	unsigned new_best_index_1;
	unsigned new_best_index_2;
	unsigned temp_index_1;
	unsigned temp_index_2;
	double new_alpha_1;
	double new_alpha_2;
	double delta_1;
	double delta_2;
	double best_gain;
	double best_gain_1;
	double best_gain_2;
	unsigned k;
	bool changed;
	unsigned thread_id;
	
	new_alpha_1 = 0.0;
	new_alpha_2 = 0.0;

	thread_id = get_thread_id();
	slack_sum_local[thread_id] = slack_sum_global[thread_id];
	
	
	if (is_first_team_member() == true)
	{
		prepare_core_solver(train_val_info);
		prepare_core_solver_generic_part(train_val_info);
		
		if (training_set_size <= CACHELINE_STEP)
			trivial_solution();

		best_index_1 = 0;
		best_index_2 = 0;
		
		if (primal_dual_gap[thread_id] > stop_eps)
			initial_iteration(best_index_1, best_index_2, new_alpha_1, new_alpha_2);

		while (primal_dual_gap[thread_id] > stop_eps)
		{
			set_NNs_search(best_index_1, best_index_2, new_alpha_1, new_alpha_2, train_val_info.train_iterations);
			
			delta_1 = new_alpha_1 - alpha_ALGD[best_index_1];
			delta_2 = new_alpha_2 - alpha_ALGD[best_index_2];

			alpha_ALGD[best_index_1] = new_alpha_1;
			alpha_ALGD[best_index_2] = new_alpha_2;

			slack_sum_local[thread_id] = 0.0;
			update_norm_etc(delta_1, delta_2, best_index_1, best_index_2, norm_etc_global[thread_id]);
			
			get_aligned_chunk(training_set_size, 2, 0, start_index, stop_index);
			inner_loop(start_index, stop_index, delta_1, delta_2, best_index_1, best_index_2, new_best_index_1, best_gain_1, slack_sum_local[thread_id]);
			get_aligned_chunk(training_set_size, 2, 1, start_index, stop_index);
			inner_loop(start_index, stop_index, delta_1, delta_2, best_index_1, best_index_2, new_best_index_2, best_gain_2, slack_sum_local[thread_id]);
			order_indices(new_best_index_1, best_gain_1, new_best_index_2, best_gain_2);

			primal_dual_gap[thread_id] = slack_sum_local[thread_id] - norm_etc_global[thread_id];

		
			best_gain = optimize_2D(alpha_ALGD[new_best_index_1], alpha_ALGD[new_best_index_2], gradient_ALGD[new_best_index_1], gradient_ALGD[new_best_index_2], weight_ALGD[new_best_index_1], weight_ALGD[new_best_index_2], training_label_ALGD[best_index_1], training_label_ALGD[best_index_2], new_alpha_1, new_alpha_2, training_kernel->entry(new_best_index_1, new_best_index_2), new_best_index_1 == new_best_index_2);


			temp_index_1 = new_best_index_1;
			temp_index_2 = new_best_index_2;
			
			changed = false;
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_alpha_1, new_alpha_2, best_gain, temp_index_1, best_index_1, changed);
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_alpha_1, new_alpha_2, best_gain, temp_index_1, best_index_2, changed);
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_alpha_1, new_alpha_2, best_gain, temp_index_2, best_index_2, changed);
			compare_pair_of_indices(new_best_index_1, new_best_index_2, new_alpha_1, new_alpha_2, best_gain, temp_index_2, best_index_1, changed);

			kernel_row1_ALGD = training_kernel->row(new_best_index_1);
			if ((NNs_search == true) and (solver_ctrl.wss_method == USE_NNs))
			{
				train_val_info.tries_2D++;
				changed = false;
				
				if (kNN_list[new_best_index_1].size() == 0)
					kNN_list[new_best_index_1] = training_kernel->get_kNNs(new_best_index_1);

				for (k=0; k<kNN_list[new_best_index_1].size(); k++)
					compare_pair_of_indices(new_best_index_1, new_best_index_2, new_alpha_1, new_alpha_2, best_gain, new_best_index_1, kNN_list[new_best_index_1][k], changed);
				
				if (changed == true)
					train_val_info.hits_2D++;
			}
			kernel_row2_ALGD = training_kernel->row(new_best_index_2);

			best_index_1 = new_best_index_1;
			best_index_2 = new_best_index_2;

			train_val_info.train_iterations++;
			train_val_info.gradient_updates = train_val_info.gradient_updates + 2;
		}
	
		build_SV_list(train_val_info);
		
		MM_CACHELINE_FLUSH(&slack_sum_local[0]);
	}
	
	sync_threads();
	slack_sum_global[thread_id] = slack_sum_local[0];
}



//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::prepare_core_solver_generic_part(Tsvm_train_val_info& train_val_info)
{
	train_val_info.train_iterations = 0;
	train_val_info.gradient_updates = 0;
	train_val_info.tries_2D = 0;
	train_val_info.hits_2D = 0;

	NNs_search = false;
	inner_optimizations = 0;
}




//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::initial_iteration(unsigned& best_index_1, unsigned& best_index_2, double& new_alpha_1, double& new_alpha_2)
{
	unsigned start_index;
	unsigned stop_index;
	double best_gain_1;
	double best_gain_2;
	
	get_aligned_chunk(training_set_size, 2, 0, start_index, stop_index);
	get_optimal_1D_direction(start_index, stop_index, best_index_1, best_gain_1);

	get_aligned_chunk(training_set_size, 2, 1, start_index, stop_index);
	get_optimal_1D_direction(start_index, stop_index, best_index_2, best_gain_2);

	order_indices(best_index_1, best_gain_1, best_index_2, best_gain_2);
	
	kernel_row1_ALGD = training_kernel->row(best_index_1);
	kernel_row2_ALGD = training_kernel->row(best_index_2);
	
	
	optimize_2D(alpha_ALGD[best_index_1], alpha_ALGD[best_index_2], gradient_ALGD[best_index_1], gradient_ALGD[best_index_2], weight_ALGD[best_index_1], weight_ALGD[best_index_2], training_label_ALGD[best_index_1], training_label_ALGD[best_index_2], new_alpha_1, new_alpha_2, training_kernel->entry(best_index_1, best_index_2), best_index_1 == best_index_2);
}


//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::get_optimal_1D_direction(unsigned start_index, unsigned stop_index, unsigned& best_index, double& best_gain)
{
	unsigned i;
	simdd__ best_gain_simdd;
	simdd__ best_index_simdd;


	best_gain_simdd = assign_simdd(-1.0);
	best_index_simdd = assign_simdd(0.0);
	for (i=start_index; i+CACHELINE_STEP <= stop_index; i+=CACHELINE_STEP)
		get_optimal_1D_CL(i, best_gain_simdd, best_index_simdd);

	argmax_simdd(best_index_simdd, best_gain_simdd, best_index, best_gain);
}




//**********************************************************************************************************************************

inline void Tsvm_2D_solver_generic_base_name::compare_pair_of_indices(unsigned& best_index_1, unsigned& best_index_2, double& new_alpha_1, double& new_alpha_2, double& best_gain, unsigned index_1, unsigned index_2, bool& changed)
{
	double gain;
	double alpha_1;
	double alpha_2;


	gain = optimize_2D(alpha_ALGD[index_1], alpha_ALGD[index_2], gradient_ALGD[index_1], gradient_ALGD[index_2], weight_ALGD[index_1], weight_ALGD[index_2], training_label_ALGD[best_index_1], training_label_ALGD[best_index_2], alpha_1, alpha_2, training_kernel->entry(index_1, index_2), index_1 == index_2);

	if (gain > best_gain)
	{
		best_index_1 = index_1;
		best_index_2 = index_2;
		new_alpha_1 = alpha_1;
		new_alpha_2 = alpha_2;
		best_gain = gain;
		changed = true;
	}
	else 
		changed = (false or changed);
}







