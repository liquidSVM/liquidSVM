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


class Tsvm_2D_solver_generic_base_name: public Tsvm_2D_solver_generic_ancestor_name
{
	protected:
		strict_inline__ void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);

		strict_inline__ void core_solver_generic_part(Tsvm_train_val_info& train_val_info);
		strict_inline__ void prepare_core_solver_generic_part(Tsvm_train_val_info& train_val_info);
		strict_inline__ void initial_iteration(unsigned& best_index_1, unsigned& best_index_2, double& new_alpha_1, double& new_alpha_2);
		
		strict_inline__ void get_optimal_1D_direction(unsigned start_index, unsigned stop_index, unsigned& best_index, double& best_gain);
		
		strict_inline__ void compare_pair_of_indices(unsigned& best_index_1, unsigned& best_index_2, double& new_alpha_1,double& new_alpha_2, double& best_gain, unsigned index_1, unsigned index_2, bool& changed);
		
		
// 		The following routines need to be implemented for all classes derived from this class.
// 		All functions that are called by these implementations, need to be declared and defined in 
// 		Tsvm_2D_solver_generic_ancestor_name
		
		strict_inline__ void trivial_solution();
		strict_inline__ void prepare_core_solver(Tsvm_train_val_info& train_val_info);
		
		strict_inline__ void get_optimal_1D_CL(unsigned index, simdd__& best_gain_simdd, simdd__& best_index_simdd);
		strict_inline__ void inner_loop(unsigned start_index, unsigned end_index, double delta_1, double delta_2, unsigned best_index_1, unsigned best_index_2, unsigned& new_best_index, double& best_gain, double& slack_sum);
		
		strict_inline__ double optimize_2D(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double weight_1, double weight_2, double label_1, double label_2, double& new_alpha_1,double& new_alpha_2, double K_ij, bool same_indices);
		
		strict_inline__ void update_norm_etc(double delta_1, double delta_2, unsigned index_1, unsigned index_2, double& norm_etc);
		
		strict_inline__ void set_NNs_search(unsigned& index_1, unsigned& index_2, double& alpha_1, double& alpha_2, unsigned iterations);
		

		
		bool NNs_search;
		unsigned inner_optimizations;
};

//**********************************************************************************************************************************


#include "sources/svm/solver/generic_2D_svm.ins.cpp"
