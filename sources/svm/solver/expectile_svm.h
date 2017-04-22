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


#if !defined (EXPECTILE_SVM_H) 
	#define EXPECTILE_SVM_H


#include "sources/svm/solver/basic_svm.h"
#include "sources/svm/solver/basic_2D_svm.h"


//**********************************************************************************************************************************


class Texpectile_svm: public Tbasic_svm, Tbasic_2D_svm
{
	public:
		Texpectile_svm();
		~Texpectile_svm();
		void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		
		void load(Tkernel* training_kernel, Tkernel* validation_kernel);
		void initialize_new_lambda_line(Tsvm_train_val_info& train_val_info);
		
	protected:
		
		void initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info);
		void core_solver(Tsvm_train_val_info& train_val_info);
		
		void build_SV_list(Tsvm_train_val_info& train_val_info);
		void build_solution(Tsvm_train_val_info& train_val_info);
		void get_train_error(Tsvm_train_val_info& train_val_info);
		
		
		double* restrict__ beta_ALGD;
		double* restrict__ gamma_ALGD;
		double* restrict__ training_label_transformed_ALGD;
		double* restrict__ gradient_beta_ALGD;
		double* restrict__ gradient_gamma_ALGD;
		
		
	private:
		
		void init_zero();
		void init_keep();
		
		strict_inline__ void get_optimal_1D_direction(unsigned start_index, unsigned stop_index, unsigned& best_index, double& best_gain);
		
		strict_inline__ double get_1D_gain(unsigned best_index,double delta, double eta);
		strict_inline__ double optimize_2D(unsigned index_1, unsigned index_2, double& new_beta_1,double& new_beta_2, double& new_gamma_1, double& new_gamma_2);
		strict_inline__ double get_2D_gain(unsigned index_1, unsigned index_2, double delta_1, double delta_2, double eta_1, double eta_2, double k);
		strict_inline__ void compare_pair_of_indices(unsigned& best_index_1, unsigned& best_index_2, double& new_beta_1, double& new_gamma_1, double& new_beta_2, double& new_gamma_2, double& best_gain, unsigned index_1, unsigned index_2, bool& changed);

		strict_inline__ void inner_loop(unsigned start_index, unsigned stop_index, unsigned index_1, unsigned index_2, double delta_1, double delta_2, double eta_1, double eta_2, double& slack_sum_local, unsigned& best_index, double& best_gain);
		
		vector <double> beta_gamma_squared_sum_local;
		vector <double> beta_gamma_squared_sum_global;
		
		double tau;
		double b1;
		double b2;
		double reciprocal_b1;
		double reciprocal_b2;
		double tau_magic_factor;
		double half_over_C_tau_1;
		double half_over_C_tau_2;
		double C_tau_magic_factor_1;
		double C_tau_magic_factor_2;
		double C_tau_magic_factor_3;
		double C_tau_magic_factor_4;
		
		bool tau_initialized;
		
};


//**********************************************************************************************************************************

#include "sources/svm/solver/expectile_svm.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/expectile_svm.cpp"
#endif

#endif

