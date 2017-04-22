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


#if !defined (LEAST_SQUARES_SVM_H) 
	#define LEAST_SQUARES_SVM_H


#include "sources/svm/solver/basic_svm.h"
#include "sources/svm/solver/basic_2D_svm.h"
#include "sources/svm/solver/generic_xD_ancestor.h"





//**********************************************************************************************************************************


class Tleast_squares_svm_generic_ancestor: public Tbasic_svm, public Tsvm_xD_ancestor
{
	protected:
		strict_inline__ void loop_with_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ neg_clipp_simdd, simdd__ pos_clipp_simdd, simdd__ C_simdd, simdd__& slack_sum_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd, double* restrict__ kernel_row1_ALGD, double* restrict__ kernel_row2_ALGD);
		strict_inline__ void loop_without_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ C_simdd, simdd__& slack_sum_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd, double* restrict__ kernel_row1_ALGD, double* restrict__ kernel_row2_ALGD);
		
		
		strict_inline__ double compute_slack_sum(unsigned start_index, unsigned stop_index);
		strict_inline__ void add_to_slack_sum_simdd(simdd__& slack_sum_simdd, simdd__ gradient_simdd, simdd__ alpha_simdd, simdd__ C_simdd);
		strict_inline__ void add_to_clipped_slack_sum_simdd(simdd__& slack_sum_simdd, simdd__ gradient_simdd, simdd__ label_simdd, simdd__ neg_clipp_simdd, simdd__ pos_clipp_simdd, simdd__ alpha_simdd, simdd__ C_simdd);
				
		
		
		
		double half_over_C;
		double C_magic_factor_1;
		double C_magic_factor_2;
		double C_magic_factor_3;
		double C_magic_factor_4;
};



//**********************************************************************************************************************************

#if defined (Tsvm_2D_solver_generic_base_name) 
	#undef Tsvm_2D_solver_generic_base_name
#endif
#define Tsvm_2D_solver_generic_base_name Tleast_squares_svm_generic_base

#if defined (Tsvm_2D_solver_generic_ancestor_name) 
	#undef Tsvm_2D_solver_generic_ancestor_name
#endif
#define Tsvm_2D_solver_generic_ancestor_name Tleast_squares_svm_generic_ancestor


#include "sources/svm/solver/generic_2D_svm.h"


//**********************************************************************************************************************************


class Tleast_squares_svm: public Tleast_squares_svm_generic_base
{
	public:
		Tleast_squares_svm();
		~Tleast_squares_svm();
		void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		void load(Tkernel* training_kernel, Tkernel* validation_kernel);

	protected:
		void initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info);
		void core_solver(Tsvm_train_val_info& train_val_info);
				
		void build_solution(Tsvm_train_val_info& train_val_info);
		void get_train_error(Tsvm_train_val_info& train_val_info);


		vector <double> alpha_squared_sum_local;
		vector <double> alpha_squared_sum_global;
		
	private:
		void init_zero();
		void init_keep();
};


//**********************************************************************************************************************************

#include "sources/svm/solver/least_squares_svm.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/least_squares_svm.cpp"
#endif

#endif

