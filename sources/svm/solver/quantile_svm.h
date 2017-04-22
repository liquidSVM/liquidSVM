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


#if !defined (QUANTILE_SVM_H) 
	#define QUANTILE_SVM_H



#include "sources/svm/solver/basic_svm.h"
#include "sources/svm/solver/basic_2D_svm.h"
#include "sources/svm/solver/generic_xD_ancestor.h"


//**********************************************************************************************************************************


class Tquantile_svm_generic_ancestor: public Tbasic_svm, public Tsvm_xD_ancestor
{
	protected:
		strict_inline__ simdd__ get_gain_simdd(simdd__ gradient_simdd, simdd__ low_weight_simdd, simdd__ up_weight_simdd, simdd__ alpha_simdd);
		
		strict_inline__ void loop_without_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__& neg_slack_sum_simdd, simdd__& pos_slack_sum_simdd, simdd__ low_weight_simdd, simdd__ up_weight_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd);
		strict_inline__ void loop_with_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__& neg_slack_sum_simdd, simdd__& pos_slack_sum_simdd, simdd__  neg_clipp_value_simdd, simdd__ pos_clipp_value_simdd, simdd__ low_weight_simdd, simdd__ up_weight_simdd, simdd__& best_gain_simdd, simdd__& new_best_index_simdd);
		
		
		strict_inline__ double clipp_to_box(double alpha);
		strict_inline__ unsigned constraint_segment(double alpha);
		strict_inline__ double gain_2D(double gradient_1, double gradient_2, double delta_1, double delta_2, double K_ij);
		strict_inline__ double optimize_2D_corner(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double border_1, double border_2, double& new_alpha_1, double& new_alpha_2, double K_ij);
		
		double tau;
		double up_weight;
		double low_weight;
		
		double* restrict__ training_label_transformed_ALGD;
};



//**********************************************************************************************************************************

#if defined (Tsvm_2D_solver_generic_base_name) 
	#undef Tsvm_2D_solver_generic_base_name
#endif
#define Tsvm_2D_solver_generic_base_name Tquantile_svm_generic_base

#if defined (Tsvm_2D_solver_generic_ancestor_name) 
	#undef Tsvm_2D_solver_generic_ancestor_name
#endif
#define Tsvm_2D_solver_generic_ancestor_name Tquantile_svm_generic_ancestor

#include "sources/svm/solver/generic_2D_svm.h"


//**********************************************************************************************************************************


class Tquantile_svm: public Tquantile_svm_generic_base
{
	public:
		Tquantile_svm();
		~Tquantile_svm();
		virtual void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		void load(Tkernel* training_kernel, Tkernel* validation_kernel);
		void initialize_new_lambda_line(Tsvm_train_val_info& train_val_info);

	protected:
		void initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info);
		void core_solver(Tsvm_train_val_info& train_val_info);
		
		void build_solution(Tsvm_train_val_info& train_val_info);
		void get_train_error(Tsvm_train_val_info& train_val_info);
		
		
		vector <unsigned> bSV_list;
		
		
	private:
		strict_inline__ void add_to_gradient(simdd__ factor_simdd, double* restrict__ kernel_row_ALGD);
		
		void init_zero(unsigned& init_iterations, unsigned& val_iterations);
		void init_keep(unsigned& init_iterations, unsigned& val_iterations);
		void scale_box(double factor, unsigned& updates, unsigned& val_iterations);
		void expand_box(unsigned& init_iterations, unsigned& val_iterations);
		void shrink_box(unsigned& init_iterations, unsigned& val_iterations);
	
		strict_inline__ double compute_slack_sum();
		strict_inline__ void compute_norm_etc();
		
		bool tau_initialized;
		
		vector <unsigned> uSV_list;
		vector <unsigned> up_SV_list;
		vector <unsigned> low_SV_list;
		vector <unsigned> lown_SV_list;
		vector <unsigned> upn_SV_list;
		
		double* restrict__ old_alpha_ALGD;
		double up_weight_old;
		double low_weight_old;
};


//**********************************************************************************************************************************

#include "sources/svm/solver/quantile_svm.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/quantile_svm.cpp"
#endif

#endif

