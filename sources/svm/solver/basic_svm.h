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


#if !defined (BASIC_SVM_H)
	#define BASIC_SVM_H


#include "sources/shared/solver/solver.h"
#include "sources/shared/basic_functions/flush_print.h"


#include "sources/svm/training_validation/svm_train_val_info.h"
#include "sources/svm/solver/svm_solver_control.h"
#include "sources/svm/decision_function/svm_solution.h"




//**********************************************************************************************************************************


class Tbasic_svm: public Tsolver<Tsvm_solution, Tsvm_train_val_info, Tsvm_solver_control>
{
	public:
		Tbasic_svm();
		virtual ~Tbasic_svm();
		
		virtual void clear();
		virtual void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		virtual void load(Tkernel* training_kernel, Tkernel* validation_kernel);
		
		virtual void initialize_new_lambda_line(Tsvm_train_val_info& train_val_info);
		virtual void initialize_new_weight_and_lambda_line(Tsvm_train_val_info& train_val_info);
		void run_solver(Tsvm_train_val_info& train_val_info, Tsvm_solution& solution);


	protected:
		virtual void clear_on_GPU();
		virtual void reserve_on_GPU(){};
		
		inline double transform_label(double label);
		inline double inverse_transform_label(double label);
		
		virtual void initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info) = 0;
		virtual void core_solver(Tsvm_train_val_info& train_val_info) = 0;
	
		virtual void build_SV_list(Tsvm_train_val_info& train_val_info);
		virtual void build_solution(Tsvm_train_val_info& train_val_info) = 0;
		
		virtual void get_val_error(Tsvm_train_val_info& train_val_info);
		virtual void get_train_error(Tsvm_train_val_info& train_val_info) = 0;
		inline void push_back_update(double delta, unsigned index, unsigned* counter = NULL);
		
		
		double* restrict__ index_ALGD;
		double* restrict__ alpha_ALGD;
		double* restrict__ gradient_ALGD;
		double* restrict__ weight_ALGD;
		
		double* restrict__ kernel_row1_ALGD;
		double* restrict__ kernel_row2_ALGD;
		
		double* restrict__ training_label_ALGD;
		double* restrict__ validation_label_ALGD;
		
		double C_old;
		double C_current;
		double solver_clipp_value;
		double validation_clipp_value;
		
		double label_offset;
		double label_spread;
		
		bool classification_data;
		double min_label;
		double max_label;
		
		vector <double> primal_dual_gap;
		vector <double> norm_etc_local;
		vector <double> norm_etc_global;
		vector <double> slack_sum_local;
		vector <double> slack_sum_global;

		vector <unsigned> SV_list;
		Tsvm_solution solution_old;
		Tsvm_solution solution_current;
		
		vector <Tsubset_info> kNN_list;


		double* coefficient_delta;
		unsigned* coefficient_changed;
		double* restrict__ prediction_ALGD;
		#ifdef  COMPILE_WITH_CUDA__
			vector <double*> validation_kernel_GPU;
			vector <Tkernel_control_GPU> kernel_control_GPU;

			vector <double*> coefficient_delta_GPU;
			vector <unsigned*> coefficient_changed_GPU;
			vector <double*> prediction_GPU;
		#endif
		unsigned number_coefficients_changed;

	private:
		void compute_val_predictions(unsigned& val_updates);
		void evaluate_val_predictions_on_GPU();
};


//**********************************************************************************************************************************

#include "sources/svm/solver/basic_svm.ins.cpp"


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/basic_svm.cpp"
#endif



#endif


