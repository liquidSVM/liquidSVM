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


#if !defined (HINGE_SVM_H) 
	#define HINGE_SVM_H


#include "sources/svm/solver/basic_svm.h"


//**********************************************************************************************************************************


class Thinge_svm: public Tbasic_svm
{
	public:
		Thinge_svm();
		virtual ~Thinge_svm();
		
		void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		void load(Tkernel* training_kernel, Tkernel* validation_kernel);
		void clear();
		
		void initialize_new_weight_and_lambda_line(Tsvm_train_val_info& train_val_info);
		
	protected:
		void initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info);
		void get_val_error(Tsvm_train_val_info& train_val_info);
		void get_train_error(Tsvm_train_val_info& train_val_info);

		void build_solution(Tsvm_train_val_info& train_val_info);
		void build_bSV_list(Tsvm_train_val_info& train_val_info);
		
		strict_inline__ simdd__ clipp_02_simdd(simdd__ arg_simdd);
		strict_inline__ double clipp_0max(double x, double max); 

		inline void compute_gap_from_scratch();

		vector <unsigned> bSV_list;
		
		unsigned neg_train_size;
		unsigned pos_train_size;
		
	private:
		inline void zero_box(unsigned& init_iterations, unsigned& val_iterations);
		inline void full_box(Tsvm_train_val_info& train_val_info);
		inline void keep_box(unsigned& init_iterations, unsigned& val_iterations);
		inline void expand_box(unsigned& init_iterations, unsigned& val_iterations);
		inline void shrink_box(unsigned& init_iterations, unsigned& val_iterations);
		inline void scale_box(double factor, unsigned& init_iterations, unsigned& val_updates);
		
		inline void scale_predictions(double factor);
		void init_full_predictions_on_GPU(Tsvm_train_val_info train_val_info);
		void init_neg_and_pos_predictions_on_GPU(double* predictions_init_GPU, double sign);
		
		void count_labels(unsigned& neg_sample_no, unsigned& pos_sample_no, double* labels, unsigned size);
		
		strict_inline__ void add_to_slack_sum_CL(simdd__& slack_sum_simdd, double* gradient, double* weight);
		strict_inline__ void add_to_gradient(simdd__ factor_simdd, double* restrict__ kernel_row_ALGD);
		
		double* restrict__ old_alpha_ALGD;
		double* restrict__ old_weights_ALGD;
		
		double* restrict__ prediction_init_neg_ALGD;
		double* restrict__ prediction_init_pos_ALGD;
		
		vector <unsigned> uSV_list;
		vector <unsigned> nuSV_list;
		vector <unsigned> nbSV_list;
		
		unsigned neg_val_size;
		unsigned pos_val_size;

		#ifdef  COMPILE_WITH_CUDA__
			vector <double*> prediction_init_neg_GPU;
			vector <double*> prediction_init_pos_GPU;
		#endif
};


//**********************************************************************************************************************************

#include "sources/svm/solver/hinge_svm.ins.cpp"


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/hinge_svm.cpp"
#endif

#endif

