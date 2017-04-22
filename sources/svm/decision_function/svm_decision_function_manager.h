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


#if !defined (SVM_DECISION_FUNCTION_MANAGER_H)
	#define SVM_DECISION_FUNCTION_MANAGER_H


#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/system_support/thread_manager.h"
#include "sources/shared/kernel/kernel_control.h"
#include "sources/shared/decision_function/decision_function_manager.h"


#include "sources/svm/training_validation/svm_train_val_info.h"
#include "sources/svm/decision_function/svm_test_info.h"
#include "sources/svm/decision_function/svm_decision_function.h"
#include "sources/svm/decision_function/svm_decision_function_GPU_control.h"

//*********************************************************************************************************************************


class Tsvm_decision_function_manager: public Tdecision_function_manager<Tsvm_decision_function, Tsvm_train_val_info, Tsvm_test_info> 
{
	public:
		Tsvm_decision_function_manager();
		Tsvm_decision_function_manager(const Tsvm_decision_function_manager& svm_decision_function_manager);
		Tsvm_decision_function_manager(const Tworking_set_manager& working_set_manager, const Tdataset& training_set, unsigned folds);
		~Tsvm_decision_function_manager();

		Tsvm_decision_function_manager& operator = (const Tsvm_decision_function_manager& svm_decision_function_manager);
		
		void clear();
		void replace_kernel_control(const Tkernel_control& new_kernel_control);
		void read_hierarchical_kernel_info_from_df_file_if_possible(unsigned task = 0, unsigned cell = 0);

		Tsvm_decision_function get_decision_function(unsigned task, unsigned cell, unsigned fold);

	protected:
		virtual void init_internal();
		virtual void clear_internal();
		virtual void setup_internal(const Tvote_control& vote_control, const Tparallel_control& parallel_control);
		
		void copy(const Tsvm_decision_function_manager& svm_decision_function_manager);
		
		unsigned get_thread_position();
		unsigned get_pre_thread_position();
		void compute_kernel_row(unsigned test_sample_number, unsigned ws_number, vector <bool>& SVs_computed);
		void clear_kernel_row_flags(unsigned test_sample_number, vector <bool>& SVs_computed, vector <bool>& pre_SVs_computed);

		
		template <class float_type> void setup_GPU(Tsvm_decision_function_GPU_control<float_type>* GPU_control, Tdataset& test_set_chunk, bool sparse_evaluation);
		template <class float_type> void clean_GPU(Tsvm_decision_function_GPU_control<float_type>* GPU_control);
		
		void convert_to_hierarchical_data_sets();
		
		void make_evaluations();
		
		unsigned size_of_largest_SV_with_gamma();
		unsigned size_of_largest_decision_function();
		
		
		unsigned kernel_type;
		double* kernel_eval;
		double* pre_kernel_eval;
		
		vector <double> gamma_list;
		vector <unsigned> gamma_indices;
		
		vector <unsigned> SVs;
		vector <vector <unsigned> > SVs_with_gamma;
		vector <vector <unsigned> > SVs_in_working_set;
		vector <vector <vector <unsigned> > > SVs_with_gamma_in_working_set;
		
		
// 	HIERARCHICAL KERNEL DEVELOPMENT
		
		unsigned full_kernel_type;
		double weights_square_sum;
		bool hierarchical_kernel_flag;
		Tkernel_control kernel_control;
		
		vector <Tdataset> hierarchical_training_set;
		vector <Tdataset> hierarchical_test_set;
		
	private:
		void find_gammas();
		void find_SVs(vector <unsigned>& SVs_list, vector <vector <unsigned> >& SVs_in_ws_list, double gamma = -1.0);
		
		void compute_pre_kernel_row(unsigned test_sample_number, unsigned ws_number, vector <bool>& SVs_computed);
		void inline adjust_counters(unsigned& small_counter, unsigned& large_counter, unsigned unit);
		void copy_internal_kernel_parameters_from_kernel_control();
		
		vector <double> kernel_init_times;
		vector <double> pre_kernel_times;
		vector <double> kernel_times;
		
		vector <unsigned> pre_kernel_eval_counter_small;
		vector <unsigned> pre_kernel_tries_counter_small;
		
		vector <unsigned> pre_kernel_eval_counter_large;
		vector <unsigned> pre_kernel_tries_counter_large;
		
		vector <unsigned> kernel_eval_counter_small;
		vector <unsigned> kernel_tries_counter_small;
		
		vector <unsigned> kernel_eval_counter_large;
		vector <unsigned> kernel_tries_counter_large;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/decision_function/svm_decision_function_manager.cpp"
#endif


#include "sources/svm/decision_function/svm_decision_function_manager.ins.cpp"


#endif
