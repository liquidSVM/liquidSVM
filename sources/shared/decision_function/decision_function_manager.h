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


#if !defined (DECISION_FUNCTION_MANAGER_H)
	#define DECISION_FUNCTION_MANAGER_H

 

#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/basic_types/dataset_info.h"
#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/system_support/full_64bit_support.h"
#include "sources/shared/system_support/thread_manager_active.h"
#include "sources/shared/decision_function/test_info.h"
#include "sources/shared/training_validation/train_val_info.h"
#include "sources/shared/training_validation/working_set_manager.h"


#include <vector>

//**********************************************************************************************************************************



enum VOTE_SCENARIOS {VOTE_CLASSIFICATION, VOTE_REGRESSION, VOTE_NPL, VOTE_SCENARIOS_MAX};


//**********************************************************************************************************************************


class Tvote_control
{
	public:
		Tvote_control();
		
		bool weighted_folds;
		bool loss_weights_are_set;
		
		unsigned scenario;
		int npl_class;
};


//**********************************************************************************************************************************

template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type> class Tdecision_function_manager: protected Tthread_manager_active
{
	public:
		Tdecision_function_manager();
		Tdecision_function_manager(const Tworking_set_manager& working_set_manager, const Tdataset& training_set, unsigned folds);
		~Tdecision_function_manager();
		
		void clear();
		void write_to_file(FILE* fp) const;
		void read_from_file(FILE* fp, const Tdataset& training_set);
		void push_back(const Tdecision_function_manager& new_decision_function_manager);
		void replace_decision_function(unsigned task, unsigned cell, unsigned fold, const Tdecision_function_type& new_decision_function);
		Tdecision_function_manager& operator = (const Tdecision_function_manager& decision_function_manager);
		
		inline unsigned size() const;
		inline unsigned folds() const;
		inline unsigned number_of_all_tasks() const;
		inline Tworking_set_manager get_working_set_manager() const;
		unsigned get_max_test_set_size(unsigned allowed_RAM_in_MB) const;
		
		vector <double> get_predictions_for_task(unsigned task) const;
		vector <double> get_predictions_for_test_sample(unsigned i) const;
		vector <Ttrain_val_info_type> compute_errors(Tloss_control loss_control, bool use_weights_from_training);
		void make_predictions(const Tdataset& test_set, const Tvote_control& vote_control, const Tparallel_control& parallel_control, Ttest_info_type& test_info);
		

	protected:
		void copy(const Tdecision_function_manager& decision_function_manager);
		void construct(const Tworking_set_manager& working_set_manager, const Tdataset& training_set, unsigned folds);
		void prepare_for_making_predictions(const Tdataset& test_set, const Tvote_control& vote_control, const Tparallel_control& parallel_control);
		void convert_evaluations_to_predictions();
		
		void reduce_number_of_decision_functions(unsigned new_size);
		void check_task(unsigned task) const;
		void check_cell(unsigned task, unsigned cell) const;

		virtual void init_internal(){};
		virtual void clear_internal(){};
		virtual void setup_internal(const Tvote_control& vote_control, const Tparallel_control& parallel_control) = 0;
				
		virtual void thread_entry();
		virtual void make_evaluations() = 0;
		
		inline unsigned decision_function_number(unsigned task, unsigned cell, unsigned fold) const;
		inline size_type_double_vector__ evaluation_position(unsigned test_sample_number, unsigned decision_function_number) const;
		
	
		vector <double> weights;
		vector <double> evaluations;
		vector <double> predictions;
		
		vector <vector <vector <unsigned> > > cell_number_test;
		vector <vector <vector <unsigned> > > cell_number_train;
		vector <Tdecision_function_type> decision_functions;
		
		Tdataset test_set;
		Tdataset training_set;
		
		Ttest_info_type test_info;
		
		Tvote_control vote_control;
		Tdataset_info test_set_info;
		Tdataset_info training_set_info;
		Tworking_set_manager working_set_manager;

		bool new_team_size;
		bool new_training_set;
		bool new_decision_functions;
		unsigned old_team_size;

	private:
		void init();
		void check_integrity();
		void compute_weights();
		void reserve(const Tdataset& training_set, unsigned folds);
		void setup(const Tvote_control& vote_control, const Tparallel_control& parallel_control);


		double vote(unsigned task, unsigned test_sample_number);
		double convert_class_probability_to_class(unsigned task, double class_probability);
		double compute_error_for_task(unsigned task, Tloss_control loss_control, bool use_weights_from_training);
		double compute_AvA_error_for_task(unsigned task);
		double compute_OvA_error_for_task(unsigned task);
		Ttrain_val_info_type compute_two_class_error_for_task(Tloss_control loss_control, unsigned task);
		Ttrain_val_info_type compute_NPL_error_for_task(Tloss_control loss_control, unsigned task, int npl_class);
		
		inline unsigned prediction_position(unsigned test_sample_number, unsigned task) const;
		inline unsigned get_task_offset() const;
		
		void make_final_predictions_bootstrap(unsigned task_offset);
		void make_final_predictions_average(unsigned task_offset);
		void make_final_predictions_most(unsigned task_offset);
		void make_final_predictions_best(unsigned task_offset);
		
		
		bool untouched;
		
		unsigned all_tasks;
		unsigned number_of_folds;
		
		Tdataset cover_dataset;
		
		vector <double> default_labels;
		vector <vector <char> > ties;
		
		double thread_start_time;
		vector <double> thread_start_times;
		vector <double> thread_stop_times;
};



//*********************************************************************************************************************************


#include "sources/shared/decision_function/decision_function_manager.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/decision_function/decision_function_manager.cpp"
#endif

#endif
