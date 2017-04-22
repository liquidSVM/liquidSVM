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


#if !defined (SVM_MANAGER_CONTROL_H)
	#define SVM_MANAGER_CONTROL_H
 


#include "sources/shared/training_validation/cv_control.h"
#include "sources/shared/training_validation/grid_control.h"
#include "sources/shared/training_validation/working_set_manager.h"
#include "sources/shared/decision_function/decision_function_manager.h"

#include "sources/svm/solver/svm_solver_control.h"
#include "sources/svm/training_validation/svm_train_val_info.h"


//**********************************************************************************************************************************


class Ttrain_control
{
	public:
		Ttrain_control();
		
		Tgrid_control grid_control;
		Tfold_control fold_control;
		Tworking_set_control working_set_control;
		
		Tparallel_control parallel_control;
		Tsvm_solver_control solver_control;

		bool full_search;
		unsigned max_number_of_increases;
		unsigned max_number_of_worse_gammas;
		
		bool scale_data;
		
		bool store_logs_internally;
		bool store_solutions_internally;
		
		string write_log_train_filename;
		string write_aux_train_filename;
		string write_sol_train_filename;
		
		string summary_log_filename;
};




//**********************************************************************************************************************************


class Tselect_control
{
	public:
		Tselect_control();
		
		void copy_to_cv_control(Tcv_control& cv_control);
		void copy_from_cv_control(const Tcv_control& cv_control);
		
		
		unsigned select_method;
		bool use_stored_logs;
		bool use_stored_solution;
		
		bool append_decision_functions;
		bool store_decision_functions_internally;
		
		bool npl;
		int npl_class;
		double npl_constraint;
		unsigned weight_number;
		
		string read_log_train_filename;
		string read_aux_train_filename;
		string read_sol_train_filename;
		
		string write_log_select_filename;
		string write_sol_select_filename;
		
		string summary_log_filename;
};



//**********************************************************************************************************************************


class Ttest_control
{
	public:
		Ttest_control();
		
		Tvote_control vote_control;
		Tloss_control loss_control;
		Tparallel_control parallel_control;
		
		unsigned max_used_RAM_in_MB;

		string read_sol_select_filename;
		string write_log_test_filename;
		
		string summary_log_filename;
};



//**********************************************************************************************************************************


class Tsvm_full_train_info
{
	public:
		double full_time;
		double file_time;
		double train_time;
		
		Tsvm_train_val_info train_val_info_log;
};



//**********************************************************************************************************************************


class Tsvm_full_test_info
{
	public:
		double full_time;
		double file_time;
		double test_time;
		
		unsigned number_of_tasks;
		unsigned number_of_all_tasks;
		
		vector <Tsvm_train_val_info> train_val_info;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/training_validation/svm_manager_control.cpp"
#endif

#endif
