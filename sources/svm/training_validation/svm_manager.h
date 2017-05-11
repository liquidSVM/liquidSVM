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


#if !defined (SVM_MANAGER_H)
	#define SVM_MANAGER_H
 
 
 
#include "sources/shared/training_validation/working_set_manager.h"

#include "sources/svm/training_validation/svm_manager_control.h"
#include "sources/svm/training_validation/svm_cv_manager.h"
#include "sources/svm/decision_function/svm_decision_function_manager.h"

#include "sources/svm/training_validation/full_run_info.h"


//**********************************************************************************************************************************

typedef vector < vector < vector <Tsvm_train_val_info> > > Ttrain_info_grid;

class Tsvm_manager 
{
	public: 
		Tsvm_manager();
		~Tsvm_manager();
		
		void clear();
		void load(const Tdataset& data_set);
		
		void train(const Ttrain_control& train_control, Tsvm_full_train_info& svm_full_train_info);
		void select(const Tselect_control& select_control, Tsvm_full_train_info& svm_full_train_info);
		void test(const Tdataset& test_set, const Ttest_control& test_control, Tsvm_full_test_info& test_info);
		
		void display_run_statistics();
		
		vector <double> get_predictions_for_task(unsigned task) const;
		vector <double> get_predictions_for_test_sample(unsigned i) const;
		vector <vector <vector < Ttrain_info_grid> > > get_list_of_train_info() const;
		vector <vector <vector < Tsvm_train_val_info> > > get_list_of_select_info() const;
	
		unsigned dim();
		unsigned size();
		unsigned decision_functions_size();
		unsigned number_of_all_tasks();
		Tworking_set_manager get_working_set_manager() const;
		Tsvm_decision_function_manager get_decision_function_manager() const;
		
		void read_train_aux_from_file(FILE* fpauxread);
		void read_decision_function_manager_from_file(FILE* fpsolread, bool& data_loaded_from_sol_file);
		void write_decision_function_manager_to_file(FILE* fpsolwrite);

		
	protected:
		void write_train_aux_to_file(FILE* fpauxwrite);
		void replace_kernel_control(const Tkernel_control& new_kernel_control);

		void read_decision_function_manager_from_file(Tsvm_decision_function_manager& decision_function_manager, FILE* fpsolread, bool& data_loaded_from_sol_file);
		void write_decision_function_manager_to_file(const Tsvm_decision_function_manager& decision_function_manager, FILE* fpsolwrite);
		
		void read_decision_function_manager_from_file(Tsvm_decision_function_manager& decision_function_manager, const string& filename, double& file_time);
		void write_decision_function_manager_to_file(const Tsvm_decision_function_manager& decision_function_manager, const string& filename, double& file_time);
	
		
		bool use_current_grid;
		bool clear_previous_train_info;
		Tfull_run_info full_run_info;
		Tsvm_decision_function_manager decision_function_manager;
		
		vector < Tgrid<Tsvm_solution, Tsvm_train_val_info> > current_grids;
		vector <vector <vector < Tgrid<Tsvm_solution, Tsvm_train_val_info> > > > list_of_grids;
		
	private:
		void clear_flags();
		void train_common(Tsvm_full_train_info& svm_full_train_info, bool select_mode);
		
		void get_train_controls(Tcv_control& cv_control, const Tdataset working_set, unsigned task, unsigned cell, double& file_time); 
		void store_train_controls(const Tcv_control& cv_control, const vector <Tsvm_train_val_info>& select_val_info, unsigned task, unsigned cell, double& file_time); 
		
		
		FILE* fp_log_train_read;
		FILE* fp_aux_train_read;
		FILE* fp_sol_train_read;

		bool read_train_log_from_file_flag;
		bool read_train_aux_from_file_flag;
		bool read_train_sol_from_file_flag;

		bool write_train_log_to_file_flag;
		bool write_train_aux_to_file_flag;

		bool write_select_log_to_file_flag;
		bool write_select_sol_to_file_flag;
		
		bool use_stored_logs;
		bool use_stored_solution;
		bool store_logs_internally;
		
		bool append_decision_functions;
		bool store_decision_functions_internally;
		
		Tdataset data_set;
		
		bool scale_data;
		vector <double> scaling;
		vector <double> translate;
		
		Ttrain_control train_control;
		Tselect_control select_control;
		
		Tworking_set_manager working_set_manager;
		vector <vector <Tfold_manager> > list_of_fold_managers;
		
		vector <vector <double> > predictions;
		vector <vector <vector < Tsvm_train_val_info > > > list_of_select_info;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/training_validation/svm_manager.cpp"
#endif
 
#endif
