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


#if !defined (SVM_MANAGER_CONTROL_CPP)
	#define SVM_MANAGER_CONTROL_CPP


#include "sources/svm/training_validation/svm_manager_control.h"

//**********************************************************************************************************************************


Ttrain_control::Ttrain_control()
{
	scale_data = false;
	
	store_logs_internally = false;
	store_solutions_internally = false;
	
	full_search = true;
	max_number_of_increases = 3;
	max_number_of_worse_gammas = 3;
}

//**********************************************************************************************************************************


Tselect_control::Tselect_control()
{
	Tcv_control cv_control;
	
	
	copy_from_cv_control(cv_control);
	
	
	use_stored_logs = false;
	use_stored_solution = false;
	append_decision_functions = true;
	store_decision_functions_internally = false;
}


//**********************************************************************************************************************************


void Tselect_control::copy_from_cv_control(const Tcv_control& cv_control)
{
	select_method = cv_control.select_method;
	use_stored_solution = cv_control.use_stored_solution;
	
	npl = cv_control.npl;
	npl_class = cv_control.npl_class;
	npl_constraint = cv_control.npl_constraint;
	weight_number = cv_control.weight_number;
}

//**********************************************************************************************************************************


void Tselect_control::copy_to_cv_control(Tcv_control& cv_control)
{
	cv_control.select_method = select_method;
	cv_control.use_stored_solution = use_stored_solution;
	
	cv_control.npl = npl;
	cv_control.npl_class = npl_class;
	cv_control.npl_constraint = npl_constraint;
	cv_control.weight_number = weight_number;
}


//**********************************************************************************************************************************



Ttest_control::Ttest_control()
{
	max_used_RAM_in_MB = 2048;
}


#endif
