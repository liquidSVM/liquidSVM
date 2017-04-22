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


#if !defined (HINGE_2D_SVM_CPP) 
	#define HINGE_2D_SVM_CPP



#include "sources/svm/solver/hinge_2D_svm.h"


#include "sources/shared/basic_functions/flush_print.h"

#ifdef _WIN32
	#include <io.h>
#endif



//**********************************************************************************************************************************


void Thinge_2D_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	solver_control.kernel_control_train.include_labels = true;
	
	if (solver_control.cold_start == SOLVER_INIT_DEFAULT)
		solver_control.cold_start = SOLVER_INIT_FULL;

	if (solver_control.warm_start == SOLVER_INIT_DEFAULT)
		solver_control.warm_start = SOLVER_INIT_EXPAND;
	

	Tsvm_2D_solver_generic_base_name::reserve(solver_control, parallel_control);
	Thinge_svm::reserve(solver_control, parallel_control);
}




//**********************************************************************************************************************************

void Thinge_2D_svm::core_solver(Tsvm_train_val_info& train_val_info)
{
	core_solver_generic_part(train_val_info);
	if (is_first_team_member() == true)
	{
		build_bSV_list(train_val_info);
		
		MM_CACHELINE_FLUSH(&slack_sum_local[0]);
	}
	
	sync_threads();
	slack_sum_global[get_thread_id()] = slack_sum_local[0];
}





#endif


