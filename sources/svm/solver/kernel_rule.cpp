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


#if !defined (KERNEL_RULE_SVM_CPP)
	#define KERNEL_RULE_SVM_CPP


#include "sources/svm/solver/kernel_rule.h"

#include "sources/shared/basic_functions/flush_print.h"



//**********************************************************************************************************************************

Tkernel_rule::Tkernel_rule()
{
};

//**********************************************************************************************************************************

Tkernel_rule::~Tkernel_rule()
{
};


//**********************************************************************************************************************************

void Tkernel_rule::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	solver_control.kernel_control_train.kNNs = 0;
	solver_control.kernel_control_train.include_labels = false;
	solver_control.kernel_control_train.memory_model_kernel = EMPTY;
	solver_control.kernel_control_train.memory_model_pre_kernel = EMPTY;
	
	if (solver_control.cold_start == SOLVER_INIT_DEFAULT)
		solver_control.cold_start = SOLVER_INIT_ZERO;
	else if (solver_control.cold_start != SOLVER_INIT_ZERO)
		flush_exit(1, "\nKernel rule must not be cold started by method %d.\n" 
			"Allowed methods are %d.", solver_control.cold_start, SOLVER_INIT_ZERO);
		
	if (solver_control.warm_start == SOLVER_INIT_DEFAULT)
		solver_control.warm_start = SOLVER_INIT_ZERO;
	else if (solver_control.warm_start != SOLVER_INIT_ZERO) 
		flush_exit(1, "\nKernel rule must not be warm started by method %d.\n" 
			"Allowed methods are %d.", solver_control.warm_start, SOLVER_INIT_ZERO);
		
	Tbasic_svm::reserve(solver_control, parallel_control);
}


//**********************************************************************************************************************************

void Tkernel_rule::core_solver(Tsvm_train_val_info& train_val_info)
{
	unsigned i;

	if (is_first_team_member() == true)
	{
		train_val_info.train_iterations = 1;
		train_val_info.gradient_updates = 0;

		for (i=0;i<training_set_size;i++)
			alpha_ALGD[i] = training_label_ALGD[i];
		
		build_SV_list(train_val_info);
	}
}

#endif


