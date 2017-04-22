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


#if !defined (TEMPLATE_SVM_CPP) 
	#define TEMPLATE_SVM_CPP

// 		CHANGE_FOR_OWN_SOLVER	

#include "sources/svm/solver/template_svm.h"


#include "sources/shared/system_support/timing.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_types/vector.h"



//**********************************************************************************************************************************

Ttemplate_svm::Ttemplate_svm()
{
}


//**********************************************************************************************************************************

Ttemplate_svm::~Ttemplate_svm()
{
}

//**********************************************************************************************************************************

void Ttemplate_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	// 	Adapt the next lines for your purposes.
	solver_control.kernel_control_train.kNNs = 0;
	solver_control.kernel_control_train.include_labels = false;


	if (solver_control.cold_start == SOLVER_INIT_DEFAULT)
		solver_control.cold_start = SOLVER_INIT_ZERO;
	else if (solver_control.cold_start != SOLVER_INIT_ZERO)
		flush_exit(1, "\nTemplate solver must not be cold started by method %d.\n" 
			"Allowed methods are %d.", solver_control.cold_start, SOLVER_INIT_ZERO);
		
	if (solver_control.warm_start == SOLVER_INIT_DEFAULT)
		solver_control.warm_start = SOLVER_INIT_RECYCLE;
	else if ((solver_control.warm_start != SOLVER_INIT_ZERO) and (solver_control.warm_start != SOLVER_INIT_RECYCLE))
		flush_exit(1, "\nTemplate solver must not be warm started by method %d.\n" 
			"Allowed methods are %d and %d.", solver_control.warm_start, SOLVER_INIT_ZERO, SOLVER_INIT_RECYCLE);

		
	// This line needs to be kept.
	Tbasic_svm::reserve(solver_control, parallel_control);
}


//**********************************************************************************************************************************

void Ttemplate_svm::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	Tbasic_svm::load(training_kernel, validation_kernel);
	
	// 	Place extra code that need to be executed when loading the kernel here.
	
	if (is_first_team_member() == true)
	{
	}
}

//**********************************************************************************************************************************

void Ttemplate_svm::get_train_error(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	double prediction;

	train_val_info.train_error = 0.0;
	for (i=0; i<training_set_size; i++)
	{
		// Replace the next line by the correct computation of the prediction
		prediction = 0.0;
		train_val_info.train_error = train_val_info.train_error + loss_function.evaluate(training_label_ALGD[i], prediction);
	}
	train_val_info.train_error = train_val_info.train_error / double(training_set_size);
}

//**********************************************************************************************************************************

void Ttemplate_svm::initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);

// This loop is likely necessary when dealing with simdd instructions. Also, extra lines may need to be added.
	for (i=training_set_size;i<training_set_size_aligned;i++)
	{
		alpha_ALGD[i] = 0.0;
		gradient_ALGD[i] = 0.0;
		training_label_ALGD[i] = 0.0;
	}

	// 	If you have more or less init methods, these lines need to be adapted.
	// 	Also, adpat the init_iterations set below if necessary.
	switch (init_method)
	{
		case SOLVER_INIT_ZERO:
			init_zero();
			train_val_info.init_iterations = 1;
			break;
		case SOLVER_INIT_RECYCLE:
			init_keep();
			train_val_info.init_iterations = 0;
			break;
		default:
			flush_exit(1, "Unknown solver initialization method %d for template solver.", init_method);
			break;
	}

	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);
}


//**********************************************************************************************************************************

void Ttemplate_svm::init_zero()
{
	// 	Please the code for initializing with zeros here.
};


//**********************************************************************************************************************************

void Ttemplate_svm::init_keep()
{
	// 	If it is possible to use the solution of a larger lambda as initialization for 
	// 	a smaller lambda, place your code here.
};


//**********************************************************************************************************************************

void Ttemplate_svm::build_solution(Tsvm_train_val_info& train_val_info)
{
	if (is_first_team_member() == true)
	{
	// 	Place the code transforming the solver solution into alpha coefficients 
	// 	for an svm solution (and its offset) here.
	}
}



//**********************************************************************************************************************************

void Ttemplate_svm::core_solver(Tsvm_train_val_info& train_val_info)
{
	train_val_info.train_iterations = 0;
	train_val_info.gradient_updates = 0;

	// The actual solver should be place here
	
	flush_info(INFO_1, "\nThe template solver is not implemented.");
	
	
	build_SV_list(train_val_info);
}



#endif







