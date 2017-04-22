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


#if !defined (SOLVER_H)
	#define SOLVER_H

 
 
#include "sources/shared/kernel/kernel.h"
#include "sources/shared/solver/solver_control.h"
#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/system_support/thread_manager.h"
#include "sources/shared/training_validation/train_val_info.h"


//**********************************************************************************************************************************


const double NO_PREVIOUS_LAMBDA = -1.0;


//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type> class Tsolver: public Tthread_manager
{
	public:
		Tsolver(){};
		~Tsolver(){};

		void clear();
		void reserve(Tsolver_control_type& solver_control, const Tparallel_control& parallel_control);
		
		void load(Tkernel* training_kernel, Tkernel* validation_kernel);

		void initialize_new_lambda_line(){};
		void initialize_new_weight_and_lambda_line(Ttrain_val_info_type& train_val_info){};
		void run_solver(Ttrain_val_info_type& train_val_info, Tsolution_type& solution){};


	protected:
		unsigned training_set_size;
		unsigned training_set_size_aligned;
		unsigned validation_set_size;
		
		double stop_eps;
		double offset;
		
		Tkernel* training_kernel;
		Tkernel* validation_kernel;
		
		Tloss_function loss_function;
		Tsolver_control_type solver_ctrl;
};


//**********************************************************************************************************************************


#include "sources/shared/solver/solver.ins.cpp"



#endif


