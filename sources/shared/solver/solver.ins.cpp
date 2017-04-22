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




#include "sources/shared/system_support/memory_allocation.h"


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type>
void Tsolver<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type>::reserve(Tsolver_control_type& solver_control, const Tparallel_control& parallel_control)
{
	reserve_threads(parallel_control);
	loss_function = Tloss_function(solver_control.loss_control);
	solver_ctrl = solver_control;
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type>
void Tsolver<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type>::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	double* dummy;
	
	Tsolver::training_kernel = training_kernel;
	Tsolver::validation_kernel = validation_kernel;
	
	training_set_size = training_kernel->get_row_set_size();
	validation_set_size = validation_kernel->get_col_set_size();
	training_set_size_aligned = allocated_memory_ALGD(&dummy, training_set_size);
}



//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type>
void Tsolver<Tsolution_type, Ttrain_val_info_type, Tsolver_control_type>::clear()
{
	clear_threads();
}
