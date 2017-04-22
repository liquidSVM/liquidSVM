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


#if !defined (SOLVER_CONTROL_CPP)
	#define SOLVER_CONTROL_CPP


#include "sources/shared/solver/solver_control.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"



//**********************************************************************************************************************************

Tsolver_control::Tsolver_control()
{
	cold_start = SOLVER_INIT_DEFAULT;
	warm_start = SOLVER_INIT_DEFAULT;

	stop_eps = 0.001;
	solver_type = 0;
	save_solution = false;
	
	clipp_value = NO_CLIPPING;
	global_clipp_value = ADAPTIVE_CLIPPING;
	
	fixed_loss = true;
	
	order_data = SOLVER_DO_NOT_ODER_DATA;
};


//**********************************************************************************************************************************


void Tsolver_control::read_from_file(FILE* fp)
{
	file_read(fp, cold_start);
	file_read(fp, warm_start);

	file_read(fp, stop_eps);
	file_read(fp, solver_type);
	
	file_read(fp, global_clipp_value);
	
	file_read(fp, fixed_loss);
	loss_control.read_from_file(fp);
	
	kernel_control_val.read_from_file(fp);
	kernel_control_train.read_from_file(fp);
}


//**********************************************************************************************************************************

void Tsolver_control::write_to_file(FILE* fp) const
{
	file_write(fp, cold_start);
	file_write(fp, warm_start);

	file_write(fp, stop_eps);
	file_write(fp, solver_type);

	file_write(fp, global_clipp_value);

	file_write(fp, fixed_loss);
	loss_control.write_to_file(fp);
	
	kernel_control_val.write_to_file(fp);
	kernel_control_train.write_to_file(fp);
};


//**********************************************************************************************************************************


void Tsolver_control::set_clipping(double max_abs_label)
{
	if (global_clipp_value == ADAPTIVE_CLIPPING)
		clipp_value = max_abs_label;
	else
		clipp_value = global_clipp_value;

	if ((max_abs_label > clipp_value) and (clipp_value != NO_CLIPPING))
		flush_exit(ERROR_DATA_MISMATCH, "Clipping %1.4f is too small for maximal label %1.4f.", clipp_value, max_abs_label);
}


#endif
