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


#if !defined (SOLVER_CONTROL_H)
	#define SOLVER_CONTROL_H


#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/kernel/kernel_control.h"

#include <cstdio>
using namespace std;


//**********************************************************************************************************************************


const int SOLVER_INIT_DEFAULT = -1;
const double NO_CLIPPING = 0.0;
const double ADAPTIVE_CLIPPING = -1.0;


enum SOLVER_INIT_DIRECTIONS {SOLVER_INIT_FORWARD, SOLVER_INIT_BACKWARD, SOLVER_INIT_NO_DIRECTION, SOLVER_INIT_DIRECTIONS_MAX};
enum SOLVER_ORDER_DATA {SOLVER_DO_NOT_ODER_DATA, SOLVER_ODER_DATA_SPATIALLY, SOLVER_ORDER_DATA_MAX};


//**********************************************************************************************************************************


class Tsolver_control
{
	public:
		Tsolver_control();
		
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;
		void set_clipping(double max_abs_label);

		
		int cold_start;
		int warm_start;
		unsigned init_direction;
		
		double stop_eps;
		unsigned solver_type;
		bool save_solution;
		
		double clipp_value;
		double global_clipp_value;
		
		bool fixed_loss;
		Tloss_control loss_control;
		
		unsigned order_data;
		
		Tkernel_control kernel_control_val;
		Tkernel_control kernel_control_train;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/solver/solver_control.cpp"
#endif


#endif
