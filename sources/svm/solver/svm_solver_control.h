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


#if !defined (SVM_SOLVER_CONTROL_H)
	#define SVM_SOLVER_CONTROL_H



#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/kernel/kernel_control.h"
#include "sources/shared/solver/solver_control.h"



#include <cstdio>
using namespace std;

//**********************************************************************************************************************************


enum WSS_METHODS {DEFAULT_WSS_METHOD, DONT_USE_NNs, USE_NNs, WSS_METHODS_MAX};

	// 		CHANGE_FOR_OWN_SOLVER
enum SOLVER_TYPES {KERNEL_RULE, SVM_LS_2D, SVM_HINGE_2D, SVM_QUANTILE, SVM_EXPECTILE_2D, SVM_TEMPLATE, SOLVER_TYPES_MAX};
enum EXPERIMENTAL_SOLVER_TYPES {SVM_LS_PAR = SOLVER_TYPES_MAX, SVM_HINGE_PAR, EXP_SOLVER_TYPES_MAX};

	#ifdef OWN_DEVELOP__
		#define AVAILABLE_SOLVER_TYPES_MAX EXP_SOLVER_TYPES_MAX
	#else
		#define AVAILABLE_SOLVER_TYPES_MAX SOLVER_TYPES_MAX
	#endif

enum SOLVER_INIT_TYPES {SOLVER_INIT_ZERO = SOLVER_INIT_DEFAULT+1, SOLVER_INIT_FULL, SOLVER_INIT_RECYCLE, SOLVER_INIT_EXPAND_UNIFORMLY, SOLVER_INIT_EXPAND, SOLVER_INIT_SHRINK_UNIFORMLY, SOLVER_INIT_SHRINK, SOLVER_INIT_TYPES_MAX};


//**********************************************************************************************************************************


class Tsvm_solver_control: public Tsolver_control
{
	public:
		Tsvm_solver_control();
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;

		
		unsigned wss_method;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/svm_solver_control.cpp"
#endif

#endif
