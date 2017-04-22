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


#if !defined (SVM_SOLVER_CONTROL_CPP)
	#define SVM_SOLVER_CONTROL_CPP



#include "sources/svm/solver/svm_solver_control.h"


#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_types/loss_function.h"




//**********************************************************************************************************************************

Tsvm_solver_control::Tsvm_solver_control()
{
	solver_type = SVM_HINGE_2D;
	wss_method = DEFAULT_WSS_METHOD;

	stop_eps = 0.001;
};


//**********************************************************************************************************************************


void Tsvm_solver_control::read_from_file(FILE* fp)
{
	file_read(fp, wss_method);
	Tsolver_control::read_from_file(fp);
}


//**********************************************************************************************************************************

void Tsvm_solver_control::write_to_file(FILE* fp) const
{
	file_write(fp, wss_method);
	Tsolver_control::write_to_file(fp);
};

#endif
