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


#if !defined (BASIC_2D_SVM_CPP) 
	#define BASIC_2D_SVM_CPP
 
 
#include "sources/svm/solver/basic_2D_svm.h"


//**********************************************************************************************************************************


void Tbasic_2D_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{	
	if (solver_control.wss_method == DEFAULT_WSS_METHOD)
		solver_control.wss_method = USE_NNs;
	if (solver_control.kernel_control_train.kNNs == DEFAULT_NN)
	{
		if (solver_control.wss_method == USE_NNs)
			solver_control.kernel_control_train.kNNs = 10;
		else
			solver_control.kernel_control_train.kNNs = 0;
	}
}

#endif

