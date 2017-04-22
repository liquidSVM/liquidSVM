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


#if !defined (BASIC_2D_SVM_H) 
	#define BASIC_2D_SVM_H

 
#include "sources/shared/system_support/simd_basics.h"
#include "sources/shared/system_support/parallel_control.h"
#include "sources/shared/basic_functions/flush_print.h"

#include "sources/svm/solver/svm_solver_control.h"


//**********************************************************************************************************************************


class Tbasic_2D_svm
{
	public:
		Tbasic_2D_svm(){};
		~Tbasic_2D_svm(){};
		void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		
		strict_inline__ void order_indices(unsigned& index_1, double& gain_1, unsigned& index_2, double& gain_2);
		strict_inline__ void get_index_with_better_gain_simdd(simdd__& best_index_simdd, simdd__& best_gain_simdd, simdd__ index_simdd, simdd__ gain_simdd);
		strict_inline__ simdd__ update_2gradients_simdd(simdd__ gradient_simdd, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ kernel_1_simdd, simdd__ kernel_2_simdd);
};


//**********************************************************************************************************************************

#include "sources/svm/solver/basic_2D_svm.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/basic_2D_svm.cpp"
#endif

#endif

