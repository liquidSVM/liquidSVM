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


#if !defined (KERNEL_RULE_SVM_H)
	#define KERNEL_RULE_SVM_H

#include "sources/svm/solver/least_squares_svm.h"


//**********************************************************************************************************************************


class Tkernel_rule: public Tleast_squares_svm
{
	public:
		Tkernel_rule();
		virtual ~Tkernel_rule();
		void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		
	protected:
		virtual void core_solver(Tsvm_train_val_info& train_val_info);
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/kernel_rule.cpp"
#endif

#endif
