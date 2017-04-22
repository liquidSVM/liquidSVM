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


#if !defined (TEMPLATE_SVM_H) 
	#define TEMPLATE_SVM_H

// 		CHANGE_FOR_OWN_SOLVER
	
#include "sources/svm/solver/basic_svm.h"


//**********************************************************************************************************************************


class Ttemplate_svm: public Tbasic_svm
{
	public:
		Ttemplate_svm();
		~Ttemplate_svm();
		virtual void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);
		void load(Tkernel* training_kernel, Tkernel* validation_kernel);

	protected:
		void initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info);
		void core_solver(Tsvm_train_val_info& train_val_info);
		
		void build_solution(Tsvm_train_val_info& train_val_info);
		void get_train_error(Tsvm_train_val_info& train_val_info);
		
	private:
		void init_zero();
		void init_keep();	
};


//**********************************************************************************************************************************

#include "sources/svm/solver/template_svm.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/template_svm.cpp"
#endif

#endif

