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


#if !defined (HINGE_2D_SVM_H) 
	#define HINGE_2D_SVM_H



#include "sources/svm/solver/hinge_svm.h"
#include "sources/svm/solver/basic_2D_svm.h"
#include "sources/svm/solver/generic_xD_ancestor.h"



//**********************************************************************************************************************************


class Thinge_svm_generic_ancestor: public Thinge_svm, public Tsvm_xD_ancestor
{
	protected:
		strict_inline__ void loop_with_clipping_CL(unsigned index, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__& slack_sum_simdd, simdd__& best_gain_simdd, simdd__& best_index_simdd, double* restrict__ kernel_row1_ALGD, double* restrict__ kernel_row2_ALGD);
		
		strict_inline__ simdd__ get_gain_simdd(simdd__ gradient_simdd, simdd__ weight_simdd, simdd__ alpha_simdd);
		
	
		strict_inline__ unsigned constraint_segment(double weight, double alpha);
		strict_inline__ double gain_2D(double gradient_1, double gradient_2, double delta_1, double delta_2, double K_ij);
		strict_inline__ double optimize_2D_corner(double current_alpha_1, double current_alpha_2, double gradient_1, double gradient_2, double border_1, double border_2, double& new_alpha_1, double& new_alpha_2, double K_ij);
};



//**********************************************************************************************************************************

#if defined (Tsvm_2D_solver_generic_base_name) 
	#undef Tsvm_2D_solver_generic_base_name
#endif
#define Tsvm_2D_solver_generic_base_name Thinge_svm_generic_base

#if defined (Tsvm_2D_solver_generic_ancestor_name) 
	#undef Tsvm_2D_solver_generic_ancestor_name
#endif
#define Tsvm_2D_solver_generic_ancestor_name Thinge_svm_generic_ancestor

#include "sources/svm/solver/generic_2D_svm.h"



//**********************************************************************************************************************************


class Thinge_2D_svm: public Thinge_svm_generic_base
{
	public:
		Thinge_2D_svm(){};
		virtual ~Thinge_2D_svm(){};
		virtual void reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control);

	protected:
		virtual void core_solver(Tsvm_train_val_info& train_val_info);
};


//**********************************************************************************************************************************


#include "sources/svm/solver/hinge_2D_svm.ins.cpp"


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/solver/hinge_2D_svm.cpp"
#endif

#endif
