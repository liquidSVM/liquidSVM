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




#include "sources/svm/solver/kernel_rule.h"
#include "sources/svm/solver/hinge_2D_svm.h"
#include "sources/svm/solver/least_squares_svm.h"
#include "sources/svm/solver/quantile_svm.h"
#include "sources/svm/solver/expectile_svm.h"
// 		CHANGE_FOR_OWN_SOLVER
#include "sources/svm/solver/template_svm.h"



#ifdef PRO_VERSION__
	#include "sources/svm/solver/hinge_svm_PAR.pro.h"
	#include "sources/svm/solver/least_squares_svm_PAR.pro.h"
#endif	



//**********************************************************************************************************************************


template <> inline void Tcv_manager<Tsvm_solution, Tsvm_train_val_info, Tsvm_solver_control, Tbasic_svm>::create_solver()
{
	switch (solver_control.solver_type)
	{
		case KERNEL_RULE:
			solver = new Tkernel_rule();
			break;
		case SVM_LS_2D:
			solver = new Tleast_squares_svm();
			break;
		case SVM_HINGE_2D:
			solver = new Thinge_2D_svm();
			break;
		case SVM_EXPECTILE_2D:
			solver = new Texpectile_svm();
			break;
		case SVM_QUANTILE:
			solver = new Tquantile_svm();
			break;
		// 		CHANGE_FOR_OWN_SOLVER
		case SVM_TEMPLATE:
			solver = new Ttemplate_svm();
			break;
		#ifdef PRO_VERSION__
			case SVM_HINGE_PAR:
				solver = new Thinge_svm_par();
				break;
			case SVM_LS_PAR:
				solver = new Tleast_squares_svm_par();
				break;
		#endif
		default:
			flush_exit(ERROR_UNSPECIFIED, "Specified solver type %d is unknown.", solver_control.solver_type);
	}
	solver->reserve(solver_control, get_parallel_control());
}


