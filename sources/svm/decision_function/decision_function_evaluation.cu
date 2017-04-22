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


#if !defined (SVM_DECISION_FUNCTION_EVALUATION_CU)
	#define SVM_DECISION_FUNCTION_EVALUATION_CU


#include "sources/svm/decision_function/decision_function_evaluation.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/kernel/kernel_computation.h"
#include "sources/shared/kernel/kernel_functions.h"
#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/system_support/cuda_memory_operations.h"

#include "sources/svm/decision_function/svm_test_info.h"
#include "sources/svm/decision_function/svm_decision_function_manager.h"



//**********************************************************************************************************************************



#define FLOAT_TYPE double
#include "sources/svm/decision_function/decision_function_evaluation.ins.cu"
#undef FLOAT_TYPE

#define FLOAT_TYPE float
#include "sources/svm/decision_function/decision_function_evaluation.ins.cu"
#undef FLOAT_TYPE






#endif
