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


#if !defined (SVM_DECISION_FUNCTION_H)
	#define SVM_DECISION_FUNCTION_H


#include "sources/shared/kernel/kernel_control.h"
#include "sources/shared/decision_function/decision_function.h"
 
#include "sources/svm/decision_function/svm_solution.h"
#include "sources/svm/decision_function/svm_test_info.h"
#include "sources/svm/training_validation/svm_train_val_info.h"


#include <utility>


//**********************************************************************************************************************************

class Tsvm_decision_function: public Tsvm_solution, public Tdecision_function
{
	public:
		Tsvm_decision_function();
		~Tsvm_decision_function();
		Tsvm_decision_function(const Tsvm_decision_function& decision_function);
		Tsvm_decision_function(const Tsvm_solution* solution, Tkernel_control kernel_control, const Tsvm_train_val_info& train_val_info, const Tsubset_info& ws_info);

		
		friend Tsvm_decision_function operator * (double scalar, const Tsvm_decision_function& decision_function);
		Tsvm_decision_function operator + (const Tsvm_decision_function& decision_function);
		Tsvm_decision_function& operator = (const Tsvm_decision_function& decision_function);
		
		void write_to_file(FILE* fp) const;
		void read_from_file(FILE* fp);
		void set_to_zero();
		
		double evaluate(double* kernel_eval, unsigned training_size, unsigned gamma_no, unsigned thread_position);
		double evaluate(Tsample* test_sample, const Tdataset& training_set);

		#ifdef  COMPILE_WITH_CUDA__
			double get_offset() const;
			double get_clipp_value() const;
		#endif
		
		unsigned kernel_type;
		double gamma;
		string hierarchical_kernel_control_read_filename;
		
	protected:
		void copy(const Tsvm_decision_function* decision_function);
		template <class Tsvm_decision_function, class Tsvm_train_val_info, class Tsvm_test_info> friend class Tdecision_function_manager;
};

//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/decision_function/svm_decision_function.cpp"
#endif


#endif


