// Copyright 2015-2017 Philipp Thomann
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

#if !defined (KERNEL_CALCULATOR_H)
	#define KERNEL_CALCULATOR_H

#define COMPILE_FOR_R__

#include "sources/shared/basic_types/dataset.h"

#include "sources/shared/system_support/thread_manager_active.h"
#include "sources/shared/solver/solver_control.h"
#include "sources/shared/kernel/kernel.h"
#include "sources/shared/kernel/kernel_control.h"
#include "sources/shared/basic_functions/random_subsets.h"
#include "sources/shared/training_validation/train_val_info.h"

class Tkernel_calculator: public Tthread_manager_active
{
	public:
		Tkernel_calculator() {order_data = SOLVER_DO_NOT_ODER_DATA;};
		~Tkernel_calculator();

		virtual void clear_threads();

		void calculate(Tkernel_control kernel_ctrl, Tdataset dataset);

		double gamma;

		Tkernel kernel;

	private:
		virtual void thread_entry();

		Tkernel_control kernel_control;
//		unsigned order_data = SOLVER_DO_NOT_ODER_DATA;
		unsigned order_data;

		Tdataset data_set;

		vector <unsigned> permutation;

};

#ifndef COMPILE_SEPERATELY__
  #include "./kernel_calculator.cpp"
#endif

#endif
