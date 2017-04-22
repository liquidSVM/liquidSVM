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


#if !defined (DECISION_FUNCTION_H)
	#define DECISION_FUNCTION_H



#include "sources/shared/training_validation/train_val_info.h"


//**********************************************************************************************************************************

class Tdecision_function
{
	public:
		Tdecision_function();
		Tdecision_function(const Tdecision_function& decision_function);
		Tdecision_function& operator = (const Tdecision_function& decision_function);
		
		virtual void write_to_file(FILE* fp) const;
		virtual void read_from_file(FILE* fp);
		virtual void set_to_zero() = 0;

		void set_error(const Ttrain_val_info& train_val_info);
		void set_labels(pair <pair <double, double>, double> labels);


	protected:
		template <class Tdecision_function_type, class Ttrain_val_info_type, class Ttest_info_type> friend class Tdecision_function_manager;
		
		void copy(const Tdecision_function* decision_function);
		
		double label1;
		double label2;
		double default_label;
		
		double error;
		double pos_error;
		double neg_error;
};

//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/decision_function/decision_function.cpp"
#endif


#endif


