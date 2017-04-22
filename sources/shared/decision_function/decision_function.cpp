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


#if !defined (DECISION_FUNCTION_CPP)
	#define DECISION_FUNCTION_CPP


#include "sources/shared/decision_function/decision_function.h"



#include "sources/shared/basic_functions/basic_file_functions.h"


//*********************************************************************************************************************************

Tdecision_function::Tdecision_function()
{
	error = NOT_EVALUATED;
	pos_error = NOT_EVALUATED;
	neg_error = NOT_EVALUATED;
}

//*********************************************************************************************************************************

Tdecision_function::Tdecision_function(const Tdecision_function& decision_function)
{
	copy(&decision_function);
}



//**********************************************************************************************************************************


Tdecision_function& Tdecision_function::operator = (const Tdecision_function& decision_function)
{
	copy(&decision_function);
	return *this;
}


//*********************************************************************************************************************************

void Tdecision_function::copy(const Tdecision_function* decision_function)
{
	error = decision_function->error;
	neg_error = decision_function->neg_error;
	pos_error = decision_function->pos_error;
	
	label1 = decision_function->label1;
	label2 = decision_function->label2;
	default_label = decision_function->default_label;
}



//**********************************************************************************************************************************

void Tdecision_function::set_error(const Ttrain_val_info& train_val_info)
{
	error = train_val_info.val_error;
	neg_error = train_val_info.neg_val_error;
	pos_error = train_val_info.pos_val_error;
}


//**********************************************************************************************************************************

void Tdecision_function::set_labels(pair <pair <double, double>, double> labels)
{
	label1 = labels.first.first;
	label2 = labels.first.second;
	default_label = labels.second;
}


//**********************************************************************************************************************************

void Tdecision_function::write_to_file(FILE* fp) const
{
	if (fp == NULL)
		return;
	
	file_write(fp, error);
	file_write(fp, neg_error);
	file_write(fp, pos_error);
	
	file_write(fp, label1);
	file_write(fp, label2);
	file_write(fp, default_label);
}



//**********************************************************************************************************************************

void Tdecision_function::read_from_file(FILE* fp)
{
	if (fp == NULL)
		return;
	
	file_read(fp, error);
	file_read(fp, neg_error);
	file_read(fp, pos_error);
	
	file_read(fp, label1);
	file_read(fp, label2);
	file_read(fp, default_label);
}

#endif



