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


#if !defined (SOLUTION_CPP)
	#define SOLUTION_CPP


#include "sources/shared/decision_function/solution.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"


//**********************************************************************************************************************************

Tsolution::Tsolution()
{
	clear();
}


//*********************************************************************************************************************************

Tsolution::Tsolution(const Tsolution& solution)
{
	copy(&solution);
}

//**********************************************************************************************************************************

void Tsolution::reserve(unsigned new_capacity)
{
	if (new_capacity > current_capacity)
		current_capacity = new_capacity;
}

//**********************************************************************************************************************************

void Tsolution::resize(unsigned new_size)
{
	current_size = new_size;
	current_capacity = max(current_capacity, new_size);
}

//**********************************************************************************************************************************

void Tsolution::clear()
{
	current_size = 0;
	current_capacity = 0;

	offset = 0.0;
	clipp_value = 0.0;
	
	pos_weight = 1.0;
	neg_weight = 1.0;
}

//**********************************************************************************************************************************

unsigned Tsolution::size() const
{
	return current_size;
}


//**********************************************************************************************************************************

unsigned Tsolution::capacity() const
{
	return current_capacity;
}



//**********************************************************************************************************************************


Tsolution& Tsolution::operator = (const Tsolution& solution)
{
	copy(&solution);
	return *this;
}


//**********************************************************************************************************************************

void Tsolution::set_prediction_modifiers(double offset, double clipp_value)
{
	Tsolution::offset = offset;
	Tsolution::clipp_value = clipp_value;
}


//**********************************************************************************************************************************

double Tsolution::get_clipp_value() const
{
	return clipp_value;
}

//**********************************************************************************************************************************

void Tsolution::set_weights(double neg_weight, double pos_weight)
{
	Tsolution::neg_weight = neg_weight;
	Tsolution::pos_weight = pos_weight;
}

//**********************************************************************************************************************************

void Tsolution::write_to_file(FILE* fp) const
{
	if (fp == NULL)
		return;

	file_write(fp, offset);
	file_write(fp, clipp_value);
	file_write(fp, neg_weight);
	file_write(fp, pos_weight);
	file_write(fp, current_size);
	file_write_eol(fp);
}

//**********************************************************************************************************************************



void Tsolution::read_from_file(FILE* fp)
{
	if (fp == NULL)
		return;

	file_read(fp, offset);
	file_read(fp, clipp_value);
	file_read(fp, neg_weight);
	file_read(fp, pos_weight);
	file_read(fp, current_size);

	this->resize(current_size);
}


//**********************************************************************************************************************************


void Tsolution::copy(const Tsolution* source_solution)
{
	this->clear();
	this->reserve(source_solution->capacity());
	this->resize(source_solution->size());

	offset = source_solution->offset;
	clipp_value = source_solution->clipp_value;
	
	pos_weight = source_solution->pos_weight;
	neg_weight = source_solution->neg_weight;
}

#endif

