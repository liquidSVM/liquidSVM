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


#if !defined (SVM_SOLUTION_CPP)
	#define SVM_SOLUTION_CPP


#include "sources/svm/decision_function/svm_solution.h"

#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_types/dataset.h"


#include "sources/shared/basic_functions/flush_print.h"

//**********************************************************************************************************************************

Tsvm_solution::Tsvm_solution()
{
	clear();
}

//**********************************************************************************************************************************

Tsvm_solution::~Tsvm_solution()
{
	flush_info(INFO_VERY_PEDANTIC_DEBUG, "\nDestroying an object of type Tsvm_solution of size %d.", size());
	clear();
}
	
//*********************************************************************************************************************************

Tsvm_solution::Tsvm_solution(const Tsvm_solution& solution)
{
	copy(&solution);
}

//**********************************************************************************************************************************

void Tsvm_solution::reserve(unsigned new_capacity)
{	
	if (new_capacity > coefficient.size())
	{
		coefficient.reserve(new_capacity);
		index.reserve(new_capacity);
		sample_number.reserve(new_capacity);
	}
	Tsolution::reserve(new_capacity);
}

//**********************************************************************************************************************************

void Tsvm_solution::resize(unsigned new_size)
{
	if (new_size > coefficient.size())
	{
		coefficient.resize(new_size);
		index.resize(new_size);
		sample_number.resize(new_size);
	}
	Tsolution::resize(new_size);
}

//**********************************************************************************************************************************

void Tsvm_solution::clear()
{
	Tsolution::clear();

	coefficient.clear();
	index.clear();
	sample_number.clear();
}

//**********************************************************************************************************************************


Tsvm_solution& Tsvm_solution::operator = (const Tsvm_solution& solution)
{
	copy(&solution);
	return *this;
}

//**********************************************************************************************************************************


void Tsvm_solution::copy(const Tsvm_solution* source_solution)
{
	unsigned i;

	Tsvm_solution::clear();
	Tsolution::copy(source_solution);

	for(i=0;i<size();i++)
	{
		coefficient[i] = source_solution->coefficient[i];
		index[i] = source_solution->index[i];
		sample_number[i] = source_solution->sample_number[i];
	}
}


//**********************************************************************************************************************************

void Tsvm_solution::write_to_file(FILE* fp) const
{
	unsigned i;

	if (fp == NULL)
		return;

	Tsolution::write_to_file(fp);
	for (i=0;i< size(); i++)
		file_write(fp, sample_number[i], coefficient[i]);

	file_write_eol(fp);
}

//**********************************************************************************************************************************



void Tsvm_solution::read_from_file(FILE* fp)
{
	unsigned i;

	if (fp == NULL)
		return;

	Tsolution::read_from_file(fp);
	for (i=0;i< size();i++)
		file_read(fp, sample_number[i], coefficient[i]);
}


#endif



