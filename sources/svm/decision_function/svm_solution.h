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


#if !defined (SVM_SOLUTION_H)
	#define SVM_SOLUTION_H

 
#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/decision_function/solution.h"

 
 
#include <cstdio>
#include <vector>
using namespace std;




//**********************************************************************************************************************************

class Tsvm_solution: public Tsolution
{
	public:
		Tsvm_solution();
		~Tsvm_solution();
		Tsvm_solution(const Tsvm_solution& solution);

		void clear();
		void resize(unsigned new_size);
		void reserve(unsigned new_capacity);
		
		void read_from_file(FILE* fp);
		void write_to_file(FILE* fp) const;
		
		Tsvm_solution& operator = (const Tsvm_solution& solution);
		
		
		Tsubset_info index;
		Tsubset_info sample_number;
		vector <double> coefficient;

		
	protected:
		void copy(const Tsvm_solution* source_solution);
};

//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/decision_function/svm_solution.cpp"
#endif

#endif


