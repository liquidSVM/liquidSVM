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


#if !defined (SOLUTION_H)
	#define SOLUTION_H



#include <cstdio>
#include <vector>
using namespace std;


//**********************************************************************************************************************************


class Tsolution
{
	public:
		Tsolution();
		Tsolution(const Tsolution& solution);

		void clear();
		virtual void resize(unsigned new_size);
		virtual void reserve(unsigned new_capacity);
		
		unsigned size() const;
		unsigned capacity() const;
		
		void read_from_file(FILE* fp);
		void write_to_file(FILE* fp) const;
		
		void set_prediction_modifiers(double offset, double clipp_value);
		void set_weights(double neg_weight, double pos_weight);
		double get_clipp_value() const;
		
		Tsolution& operator = (const Tsolution& solution);

	protected:
		void copy(const Tsolution* source_solution);
		

		double offset;
		double clipp_value;
		
		double pos_weight;
		double neg_weight;
		
	private:
		unsigned current_size;
		unsigned current_capacity;
};


//**********************************************************************************************************************************



#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/decision_function/solution.cpp"
#endif

#endif
