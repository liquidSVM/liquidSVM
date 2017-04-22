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


#if !defined (SMALL_ORDERED_INDEX_SET_H) 
	#define SMALL_ORDERED_INDEX_SET_H

//**********************************************************************************************************************************


#include <vector>
using namespace std;


//**********************************************************************************************************************************


class Tordered_index_set
{
	public: 
		Tordered_index_set();
		Tordered_index_set(unsigned size, bool increasing = false);
		
		void resize(unsigned size);
		inline unsigned size() const;
		inline unsigned capacity() const;
		void clear(bool increasing = false);
		
		inline void insert(unsigned new_index, double new_value);
		inline unsigned operator [] (unsigned i) const;

	
	private:
		vector <double> value; 
		vector <unsigned> index;
		
		bool increasing;
		unsigned current_size;
};


//**********************************************************************************************************************************


#include "sources/shared/basic_types/small_ordered_index_set.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/small_ordered_index_set.cpp"
#endif

#endif

