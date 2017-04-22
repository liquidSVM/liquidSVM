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


#if !defined (SMALL_ORDERED_INDEX_SET_CPP) 
	#define SMALL_ORDERED_INDEX_SET_CPP


#include "sources/shared/basic_types/small_ordered_index_set.h"


#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/basic_functions/flush_print.h"


#include <limits>
using namespace std;





//**********************************************************************************************************************************


Tordered_index_set::Tordered_index_set() 
{
	increasing = false;
	current_size = 0;
}

//**********************************************************************************************************************************

	
Tordered_index_set::Tordered_index_set(unsigned size, bool increasing)
{
	value.resize(size);
	index.resize(size);
	
	clear(increasing);
}	


//**********************************************************************************************************************************

		
void Tordered_index_set::resize(unsigned size)
{
	unsigned i;
	unsigned old_size;
	
	old_size = unsigned(value.capacity());
	value.resize(size);
	index.resize(size);
	if (size > old_size)
		for (i=old_size; i<size; i++)
		{
			index[i] = 0;
			if (increasing == true)
				value[i] = numeric_limits<double>::max();
			else
				value[i] = -numeric_limits<double>::max();
		}
}


//**********************************************************************************************************************************

		
void Tordered_index_set::clear(bool increasing)
{
	Tordered_index_set::increasing = increasing;
	
	index.assign(capacity(), 0);
	if (increasing == true)
		value.assign(capacity(), numeric_limits<double>::max());
	else
		value.assign(capacity(), -numeric_limits<double>::max());
	
	current_size = 0;
}

#endif
