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



#include "sources/shared/basic_functions/flush_print.h"


//**********************************************************************************************************************************


inline unsigned Tordered_index_set::capacity() const
{
	return unsigned(value.size());
}

//**********************************************************************************************************************************



inline unsigned Tordered_index_set::size() const
{
	return current_size;
}

//**********************************************************************************************************************************


inline unsigned Tordered_index_set::operator [] (unsigned i) const
{
	return index[i];
}


//**********************************************************************************************************************************
		
inline void Tordered_index_set::insert(unsigned new_index, double new_value)
{
	int i;
	
	i = int(size()) - 1;
	if (i >= 0)
	{
		if (increasing == true)
			while (new_value < value[i]) //and (i >= 0))
			{
				if (i+1 < int(capacity()))
				{
					value[i + 1] = value[i];
					index[i + 1] = index[i];
				}
				i--;
				if (i < 0)
					break;
			}
		else
			while (new_value > value[i]) //and (i >= 0))
			{
				if (i+1 < int(capacity()))
				{
					value[i + 1] = value[i];
					index[i + 1] = index[i];
				}
				i--;
				if (i < 0)
					break;
			}
	}

	if (i+1 < int(capacity()))
	{
		value[i + 1] = new_value;
		index[i + 1] = new_index;
	}
	
	current_size = min(current_size + 1, capacity());
}
