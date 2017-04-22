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


#if !defined (VECTOR_CPP)
	#define VECTOR_CPP

	
#include "sources/shared/basic_types/vector.h"


#include <set>
#include <iterator>
using namespace std;

//**********************************************************************************************************************************


vector <int> is_categorial(const vector<double> vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	bool all_labels_integral;
	set <int> list_of_different_values;
	vector <int> return_vector;
	
	stop_index = determine_stop_index(vec, start_index, length);
	
	
	i = 0;
	all_labels_integral = true;
	do
	{
		if (double(int(vec[i])) != vec[i])
		{
			all_labels_integral = false;
			list_of_different_values.clear();
		}
		else
			list_of_different_values.insert(int(vec[i]));
		i++;
	}
	while ((i < stop_index) and (all_labels_integral == true));
	
	if (all_labels_integral == true)
		copy(list_of_different_values.begin(), list_of_different_values.end(), inserter(return_vector, return_vector.begin()));
	
	return return_vector;
}

#endif
