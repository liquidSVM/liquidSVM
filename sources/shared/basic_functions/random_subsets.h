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


#if !defined (RANDOM_SUBSETS_H) 
	#define RANDOM_SUBSETS_H

 

#include <vector>
using namespace std;


//**********************************************************************************************************************************
  

vector <unsigned> id_permutation(unsigned size);
vector <unsigned> random_permutation(unsigned size, int random_seed = -1, unsigned extra_seed = 0);
vector <unsigned> random_subset(const vector <unsigned>& set, unsigned subset_size, int random_seed = -1, unsigned extra_seed = 0);
vector <unsigned> random_multiset(const vector <unsigned>& set, unsigned multiset_size, int random_seed = -1, unsigned extra_seed = 0);

void random_shuffle(vector <unsigned>& set, unsigned start_index, unsigned stop_index, int random_seed = -1, unsigned extra_seed = 0);

//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_functions/random_subsets.cpp"
#endif

#endif



