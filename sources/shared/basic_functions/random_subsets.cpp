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


#if !defined (RANDOM_SUBSETS_CPP) 
	#define RANDOM_SUBSETS_CPP
 
 
#include "sources/shared/basic_functions/random_subsets.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/random_generator.h"

#include <cstdlib>
using namespace std;


//**********************************************************************************************************************************


vector <unsigned> id_permutation(unsigned size)
{
	unsigned i;
	vector <unsigned> permutation;
	
	permutation.resize(size);
	
	for (i=0;i<size;i++)
		permutation[i] = i;
	return permutation;
}



//**********************************************************************************************************************************

vector <unsigned> random_permutation(unsigned size, int random_seed, unsigned extra_seed)
{
// This O(size) algorithm is based on the Knuth-Shuffle algorithm.
	
	unsigned i;
	vector <unsigned> permutation;

	
	permutation = id_permutation(size);
	init_random_generator(random_seed, extra_seed);
	
	for (i=1;i<size;i++)
		swap(permutation[i], permutation[get_random_number() % i]);
	return permutation;
}



//**********************************************************************************************************************************

vector <unsigned> random_subset(const vector <unsigned>& set, unsigned subset_size, int random_seed, unsigned extra_seed)
{	
	unsigned i;
	vector <unsigned> subset;
	vector <unsigned> random_vector;


	random_vector = random_permutation(unsigned(set.size()), random_seed, extra_seed);

	subset.resize(subset_size);
	for (i=0;i<subset_size;i++)
		subset[i] = set[random_vector[i]];
	return subset;
}

//**********************************************************************************************************************************


vector <unsigned> random_multiset(const vector <unsigned>& set, unsigned multiset_size, int random_seed, unsigned extra_seed)
{
	unsigned i;
	vector <unsigned> multiset;
	
	
	init_random_generator(random_seed, extra_seed);
	
	multiset.resize(multiset_size);	
	for (i=0;i<multiset_size;i++)
		multiset[i] = set[get_random_number() % set.size()];
	return multiset;
}


//**********************************************************************************************************************************


void random_shuffle(vector <unsigned>& set, unsigned start_index, unsigned stop_index, int random_seed, unsigned extra_seed)
{
	unsigned i;
	unsigned size;
	vector <unsigned> permutation;
	vector <unsigned> copy;
	
	size = stop_index - start_index;
	copy = set;
	permutation = random_permutation(size, random_seed, extra_seed);
	for (i=0; i<size; i++)
		set[start_index + i] = copy[start_index + permutation[i]];
}



#endif
