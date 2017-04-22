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


#if !defined (RANDOM_GENERATOR_CPP) 
	#define RANDOM_GENERATOR_CPP



#include <ctime>

 
#include "sources/shared/basic_functions/random_generator.h"


#ifndef __RAND__
  #define __RAND__() rand();
#endif

#ifndef __SRAND__
  #define __SRAND__(i) srand(i);
#endif

//**********************************************************************************************************************************



void fix_random_seed(int& random_seed)
{
	if (random_seed < 0)
		random_seed = unsigned(time(NULL));
}

//**********************************************************************************************************************************

void init_random_generator(int random_seed, unsigned extra_seed)
{
	fix_random_seed(random_seed);
	__SRAND__(random_seed + extra_seed);
}

//**********************************************************************************************************************************


int get_random_number(int min_value, int max_value)
{
	int random_value;
	
	random_value = __RAND__();
	if ((min_value == 0) and (max_value == RAND_MAX-1))
		return random_value;
	else
		return random_value % (max_value - min_value + 1) + min_value;
}

//**********************************************************************************************************************************

double get_uniform_random_number()
{
	return (double(get_random_number()) / double(RAND_MAX-1));
}


//**********************************************************************************************************************************

unsigned get_bernoulli_random_number(double p)
{
	if (get_uniform_random_number() < p)
		return 1;
	else
		return 0;
}


//**********************************************************************************************************************************

void get_random_distribution(vector <double>& distribution, unsigned size)
{
	unsigned i;
	double sum;

	sum = 0.0;	
	distribution.resize(size);

	for (i=0; i<size; i++)
	{
		distribution[i] = get_uniform_random_number();
		sum = sum + distribution[i];
	}
	
	for (i=0; i<size; i++)
		distribution[i] = distribution[i] / sum;
}

//**********************************************************************************************************************************


void get_random_vector(vector <double>& random_vector, unsigned size, double min_value, double max_value)
{
	unsigned i;
	
	random_vector.resize(size);
	for (i=0; i<size; i++)
		random_vector[i] = (max_value - min_value) * (double(get_random_number()) / double(RAND_MAX)) + min_value;
}

#endif
