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


#if !defined (RANDOM_GENERATOR_H) 
	#define RANDOM_GENERATOR_H


#include <cstdlib>
#include <vector>
using namespace std;
 
//********************************************************************************************************************************** 


void fix_random_seed(int& random_seed);
void init_random_generator(int random_seed = -1, unsigned extra_seed = 0);

int get_random_number(int min_value = 0, int max_value = RAND_MAX-1);
double get_uniform_random_number();
unsigned get_bernoulli_random_number(double p);
void get_random_distribution(vector <double>& distribution, unsigned size);
void get_random_vector(vector <double>& random_vector, unsigned size, double min_value = 0.0, double max_value = 1.0);


//**********************************************************************************************************************************

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_functions/random_generator.cpp"
#endif


#endif
