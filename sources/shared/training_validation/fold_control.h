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


#if !defined (FOLD_CONTROL_H)
	#define FOLD_CONTROL_H

 
 
#include <cstdio>
using namespace std;


//********************************************************************************************************************************** 


const double FRACTION_NOT_ASSIGNED = -1.0;
enum FOLD_CREATION_TYPES {FROM_FILE, BLOCKS, ALTERNATING, RANDOM, STRATIFIED, RANDOM_SUBSET, FOLD_CREATION_TYPES_MAX};


//**********************************************************************************************************************************


class Tfold_control
{
	public:
		Tfold_control();
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;


		unsigned number;
		unsigned kind;
		double train_fraction;
		double negative_fraction;
		int random_seed;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/fold_control.cpp"
#endif


#endif
