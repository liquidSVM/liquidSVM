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


#if !defined (FOLD_CONTROL_CPP)
	#define FOLD_CONTROL_CPP


#include "sources/shared/training_validation/fold_control.h"



#include "sources/shared/basic_functions/basic_file_functions.h"


//**********************************************************************************************************************************


Tfold_control::Tfold_control()
{
	number = 5;
	kind = RANDOM;
	train_fraction = 1.0;
	negative_fraction = FRACTION_NOT_ASSIGNED;
	random_seed = -1;
}

//**********************************************************************************************************************************


void Tfold_control::read_from_file(FILE *fp)
{
	file_read(fp, number);
	file_read(fp, kind);
	file_read(fp, train_fraction);
	file_read(fp, negative_fraction);
	file_read(fp, random_seed);
};

//**********************************************************************************************************************************

void Tfold_control::write_to_file(FILE *fp) const
{
	file_write(fp, number);
	file_write(fp, kind);
	file_write(fp, train_fraction);
	file_write(fp, negative_fraction);
	file_write(fp, random_seed);
	file_write_eol(fp);
};

#endif






