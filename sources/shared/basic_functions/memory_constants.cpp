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


#if !defined (MEMORY_CONSTANTS_CPP) 
	#define MEMORY_CONSTANTS_CPP
 

#include "sources/shared/basic_functions/memory_constants.h"


//**********************************************************************************************************************************


unsigned convert_to_KB(size_t size)
{
	return unsigned(double(size)/double(KILOBYTE));
}

//**********************************************************************************************************************************

unsigned convert_to_MB(size_t size)
{
	return unsigned(double(size)/double(MEGABYTE));	
}

//**********************************************************************************************************************************

unsigned convert_to_GB(size_t size)
{
	return unsigned(double(size)/double(GIGABYTE));
}

#endif

