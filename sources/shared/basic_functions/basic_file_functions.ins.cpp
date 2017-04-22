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




#include <typeinfo>
using namespace std;

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/system_support/os_specifics.h"

#ifdef _WIN32
#ifndef __MINGW32__
	#pragma warning(disable:4996)
#endif
	#define _CRT_SECURE_NO_DEPRECATE
	#define _CRT_SECURE_NO_WARNINGS
#endif



//**********************************************************************************************************************************


template <typename Template_type> void file_read(FILE* fp, vector <Template_type>& input)
{
	unsigned i;
	unsigned size;
	
	file_read(fp, size);
	input.resize(size);
	
	for (i=0; i<size; i++)
		file_read(fp, input[i]);
}


//**********************************************************************************************************************************


template <typename Template_type> void file_write(FILE* fp, vector <Template_type> output, string separator)
{
	unsigned i;
	
	if (fp != NULL)
	{
		file_write(fp, unsigned(output.size())); 
		file_write_eol(fp);
		
		for (i=0; i<output.size(); i++)
			file_write(fp, output[i], separator);
		if (output.size() > 0)
			file_write_eol(fp);
	}
}

