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


#if !defined (EXTRA_STRING_FUNCTIONS_CPP) 
	#define EXTRA_STRING_FUNCTIONS_CPP


#include "sources/shared/basic_functions/extra_string_functions.h"

#include <stdlib.h>
#include <ctype.h>
#include <string.h> 




//**********************************************************************************************************************************

bool is_integer(char* string)
{
	unsigned i;
	unsigned first_digit;
	bool all_digits;

	
	if (strlen(string) == 0)
		return false;
	
	first_digit = 0;
	if (string[0] == '-')
		first_digit = 1;
	
	all_digits = true;
	for (i=first_digit; (i<strlen(string) and all_digits); i++)
		all_digits = (isdigit(string[i]) > 0);

	return all_digits;
}


//**********************************************************************************************************************************

bool is_real(char* string)
{
	unsigned i;
	unsigned first_digit;
	bool all_digits;
	unsigned dots;
	

	if (strlen(string) == 0)
		return false;	
	
	first_digit = 0;
	if (string[0] == '-')
		first_digit = 1;
	
	dots = 0;
	all_digits = true;	
	for (i=first_digit; (i<strlen(string) and all_digits); i++)
		if (string[i] == '.')
			dots++;
		else
			all_digits = (isdigit(string[i]) > 0);
	
	return (all_digits and dots <=1);
}


//**********************************************************************************************************************************

string reduce(const string& original_string)
{
	size_t first_none_space;
	
	first_none_space = original_string.find_first_not_of(" \t");
	if (first_none_space == std::string::npos)
		return ""; 
	else
		return original_string.substr(first_none_space, std::string::npos);
}


#endif
