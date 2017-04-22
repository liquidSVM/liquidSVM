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



#include "sources/shared/basic_functions/flush_print.h"

#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>
#include <sstream>
using namespace std;



//**********************************************************************************************************************************


template <typename Template_type> bool string_to_number(char* string, Template_type& number)
{	
	if ((typeid(number) == typeid(bool)) or (typeid(number) == typeid(int)) or (typeid(number) == typeid(unsigned)))
	{
		if (is_integer(string) == false)
		{
			number = 0;
			return false;
		}
		else
		{	
			number = atoi(string);
			return true;
		}
	}
	else if ((typeid(number) == typeid(float)) or (typeid(number) == typeid(double)))
	{
		if (is_real(string) == false)
		{
			number = 0.0;
			return false;
		}
		else
		{	
			number = atof(string);
			return true;
		}
	}
	else
	{
		number = 0;
		return false;
	}
}


//**********************************************************************************************************************************


template <typename Template_type> bool string_to_number(char* string, Template_type& number, Template_type min, Template_type max)
{
	return (string_to_number(string, number) and (number >= min) and (number <= max));
}


//**********************************************************************************************************************************


template <typename Template_type> bool string_to_number_no_limits(char* string, Template_type& number, Template_type min, Template_type max)
{
	return (string_to_number(string, number) and (number > min) and (number < max));
}

//**********************************************************************************************************************************


template <typename Template_type> bool string_to_number_no_lower_limits(char* string, Template_type& number, Template_type min, Template_type max)
{
	return (string_to_number(string, number) and (number > min) and (number <= max));
}

//**********************************************************************************************************************************


template <typename Template_type> bool string_to_number_no_upper_limits(char* string, Template_type& number, Template_type min, Template_type max)
{
	return (string_to_number(string, number) and (number >= min) and (number < max));
}


//**********************************************************************************************************************************


template <typename Template_type> string number_to_string(Template_type number, unsigned precision, unsigned mode)
{
	stringstream output_stream;
	
	if ((typeid(number) == typeid(float)) or (typeid(number) == typeid(double)))
	{
		output_stream.precision(precision);
		output_stream.fill(' ');
		if (mode == DISPLAY_SCIENTIFIC)
			output_stream.setf(ios_base::scientific, ios_base::floatfield);
		else 
			output_stream.setf(ios_base::fixed, ios_base::floatfield);
		output_stream << number;
	}
	else
	{
		output_stream.width(precision);
		output_stream << number;
	}	
	return output_stream.str();
}



//**********************************************************************************************************************************


template <typename Template_type> string number_to_adjusted_string(Template_type number, unsigned precision, unsigned mode)
{
	string output;
	
	
	output = number_to_string(number, precision, mode);
	if (output[0] != '-')
		output = " " + output;
		
	return output;
}

//**********************************************************************************************************************************


template <typename Template_type> string pos_number_to_string(Template_type number, unsigned precision)
{
	string output;
	
	if (double(number) >= 0.0)
		return number_to_string(number, precision, DISPLAY_FLOAT);
	else
	{
		output.assign(precision -1, ' ');
		return output + "---";
	}
}
