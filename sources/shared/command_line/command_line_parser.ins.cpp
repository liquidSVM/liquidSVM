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



#include "sources/shared/basic_functions/extra_string_functions.h"


#include "sources/shared/basic_functions/flush_print.h"


#include <limits>
using namespace std; 

//**********************************************************************************************************************************
//
// This little beauty is because of a compiler bug in Visual C++.


template <typename Template_type> Template_type get_limits_max() 
{ 
	return std::numeric_limits<Template_type>::max(); 
}


//**********************************************************************************************************************************


template <typename Template_type> Template_type Tcommand_line_parser::get_next_number(unsigned error_code, Template_type min, Template_type max)
{
	Template_type number;

	check_parameter_position(error_code);

	if (string_to_number(parameter_list[current_position], number, min, max) == false)
		this->exit_with_help(error_code);
	return number;
}

//**********************************************************************************************************************************


template <typename Template_type> Template_type  Tcommand_line_parser::get_next_number_no_limits(unsigned error_code, Template_type min, Template_type max)
{
	Template_type number;

	check_parameter_position(error_code);

	if (string_to_number_no_limits(parameter_list[current_position], number, min, max) == false)
		this->exit_with_help(error_code);
	return number;
}

//**********************************************************************************************************************************


template <typename Template_type> Template_type  Tcommand_line_parser::get_next_number_no_lower_limits(unsigned error_code, Template_type min, Template_type max)
{
	Template_type number;

	check_parameter_position(error_code);

	if (string_to_number_no_lower_limits(parameter_list[current_position], number, min, max) == false)
		this->exit_with_help(error_code);
	return number;
}

//**********************************************************************************************************************************


template <typename Template_type> Template_type  Tcommand_line_parser::get_next_number_no_upper_limits(unsigned error_code, Template_type min, Template_type max)
{
	Template_type number;

	check_parameter_position(error_code);

	if (string_to_number_no_upper_limits(parameter_list[current_position], number, min, max) == false)
		this->exit_with_help(error_code);
	return number;
}


//**********************************************************************************************************************************

template <typename Template_type> vector <Template_type> Tcommand_line_parser::get_next_list(unsigned error_code, Template_type min, Template_type max)
{
	vector <Template_type> return_list;


	if (next_parameter_equals('[') == false)
		this->exit_with_help(error_code);
	
	current_position++;
	do
		return_list.push_back(get_next_number(error_code, min, max));
	while (next_parameter_equals(']') == false);
	current_position++;
	
	return return_list;
}
