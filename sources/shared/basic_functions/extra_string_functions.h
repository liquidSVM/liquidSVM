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


#if !defined (EXTRA_STRING_FUNCTIONS_H) 
	#define EXTRA_STRING_FUNCTIONS_H
   
  
#include <limits>
#include <string>
#include <cstring>
using namespace std;


//**********************************************************************************************************************************


enum {DISPLAY_SCIENTIFIC, DISPLAY_FLOAT};

 
bool is_integer(char* string);
bool is_real(char* string);

template <typename Template_type> bool string_to_number(char* string, Template_type& number);
template <typename Template_type> bool string_to_number(char* string, Template_type& number, Template_type min, Template_type max = numeric_limits<Template_type>::max( ));
template <typename Template_type> bool string_to_number_no_limits(char* string, Template_type& number, Template_type min, Template_type max = numeric_limits<Template_type>::max( ));
template <typename Template_type> bool string_to_number_no_lower_limits(char* string, Template_type& number, Template_type min, Template_type max = numeric_limits<Template_type>::max( ));
template <typename Template_type> bool string_to_number_no_upper_limits(char* string, Template_type& number, Template_type min, Template_type max = numeric_limits<Template_type>::max( ));


template <typename Template_type> string number_to_string(Template_type number, unsigned precision, unsigned mode = DISPLAY_SCIENTIFIC);
template <typename Template_type> string number_to_adjusted_string(Template_type number, unsigned precision, unsigned mode = DISPLAY_SCIENTIFIC);
template <typename Template_type> string pos_number_to_string(Template_type number, unsigned precision);


string reduce(const string& original_string);

//**********************************************************************************************************************************


#include "sources/shared/basic_functions/extra_string_functions.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_functions/extra_string_functions.cpp"
#endif

#endif
