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


#if !defined (FLUSH_PRINT_H)
	#define FLUSH_PRINT_H
  
 
#include <vector>
#include <string>
using namespace std;




#include "sources/shared/system_support/simd_basics.h"


 
//**********************************************************************************************************************************


enum INFO_LEVELS {INFO_SILENCE, INFO_1, INFO_2, INFO_3, INFO_DEBUG, INFO_PEDANTIC_DEBUG, INFO_VERY_PEDANTIC_DEBUG, INFO_EXTREMELY_PEDANTIC_DEBUG, INFO_LEVELS_MAX};
enum WARN_LEVELS {WARN_SILENCE, WARN_MEDIUM, WARN_ALL, WARN_LEVELS_MAX};
enum ERROR_CAUSES {ERROR_SILENT, ERROR_UNSPECIFIED, ERROR_IO, ERROR_DATA_MISMATCH, ERROR_DATA_STRUCTURE, ERROR_OUT_OF_MEMORY, ERROR_COMMAND_LINE, ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, ERROR_RUNTIME, ERROR_CAUSES_MAX};

extern unsigned info_mode;
extern unsigned warn_mode;


void flush_info(const char* message_format,...);
void flush_info(unsigned level, const char* message_format,...);
bool flush_info_will_show(unsigned level);

void flush_warn(unsigned level, const char* message_format,...);
void flush_exit(int error_code, const char* message_format,...);

void deactivate_display();
void reactivate_display();


void dump();
void dump(simdd__ input_simdd);
template <typename Template_type> void dump(Template_type* input);
template <typename Template_type> void dump(Template_type* input, unsigned size);
template <typename Template_type> void dump(Template_type input);
template <typename Template_type> void dump(vector <Template_type> input);

template <typename Template_type1, typename Template_type2> void dump(Template_type1 input1, Template_type2 input2);
template <typename Template_type1, typename Template_type2, typename Template_type3> void dump(Template_type1 input1, Template_type2 input2, Template_type3 input3);
template <typename Template_type1, typename Template_type2, typename Template_type3, typename Template_type4> void dump(Template_type1 input1, Template_type2 input2, Template_type3 input3, Template_type4 input4);



//**********************************************************************************************************************************


#include "sources/shared/basic_functions/flush_print.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_functions/flush_print.cpp"
#endif

#endif
