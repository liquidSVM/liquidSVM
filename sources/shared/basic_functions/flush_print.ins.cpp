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







#include "sources/shared/basic_types/sample.h"



#include <typeinfo>
using namespace std;



//**********************************************************************************************************************************


#if defined(_WIN32) && !defined(__MINGW32__)
	#define COMPILE_WITHOUT_EXCEPTIONS__
	#define COMPILE_FOR_COMMAND_LINE__
#endif



#ifdef COMPILE_FOR_COMMAND_LINE__ 
	#define VPRINTF(message_format) va_list arguments; \
									va_start(arguments, message_format); \
									vprintf(message_format, arguments); \
									va_end(arguments); \
									fflush(stdout);
#elif !defined(VPRINTF)
	#define VPRINTF(message_format)
#endif


//**********************************************************************************************************************************


void ddump(string input);
void ddump(char* input);
void ddump(bool input);
void ddump(int input);
void ddump(unsigned input);
#ifdef OWN_DEVELOP__
	void ddump(size_t input);
#endif
void ddump(float input);
void ddump(double input);
void ddump(double* pointer);
void ddump(const Tsample& input_sample);


//**********************************************************************************************************************************


template <typename Template_type> void dump(Template_type input)
{
	flush_info("\n");
	ddump(input);
}


//**********************************************************************************************************************************


template <typename Template_type> void dump(Template_type* input)
{
	flush_info("\n%p ", input);
}


//**********************************************************************************************************************************


template <typename Template_type> void dump(vector <Template_type> input)
{
	unsigned i;
	
	
	flush_info("\n");
	for (i=0; i<input.size(); i++)
		ddump(input[i]);
}

//**********************************************************************************************************************************


template <typename Template_type> void dump(Template_type* input, unsigned size)
{
	unsigned i;
	
	
	flush_info("\n");
	for (i=0; i<size; i++)
		ddump(input[i]);
}

//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> void dump(Template_type1 input1, Template_type2 input2)
{
	flush_info("\n");
	ddump(input1);
	ddump(input2);
}



//**********************************************************************************************************************************



template <typename Template_type1, typename Template_type2, typename Template_type3> void dump(Template_type1 input1, Template_type2 input2, Template_type3 input3)
{
	flush_info("\n");
	ddump(input1);
	ddump(input2);
	ddump(input3);
}


//**********************************************************************************************************************************



template <typename Template_type1, typename Template_type2, typename Template_type3, typename Template_type4> void dump(Template_type1 input1, Template_type2 input2, Template_type3 input3, Template_type4 input4)
{
	flush_info("\n");
	ddump(input1);
	ddump(input2);
	ddump(input3);
	ddump(input4);
}






