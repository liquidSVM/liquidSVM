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


#if !defined (FLUSH_PRINT_CPP)
	#define FLUSH_PRINT_CPP



#include "sources/shared/basic_functions/flush_print.h"




#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>


#ifdef _WIN32
	#ifndef __MINGW32__
		#pragma warning(disable:4996)
	#endif
	#define _CRT_SECURE_NO_DEPRECATE
	#define _CRT_SECURE_NO_WARNINGS
#endif





//**********************************************************************************************************************************


unsigned info_mode = INFO_1;
unsigned warn_mode = WARN_MEDIUM;
unsigned info_mode_back_up;
bool info_mode_back_up_activated = false;


//**********************************************************************************************************************************


void my_exit(int error_code)
{
	#ifdef COMPILE_FOR_COMMAND_LINE__ 
		exit(error_code);
	#else
		
	#endif
}


//**********************************************************************************************************************************


void deactivate_display()
{
	if (info_mode_back_up_activated == false)
	{
		info_mode_back_up = info_mode;
		if (info_mode < INFO_DEBUG)
			info_mode = INFO_SILENCE;
		info_mode_back_up_activated = true;
	}
	else
		flush_exit(ERROR_UNSPECIFIED, "Trying to deactivate display that has already been deactivated.");
}



//**********************************************************************************************************************************


void reactivate_display()
{
	if (info_mode_back_up_activated == true)
	{
		info_mode = info_mode_back_up;
		info_mode_back_up_activated = false;
	}
	else
		flush_exit(ERROR_UNSPECIFIED, "Trying to re-activated display without having deactivated it.");
}



//**********************************************************************************************************************************

void flush_info(const char* message_format,...)
{
	VPRINTF(message_format)
}

//**********************************************************************************************************************************


bool flush_info_will_show(unsigned level)
{
	return (level <= info_mode);
}


//**********************************************************************************************************************************

void flush_info(unsigned level, const char* message_format,...)
{
	if (flush_info_will_show(level) == true)
	{
		VPRINTF(message_format)
	}
}


//**********************************************************************************************************************************

void flush_warn(unsigned level, const char* message_format,...)
{
	if (level <= warn_mode)
	{
		flush_info("\nWARNING: ");	
		VPRINTF(message_format)
	}
}


//**********************************************************************************************************************************

void flush_exit(int error_code, const char* message_format,...)
{
	#if !defined(COMPILE_WITHOUT_EXCEPTIONS__)
		va_list arguments_for_exceptions;
	#endif
	
	switch (error_code)
	{
		case ERROR_SILENT:
			break;
		
		case ERROR_IO:
			flush_info("\n\nIO ERROR:\n");
			break;

		case ERROR_DATA_MISMATCH:
			flush_info("\n\nDATA MISMATCH ERROR:\n");
			break;

		case ERROR_DATA_STRUCTURE:
			flush_info("\n\nDATA STRUCTURE ERROR:\n");
			break;
			
		case ERROR_OUT_OF_MEMORY:
			flush_info("\n\nERROR_OUT_OF_MEMORY:\n");
			break;
			
		case ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS:
			flush_info("\n\nERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS:\n");
			break;

		case ERROR_RUNTIME:
			flush_info("\n\nRUN TIME ERROR:\n");
			break;
			
		default:
			flush_info("\n\nERROR:\n");
			break;
	}
	
	VPRINTF(message_format)
	
	#if !defined(COMPILE_WITHOUT_EXCEPTIONS__)
		flush_warn(WARN_ALL, "This should usually not happen. It is more safe to restart your process right now.\n\n");
		char buffer[256];
		va_start(arguments_for_exceptions, message_format);
		vsprintf(buffer, message_format, arguments_for_exceptions);
		string exception = buffer;
		va_end(arguments_for_exceptions);
		throw exception;
	#else

	if (error_code != ERROR_SILENT)
		flush_info("\n\n");
	my_exit(error_code);

	#endif
}


//**********************************************************************************************************************************


void ddump(string input)
{
	flush_info("%s ", input.c_str());
}


//**********************************************************************************************************************************


void ddump(char* input)
{
	flush_info("%s ", input);
}


//**********************************************************************************************************************************


void ddump(bool input)
{
	if (input == true)
		flush_info("%s ", "true");
	else
		flush_info("%s ", "false");
}

//**********************************************************************************************************************************

void ddump(int input)
{
	flush_info("%d ", input);
}

//**********************************************************************************************************************************


void ddump(unsigned input)
{
	flush_info("%u ", input);
}

//**********************************************************************************************************************************

#ifdef OWN_DEVELOP__
	void ddump(size_t input)
	{
		flush_info("%zu ", input);
	}
#endif

//**********************************************************************************************************************************


void ddump(float input)
{
	flush_info("%f ", input);
}

//**********************************************************************************************************************************


void ddump(double input)
{
	flush_info("%2.10lf ", input);
}

//**********************************************************************************************************************************

void ddump(double* pointer)
{
	flush_info("%p ", pointer);
}


//**********************************************************************************************************************************


void dump(simdd__ input_simdd)
{
	#ifdef AVX__
		double* input_ptr;
		
		input_ptr = (double*)&input_simdd;
		flush_info("\n%lf %lf %lf %lf", input_ptr[0], input_ptr[1], input_ptr[2], input_ptr[3]);
	#elif defined SSE2__	
		double* input_ptr;
		
		input_ptr = (double*)&input_simdd;
		flush_info("\n%lf %lf", input_ptr[0], input_ptr[1]);
	#else
		flush_info("\n%lf ", input_simdd);
	#endif
}

//**********************************************************************************************************************************


void ddump(const Tsample& input_sample)
{
	unsigned i;
	
	flush_info("%2.4f:  ", input_sample.label);
	for (i=0; i<input_sample.dim(); i++)
		flush_info("%2.4f  ", input_sample[i]);
}


//**********************************************************************************************************************************


void dump()
{
	flush_info("\n-------------------------- Dump point passed ----------------------------------\n");
}


#endif
