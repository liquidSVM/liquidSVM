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


#if !defined (BASIC_FILE_FUNCTIONS_CPP) 
	#define BASIC_FILE_FUNCTIONS_CPP



#include "sources/shared/basic_functions/basic_file_functions.h"

#include "sources/shared/system_support/os_specifics.h"
#include "sources/shared/basic_functions/flush_print.h"


#include <stdio.h>
#include <cmath>  
#include <map>
using namespace std;

#ifdef _WIN32
#ifndef __MINGW32__
	#pragma warning(disable:4996)
#endif
	#define _CRT_SECURE_NO_DEPRECATE
	#define _CRT_SECURE_NO_WARNINGS
#endif

//**********************************************************************************************************************************


map <FILE*, string> openfiles;


//**********************************************************************************************************************************

void exit_on_file_error(int error_type, const string& filename)
{
	switch (error_type)
	{
		case FILE_CORRUPTED:
			flush_exit(ERROR_IO, "File '%s' is corrupted.", filename.c_str());
		case LSV_FILE_CORRUPTED:
			flush_exit(ERROR_IO, "File '%s' is corrupted since the dimension indices are not increasing.", filename.c_str());
		case INPUT_TYPE_UNKNOWN:
			flush_exit(ERROR_IO, "Trying to read a C++ type from file '%s' not covered by function file_read(...).", filename.c_str());
	}
}


//**********************************************************************************************************************************

void exit_on_file_error(int error_type, FILE* fp)
{
	exit_on_file_error(error_type, openfiles[fp]);
}

//**********************************************************************************************************************************

void check_data_filename(const string& filename)
{
	unsigned filetype;
	
	filetype = get_filetype(filename);
	if ((filetype != CSV) and (filetype != WSV) and (filetype != LSV) and (filetype != UCI) and (filetype != NLA))
		flush_exit(ERROR_IO, "Data file '%s' does not have one of the allowed types: '.lsv', '.csv', '.wsv', .uci', or '.nla'.", filename.c_str());
}

//**********************************************************************************************************************************

void check_labeled_data_filename(const string& filename)
{
	unsigned filetype;
	
	filetype = get_filetype(filename);
	if ((filetype != CSV) and (filetype != WSV) and (filetype != LSV) and (filetype != UCI))
		flush_exit(ERROR_IO, "Labeled data file '%s' does not have one of the allowed types: '.lsv', '.csv', '.wsv', or '.uci'.", filename.c_str());
}

//**********************************************************************************************************************************

void check_unlabeled_data_filename(const string& filename)
{
	unsigned filetype;
	
	filetype = get_filetype(filename);
	if (filetype != NLA)
		flush_exit(ERROR_IO, "Unlabeled data file '%s' does not have one of the allowed types: '.nla'.", filename.c_str());
}


//**********************************************************************************************************************************

void check_log_filename(const string& filename)
{
	unsigned filetype;
	
	filetype = get_filetype(filename);
	if (filetype != LOG)
		flush_exit(ERROR_IO, "Log file '%s' does not have one of the allowed types: '.log'.", filename.c_str());
}


//**********************************************************************************************************************************

void check_aux_filename(const string& filename)
{
	unsigned filetype;
	
	filetype = get_filetype(filename);
	if (filetype != AUX)
		flush_exit(ERROR_IO, "Aux file '%s' does not have one of the allowed types: '.aux'.", filename.c_str());
}

//**********************************************************************************************************************************

void check_solution_filename(const string& filename)
{
	unsigned filetype;
	
	filetype = get_filetype(filename);
	if ((filetype != SOL) and (filetype != FSOL))
		flush_exit(ERROR_IO, "Solution file '%s' does not have one of the allowed types: '.sol' or '.fsol'.", filename.c_str());
}

//**********************************************************************************************************************************


bool file_exists(const string& filename)
{
	FILE* fp;
	bool exists;

	fp = fopen(filename.c_str(), "r");
	if (fp == NULL)
		exists = false;
	else
	{
		exists = true;
		fclose(fp);
	}
	return exists;
}


//**********************************************************************************************************************************


FILE* open_file(const string& filename, const char* mode)
{
	FILE* fp;

	if (filename.size() == 0)
		return NULL;

	#if !defined(_WIN32) || defined (__MINGW32__)
		if ((mode[0] == 'r') and (filename.substr(std::max(size_t(3), filename.size()) - 3,string::npos) == ".gz"))
			fp = popen(("gzip -cd "+filename).c_str(), mode);
		else
	#endif
			fp = fopen(filename.c_str(), mode);
	
	if (fp == NULL)
		flush_exit(ERROR_IO, "File '%s' cannot be opened.", filename.c_str());

	openfiles[fp] = filename;
	return fp;
}


//**********************************************************************************************************************************


void close_file(FILE* fp)
{
	if (fp != NULL)
	{
		openfiles.erase(fp);
		fclose(fp);
	}
}

//**********************************************************************************************************************************

string get_filename_of_fp(FILE* fp)
{
	string filename; 
	std::map<FILE*, string>::iterator it;
	
	it = openfiles.find(fp);
  if (it != openfiles.end())
		filename = openfiles[fp];
	
	return filename;
}


//**********************************************************************************************************************************


string convert_log_to_aux_filename(const string& filename)
{
	check_log_filename(filename);
	return filename.substr(0, filename.length() - 4) + ".aux";
}

//**********************************************************************************************************************************

unsigned get_filetype(const string& filename)
{
	unsigned last_dot_position;
	string extension;
	
	
	if (filename.size() < 4)
		return UNKNOWN_FILETYPE;
		
	last_dot_position = unsigned(filename.find_last_of('.'));
	if (last_dot_position == unsigned(string::npos))
		return UNKNOWN_FILETYPE;

	extension = filename.substr(last_dot_position, filename.length() - last_dot_position);
	if (extension == ".nla")
		return NLA;
	if (extension == ".csv")
		return CSV;
	if (extension == ".wsv")
		return WSV;
	if (extension == ".lsv")
		return LSV;
	if (extension == ".uci")
		return UCI;
	if (extension == ".log")
		return LOG;
	if (extension == ".aux")
		return AUX;
	if (extension == ".sol")
		return SOL;
	if (extension == ".fsol")
		return FSOL;
	
	#if !defined(_WIN32) || defined (__MINGW32__)
		if (extension == ".gz")
			return get_filetype(filename.substr(0, filename.size() - 3));
	#endif

	return UNKNOWN_FILETYPE;
}



//**********************************************************************************************************************************


void file_read(FILE* fp, unsigned& i, double& x)
{
	int io_return;
	
	io_return = fscanf(fp, "%u:%lf",&i, &x);
	if ((io_return == 0) or (io_return == EOF))
		exit_on_file_error(FILE_CORRUPTED, fp);
}


//**********************************************************************************************************************************


void file_read(FILE* fp, bool& input)
{
	int io_return;
	int idummy;
	
	io_return = fscanf(fp, "%d", &idummy);
	input = bool(idummy);
	if ((io_return == 0) or (io_return == EOF))
		exit_on_file_error(FILE_CORRUPTED, fp);
}

//**********************************************************************************************************************************


void file_read(FILE* fp, int& input)
{
	int io_return;
	
	io_return = fscanf(fp, "%d", &input);
	if ((io_return == 0) or (io_return == EOF))
		exit_on_file_error(FILE_CORRUPTED, fp);
}

//**********************************************************************************************************************************

void file_read(FILE* fp, unsigned& input)
{
	int io_return;
	
	io_return = fscanf(fp, "%u", &input);
	if ((io_return == 0) or (io_return == EOF))
		exit_on_file_error(FILE_CORRUPTED, fp);
}

//**********************************************************************************************************************************


void file_read(FILE* fp, double& input)
{
	int io_return;
	
	io_return = fscanf(fp, "%lf", &input);
	if ((io_return == 0) or (io_return == EOF))
		exit_on_file_error(FILE_CORRUPTED, fp);
}

//**********************************************************************************************************************************

void file_read(FILE* fp, string& input)
{
	char c;

	do
		c = getc(fp);
	while ((c != 34) and (c != EOF));
	
	if (c == EOF)
		exit_on_file_error(FILE_CORRUPTED, fp);
		
	input.clear();
	do 
	{
		c = getc(fp);
		if (c != 34)
			input.push_back(c);
	}
	while ((c != 34) and (c != EOF));
	
	if (c == EOF)
		exit_on_file_error(FILE_CORRUPTED, fp);
	
	c = getc(fp);
}


//**********************************************************************************************************************************


void file_read(FILE* fp, vector <double>& input)
{
	unsigned i;
	unsigned size;
	
	file_read(fp, size);
	input.resize(size);
	
	for (i=0; i<size; i++)
		file_read(fp, input[i]);
}

//**********************************************************************************************************************************


void file_write_eol(FILE* fp)
{
	if (fp != NULL)
		fprintf(fp, "\n");
}

//**********************************************************************************************************************************


void file_write(FILE* fp, double output, string format, string separator)
{
	if (fp != NULL)
	{
		if (abs(output) < 1.0e-14)
			output = 0.0;
		
		fprintf(fp, format.c_str(), output);
		if (separator.size() > 0)
			fprintf(fp, "%s", separator.c_str());
	}
}

//**********************************************************************************************************************************


void file_write(FILE* fp, bool output, string separator)
{
	if (fp != NULL)
	{
		fprintf(fp, "%d", output);
		fprintf(fp, "%s", separator.c_str());
	}
}

//**********************************************************************************************************************************



void file_write(FILE* fp, int output, string separator)
{
	if (fp != NULL)
	{
		fprintf(fp, "%d", output);
		fprintf(fp, "%s", separator.c_str());
	}
}



//**********************************************************************************************************************************

void file_write(FILE* fp, unsigned output, string separator)
{
	if (fp != NULL)
	{
		fprintf(fp, "%d", output);
		fprintf(fp, "%s", separator.c_str());
	}
}


//**********************************************************************************************************************************

void file_write(FILE* fp, unsigned index, double output, string separator)
{
	if (fp != NULL)
	{
		file_write(fp, index, ":");
		file_write(fp, output);
		fprintf(fp, "%s", separator.c_str());
	}
}


//**********************************************************************************************************************************


void file_write(FILE* fp, string output, string separator)
{
	if (fp != NULL)
	{
		fprintf(fp, "\"%s\"", output.c_str());
		fprintf(fp, "%s", separator.c_str());
	}
}


//**********************************************************************************************************************************


void file_write(FILE* fp, vector <double> output, string format, string separator)
{
	unsigned i;
	
	if (fp != NULL)
	{
		
		file_write(fp, unsigned(output.size()), separator); 
		file_write_eol(fp);
		
		for (i=0; i<output.size(); i++)
			file_write(fp, output[i], format, separator);
		if (output.size() > 0)
			file_write_eol(fp);
	}
}


#endif


