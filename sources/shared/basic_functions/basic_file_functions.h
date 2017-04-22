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


#if !defined (BASIC_FILE_FUNCTIONS_H) 
	#define BASIC_FILE_FUNCTIONS_H


#include <vector>
#include <cstdio>
#include <string> 
using namespace std;


//**********************************************************************************************************************************
 

enum FILETYPES {UNKNOWN_FILETYPE, NLA, CSV, WSV, LSV, UCI, LOG, AUX, SOL, FSOL, FILETYPES_MAX};
enum FILE_ERROR_MESSAGES {FILE_OP_OK, END_OF_LINE, END_OF_FILE, ILLEGAL_FILETYPE, FILE_CORRUPTED, LSV_FILE_CORRUPTED, INPUT_TYPE_UNKNOWN, FILE_ERROR_MESSAGES_MAX};


void exit_on_file_error(int error_type, FILE* fp);
void exit_on_file_error(int error_type, const string& filename);

unsigned get_filetype(const string& filename);

void check_data_filename(const string& filename);
void check_labeled_data_filename(const string& filename);
void check_unlabeled_data_filename(const string& filename);
void check_log_filename(const string& filename);
void check_aux_filename(const string& filename);
void check_solution_filename(const string& filename);

string convert_log_to_aux_filename(const string& filename);

bool file_exists(const string& filename);
FILE* open_file(const string& filename, const char* mode);
void close_file(FILE* fp);
string get_filename_of_fp(FILE* fp);


void file_read(FILE* fp, bool& input);
void file_read(FILE* fp, int& input);
void file_read(FILE* fp, unsigned& input);
void file_read(FILE* fp, double& input);
void file_read(FILE* fp, unsigned& i, double& x);
void file_read(FILE* fp, string& input);


void file_read(FILE* fp, vector <double>& input);
template <typename Template_type> void file_read(FILE* fp, vector <Template_type>& input);


void file_write_eol(FILE* fp);
void file_write(FILE* fp, bool output, string separator = " ");
void file_write(FILE* fp, int output, string separator = " ");
void file_write(FILE* fp, unsigned output, string separator = " ");
void file_write(FILE* fp, double output, string format = "%3.15f ", string separator = " ");
void file_write(FILE* fp, unsigned index, double output, string separator = "");
void file_write(FILE* fp, string output, string separator = " ");


void file_write(FILE* fp, vector <double> output, string format = "%3.15f ", string separator = "");
template <typename Template_type> void file_write(FILE* fp, vector <Template_type> output, string separator = " ");


//**********************************************************************************************************************************


#include "sources/shared/basic_functions/basic_file_functions.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_functions/basic_file_functions.cpp"
#endif


#endif

