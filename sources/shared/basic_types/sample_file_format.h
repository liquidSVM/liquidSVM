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


#if !defined (SAMPLE_FILE_FORMAT_H) 
	#define SAMPLE_FILE_FORMAT_H


#include <vector>
#include <cstdio>
#include <string>
using namespace std;


	
//**********************************************************************************************************************************
	
	
	
class Tsample_file_format
{
	public:
		Tsample_file_format();
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;
		void display(unsigned level) const;
		void clear();


		unsigned count_extra_positions();
		void compute_extra_position_list(unsigned number_of_columns);
		void compute_full_include_list(unsigned number_of_columns);
		void compute_full_exclude_list(unsigned number_of_columns);
		unsigned get_true_column(int column, unsigned number_of_columns);
		void update_filetype();
		void check_filetype();
		
		
		int label_position;
		int weight_position;
		int id_position;
		int group_id_position;
		
		string filename;
		unsigned filetype;
		unsigned dataset_dim;
		
		
		vector <unsigned> user_include_list;
		vector <unsigned> user_exclude_list;
		
		vector <unsigned> full_include_list;
		vector <unsigned> full_exclude_list;
		vector <unsigned> extra_position_list;
};
	
	
	
	
//**********************************************************************************************************************************



#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/sample_file_format.cpp"
#endif
	
	
#endif

