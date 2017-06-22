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


#if !defined (SAMPLE_FILE_FORMAT_CPP) 
	#define SAMPLE_FILE_FORMAT_CPP


#include "sources/shared/basic_types/sample_file_format.h"


	
#include "sources/shared/basic_functions/flush_print.h"	
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_types/vector.h"



	
//**********************************************************************************************************************************
	

Tsample_file_format::Tsample_file_format()
{
	clear();
};


//**********************************************************************************************************************************


void Tsample_file_format::read_from_file(FILE *fp)
{
	file_read(fp, label_position);
	file_read(fp, weight_position);
	file_read(fp, id_position);
	file_read(fp, group_id_position);
}

//**********************************************************************************************************************************


void Tsample_file_format::write_to_file(FILE *fp) const
{
	file_write(fp, label_position);
	file_write(fp, weight_position);
	file_write(fp, id_position);
	file_write(fp, group_id_position);
	file_write_eol(fp);
}
	

	
//**********************************************************************************************************************************


void Tsample_file_format::display(unsigned level) const
{
	flush_info(level, "\n\nLabel position:    %5d", label_position);
	flush_info(level, "\nWeight position:   %5d", weight_position);
	flush_info(level, "\nID position:       %5d", id_position);
	flush_info(level, "\nGroup ID position: %5d\n", group_id_position);
}
	

//**********************************************************************************************************************************


void Tsample_file_format::clear()
{
	label_position = 1;
	weight_position = 0;
	id_position = 0;
	group_id_position = 0;
	
	filetype = CSV;
	dataset_dim = 0;
	
	user_include_list.clear();
	user_exclude_list.clear();

	full_include_list.clear();
	full_exclude_list.clear();
	extra_position_list.clear();
	
	filename.clear();
}


//**********************************************************************************************************************************

void Tsample_file_format::update_filetype()
{
	filetype = get_filetype(filename);
}

//**********************************************************************************************************************************

void Tsample_file_format::check_filetype()
{
	unsigned filetype_tmp;
	
	if (filename.size() > 0)
	{
		filetype_tmp = get_filetype(filename);
		
		if ((filetype == LSV) and (filetype_tmp == CSV))
			flush_exit(ERROR_DATA_MISMATCH, "Data file %s is not of type LSV as specified in Tsample_file_format object.", filename.c_str());
		
		if ((filetype == CSV) and (filetype_tmp == LSV))
			flush_exit(ERROR_DATA_MISMATCH, "Data file %s is not of type CSV as specified in Tsample_file_format object.", filename.c_str());
	}
}
	

//**********************************************************************************************************************************


void Tsample_file_format::compute_full_include_list(unsigned number_of_columns)
{
	unsigned i;
	my_unordered_set <unsigned> include_set_tmp;
	
	
	if ((user_include_list.size() > 0) and (user_exclude_list.size() > 0))
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Cannot handle non-empty include and exclude lists simultaneously."); 
	
	
	full_include_list.clear();
	if (user_include_list.size() > 0)
		for (i=0; i<user_include_list.size(); i++)
			full_include_list.push_back(get_true_column(user_include_list[i], number_of_columns));
	else
	{
		for (i=0; i<number_of_columns; i++)
			include_set_tmp.insert(i+1);
		
		for (i=0; i<user_exclude_list.size(); i++)
			include_set_tmp.erase(get_true_column(user_exclude_list[i], number_of_columns));
		
		copy(include_set_tmp.begin(), include_set_tmp.end(), inserter(full_include_list, full_include_list.begin()));
	}
	sort_up(full_include_list);
}


//**********************************************************************************************************************************


void Tsample_file_format::compute_full_exclude_list(unsigned number_of_columns)
{
	unsigned i;
	my_unordered_set <unsigned> exclude_set_tmp;
	
	
	if ((user_include_list.size() > 0) and (user_exclude_list.size() > 0))
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Cannot handle non-empty include and exclude lists simultaneously."); 
	
	full_exclude_list.clear();
	if (user_exclude_list.size() > 0)
		for (i=0; i<user_exclude_list.size(); i++)
			exclude_set_tmp.insert(get_true_column(user_exclude_list[i], number_of_columns));
	else if (user_include_list.size() > 0)
	{
		for (i=0; i<number_of_columns; i++)
			exclude_set_tmp.insert(i+1);
		
		for (i=0; i<user_include_list.size(); i++)
			exclude_set_tmp.erase(get_true_column(user_include_list[i], number_of_columns));
	}
	
	compute_extra_position_list(number_of_columns);
	for (i=0; i<extra_position_list.size(); i++)
		exclude_set_tmp.insert(extra_position_list[i]);
	
	copy(exclude_set_tmp.begin(), exclude_set_tmp.end(), inserter(full_exclude_list, full_exclude_list.begin()));
	sort_up(full_exclude_list);
}
	

//**********************************************************************************************************************************

	
unsigned Tsample_file_format::get_true_column(int column, unsigned number_of_columns)
{
	unsigned true_column;
	
	
	true_column = 0;
	if (column == 0)
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Cannot consider column 0."); 
	
	if (column > 0)
	{
		if (column > int(number_of_columns))
			flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Cannot consider column %d for samples with %d columns.", column, number_of_columns);
		else
			true_column = unsigned(column);
	}
	
	if (column < 0)
	{
		if (int(number_of_columns + 1) + column < 1)
			flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Cannot consider column %d for samples with %d columns", column, number_of_columns);
		else
			true_column = unsigned(int(number_of_columns + 1) + column);
	}
	
	return true_column;
}


//**********************************************************************************************************************************

	
void Tsample_file_format::compute_extra_position_list(unsigned number_of_columns)
{
	unsigned extra_positions;


	extra_position_list.clear();
	if (label_position != 0)
		extra_position_list.push_back(get_true_column(label_position, number_of_columns));
	
	if (weight_position != 0)
		extra_position_list.push_back(get_true_column(weight_position, number_of_columns));
	
	if (id_position != 0)
		extra_position_list.push_back(get_true_column(id_position, number_of_columns));
	
	if (group_id_position != 0)
		extra_position_list.push_back(get_true_column(group_id_position, number_of_columns));

	extra_positions = extra_position_list.size();
	extra_position_list = get_unique_entries(extra_position_list);
	if (extra_positions != extra_position_list.size())
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Some extra positions for samples occured twice."); 
		
	sort_up(extra_position_list);
}

//**********************************************************************************************************************************

	
unsigned Tsample_file_format::count_extra_positions()
{
	unsigned extra_positions;


	extra_positions = 0;
	if (label_position != 0)
		extra_positions++;
	
	if (weight_position != 0)
		extra_positions++;
	
	if (id_position != 0)
		extra_positions++;
	
	if (group_id_position != 0)
		extra_positions++;

	return extra_positions;
}


//**********************************************************************************************************************************




	
#endif
