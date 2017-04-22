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


#if !defined (WORKING_SET_CONTROL_CPP)
	#define WORKING_SET_CONTROL_CPP


#include "sources/shared/training_validation/working_set_control.h"


#include "sources/shared/basic_functions/basic_file_functions.h"

//**********************************************************************************************************************************


Tworking_set_control::Tworking_set_control()
{
	classification = true;

	size_of_tasks = 0;
	number_of_tasks = 0;
	
	working_set_selection_method = FULL_SET;
	partition_method = FULL_SET;
	number_of_covers = 1;
	reduce_covers = true;
	max_ignore_factor = 0.5;
	
	tree_reduction_factor = 2.0;
	max_tree_depth = 4;
	max_theoretical_node_width = 20;
	
	size_of_cells = 0;
	number_of_cells = 1;
	size_of_dataset_to_find_partition = 0;
	radius = 0.0;
	random_seed = -1;
};

//**********************************************************************************************************************************


void Tworking_set_control::set_partition_method_with_defaults(unsigned partition_method)
{
	Tworking_set_control::partition_method = partition_method;
	switch (partition_method)
	{
		case NO_PARTITION:
			break;
				
		case RANDOM_CHUNK_BY_SIZE:
			size_of_cells = 2000;
			break;
					
		case RANDOM_CHUNK_BY_NUMBER:
			number_of_cells = 10;
			break;
			
		case VORONOI_BY_RADIUS:
			radius = 1.0;
			size_of_dataset_to_find_partition = 0;
			size_of_dataset_to_find_partition = 0;
			break;
			
		case VORONOI_BY_SIZE:
			size_of_cells = 2000;
			reduce_covers = true;
			size_of_dataset_to_find_partition = 50000;
			break;
			
		case OVERLAP_BY_SIZE:
			size_of_cells = 2000;
			max_ignore_factor = 0.5;
			size_of_dataset_to_find_partition = 50000;
			number_of_covers = 1;
			break;
			
		case VORONOI_TREE_BY_SIZE:
			size_of_cells = 2000;
			reduce_covers = true;
			size_of_dataset_to_find_partition = 50000;
			number_of_covers = 1;
			tree_reduction_factor = 2.0;
			max_tree_depth = 4;
			max_theoretical_node_width = 20;
			break;
			
		default:
			flush_exit(ERROR_DATA_MISMATCH, "Trying to use partition method %d% that does not exist.", partition_method);
	}
}


//**********************************************************************************************************************************

void Tworking_set_control::write_to_file(FILE *fp) const
{
	file_write(fp, classification);

	file_write(fp, size_of_tasks);
	file_write(fp, number_of_tasks);
	file_write(fp, working_set_selection_method);
	file_write(fp, partition_method);
	file_write(fp, number_of_covers);
	file_write(fp, reduce_covers);
	file_write(fp, max_ignore_factor);
	
	file_write(fp, size_of_cells);
	file_write(fp, number_of_cells);
	file_write(fp, size_of_dataset_to_find_partition);
	file_write(fp, radius);
	file_write(fp, random_seed);
	
	file_write_eol(fp);
};


//**********************************************************************************************************************************


void Tworking_set_control::read_from_file(FILE *fp)
{
	file_read(fp, classification);

	file_read(fp, size_of_tasks);
	file_read(fp, number_of_tasks);
	file_read(fp, working_set_selection_method);
	file_read(fp, partition_method);
	file_read(fp, number_of_covers);
	file_read(fp, reduce_covers);
	file_read(fp, max_ignore_factor);
	
	file_read(fp, size_of_cells);
	file_read(fp, number_of_cells);
	file_read(fp, size_of_dataset_to_find_partition);
	file_read(fp, radius);
	file_read(fp, random_seed);
}

#endif

