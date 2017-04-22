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


#if !defined (WORKING_SET_CONTROL_H)
	#define WORKING_SET_CONTROL_H


 
#include "sources/shared/basic_types/dataset_info.h"
 
#include <cstdio>
using namespace std;


//**********************************************************************************************************************************


enum WORKING_SET_SELECTION_TYPES {FULL_SET, MULTI_CLASS_ALL_VS_ALL, MULTI_CLASS_ONE_VS_ALL, BOOT_STRAP, WORKING_SET_SELECTION_TYPES_MAX};
enum PARTITION_TYPES {NO_PARTITION, RANDOM_CHUNK_BY_SIZE, RANDOM_CHUNK_BY_NUMBER, VORONOI_BY_RADIUS, VORONOI_BY_SIZE, OVERLAP_BY_SIZE, VORONOI_TREE_BY_SIZE, PARTITION_TYPES_MAX};


//**********************************************************************************************************************************


class Tworking_set_control
{
	public:
		Tworking_set_control();
		void write_to_file(FILE *fp) const;
		void read_from_file(FILE *fp);
		void set_partition_method_with_defaults(unsigned partition_method);
		
		bool classification;

		unsigned size_of_tasks;
		unsigned number_of_tasks;
		
		unsigned working_set_selection_method;
		unsigned partition_method;
		unsigned number_of_covers;
		bool reduce_covers;
		double max_ignore_factor;
		
		double tree_reduction_factor;
		unsigned max_tree_depth;
		unsigned max_theoretical_node_width;
		
		double radius;
		unsigned size_of_cells;
		unsigned number_of_cells;
		unsigned size_of_dataset_to_find_partition;
		int random_seed;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/working_set_control.cpp"
#endif

#endif
