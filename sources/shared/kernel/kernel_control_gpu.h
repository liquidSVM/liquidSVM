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


#if !defined (KERNEL_CONTROL_GPU_H)
	#define KERNEL_CONTROL_GPU_H


 




//**********************************************************************************************************************************


struct Tkernel_control_GPU
{
	unsigned dim;
	size_t size;
	
	unsigned row_start;
	unsigned row_stop;
	unsigned row_set_size;
	
	unsigned col_start;
	unsigned col_stop;
	unsigned col_set_size;
	unsigned col_set_size_aligned;
	
	double* row_labels;
	double* row_data_set;
	
	double* col_labels;
	double* col_data_set;
	
	double kernel_offset;
	double gamma_factor;
	
	unsigned kernel_type;
	unsigned full_kernel_type;
	
	double* pre_kernel_matrix;
	double* kernel_matrix;
	
	
	double weights_square_sum;
	unsigned number_of_nodes;
	unsigned total_number_of_hierarchical_coordinates;
	unsigned* hierarchical_coordinate_intervals;
	double* hierarchical_weights_squared;
};

//**********************************************************************************************************************************



#endif

