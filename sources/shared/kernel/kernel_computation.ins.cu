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



#include "sources/shared/kernel/kernel_control.h"




//**********************************************************************************************************************************


template <class float_type> __device__ inline float_type squared_distance(unsigned dim, unsigned row_size, float_type* row_data_set, unsigned row_pos, unsigned col_size, float_type* col_data_set, unsigned col_pos)
{
	unsigned k;
	float_type diff;
	float_type distance;
	
	
	distance = 0.0; 
	for (k=0; k<dim; k++)
	{
		diff = row_data_set[feature_pos_on_GPU(row_size, row_pos, k)] - col_data_set[feature_pos_on_GPU(col_size, col_pos, k)];
		distance = distance + diff * diff;
	}
	return distance;
}


//**********************************************************************************************************************************


template <class float_type> __device__ inline float_type pre_kernel_init_value(unsigned full_kernel_type, float_type weights_square_sum)
{
	if (full_kernel_type == HIERARCHICAL_GAUSS)
		return weights_square_sum;
	else
		return 0.0;
}


//**********************************************************************************************************************************


template <class float_type> __device__ inline float_type pre_kernel_value_summand(unsigned full_kernel_type, float_type weight, float_type pre_kernel_l_value)
{
	if (full_kernel_type == HIERARCHICAL_GAUSS)
		return -weight * pre_kernel_l_value;
	else
		return pre_kernel_l_value;
}


//**********************************************************************************************************************************


template <class float_type> __device__ inline float_type pre_kernel_l_value_conversion(unsigned full_kernel_type, float_type squared_distance)
{
	if (full_kernel_type == HIERARCHICAL_GAUSS)
		return exp( -squared_distance);
	else
		return squared_distance;
}

//**********************************************************************************************************************************


template <class float_type> __device__ inline float_type pre_kernel_update_l_value_conversion(unsigned full_kernel_type, float_type squared_distance, float_type old_pre_kernel_l_value)
{
	if (full_kernel_type == HIERARCHICAL_GAUSS)
		return old_pre_kernel_l_value * exp( -squared_distance);
	else
		return old_pre_kernel_l_value + squared_distance;
}


//**********************************************************************************************************************************



template <class float_type> __device__ inline float_type hierarchical_pre_kernel(unsigned full_kernel_type, unsigned number_of_nodes, unsigned number_of_coordinates, unsigned* coordinate_starts, float_type* weights, float_type weights_square_sum, unsigned row_size, float_type* row_data_set, unsigned row_pos, unsigned col_size, float_type* col_data_set, unsigned col_pos)
{
	unsigned l;
	unsigned k;
	unsigned current_row_feature_pos;
	unsigned current_col_feature_pos;
	float_type diff;
	float_type distance;
	float_type pre_kernel_value;
	float_type pre_kernel_l_value;

	
	pre_kernel_value = pre_kernel_init_value(full_kernel_type, weights_square_sum);

	current_row_feature_pos = feature_pos_on_GPU(row_size, row_pos, 0);
	current_col_feature_pos = feature_pos_on_GPU(col_size, col_pos, 0);
	for (l=0; l<number_of_nodes; l++)
	{
		distance = 0.0; 
		for (k=coordinate_starts[l]; k<coordinate_starts[l+1]; k++)
		{
			diff = row_data_set[current_row_feature_pos] - col_data_set[current_col_feature_pos];
			distance = distance + diff * diff;

			current_row_feature_pos = next_feature_pos_on_GPU(row_size, current_row_feature_pos);
			current_col_feature_pos = next_feature_pos_on_GPU(col_size, current_col_feature_pos);
		}
		
		pre_kernel_l_value = pre_kernel_l_value_conversion(full_kernel_type, distance);
		pre_kernel_value = pre_kernel_value + pre_kernel_value_summand(full_kernel_type, weights[l], pre_kernel_l_value);
	}
	return pre_kernel_value;
}
