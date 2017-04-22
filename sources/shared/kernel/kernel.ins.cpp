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


#include "sources/shared/basic_functions/flush_print.h"


//**********************************************************************************************************************************


inline bool Tkernel::all_kNN_assigned() const
{
	return all_kNNs_assigned;
}

//**********************************************************************************************************************************

inline unsigned Tkernel::get_max_kNNs() const
{
	return max_kNNs;
}


//**********************************************************************************************************************************


inline Tsubset_info Tkernel::get_kNNs(unsigned i)
{
	unsigned j;
	Tsubset_info kNNs;
	
	find_kNNs(i, i);
	kNNs.resize(min(kNN_list[i]->size(), unsigned(max(int(col_set_size) - 1, 0))));
	
	for (j=0; j<kNNs.size(); j++)
		kNNs[j] = (*kNN_list[i])[j];
	
	return kNNs;
}

//**********************************************************************************************************************************


inline double Tkernel::entry(unsigned row, unsigned column)
{
	if (assigned == false)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to access the kernel matrix without having assigned values.");
	
	switch (kernel_control.memory_model_kernel) 
	{
		case LINE_BY_LINE:
			return kernel_row[row][column];
		case BLOCK:
			return kernel_row[row][column];
		case CACHE:
			return compute_entry(row, column);
		case EMPTY:
			return compute_entry(row, column);
		default:
			return 0.0;
	}
}



//**********************************************************************************************************************************


inline double* Tkernel::pre_row_from_cache(unsigned i)
{
	unsigned j;
	unsigned l;
	
	
	if (kernel_control.memory_model_pre_kernel == CACHE)
	{
		if (pre_cache.exists(i) == true)
			l = pre_cache[i];
		else 
		{
			l = pre_cache.insert(i);

			for(j=0; j<col_set_size; j++)
				pre_kernel_row[l][j] = pre_kernel_function(kernel_control.kernel_type, row_data_set[i], col_data_set[j]);
		}
		return pre_kernel_row[l];
	}
	
	flush_exit(1, "Undefined kernel mode!");
	return NULL;
}



//**********************************************************************************************************************************


inline double* Tkernel::row(unsigned i)
{
	unsigned j;
	unsigned l;
	double* pre_kernel_row_tmp;
	
	
	if (assigned == false)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to access the kernel matrix without having assigned values.");
	
	if (i >=row_set_size)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to access kernel row %i of a kernel matrix that only has %d rows.", i, row_set_size);

	if (kernel_control.is_full_matrix_model() == true)
		return kernel_row[i];
	else if (kernel_control.is_full_matrix_pre_model() == true)
	{
		for(j=0; j<col_set_size; j++)
			kernel_row[i][j] = kernel_function(kernel_control.kernel_type, gamma_factor, pre_kernel_row[i][j]);
		
		for(j=col_set_size; j<max_aligned_col_set_size; j++)
			kernel_row[i][j] = 0.0;
		
		find_kNNs(i, i);
		return kernel_row[i];
	}
	else if (kernel_control.memory_model_kernel == CACHE)
	{
		if (cache.exists(i) == true)
			l = cache[i];
		else 
		{
			l = cache.insert(i);
			if (kernel_control.memory_model_pre_kernel == CACHE)
			{
				pre_kernel_row_tmp = pre_row_from_cache(i);
				for(j=0; j<col_set_size; j++)
					kernel_row[l][j] = compute_entry(i, j, pre_kernel_row_tmp[j]);
			}
			else
				for(j=0; j<col_set_size; j++)
					kernel_row[l][j] = compute_entry(i, j);

			for(j=col_set_size; j<max_aligned_col_set_size; j++)
				kernel_row[l][j] = 0.0;

			find_kNNs(i, l);
		}
		return kernel_row[l];
	}
	
	flush_exit(1, "Undefined kernel mode!");
	return NULL;
}



//**********************************************************************************************************************************


inline double* Tkernel::row(unsigned i, unsigned start_col, unsigned end_col)
{	
	#ifndef  __CUDACC__
		unsigned j;

		
		if (assigned == false)
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to access the kernel matrix without having assigned values.");
		
		if (kernel_control.is_full_matrix_model() == true)
			return kernel_row[i];
		else if (kernel_control.is_full_matrix_pre_model() == true)
			for(j=start_col;j<end_col;j++)
				kernel_row_ALGD[j] = kernel_function(kernel_control.kernel_type, gamma_factor, pre_kernel_row[i][j]);
		else
			for(j=start_col;j<end_col;j++)
				kernel_row_ALGD[j] = kernel_function(kernel_control.kernel_type, gamma_factor, row_data_set[i], col_data_set[j]);

		for(j=col_set_size; j<max_aligned_col_set_size; j++)
			kernel_row_ALGD[j] = 0.0;

		return kernel_row_ALGD;
	#else
		return NULL;
	#endif
}



//**********************************************************************************************************************************


inline double Tkernel::compute_entry(unsigned i, unsigned j)
{
	#ifndef  __CUDACC__
		double tmp;
		
		if ((kernel_control.memory_model_pre_kernel == EMPTY) or (kernel_control.memory_model_pre_kernel == CACHE))
			tmp = kernel_function(kernel_control.kernel_type, gamma_factor, row_data_set[i], col_data_set[j]);
		else
			tmp = kernel_function(kernel_control.kernel_type, gamma_factor, pre_kernel_row[i][j]);

		tmp = (row_labels_ALGD[i] * col_labels_ALGD[j] + kernel_offset) * tmp;

		return tmp;
	#else
		return 0.0;
	#endif
}

//**********************************************************************************************************************************


inline double Tkernel::compute_entry(unsigned i, unsigned j, double pre_kernel_value)
{
	#ifndef  __CUDACC__
		double tmp;
		
		tmp = kernel_function(kernel_control.kernel_type, gamma_factor, pre_kernel_value);
		tmp = (row_labels_ALGD[i] * col_labels_ALGD[j] + kernel_offset) * tmp;

		return tmp;
	#else
		return 0.0;
	#endif
}
