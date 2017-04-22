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

#ifdef  COMPILE_WITH_CUDA__
	#include "sources/shared/kernel/kernel_computation.h"
	#include "sources/shared/system_support/cuda_memory_operations.h"
	#if defined(COMPILE_SEPERATELY__CUDA)
		#include "sources/shared/kernel/kernel_computation_joint_cpugpu.h"
	#endif
#endif

//**********************************************************************************************************************************


inline unsigned Tdataset::size() const
{
	return data_size;
}


//**********************************************************************************************************************************


inline unsigned Tdataset::dim() const
{
	unsigned i;
	unsigned dimension;
	
	
	if (size() == 0)
		return 0;
	
	dimension = sample_list[0]->dim();
	for (i=1; i<size(); i++)
		dimension = max(dimension, sample_list[i]->dim());
	
	return dimension;
}

//**********************************************************************************************************************************


inline bool Tdataset::has_ownership() const
{
	return owns_samples;
}


//**********************************************************************************************************************************



inline void Tdataset::check_index(unsigned index) const
{
	if (index >= size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to access sample %d in a dataset containing only %d samples.", index, size());
}


//**********************************************************************************************************************************


inline Tsample* Tdataset::sample(unsigned index) const
{
	check_index(index);
	sample_list[index]->blocked_destruction = true;
	
	return sample_list[index];
}


//**********************************************************************************************************************************


template <typename float_type> float_type* Tdataset::convert_to_GPU_format(unsigned start_index, unsigned end_index) const
{
	#if defined(COMPILE_WITH_CUDA__) && (!defined(__CUDACC__) || defined(COMPILE_SEPERATELY__CUDA))
		unsigned i;
		unsigned j;
		unsigned length;
		unsigned dim_max;
		float_type* array;
		double* xpart_tmp;

		if (start_index > end_index)
			flush_exit(ERROR_DATA_STRUCTURE, "Cannot convert described part of dataset to array");
		else if (start_index == end_index)
			return NULL;

		length = end_index - start_index;
		dim_max = dim();
		array = new float_type[length * dim_max];
		xpart_tmp = new double[dim_max];
		
	// Make sure the array contains zero coordinates for samples, whose dimension is smaller than dim().
		
		for (i=0; i<length * dim_max; i++)
			array[i] = float_type(0.0);
		
	// Now copy the samples into the array
		
		for(i=0;i<length;i++)
		{
			for (j=0; j<dim_max; j++)
				xpart_tmp[j] = 0.0;
				
			sample_list[start_index + i]->get_x_part(xpart_tmp);
			
			for (j=0; j<dim_max; j++)
				array[feature_pos_on_GPU(length, i, j)] = float_type(xpart_tmp[j]);
		}
		
		delete xpart_tmp;
		
		return array;
	#else
		return NULL;
	#endif
}



//**********************************************************************************************************************************


template <typename float_type> float_type* Tdataset::upload_to_GPU(unsigned start_index, unsigned end_index) const
{
	#ifdef  COMPILE_WITH_CUDA__
		float_type* data_set;
		float_type* data_set_on_GPU;

		
		data_set_on_GPU = NULL;
		data_set = convert_to_GPU_format<float_type>(start_index, end_index);
		my_alloc_GPU(&data_set_on_GPU, required_memory_on_GPU(start_index, end_index));
		copy_to_GPU(data_set, data_set_on_GPU, (end_index - start_index) * dim());
		delete data_set;

		return data_set_on_GPU;
	#else
		return NULL;
	#endif
}



