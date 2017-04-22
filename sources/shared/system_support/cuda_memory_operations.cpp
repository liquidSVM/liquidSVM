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


#if !defined (CUDA_MEMORY_OPERATIONS_CPP)
	#define CUDA_MEMORY_OPERATIONS_CPP


	
#include "sources/shared/system_support/cuda_memory_operations.h"


//**********************************************************************************************************************************

void copy_to_GPU(vector <double> data, float* data_on_GPU)
{
	copy_to_GPU(&(data[0]), data_on_GPU, data.size());
}


//**********************************************************************************************************************************


void copy_from_GPU(vector <double> data, float* data_on_GPU)
{
	copy_from_GPU(&(data[0]), data_on_GPU, data.size());
}



//**********************************************************************************************************************************


void copy_to_GPU(double* data, float* data_on_GPU, size_t size)
{
	unsigned i;
	vector <float> data_tmp;
	
	data_tmp.resize(size);
	for (i=0; i<data_tmp.size(); i++)
		data_tmp[i] = float(data[i]);
	
	copy_to_GPU(data_tmp, data_on_GPU);
}

//**********************************************************************************************************************************


void copy_to_GPU(double** data, float** data_on_GPU, size_t size)
{
	unsigned i;
	vector <float*> data_tmp;
	
	data_tmp.resize(size);
	for (i=0; i<data_tmp.size(); i++)
		data_tmp[i] = (float*)(data[i]);
	
	copy_to_GPU(data_tmp, data_on_GPU);
}



//**********************************************************************************************************************************

void copy_from_GPU(double* data, float* data_on_GPU, size_t size)
{
	unsigned i;
	vector <float> data_tmp;
	
	data_tmp.resize(size);
	copy_from_GPU(data_tmp, data_on_GPU);
	
	for (i=0; i<data_tmp.size(); i++)
		data[i] = double(data_tmp[i]);
}






#endif
