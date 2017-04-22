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


#if !defined (CUDA_MEMORY_OPERATIONS_H)
	#define CUDA_MEMORY_OPERATIONS_H

 

#include <vector>
using namespace std;


//**********************************************************************************************************************************

void copy_to_GPU(vector <double> data, float* data_on_GPU);
void copy_to_GPU(double* data, float* data_on_GPU, size_t size);
void copy_to_GPU(double** data, float** data_on_GPU, size_t size);

void copy_from_GPU(vector <double> data, float* data_on_GPU);
void copy_from_GPU(double* data, float* data_on_GPU, size_t size);



template <typename Template_type> void my_alloc_GPU(Template_type** pointer, size_t size);
template <typename Template_type> void my_realloc_GPU(Template_type** pointer, size_t size);
template <typename Template_type> void my_dealloc_GPU(Template_type** pointer);

template <typename Template_type> void copy_to_GPU(Template_type* data, Template_type* data_on_GPU, size_t size);
template <typename Template_type> void copy_to_GPU(vector <Template_type> data, Template_type* data_on_GPU);
template <typename Template_type> void copy_to_GPU(Template_type data, Template_type* data_on_GPU);


template <typename Template_type> void copy_from_GPU(Template_type* data, Template_type* data_on_GPU, size_t size);
template <typename Template_type> void copy_from_GPU(vector <Template_type>& data, Template_type* data_on_GPU);



//**********************************************************************************************************************************


#include "sources/shared/system_support/cuda_memory_operations.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/system_support/cuda_memory_operations.cpp"
#endif

#endif
