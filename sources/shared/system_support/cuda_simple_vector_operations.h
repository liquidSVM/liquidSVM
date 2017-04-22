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


#if !defined (CUDA_SIMPLE_VECTOR_OPERATIONS_H)
	#define CUDA_SIMPLE_VECTOR_OPERATIONS_H
 


 
 
void init_vector_on_GPU(double* vector_GPU, unsigned size, double value = 0.0); 
void mult_vector_on_GPU(double coefficient, double* vector_GPU, unsigned size);


//**********************************************************************************************************************************


#if !defined(COMPILE_SEPERATELY__) && !defined(COMPILE_SEPERATELY__CUDA)
	#include "sources/shared/system_support/cuda_simple_vector_operations.cu"
#endif

#endif
