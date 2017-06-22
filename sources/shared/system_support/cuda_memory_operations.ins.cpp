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


#ifdef  COMPILE_WITH_CUDA__
	#include <cuda_runtime.h>
#endif

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/memory_constants.h"

//**********************************************************************************************************************************


template <typename Template_type> void my_alloc_GPU(Template_type** pointer, size_t size)
{
	#ifdef  COMPILE_WITH_CUDA__
		void* cuda_ptr;
		cudaError_t error_code;

		if (size > 0)
		{
			error_code = cudaMalloc(&cuda_ptr, sizeof(Template_type) * size);
			if (error_code == cudaErrorMemoryAllocation)
				flush_exit(ERROR_OUT_OF_MEMORY, "It was impossible to allocate %d MB on the GPU.", convert_to_MB(sizeof(Template_type) * size));
			else
				*pointer = (Template_type*) cuda_ptr;
		}
		else
			*pointer = NULL;
	#else
		*pointer = NULL;
	#endif
}


//**********************************************************************************************************************************


template <typename Template_type> void my_dealloc_GPU(Template_type** pointer)
{
	#ifdef  COMPILE_WITH_CUDA__
		cudaError_t error_code;
		
		
		if (*pointer != NULL)
		{
			error_code = cudaFree(*pointer);
			
			if (error_code == cudaErrorInvalidDevicePointer)
				flush_exit(ERROR_UNSPECIFIED, "Unable to free memory on the GPU.");
			if (error_code == cudaErrorInitializationError)
				flush_exit(ERROR_UNSPECIFIED, "Error while freeing memory on the GPU, since CUDA driver and runtime could not be initialized.");
		}
	#endif
		
	*pointer = NULL;
}


//**********************************************************************************************************************************


template <typename Template_type> void my_realloc_GPU(Template_type** pointer, size_t size)
{
	my_dealloc_GPU(pointer);
	my_alloc_GPU(pointer, size);
}


//*********************************************************************************************************************************


template <typename Template_type> void copy_to_GPU(Template_type* data, Template_type* data_on_GPU, size_t size)
{
	if (size == 0)
		return;

	#ifdef  COMPILE_WITH_CUDA__
		cudaError_t error_code;

		if (data == NULL)
			flush_exit(ERROR_RUNTIME, "Cannot copy data from NULL pointer onto the GPU.");
		if (data_on_GPU == NULL)
			flush_exit(ERROR_RUNTIME, "Cannot copy data to NULL pointer on the GPU.");
			
		error_code = cudaMemcpy(data_on_GPU, data, sizeof(Template_type) * size, cudaMemcpyHostToDevice);

		if (error_code != cudaSuccess)
			flush_exit(ERROR_UNSPECIFIED, "Error while copying %d KB onto the GPU. CUDA code %d.", convert_to_KB(sizeof(Template_type) * size), error_code);
	#endif
}


//*********************************************************************************************************************************

template <typename Template_type> void copy_to_GPU(vector <Template_type> data, Template_type* data_on_GPU)
{
	copy_to_GPU(&(data[0]), data_on_GPU, data.size());
}


//*********************************************************************************************************************************

template <typename Template_type> void copy_to_GPU(Template_type data, Template_type* data_on_GPU)
{
	copy_to_GPU(&data, data_on_GPU, 1);
}


//*********************************************************************************************************************************

template <typename Template_type> void copy_from_GPU(Template_type* data, Template_type* data_on_GPU, size_t size)
{
	if (size == 0)
		return;
	
	#ifdef  COMPILE_WITH_CUDA__
		cudaError_t error_code;
		
		if (data == NULL)
			flush_exit(ERROR_RUNTIME, "Cannot copy data from GPU to NULL pointer.");
		if (data_on_GPU == NULL)
			flush_exit(ERROR_RUNTIME, "Cannot copy data from NULL pointer on GPU to HOST.");
		
		error_code = cudaMemcpy(data, data_on_GPU, sizeof(Template_type) * size, cudaMemcpyDeviceToHost);

		if (error_code != cudaSuccess)
			flush_exit(ERROR_UNSPECIFIED, "Error while copying %d KB from the GPU. CUDA code %d.", convert_to_KB(sizeof(Template_type) * size), error_code);
	#endif
}

//*********************************************************************************************************************************


template <typename Template_type> void copy_from_GPU(vector <Template_type>& data, Template_type* data_on_GPU)
{
	copy_from_GPU(&(data[0]), data_on_GPU, data.size());
}



