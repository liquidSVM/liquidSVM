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


#if !defined (CUDA_BASICS_H)
	#define CUDA_BASICS_H

	

#ifdef  COMPILE_WITH_CUDA__
	#include <cuda_runtime.h>
#endif

	
	
//**********************************************************************************************************************************


#ifdef __CUDACC__
	#define __target_device__   __device__ __host__
#else
	#define __target_device__   
#endif

#define WARP_SIZE 32


//**********************************************************************************************************************************


class Tcuda_timer
{
	public:
		Tcuda_timer();
		~Tcuda_timer();
		
		void clear();
		void inline start_timing(double initial_time = 0.0);
		void inline stop_timing(double& final_time);
		
	private:
		double start_time_stored;
		#ifdef COMPILE_WITH_CUDA__
			cudaEvent_t start_event;
			cudaEvent_t stop_event;
		#endif
};


//**********************************************************************************************************************************


#include "sources/shared/system_support/cuda_basics.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/system_support/cuda_basics.cpp"
#endif


#endif
