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


#if !defined (CUDA_BASICS_CPP)
	#define CUDA_BASICS_CPP


#include "sources/shared/system_support/cuda_basics.h"

	
	
//**********************************************************************************************************************************


Tcuda_timer::Tcuda_timer()
{
	clear();
	#ifdef COMPILE_WITH_CUDA__
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
	#endif
}


//**********************************************************************************************************************************


Tcuda_timer::~Tcuda_timer()
{
	#ifdef COMPILE_WITH_CUDA__
		cudaEventDestroy(start_event);
		cudaEventDestroy(stop_event);
	#endif
}



//**********************************************************************************************************************************


void Tcuda_timer::clear()
{
	start_time_stored = 0.0;
}


#endif
