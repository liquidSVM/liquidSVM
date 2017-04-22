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


void inline Tcuda_timer::start_timing(double initial_time)
{
	#ifdef COMPILE_WITH_CUDA__
		cudaEventRecord(start_event, 0);
	#endif
	start_time_stored = initial_time;
}


//**********************************************************************************************************************************


void inline Tcuda_timer::stop_timing(double& final_time)
{
	#ifdef COMPILE_WITH_CUDA__
		float measured_time;
	#endif
		
		

	#ifdef COMPILE_WITH_CUDA__
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&measured_time, start_event, stop_event);
		final_time = start_time_stored + double(measured_time)/1000.0;
	#else
		final_time = start_time_stored;
	#endif
		
}
