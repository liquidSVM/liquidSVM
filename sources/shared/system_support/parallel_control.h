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


#if !defined (PARALLEL_CONTROL_H)
	#define PARALLEL_CONTROL_H



#include <cstdio>
using namespace std;


//**********************************************************************************************************************************

class Tparallel_control
{
	public:
		Tparallel_control();
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;
		
		Tparallel_control set_to_single_threaded(bool adaptive_shrink) const;
		
		
		int requested_team_size;
		unsigned core_number_offset;
		unsigned GPUs;
		unsigned GPU_number_offset;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/system_support/parallel_control.cpp"
#endif

#endif
