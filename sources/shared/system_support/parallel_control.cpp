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


#if !defined (PARALLEL_CONTROL_CPP)
	#define PARALLEL_CONTROL_CPP


#include "sources/shared/system_support/parallel_control.h"

#include "sources/shared/basic_functions/basic_file_functions.h"


#ifdef __linux__
	#include <sys/sysinfo.h>
#endif


//**********************************************************************************************************************************

Tparallel_control::Tparallel_control()
{
	requested_team_size = 0;
	core_number_offset = 0;
	GPUs = 0;
	GPU_number_offset = 0;
};


//**********************************************************************************************************************************

void Tparallel_control::write_to_file(FILE* fp) const
{
	file_write(fp, requested_team_size);
	file_write(fp, core_number_offset);
	file_write(fp, GPUs);
	file_write(fp, GPU_number_offset);
	
	file_write_eol(fp);
};

//**********************************************************************************************************************************


void Tparallel_control::read_from_file(FILE* fp)
{
	file_read(fp, requested_team_size);
	file_read(fp, core_number_offset);
	file_read(fp, GPUs);
	file_read(fp, GPU_number_offset);
}


//**********************************************************************************************************************************


Tparallel_control Tparallel_control::set_to_single_threaded(bool adaptive_shrink) const
{
	Tparallel_control parallel_control;

	parallel_control = *this;
	if (adaptive_shrink == true)
	{
		parallel_control.requested_team_size = 1;
		parallel_control.GPUs = min(1, int(parallel_control.GPUs));
	}
	
	return parallel_control;
}

#endif

