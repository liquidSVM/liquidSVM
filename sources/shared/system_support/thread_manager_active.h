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


#if !defined (THREAD_MANAGER_ACTIVE_H)
	#define THREAD_MANAGER_ACTIVE_H



#include "sources/shared/system_support/thread_manager.h"


//**********************************************************************************************************************************


class Tthread_manager_active: public Tthread_manager
{
	public:
		void reserve_threads(Tparallel_control parallel_ctrl);

		void start_threads();
		virtual void thread_entry() = 0;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/system_support/thread_manager_active.cpp"
#endif


#endif

