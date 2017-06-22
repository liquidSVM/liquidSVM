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


#if !defined (MY_UNORDERED_MAP_AND_SET_H)
	#define MY_UNORDERED_MAP_AND_SET_H


#include "sources/shared/system_support/compiler_specifics.h"
	
//**********************************************************************************************************************************

// This little header file defines my_unordered_map by C++11 unordered_map if possible
// and by STL map otherwise. It also includes the necessary header files. The same is
// done for unordered_set.

//**********************************************************************************************************************************


#ifdef USE_TR1_CODE
	#define my_unordered_map unordered_map
	#include <tr1/unordered_map>
	using namespace std::tr1; 
	
	#define my_unordered_set unordered_set
	#include <tr1/unordered_set>
	using namespace std::tr1; 
#else
	#ifdef FALL_BACK_MAP
		#define my_unordered_map map
		#include <map>

		#define my_unordered_set set
		#include <set>
	#else
		#define my_unordered_map unordered_map
		#include <unordered_map>
	
		#define my_unordered_set unordered_set
		#include <unordered_set>
	#endif
#endif


#endif
	
	
