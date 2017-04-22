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


#if !defined (CACHE_LRU_CPP)
	#define CACHE_LRU_CPP



#include "sources/shared/basic_types/cache_lru.h"


#include "sources/shared/basic_functions/flush_print.h"



//**********************************************************************************************************************************



Tcache_lru::Tcache_lru()
{
	clear_stats();
}



//**********************************************************************************************************************************



Tcache_lru::Tcache_lru(unsigned size)
{
	reserve(size);
	clear_stats();
}

//**********************************************************************************************************************************


Tcache_lru::~Tcache_lru()
{
}



//**********************************************************************************************************************************



void Tcache_lru::reserve(unsigned size)
{
	clear();
	size_of_cache = size;
}


//**********************************************************************************************************************************



void Tcache_lru::clear()
{
	clear_stats();
	lru_list.clear();
	hash_map.clear();
}



//**********************************************************************************************************************************



void Tcache_lru::clear_stats()
{
	hits = 0;
	misses = 0;
}

//**********************************************************************************************************************************



void Tcache_lru::get_stats(unsigned& hits, unsigned& misses) const
{
	hits = Tcache_lru::hits;
	misses = Tcache_lru::misses;
}


#endif

