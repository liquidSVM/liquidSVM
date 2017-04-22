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


#if !defined (CACHE_LRU_H)
	#define CACHE_LRU_H

	

#include "sources/shared/system_support/compiler_specifics.h"
 
 
#include <list>
using namespace std;


#ifdef USE_TR1_CODE
	#include <tr1/unordered_map>
	using namespace std::tr1; 
#else
	#include <unordered_map>
#endif


//**********************************************************************************************************************************


class Tcache_lru
{
	public:
		Tcache_lru();
		Tcache_lru(unsigned size);
		~Tcache_lru();

		void clear();
		void reserve(unsigned size);
	
		inline bool is_full() const;
		inline unsigned size() const;
		
		inline bool exists(unsigned key);
		inline unsigned insert(unsigned key);
		inline unsigned operator[] (unsigned key);

		void clear_stats();
		void get_stats(unsigned& hits, unsigned& misses) const;
		

	private:
		inline void pop_back();
		inline void push_front(unsigned key, unsigned position);
		inline unsigned get_last() const;
		
		typedef std::pair <unsigned, unsigned> Tlist_entry;
		typedef std::list <Tlist_entry> Tlist;
		
		typedef std::pair <unsigned, Tlist::iterator> Thash_map_entry;
		typedef unordered_map <unsigned, Thash_map_entry> Thash_map;
		
		inline bool exists__(unsigned key);
		inline void update(unsigned key);

		unsigned hits;
		unsigned misses;
		
		Tlist lru_list;
		Thash_map hash_map;
		
		unsigned size_of_cache;
};


//**********************************************************************************************************************************


#include "sources/shared/basic_types/cache_lru.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/cache_lru.cpp"
#endif

//**********************************************************************************************************************************


#endif
