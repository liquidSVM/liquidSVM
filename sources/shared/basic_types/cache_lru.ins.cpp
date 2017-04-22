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





#include "sources/shared/basic_types/cache_lru.h"



#include "sources/shared/basic_functions/flush_print.h"




//**********************************************************************************************************************************



inline bool Tcache_lru::is_full() const
{
	return (size() >= size_of_cache);
}


//**********************************************************************************************************************************



inline unsigned Tcache_lru::size() const
{
	return unsigned(hash_map.size());
}

//**********************************************************************************************************************************



inline bool Tcache_lru::exists(unsigned key)
{
	if (hash_map.count(key) == 1)
		hits++;
	else
		misses++;
	
	return (hash_map.count(key) == 1);
}

//**********************************************************************************************************************************



inline bool Tcache_lru::exists__(unsigned key)
{
	return (hash_map.count(key) == 1);
}

//**********************************************************************************************************************************



inline unsigned Tcache_lru::get_last() const
{
	if (size() > 0)
		return (*(--lru_list.end())).second;
	else
		return 0;
}


//**********************************************************************************************************************************



inline void Tcache_lru::update(unsigned key)
{
	Thash_map_entry hash_map_entry;

	if (exists__(key) == true)
	{
		hash_map_entry = hash_map[key];
		lru_list.splice(lru_list.begin(), lru_list, hash_map_entry.second);
	}
}

//**********************************************************************************************************************************



inline unsigned Tcache_lru::operator[] (unsigned key)
{
	Thash_map_entry hash_map_entry;

	if (exists__(key) == true)
	{
		update(key);
		return hash_map[key].first;
	}
	else
		return 0;
}


//**********************************************************************************************************************************



inline unsigned Tcache_lru::insert(unsigned key)
{
	unsigned position;

	if (exists__(key) == true)
		return (*this)[key];

	if (is_full() == true)
		position = get_last();
	else
		position = size();
	push_front(key, position);

	return position;
}


//**********************************************************************************************************************************



inline void Tcache_lru::push_front(unsigned key, unsigned position)
{
	Thash_map_entry hash_map_entry;

	if (exists__(key) == true)
		update(key);
	else
	{
		lru_list.push_front(Tlist_entry(key, position));
		hash_map.insert(std::pair <unsigned, Thash_map_entry> (key, Thash_map_entry(position, lru_list.begin())));

		if (size() > size_of_cache)
			pop_back();
	}
}

//**********************************************************************************************************************************



inline void Tcache_lru::pop_back()
{
	hash_map.erase((*(--lru_list.end())).first);
	lru_list.pop_back();
}



