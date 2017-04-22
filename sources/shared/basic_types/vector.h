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


#if !defined (VECTOR_H)
	#define VECTOR_H

 

#include <map>
#include <vector>
using namespace std;


//**********************************************************************************************************************************


template <typename Template_type> void push_back_mem_safe(vector <Template_type>& vec, Template_type element);

template <typename Template_type> vector <Template_type> convert_to_vector(Template_type* array, unsigned size);

template <typename Template_type> unsigned argmax(const vector <Template_type>& vec, unsigned start_index = 0, int length = -1);
template <typename Template_type> unsigned argmin(const vector <Template_type>& vec, unsigned start_index = 0, int length = -1);
template <typename Template_type> std::vector<unsigned> find(const vector <Template_type>& vec, Template_type value, unsigned start_index = 0, int length = -1);

template <typename Template_type> double sum(const vector <Template_type>& vec, unsigned start_index = 0, int length = -1);
template <typename Template_type> double abs_sum(const vector <Template_type>& vec, double offset = 0.0, unsigned start_index = 0, int length = -1);
template <typename Template_type> double square_sum(const vector <Template_type>& vec, double offset = 0.0, unsigned start_index = 0, int length = -1);


template <typename Template_type> double mean(const vector <Template_type>& vec, unsigned start_index = 0, int length = -1);
template <typename Template_type> double variance(const vector <Template_type>& vec, unsigned start_index = 0, int length = -1);
template <typename Template_type> double expectile(const vector <Template_type>& vec, double tau, unsigned start_index = 0, int length = -1);
template <typename Template_type> std::pair<double, double> quantile(const vector <Template_type>& vec, double tau, unsigned start_index = 0, int length = -1);

vector <int> is_categorial(const vector<double> vec, unsigned start_index = 0, int length = -1);


template <typename Template_type> void get_k_smallest(vector <Template_type>& value, unsigned k);
template <typename Template_type> void get_k_largest(vector <Template_type>& value, unsigned k);

template <typename Template_type> void sort_up(vector <Template_type>& value);
template <typename Template_type> void sort_down(vector <Template_type>& value);

template <typename Template_type1, typename Template_type2> void sort_up(vector <Template_type1>& value, vector <Template_type2>& index);
template <typename Template_type1, typename Template_type2> void sort_down(vector <Template_type1>& value, vector <Template_type2>& index);


template <typename Template_type1, typename Template_type2> void merge_sort_up(vector <Template_type1>& value, vector <Template_type2>& index);
template <typename Template_type1, typename Template_type2> void merge_sort_down(vector <Template_type1>& value, vector <Template_type2>& index);

template <typename Template_type> map <Template_type, unsigned> create_map(const vector <Template_type>& vec, unsigned start_index = 0, int length = -1);

template <typename Template_type> void apply_permutation(vector <Template_type>& vec, vector <unsigned> permutation);


//**********************************************************************************************************************************

#include "sources/shared/basic_types/vector.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/vector.cpp"
#endif


#endif 
