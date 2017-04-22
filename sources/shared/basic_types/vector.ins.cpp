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


#include <cmath>
#include <limits>
#include <vector>
#include <utility>
#include <functional>
using namespace std; 


//**********************************************************************************************************************************


template <typename Template_type> void push_back_mem_safe(vector <Template_type>&  vec, Template_type element)
{
	if (vec.size() < vec.max_size())
		vec.push_back(element);
	else
		flush_exit(ERROR_OUT_OF_MEMORY, "Unsufficient memory for adding an element to vector of size %d.", vec.size());
}

//**********************************************************************************************************************************


template <typename Template_type> vector <Template_type> convert_to_vector(Template_type* array, unsigned size)
{
	unsigned i;
	vector <Template_type> vec;
	
	vec.resize(size);
	for (i=0; i<size; i++)
		vec[i] = array[i];
	
	return vec;
}



//**********************************************************************************************************************************


template <typename Template_type> unsigned determine_stop_index(const vector <Template_type>& vec, unsigned start_index, int length)
{
	if (length < 0)
		return vec.size();
	else
		return min(start_index + unsigned(length), unsigned(vec.size()));
}


//**********************************************************************************************************************************


template <typename Template_type> std::vector<unsigned> find(const vector <Template_type>& vec, Template_type value, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	vector <unsigned> indices;
	
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
		if (vec[i] == value)
			indices.push_back(i);
		
	return indices;
}


//**********************************************************************************************************************************


template <typename Template_type> unsigned argmax(const vector <Template_type>& vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned best_i;
	unsigned stop_index;
	Template_type max_value;
	
	
	best_i = start_index;
	max_value = vec[start_index];
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index+1; i<stop_index; i++)
		if (vec[i] > max_value)
		{
			best_i = i;
			max_value = vec[i];
		}
	return best_i;
}



//**********************************************************************************************************************************


template <typename Template_type> unsigned argmin(const vector <Template_type>& vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned best_i;
	unsigned stop_index;
	Template_type min_value;
	

	best_i = start_index;
	min_value = vec[start_index];
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index+1; i<stop_index; i++)
	{
		if (vec[i] < min_value)
		{
			best_i = i;
			min_value = vec[i];
		}
	}
	return best_i;
}


//**********************************************************************************************************************************


template <typename Template_type> double sum(const vector <Template_type>& vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	double sum;
	

	sum = 0.0;
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
		sum = sum + double(vec[i]);

	return sum;
}


//**********************************************************************************************************************************


template <typename Template_type> double abs_sum(const vector <Template_type>& vec, double offset, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	double sum;
	

	sum = 0.0;
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
		sum = sum + abs(double(vec[i] - offset));

	return sum;
}


//**********************************************************************************************************************************


template <typename Template_type> double square_sum(const vector <Template_type>& vec, double offset, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	double sum;
	

	sum = 0.0;
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
		sum = sum + double(vec[i] - offset) * double(vec[i] - offset);

	return sum;
}



//**********************************************************************************************************************************


template <typename Template_type> double mean(const vector <Template_type>& vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	double average;
	

	average = 0.0;
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
		average = average + double(vec[i]);
	
	if (start_index < stop_index)
		average = average/double(stop_index - start_index);

	return average;
}


//**********************************************************************************************************************************


template <typename Template_type> double variance(const vector <Template_type>& vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned n;
	unsigned stop_index;
	double delta;
	double average;
	double variance;
	
	n = 0;
	average = 0.0;
	variance = 0.0;
	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
	{
		n++;
		delta = double(vec[i]) - average;
		average = average + delta/double(n);
		variance = variance + delta * (double(vec[i]) - average);
	}
 
	if (n < 2)
		return 0.0;
	else 
		return variance/double(int(n) - 1);
}



//**********************************************************************************************************************************


template <typename Template_type> double expectile(const vector <Template_type>& vec, double tau, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	double exp_left;
	double exp_right;
	unsigned left_count;
	unsigned right_count;
	double exp_min;
	double exp_max;
	double exp_tau;
	double exp_taum;
	double exp_taup;
	double exp_taum_old;
	double exp_taup_old;
	double best_diffm;
	double best_diffp;
	double tstar;

	
	stop_index = determine_stop_index(vec, start_index, length);

	i = argmin(vec, start_index, length);
	exp_min = vec[i];

	i = argmax(vec, start_index, length);
	exp_max = vec[i];
	
	if (tau == 0.0)
		return exp_min;
	if (tau == 1.0)
		return exp_max;
	
	
// 	In the first part, we look for two consequetive values in the vector, in between the true quantile lies 

	exp_taup = 0.0;
	exp_taum = 1.0;
	
	do
	{
		exp_taum_old = exp_taum;
		exp_taup_old = exp_taup;
		
		exp_tau = 0.5 * (exp_min + exp_max);
		exp_left = 0.0;
		exp_right = 0.0;
		
		exp_taum = exp_min;
		exp_taup = exp_max;
		
		best_diffm = std::numeric_limits<double>::max();
		best_diffp = std::numeric_limits<double>::max();
		
		
		for(i=start_index; i<stop_index; i++)
		{
			if (vec[i] <= exp_tau)
			{
				exp_left = exp_left + (exp_tau - vec[i]);
				if (exp_tau - vec[i] < best_diffm)
				{
					best_diffm = exp_tau - vec[i];
					exp_taum = vec[i];
				}
			}
			if (vec[i] >= exp_tau)
			{
				exp_right = exp_right + (exp_tau - vec[i]);
				if (vec[i] - exp_tau < best_diffp)
				{
					best_diffp = vec[i] - exp_tau;
					exp_taup = vec[i];
				}
			}
		}
		
		if ( (1-tau) * exp_left + tau * exp_right > 0.0)
			exp_max = exp_tau;
		else if ( (1-tau) * exp_left + tau * exp_right < 0.0)
			exp_min = exp_tau;
	}
	while ( (exp_taum_old != exp_taum) or ( exp_taup_old != exp_taup) );
	
	
// 	Having found these values, we can now make an exact calculation
	
	left_count = 0;
	right_count = 0;
	
	exp_left = 0.0;
	exp_right = 0.0;
	
	for(i=start_index; i<stop_index; i++)
	{
		if (vec[i] <= exp_taum)
		{
			left_count++;
			exp_left = exp_left + vec[i];
		}
		
		if (vec[i] >= exp_taup)
		{
			right_count++;
			exp_right = exp_right + vec[i];
		}
	}
	
	tstar = (tau * exp_right + (1.0 - tau) * exp_left) / (tau * double(right_count) + (1.0 - tau) * double(left_count));
	
	if (tstar < exp_taum)
		return exp_taum;
	else if (tstar > exp_taup)
		return exp_taup;
	else
		return tstar;
}



//**********************************************************************************************************************************


template <typename Template_type> std::pair<double, double> quantile(const vector <Template_type>& vec, double tau, unsigned start_index, int length)
{
	unsigned i;
	unsigned size;
	unsigned stop_index;
	unsigned qm_count;
	unsigned qp_count;
	double qmin;
	double qmax;
	double qtau;
	double qtaum;
	double qtaup;
	double best_diffm;
	double best_diffp;

	
	stop_index = determine_stop_index(vec, start_index, length);
	size = unsigned(int(stop_index) - int(start_index));
	
	i = argmin(vec, start_index, length);
	qmin = vec[i];

	i = argmax(vec, start_index, length);
	qmax = vec[i];
	
	if (tau == 0.0)
		return std::pair<double, double> (qmin, qmin);
	if (tau == 1.0)
		return std::pair<double, double> (qmax, qmax);
	
	do
	{
		qtau = 0.5 * (qmin + qmax);
		qm_count = 0;
		qp_count = 0;
		
		qtaum = qmin;
		qtaup = qmax;
		
		best_diffm = std::numeric_limits<double>::max();
		best_diffp = std::numeric_limits<double>::max();
		
		
		for(i=start_index; i<stop_index; i++)
		{
			if (vec[i] <= qtau)
			{
				qm_count++;
				if (qtau - vec[i] < best_diffm)
				{
					best_diffm = qtau - vec[i];
					qtaum = vec[i];
				}
			}	
			if (vec[i] >= qtau)
			{
				qp_count++;
				if (vec[i] - qtau < best_diffp)
				{
					best_diffp = vec[i] - qtau;
					qtaup = vec[i];
				}
			}
		}

		if (double(qm_count) / double(size) < tau)
			qmin = qtaup;
		else
			qmax = qtaum;

	}
	while ( ((double(qm_count) / double(size) < tau) or (double(qp_count) / double(size) < 1.0 - tau)) and (qmin < qmax));
	
	
// Finally, recompute qtaum and qtaup and check whether qtaum and qtaup are also tau-quantiles, and, if not, replace then by qtau,

	best_diffm = std::numeric_limits<double>::max();
	best_diffp = std::numeric_limits<double>::max();
	for(i=start_index; i<stop_index; i++)
	{
		if ((vec[i] < qtau) and (qtau - vec[i] < best_diffm))
		{
			best_diffm = qtau - vec[i];
			qtaum = vec[i];
		}
		if ((vec[i] > qtau) and (vec[i] - qtau < best_diffp))
		{
			best_diffp = vec[i] - qtau;
			qtaup = vec[i];
		}
	}

	qm_count = 0;
	qp_count = 0;
	for(i=start_index; i<stop_index; i++)
	{
		if (vec[i] <= qtaum)
			qm_count++;

		if (vec[i] >= qtaum)
			qp_count++;
	}
	if ( (double(qm_count) / double(size) < tau) or (double(qp_count) / double(size) < 1.0 - tau) )
		qtaum = qtau;
	
	
	qm_count = 0;
	qp_count = 0;
	for(i=start_index; i<stop_index; i++)
	{
		if (vec[i] <= qtaup)
			qm_count++;

		if (vec[i] >= qtaup)
			qp_count++;
	}
	if ( (double(qm_count) / double(size) < tau) or (double(qp_count) / double(size) < 1.0 - tau) )
		qtaup = qtau;
		
	return std::pair<double, double> (qtaum, qtaup);
}


//**********************************************************************************************************************************

template <typename Template_type> map <Template_type, unsigned> create_map(const vector <Template_type>& vec, unsigned start_index, int length)
{
	unsigned i;
	unsigned stop_index;
	map <Template_type, unsigned> return_map;
	

	stop_index = determine_stop_index(vec, start_index, length);
	
	for(i=start_index; i<stop_index; i++)
		return_map.insert( std::pair<Template_type, unsigned> (vec[i], i) );
	
	return return_map;
}


//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> void merge_sort_up(vector <Template_type1>& value, vector <Template_type2>& index)
{
	int i;
	int size;
	bool swapped;
	
	// This function uses the function template std::swap defined in <utility>
 	
	if (value.size() != value.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to sort two vector of size %d and %d", value.size(), value.size());
	else
		size = value.size();
	
	swapped = true;
	while(swapped)
	{
		swapped = false;

		for(i=0;i<size-1;)
		{
			if(value[i] > value[i+1])
			{
				swap(value[i], value[i+1]);
				swap(index[i], index[i+1]);
				swapped = true;
			}
			i = i + 2;
		}

		for(i=0;i<size-3;)
		{
			if(value[i] > value[i+2])
			{
				swap(value[i], value[i+2]);
				swap(index[i], index[i+2]);
				swapped = true;
			}
			i = i + 1;
		}

		for(i=1;i<size-1;)
		{
			if(value[i] > value[i+1])
			{
				swap(value[i], value[i+1]);
				swap(index[i], index[i+1]);
				swapped = true;
			}
			i = i + 2;
		}
	}
} 

//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> void merge_sort_down(vector <Template_type1>& value, vector <Template_type2>& index)
{
	int i;
	int size;
	bool swapped;
	
	// This function uses the function template std::swap defined in <utility>
 	
	if (value.size() != value.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to sort two vector of size %d and %d", value.size(), value.size());
	else
		size = value.size();
	
	swapped = true;
	while(swapped)
	{
		swapped = false;

		for(i=0;i<size-1;)
		{
			if(value[i] < value[i+1])
			{
				swap(value[i], value[i+1]);
				swap(index[i], index[i+1]);
				swapped = true;
			}
			i = i + 2;
		}

		for(i=0;i<size-3;)
		{
			if(value[i] < value[i+2])
			{
				swap(value[i], value[i+2]);
				swap(index[i], index[i+2]);
				swapped = true;
			}
			i = i + 1;
		}

		for(i=1;i<size-1;)
		{
			if(value[i] < value[i+1])
			{
				swap(value[i], value[i+1]);
				swap(index[i], index[i+1]);
				swapped = true;
			}
			i = i + 2;
		}
	}
} 



//**********************************************************************************************************************************


template <typename Template_type> void get_k_smallest(vector <Template_type>& value, unsigned k)
{
	if (k > value.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to partially order a vector of size %d up to %d", value.size(), k);
	
	std::nth_element(value.begin(), value.begin() + k, value.end());
	
}

//**********************************************************************************************************************************


template <typename Template_type> void get_k_largest(vector <Template_type>& value, unsigned k)
{
	if (k > value.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to partially order a vector of size %d up to %d", value.size(), k);
	
	std::nth_element(value.begin(), value.begin() + k, value.end(), greater<Template_type>());
}

//**********************************************************************************************************************************


template <typename Template_type> void sort_up(vector <Template_type>& value)
{
	std::sort(value.begin(), value.end()); 
}


//**********************************************************************************************************************************


template <typename Template_type> void sort_down(vector <Template_type>& value)
{
	std::sort(value.begin(), value.end(),  greater<Template_type>()); 
}


//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> bool smaller (const std::pair<Template_type1, Template_type2>& left, const std::pair<Template_type1, Template_type2>& right)
{
	return (left.first < right.first);
}

//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> bool larger (const std::pair<Template_type1, Template_type2>& left, const std::pair<Template_type1, Template_type2>& right)
{
	return (left.first > right.first);
}

//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> void sort_up(vector <Template_type1>& value, vector <Template_type2>& index)
{
	unsigned i;
	vector <std::pair<Template_type1, Template_type2> > paired_vector;
	
	
	paired_vector.resize(value.size());
	for (i=0; i<value.size(); i++)
		paired_vector[i] = std::pair<Template_type1, Template_type2>(value[i], index[i]);
	
	std::sort(paired_vector.begin(), paired_vector.end(), smaller<Template_type1, Template_type2>);
	
	for (i=0; i<value.size(); i++)
	{
		value[i] = paired_vector[i].first;
		index[i] = paired_vector[i].second;
	}
}


//**********************************************************************************************************************************


template <typename Template_type1, typename Template_type2> void sort_down(vector <Template_type1>& value, vector <Template_type2>& index)
{
	unsigned i;
	vector <std::pair<Template_type1, Template_type2> > paired_vector;
	
	
	paired_vector.resize(value.size());
	for (i=0; i<value.size(); i++)
		paired_vector[i] = std::pair<Template_type1, Template_type2>(value[i], index[i]);
	
	std::sort(paired_vector.begin(), paired_vector.end(), larger<Template_type1, Template_type2>);
	
	for (i=0; i<value.size(); i++)
	{
		value[i] = paired_vector[i].first;
		index[i] = paired_vector[i].second;
	}
}



//**********************************************************************************************************************************



template <typename Template_type> void apply_permutation(vector <Template_type>& vec, vector <unsigned> permutation)
{
	unsigned i;
	vector <Template_type> copy_of_vec;
	
	if (vec.size() != permutation.size())
		flush_exit(ERROR_DATA_MISMATCH, "Trying to apply a permutation to a vector of mismatching size.");
	
	copy_of_vec = vec;
	for (i=0; i<vec.size(); i++)
		vec[i] = copy_of_vec[permutation[i]];
	
	
}
