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



#include "sources/shared/kernel/kernel_control.h"

#include <cmath>


//**********************************************************************************************************************************


template <class float_type> inline float_type compute_gamma_factor(unsigned type, float_type gamma)
{
	switch (type)
	{
		case GAUSS_RBF:
			return -1.0/(gamma * gamma);

		case POISSON:
			return -1.0/gamma;
	}
	return 0.0;
}


//**********************************************************************************************************************************

inline double pre_kernel_function(unsigned type, Tsample* sample1, Tsample* sample2)
{
	return squared_distance(sample1, sample2);
}


//**********************************************************************************************************************************

template <class float_type> __target_device__ inline float_type kernel_function(unsigned type, float_type gamma_factor, float_type pre_kernel_value)
{
	switch (type)
	{
		case GAUSS_RBF:
			return exp(gamma_factor * pre_kernel_value);
		case POISSON:
			return exp(gamma_factor * sqrt(pre_kernel_value));
	}
	return 1.0;
}


//**********************************************************************************************************************************

inline double kernel_function(unsigned type, double gamma_factor, Tsample* sample1, Tsample* sample2)
{
	return kernel_function(type, gamma_factor, pre_kernel_function(type, sample1, sample2));
}


//**********************************************************************************************************************************


// HIERARCHICAL KERNEL DEVELOPMENT

template <class float_type> inline float_type hierarchical_pre_kernel_function(float_type weights_square_sum, const vector <float_type>& hierarchical_weights_squared, const Tdataset& dataset1, const Tdataset& dataset2)
{
	unsigned l;
	float_type pre_kernel_value;
	float_type projected_distance;
	
	pre_kernel_value = weights_square_sum;
	for (l=0; l<hierarchical_weights_squared.size(); l++)
	{
		projected_distance = pre_kernel_function(GAUSS_RBF, dataset1.sample(l), dataset2.sample(l));
		pre_kernel_value = pre_kernel_value - hierarchical_weights_squared[l] * exp( -projected_distance);
	}
	return pre_kernel_value;
}
