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


#if !defined (KERNEL_FUNCTIONS_H) 
	#define KERNEL_FUNCTIONS_H





#include "sources/shared/basic_types/sample.h"
#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/system_support/cuda_basics.h"

//**********************************************************************************************************************************

inline double pre_kernel_function(unsigned type, Tsample* sample1, Tsample* sample2);
inline double kernel_function(unsigned type, double gamma_factor, Tsample* sample1, Tsample* sample2);

template <class float_type> inline float_type compute_gamma_factor(unsigned type, float_type gamma);
template <class float_type> __target_device__ inline float_type kernel_function(unsigned type, float_type gamma_factor, float_type pre_kernel_value);


//**********************************************************************************************************************************

// HIERARCHICAL KERNEL DEVELOPMENT

template <class float_type> inline float_type hierarchical_pre_kernel_function(float_type weights_square_sum, const vector <float_type>& hierarchical_weights_squared, const Tdataset& dataset1, const Tdataset& dataset2);


//**********************************************************************************************************************************


#include "sources/shared/kernel/kernel_functions.ins.cpp"

#endif


