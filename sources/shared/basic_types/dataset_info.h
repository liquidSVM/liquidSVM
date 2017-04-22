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


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/dataset.h"
#endif


#if !defined (DATASET_INFO_H) 
	#define DATASET_INFO_H

	
#ifdef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/dataset.h"
#endif

#include <set>
#include <vector>
using namespace std;


//**********************************************************************************************************************************


enum LEARNING_SCENARIOS {CLASSIFICATION, REGRESSION, UNSUPERVISED, LEARNING_SCENARIOS_MAX};


//**********************************************************************************************************************************


class Tdataset_info
{
	public:
		Tdataset_info(){};
		Tdataset_info(const Tdataset& dataset, bool quick_info, double tau = -1.0, double label_tau = 0.05);
		
		unsigned get_label_number(double label);
		
		
		unsigned size;
		unsigned dim;
		unsigned kind;
		
		vector <int> label_list;
		vector <unsigned> label_count;
		unsigned most_frequent_label_number;
		
		double min_label;
		double max_label;
		double max_abs_label;
		
		double mean_label;
		double median_label;
		double abs_label_error;
		double square_label_error;
		double lower_label_quantile;
		double upper_label_quantile;
		
		vector <double> means;
		vector <double> minima;
		vector <double> maxima;
		vector <double> variances;
		vector <double> lower_quantiles;
		vector <double> upper_quantiles;
		
		vector <unsigned> list_of_categorial_coordinates;
		vector <vector <int> > categorial_values_of_coordinates; 
		
		vector <unsigned> coordinates_with_positive_quantile_distance;
		set <unsigned> sample_indices_outside_quantile_box;
		
	private:
		vector <unsigned> label_numbers;
};


//**********************************************************************************************************************************

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/dataset_info.cpp"
#endif


#endif

