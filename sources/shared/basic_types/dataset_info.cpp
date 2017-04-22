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


#if !defined (DATASET_INFO_CPP) 
	#define DATASET_INFO_CPP



#include "sources/shared/basic_types/dataset_info.h"


#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_functions/flush_print.h"


#include <cmath>
#include <limits>


//**********************************************************************************************************************************


Tdataset_info::Tdataset_info(const Tdataset& dataset, bool quick_info, double tau, double label_tau)
{
	unsigned i;
	unsigned j;
	unsigned istar;
	unsigned largest_label_count;
	set <int> label_list_tmp;
	vector <double> labels;
	std::pair<double, double> label_median_tmp;
	vector <double> coordinates;
	vector <int> categorial_values;


	size = dataset.size();
	dim = dataset.dim();

	labels = dataset.get_labels();

	min_label = labels[argmin(labels)];
	max_label = labels[argmax(labels)];
	max_abs_label = max(abs(min_label), abs(max_label));	
	
	if (dataset.is_classification_data())
	{
		kind = CLASSIFICATION;
		
		for (i=0; i<size; i++)
			label_list_tmp.insert(int(labels[i]));
		copy(label_list_tmp.begin(), label_list_tmp.end(), inserter(label_list, label_list.begin()));

		for (i=0; i<label_list.size(); i++)
			label_count.push_back(unsigned(dataset.create_subset_info_with_label(label_list[i]).size()));
		
		label_numbers.assign(label_list[label_list.size()-1] - label_list[0] + 1, 0);
		for (i=0; i<label_list.size(); i++)
			label_numbers[label_list[i] - label_list[0]] = i;

		most_frequent_label_number = 0;
		largest_label_count = label_count[0];
		for (i=1; i<label_list.size(); i++)
			if (largest_label_count < label_count[i])
			{
				most_frequent_label_number = i;
				largest_label_count = label_count[i];
			}
	}
	else
	{
		kind = REGRESSION;

		mean_label = mean(labels);
		square_label_error = square_sum(labels, mean_label) / double(size);

		label_median_tmp = quantile(labels, 0.5);
		median_label = 0.5 * (label_median_tmp.first + label_median_tmp.second);
		abs_label_error = abs_sum(labels, median_label) / double(size);

		lower_label_quantile = quantile(labels, label_tau).first;
		upper_label_quantile = quantile(labels, 1.0 - label_tau).second;
	}

	if (quick_info == false)
	{
		means.resize(dim);
		minima.resize(dim);
		maxima.resize(dim);
		variances.resize(dim);
		lower_quantiles.resize(dim);
		upper_quantiles.resize(dim);
		coordinates.resize(size);
		
		for (j=0; j<dim; j++)
		{
			for (i=0; i<size; i++)
				coordinates[i] = dataset.sample(i)->coord(j);

			means[j] = mean(coordinates);
			variances[j] = variance(coordinates);
			
			categorial_values = is_categorial(coordinates);
			if (categorial_values.size() > 0)
			{
				list_of_categorial_coordinates.push_back(j);
				categorial_values_of_coordinates.push_back(categorial_values);
			}
			
			if ((tau >= 0.0) and (tau <= 0.5))
			{
				lower_quantiles[j] = quantile(coordinates, tau).first;
				upper_quantiles[j] = quantile(coordinates, 1.0 - tau).second;

				for (i=0; i<size; i++)
					if ((coordinates[i] < lower_quantiles[j]) or (coordinates[i] > upper_quantiles[j]))
						sample_indices_outside_quantile_box.insert(i);

				if (lower_quantiles[j] < upper_quantiles[j])
					coordinates_with_positive_quantile_distance.push_back(j);
			}
			else
			{
				lower_quantiles[j] = std::numeric_limits<double>::max();
				upper_quantiles[j] = -std::numeric_limits<double>::max();
			}
			
			istar = argmin(coordinates);
			minima[j] = coordinates[istar];
			
			istar = argmax(coordinates);
			maxima[j] = coordinates[istar];
		}
	}
}



//**********************************************************************************************************************************


unsigned Tdataset_info::get_label_number(double label)
{
	if (kind == CLASSIFICATION)
		return label_numbers[int(label) - label_list[0]];
	else
		return 0;
}

#endif


