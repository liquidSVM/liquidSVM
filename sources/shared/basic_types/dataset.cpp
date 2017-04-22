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


#if !defined (DATASET_CPP)
	#define DATASET_CPP




#include "sources/shared/basic_types/dataset.h"


#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_types/dataset_info.h"

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/random_subsets.h"
#include "sources/shared/basic_functions/random_generator.h"
#include "sources/shared/basic_functions/basic_file_functions.h"

#include "sources/shared/system_support/memory_allocation.h"



#include <list>


//**********************************************************************************************************************************


Tsubset_info compose(Tsubset_info info1, Tsubset_info info2)
{
	unsigned i;
	Tsubset_info result_info;
	
	for (i=0; i<info2.size(); i++)
		if (info2[i] < info1.size())
			result_info.push_back(info1[info2[i]]);
		else
			flush_exit(ERROR_DATA_MISMATCH, "Trying to compose two Tsubset_info objects that cannot be composed.");
	
	return result_info;
}


//**********************************************************************************************************************************


Tdataset::Tdataset()
{
	owns_samples = false;
	
	data_size = 0;
	
	original_label1 = 0.0;
	original_label2 = 0.0;
	original_most_frequent_label = 0.0;
}


//**********************************************************************************************************************************

Tdataset::Tdataset(const Tdataset& dataset)
{
	owns_samples = false;
	copy(dataset);
}



//**********************************************************************************************************************************

Tdataset::Tdataset(const double* data_array, unsigned size, unsigned dim, const double* labels, bool array_transposed)
{
	unsigned i;
	unsigned j;
	Tsample dummy_sample;
	
	data_size = 0;
	owns_samples = true;
	
	if (array_transposed == false)
		for(i=0; i<size; i++)
		{
			dummy_sample = Tsample(&data_array[i * dim], dim, 0.0);
			if (labels != NULL)
				dummy_sample.label = labels[i];
			else
				dummy_sample.labeled = false;

			push_back(&dummy_sample);
		}  
	else
	{
		dummy_sample = Tsample(CSV, dim);
		for(i=0; i<size; i++)
		{
			for (j=0; j<dim; j++)
				dummy_sample.change_coord(j, data_array[j * size + i]);

			if (labels != NULL)
				dummy_sample.label = labels[i];
			else
				dummy_sample.labeled = false;

			push_back(&dummy_sample);
		}  
	}
}


//**********************************************************************************************************************************


Tdataset::~Tdataset()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tdataset of size %d ...", size());
	clear();
	flush_info(INFO_PEDANTIC_DEBUG, "\nTdataset destroyed.");
}


//**********************************************************************************************************************************


void Tdataset::clear()
{
	unsigned i;

	if (owns_samples == true)
		for (i=0; i<size(); i++)
		{
			sample_list[i]->blocked_destruction = false;
			delete sample_list[i];
		}

	sample_list.clear();

	data_size = 0;
	
	original_label1 = 0.0;
	original_label2 = 0.0;
	original_most_frequent_label = 0.0;
	
	owns_samples = false;
}



//**********************************************************************************************************************************


void Tdataset::enforce_ownership()
{
	unsigned i;

	if (owns_samples == false)
	{
		flush_info(INFO_PEDANTIC_DEBUG, "\nEnforcing ownership for an object of type Tdataset of size %d.", size());
		for (i=0; i<size(); i++)
			sample_list[i] = new Tsample(sample_list[i]);
	}
	
	owns_samples = true;
}



//**********************************************************************************************************************************


Tdataset& Tdataset::operator = (const Tdataset& dataset)
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nCopying an object of type Tdataset of size %d.", dataset.size());
	copy(dataset);

	return *this;
}


//**********************************************************************************************************************************


void Tdataset::push_back(const Tsample& new_sample)
{
	if (owns_samples == true)
		push_back_mem_safe(sample_list, new Tsample(&new_sample));
	else
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to push a sample into a data set, that does not own its samples.");

	data_size++;
}


//**********************************************************************************************************************************


void Tdataset::push_back(Tsample* new_sample)
{
	if (owns_samples == true)
		push_back_mem_safe(sample_list, new Tsample(new_sample));
	else
		sample_list.push_back(new_sample);

	data_size++;
}



//**********************************************************************************************************************************


void Tdataset::push_back(const Tdataset& dataset)
{
	unsigned i;

	flush_info(INFO_PEDANTIC_DEBUG, "\nAppending a dataset of size %d to a dataset of size %d.", dataset.size(), size());
	for (i=0; i<dataset.size(); i++)
		push_back(const_cast <Tsample*> (dataset.sample(i)));
}



//**********************************************************************************************************************************


void Tdataset::read_from_file(FILE* fpread, unsigned filetype, unsigned size, unsigned dim)
{
	unsigned i;
	unsigned j;
	int read_status;
	Tsample dummy_sample;
	
	
	clear();
	enforce_ownership();


	if (size == 0)
		size = numeric_limits<unsigned>::max();
	
	i = 0;
	j = 0;
	do
	{
		read_status = dummy_sample.read_from_file(fpread, filetype, dim);
		if (read_status == FILE_OP_OK)
		{
			dummy_sample.number = i;
			push_back(&dummy_sample);
			i++;
			j++;
		}
		else if (read_status == FILE_CORRUPTED)
			exit_on_file_error(FILE_CORRUPTED, fpread);
	}
	while ((read_status != END_OF_FILE) and (i < size));
}



//**********************************************************************************************************************************


void Tdataset::read_from_file(string filename)
{
	unsigned dim;
	int filetype;
	FILE *fpread;
	Tsample dummy_sample;
	
	filetype = get_filetype(filename);
	check_data_filename(filename);

	fpread = open_file(filename,"r");
	
	dim = 0;
	if (filetype != LSV)
		dummy_sample.get_dim_from_file(fpread, filetype, dim);
	
	read_from_file(fpread, filetype, 0, dim);
	close_file(fpread);
	
	flush_info(INFO_2, "\nLoaded %d samples of dimension %d from file %s", size(), dim, filename.c_str());
}


//**********************************************************************************************************************************


void Tdataset::write_to_file(FILE* fpwrite, unsigned filetype) const
{
	unsigned i;
	unsigned data_dim;

	
	data_dim = dim();
	for (i=0;i<size();i++)
		sample_list[i]->write_to_file(fpwrite, filetype, data_dim);
}


//**********************************************************************************************************************************


void Tdataset::write_to_file(string filename) const
{
	FILE* fpwrite;
	int filetype;


	filetype = get_filetype(filename);
	check_data_filename(filename);
	fpwrite = open_file(filename, "w");
	
	flush_info(INFO_2, "\nWriting %d samples of dimension %d to file %s", size(), dim(), filename.c_str());
	write_to_file(fpwrite, filetype);
	
	close_file(fpwrite);
}


//**********************************************************************************************************************************


bool Tdataset::is_classification_data() const
{
	unsigned i;
	bool return_value;

	i = 0;
	return_value = true;
	do
	{
		if (double(int(sample_list[i]->label)) != sample_list[i]->label)
			return_value = false;
		i++;
	}
	while ((i < size()) and (return_value == true));

	return ((return_value == true) and (is_unsupervised_data() == false));
}


//**********************************************************************************************************************************


bool Tdataset::is_unsupervised_data() const
{
	unsigned i;
	bool labeled;
	

	labeled = true;	
	for (i=0; i<size(); i++)
		labeled = labeled and sample_list[i]->labeled;

	return not labeled;
}

//**********************************************************************************************************************************


vector <double> Tdataset::get_labels() const
{
	unsigned i;
	vector <double> labels;

	labels.resize(size());
	for(i=0;i<size();i++)
		labels[i] = sample_list[i]->label;
	return labels;
}


//**********************************************************************************************************************************


void Tdataset::change_labels(double old_label, double new_label)
{
	unsigned i;
	
	for (i=0; i<size(); i++)
		if (sample_list[i]->label == old_label)
			sample_list[i]->label = new_label;
}


//**********************************************************************************************************************************


void Tdataset::set_label_of_sample(unsigned index, double new_label)
{
	check_index(index);
	sample_list[index]->label = new_label;
}

//**********************************************************************************************************************************


void Tdataset::store_original_labels()
{
	Tdataset_info dataset_info;
	

	dataset_info = Tdataset_info(*this, true);
	
	if (is_classification_data() == false)
		flush_exit(ERROR_DATA_MISMATCH, "Trying to store true classification labels for dataset, which is not of classification type.");
	
	if (dataset_info.label_list.size() > 2)
		flush_exit(ERROR_DATA_MISMATCH, "Trying to store true binary classification labels for dataset, which has more than two labels.");

	
	if (dataset_info.label_list.size() == 1)
	{
		original_label1 = dataset_info.label_list[0];
		original_label2 = dataset_info.label_list[0];
		original_most_frequent_label = dataset_info.label_list[0];
	}
	else
	{
		original_label1 = dataset_info.label_list[0];
		original_label2 = dataset_info.label_list[1];
		original_most_frequent_label = dataset_info.label_list[dataset_info.most_frequent_label_number];
	}
}


//**********************************************************************************************************************************


pair <pair <double, double>, double> Tdataset::get_original_labels() const
{
	pair <double, double> label_pair;

	label_pair = pair <double, double> (original_label1, original_label2);
	return pair <pair <double, double>, double> (label_pair, original_most_frequent_label);
}


//**********************************************************************************************************************************


void Tdataset::compute_scaling(vector <double>& scaling, vector <double>& translate, double tau, unsigned type, bool uniform_scaling, bool scale_to_01) const
{
	unsigned i;
	double std_dev;
	double quantile_distance;
	Tdataset_info data_set_info;
	
	
	scaling.resize(dim());
	translate.resize(dim());
	data_set_info = Tdataset_info(*this, false, tau);
	
	std_dev = 0.0;
	quantile_distance = 0.0;
	if (uniform_scaling == true)
		for (i=0; i<dim(); i++)
		{
			std_dev = max(std_dev, sqrt(data_set_info.variances[i]));
			quantile_distance = max(quantile_distance, data_set_info.upper_quantiles[i] - data_set_info.lower_quantiles[i]);
		}

	for (i=0; i<dim(); i++)
	{
		if (type == QUANTILE)
		{
			if (uniform_scaling == false)
				quantile_distance = data_set_info.upper_quantiles[i] - data_set_info.lower_quantiles[i];
			
			if (quantile_distance > 0.0)
			{
				if (scale_to_01 == true)
					scaling[i] = 1.0 / quantile_distance;
				else
					scaling[i] = 2.0 / quantile_distance;
				translate[i] = 1.0 - scaling[i] * data_set_info.upper_quantiles[i];
			}
			else
			{
				scaling[i] = 1.0;
				translate[i] = - data_set_info.upper_quantiles[i];
			}
		}
		else
		{
			if (uniform_scaling == false)
				std_dev = sqrt(data_set_info.variances[i]);
			if (std_dev > 0.0)
			{
				scaling[i] = 1.0 / std_dev;
				translate[i] = - data_set_info.means[i] / std_dev;
			}
			else
			{
				scaling[i] = 1.0;
				translate[i] = - data_set_info.means[i];
			}
		}
	}
}


//**********************************************************************************************************************************

void Tdataset::apply_scaling(const vector <double>& scaling, const vector <double>& translate)
{
	unsigned i;
	Tsample dummy_sample;
	Tdataset scaled_data_set;
	
	
	if ((scaling.size() != dim()) or (translate.size() != dim()))
		flush_exit(ERROR_DATA_MISMATCH, "Trying to scale a data set of dimension %d by scale and translate vectors of size %d and %d.", dim(), scaling.size(), translate.size());
	
	enforce_ownership();
	scaled_data_set.enforce_ownership();
	for (i=0; i<size(); i++)
	{
		dummy_sample = scaling * (*sample(i));
		scaled_data_set.push_back(translate + dummy_sample);
	}
	
	copy(scaled_data_set);
}

//**********************************************************************************************************************************


void Tdataset::check_whether_complete_and_ordered() const
{
	unsigned i;
	
	for (i=0; i<size(); i++)
		if (sample_list[i]->number != i)
			flush_exit(ERROR_DATA_STRUCTURE, "Dataset of size %d is either not complete or not ordered.", size());
}


//**********************************************************************************************************************************


void Tdataset::create_subset(Tdataset& data_subset, Tsubset_info subset_info, bool give_ownership) const
{
	unsigned i;
	Tdataset_info dataset_info;
	
	
	data_subset.clear();
	if (give_ownership == true)
		data_subset.enforce_ownership();
	
	for(i=0;i<subset_info.size();i++)
	{
		if (subset_info[i] < sample_list.size())
			data_subset.push_back(sample_list[subset_info[i]]);
		else
			flush_exit(ERROR_DATA_STRUCTURE, "Trying to get sample number %i of a dataset of size %d", subset_info[i], sample_list.size());
	}
		
	dataset_info = Tdataset_info(data_subset, true);
	if ((dataset_info.kind == CLASSIFICATION) and (dataset_info.label_list.size() <= 2))
		data_subset.store_original_labels();
}

//**********************************************************************************************************************************


vector <unsigned> Tdataset::get_sample_numbers() const
{
	unsigned i;
	vector <unsigned> sample_list_tmp;
	
	
	for (i=0; i<size(); i++)
		sample_list_tmp.push_back(sample_list[i]->number);
	
	return sample_list_tmp;
}


//**********************************************************************************************************************************


unsigned Tdataset::get_index_of_closest_sample(Tsample* new_sample) const
{
	unsigned i;
	unsigned best_i;
	double distance;
	double smallest_distance;


	if (size() == 0)
		flush_exit(ERROR_DATA_MISMATCH, "Trying to get the closest sample in an empty data set.");
	
	best_i = 0;
	smallest_distance = squared_distance(sample_list[0], new_sample);
	
	for(i=1; i<size(); i++)
	{
		distance = squared_distance(sample_list[i], new_sample); 
		if (distance < smallest_distance)
		{
			smallest_distance = distance;
			best_i = i;
		}
	}
	return best_i;
}



//**********************************************************************************************************************************


Tsubset_info Tdataset::create_cover_subset_info_by_radius(double radius, int random_seed, unsigned subset_size) const
{
	unsigned i;
	unsigned farthest_index;
	Tdataset data_subset;
	Tsubset_info subset_info;
	Tsubset_info cover;
	vector <unsigned> permutation;
	vector <double> distances;
	
	
// If subset_size is set correctly, then create a subset and 
// create the cover from this subset and return it.

	if ((subset_size > 0) and (subset_size <= size()))
	{
		subset_info = id_permutation(size());
		subset_info = random_subset(subset_info, subset_size, random_seed);
		
		create_subset(data_subset, subset_info);
		cover = data_subset.create_cover_subset_info_by_radius(radius, random_seed, 0);
		
		for (i=0; i<cover.size(); i++)
			cover[i] = subset_info[cover[i]];
		
		return cover;
	}
	
	
// Otherwise, create the cover ...
// First, we do some preparations.
	
	init_random_generator(random_seed);
	farthest_index = (get_random_number() % size());
	distances.assign(size(), numeric_limits<double>::max());	
	
	
// 	Then we use the arthest-first traversal algorithm ...
	
	while (distances[farthest_index] > radius * radius)
	{
		cover.push_back(farthest_index);
		for (i=0; i<size(); i++)
			distances[i] = min(distances[i], squared_distance(sample_list[i], sample_list[cover.back()]));

		farthest_index = argmax(distances);
	}

	return cover;
}

//**********************************************************************************************************************************


Tsubset_info Tdataset::create_cover_subset_info_by_kNN(unsigned NNs, int random_seed, bool reduce, vector <double>& radii, unsigned subset_size) const
{
	unsigned i;
	unsigned j;
	unsigned l;
	unsigned reduced_NNs;
	unsigned pos;
	unsigned farthest_index;
	unsigned max_size;
	double max_distance;
	double reduction_fraction;
	unsigned largest_cell;
	vector <unsigned> cell_size;
	vector <unsigned> cell_affiliation;
	vector <unsigned> cell_affiliation_tmp;
	Tdataset data_subset;
	Tsubset_info subset_info;
	Tsubset_info cover;
	vector <unsigned> permutation;
	vector <double> distance_to_cover;
	vector <vector <double> > distances_to_cover_centers;
	


// If subset_size is set correctly, then create a subset and 
// create the cover from this subset and return it.

	if ((subset_size > 0) and (subset_size <= size()))
	{
		subset_info = id_permutation(size());
		subset_info = random_subset(subset_info, subset_size, random_seed);
		
		reduction_fraction = double(subset_info.size()) / double(size());
		reduced_NNs = unsigned(reduction_fraction * double(NNs));
		
		create_subset(data_subset, subset_info);
		cover = data_subset.create_cover_subset_info_by_kNN(reduced_NNs, random_seed, reduce, radii, 0);
		
		for (i=0; i<cover.size(); i++)
			cover[i] = subset_info[cover[i]];
		
		return cover;
	}
	
	
// Otherwise, create the cover ...
// First, we do some preparations.

	cell_affiliation.resize(size());
	distance_to_cover.assign(size(), numeric_limits<double>::max());

	init_random_generator(random_seed);
	farthest_index = (get_random_number() % size());
	
	j=0;
	largest_cell = 0;
	cell_size.resize(1);
	cell_size[0] = size();

	
// Now, we generate cover centers until all cells are smaller than the 
// desired number of nearest neighbors. We do this by splitting the largest
// cell. This splitting is done by choosing the sample in the cell that is
// farest away from the center. This way, the diameter of the cell is 
// aggressively reduced, which is likely good for learning. However, the 
// number of cells may be larger than necessary.

	if (size() <= NNs)
	{
		cover.push_back(farthest_index);

		radii.resize(1);
		distances_to_cover_centers.resize(1);
		distances_to_cover_centers[0].resize(size());
		for (i=0; i<size(); i++)
			distances_to_cover_centers[0][i] = squared_distance(sample_list[i], sample_list[farthest_index]);
		radii[0] = distances_to_cover_centers[0][argmax(distances_to_cover_centers[0])];

		return cover;
	}
	
	while (cell_size[largest_cell] > NNs)
	{
		cover.push_back(farthest_index);
		distances_to_cover_centers.resize(j+1);
		distances_to_cover_centers[j].resize(size());
		cell_size.assign(j+1, 0);
	
		max_distance = 0.0;
		for (i=0; i<size(); i++)
		{
			distances_to_cover_centers[j][i] = squared_distance(sample_list[i], sample_list[cover[j]]);
			
			if (distances_to_cover_centers[j][i] < distance_to_cover[i])
			{
				distance_to_cover[i] = distances_to_cover_centers[j][i];
				cell_affiliation[i] = j;
			}
			cell_size[cell_affiliation[i]]++;

			if ((distances_to_cover_centers[largest_cell][i] > max_distance) and (cell_affiliation[i] == largest_cell))
			{
				max_distance = distance_to_cover[i];
				farthest_index = i;
			}
		}

		largest_cell = argmax(cell_size);
		j++;
		if (5 * j > size())
			flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Found %d centers for a dataset of size %d", j , size());
	}
	flush_info(INFO_1, "\nFound %d centers for a dataset of size %d.", cover.size(), size());
	

	if (reduce == true)
	{
	// ... otherwise ...
	// Now that all cells are small enough, we try to merge cells, so that the number of
	// cells becomes smaller. We do this by repeatedly checking each cell for a merger.
	// The process stops when no cell can be merged anymore.
		
		pos = 0;
		while (pos < cover.size())
		{
			pos = 0;
			for (j=0; j<cover.size(); j++)
			{
				cell_size.assign(cover.size(), 0);
				distance_to_cover.assign(size(), numeric_limits<double>::max());
				cell_affiliation_tmp = cell_affiliation;
				
				for (i=0; i<size(); i++)
				{
					if (cell_affiliation[i] == pos)
						for (l=0; l<cover.size(); l++)
							if ((distances_to_cover_centers[l][i] < distance_to_cover[i]) and (pos != l))
							{
								distance_to_cover[i] = distances_to_cover_centers[l][i];
								cell_affiliation_tmp[i] = l;
							}
					cell_size[cell_affiliation_tmp[i]]++;
				}
				
				max_size = cell_size[argmax(cell_size)];

				if (max_size <= NNs)
				{
					cover.erase(cover.begin() + pos);
					distances_to_cover_centers.erase(distances_to_cover_centers.begin() + pos);

					for (i=0; i<size(); i++)
						if (cell_affiliation_tmp[i] < pos)
							cell_affiliation[i] = cell_affiliation_tmp[i];
						else 
							cell_affiliation[i] = cell_affiliation_tmp[i]-1;
				}
				else
					pos++;
			}
		}
		flush_info(INFO_1, " Reduced to %d centers.", cover.size());
	}

	
// Now we need to determine the radius for each cell and finally finish the job
	
	radii.resize(cover.size());
	for (j=0; j<cover.size(); j++)
	{
		get_k_smallest(distances_to_cover_centers[j], NNs-1);
		radii[j] = distances_to_cover_centers[j][NNs-1];
	}

	return cover;
}


//**********************************************************************************************************************************


Tsubset_info Tdataset::create_region_subset_info(unsigned NNs, int random_seed, double max_ignore_factor, vector <double>& radii, unsigned subset_size) const
{
	unsigned i;
	unsigned j;
	unsigned reduced_NNs;
	unsigned max_ignore;
	unsigned farthest_index;
	double max_distance;
	double reduction_fraction;
	Tdataset data_subset;
	Tsubset_info subset_info;
	Tsubset_info cover;
	vector <unsigned> permutation;
	vector <double> distance_to_cover;
	vector <vector <double> > distances_to_cover_centers;
	vector <double> current_distances;
	list <unsigned> unassigned_samples; 
	list <unsigned>::iterator iterator;

	
// If subset_size is set correctly, then create a subset and 
// create the cover from this subset and return it.

	if ((subset_size > 0) and (subset_size <= size()))
	{
		subset_info = id_permutation(size());
		subset_info = random_subset(subset_info, subset_size, random_seed);
		
		reduction_fraction = double(subset_info.size()) / double(size());
		reduced_NNs = unsigned(reduction_fraction * double(NNs));
		
		create_subset(data_subset, subset_info);
		cover = data_subset.create_region_subset_info(reduced_NNs, random_seed, max_ignore_factor, radii, 0);
		
		for (i=0; i<cover.size(); i++)
			cover[i] = subset_info[cover[i]];
		
		return cover;
	}
	radii.clear();
	
// Otherwise, create the cover ...
// First, we do some preparations.

	distance_to_cover.assign(size(), numeric_limits<double>::max());

	init_random_generator(random_seed);
	farthest_index = (get_random_number() % size());

	for (i=0; i<size(); i++)
		if (i != farthest_index)
			unassigned_samples.push_back(i);
		
	
// Now, we generate cover centers until all cells are smaller than the 
// desired number of nearest neighbors. We do this by splitting the largest
// cell. This splitting is done by choosing the sample in the cell that is
// farest away from the center. This way, the diameter of the cell is 
// aggressively reduced, which is likely good for learning. However, the 
// number of cells may be larger than necessary.

		
	if (size() <= NNs)
	{
		cover.push_back(farthest_index);

		radii.resize(1);
		distances_to_cover_centers.resize(1);
		distances_to_cover_centers[0].resize(size());
		for (i=0; i<size(); i++)
			distances_to_cover_centers[0][i] = squared_distance(sample_list[i], sample_list[farthest_index]);
		radii[0] = distances_to_cover_centers[0][argmax(distances_to_cover_centers[0])];

		return cover;
	}

	j=0;
	if (max_ignore_factor > 0.0)
		max_ignore = unsigned(double(NNs) * max_ignore_factor);
	else
		max_ignore = 0;
	while (unassigned_samples.size() > max_ignore)
	{
		cover.push_back(farthest_index);
		distances_to_cover_centers.resize(j+1);
		distances_to_cover_centers[j].resize(size());
	
		max_distance = 0.0;
		for (i=0; i<size(); i++)
		{
			distances_to_cover_centers[j][i] = squared_distance(sample_list[i], sample_list[cover[j]]);
			
			if (distances_to_cover_centers[j][i] < distance_to_cover[i])
				distance_to_cover[i] = distances_to_cover_centers[j][i];

			if (distance_to_cover[i] > max_distance)
			{
				max_distance = distance_to_cover[i];
				farthest_index = i;
			}
		}
		
		current_distances = distances_to_cover_centers[j];
		get_k_smallest(current_distances, NNs-1);
		radii.push_back(current_distances[NNs-1]);
		
		for (iterator = unassigned_samples.begin(); iterator != unassigned_samples.end();)
		{
			if (distances_to_cover_centers[j][*iterator] <= radii[j])
				unassigned_samples.erase(iterator++);
			else
				++iterator;
		}
		j++;
	}
	flush_info(INFO_1, "\nFound %d centers for covering the training set.", cover.size());
	
	
// Now we need to determine the radius for each cell and finally finish the job
	
	radii.resize(cover.size());
	for (j=0; j<cover.size(); j++)
	{
		get_k_smallest(distances_to_cover_centers[j], NNs-1);
		radii[j] = distances_to_cover_centers[j][NNs-1];
	}

	return cover;
}


//**********************************************************************************************************************************


double Tdataset::get_approximate_radius() const
{	
	unsigned i;
	unsigned k;
	unsigned extreme_i;
	unsigned current_farthest_i;
	bool improvement;
	double farthest_distance;
	vector <double> distances;	
	
	
	init_random_generator(1);
	extreme_i = (get_random_number() % size());
	distances.assign(size(), numeric_limits<double>::max());	
	
	k = 0;
	improvement = true;
	farthest_distance = 0.0;
	
	while ((k < 10) and (improvement == true))
	{
		for (i=0; i<size(); i++)
			distances[i] = squared_distance(sample_list[i], sample_list[extreme_i]);

		current_farthest_i = argmax(distances);
		if (distances[current_farthest_i] > farthest_distance)
		{
			extreme_i = current_farthest_i;
			farthest_distance = distances[current_farthest_i];
		}
		else
			improvement = false;
			
		k++;
	}

	return 0.5 * farthest_distance;
	
}


//**********************************************************************************************************************************


void Tdataset::sort_by_number()
{
	unsigned i;
	vector <unsigned> numbers;
	
	for (i=0; i<size(); i++)
		numbers.push_back(sample_list[i]->number);
	
	merge_sort_up(numbers, sample_list);
}

//**********************************************************************************************************************************


void Tdataset::group_spatially(unsigned chunk_size_aligned, unsigned chunks, vector <unsigned>& permutation)
{
	unsigned i;
	unsigned j;
	unsigned k;
	unsigned l;
	vector <double> radii;
	vector <unsigned> groups; 
	vector <unsigned> groups_sizes;
	vector <unsigned> centers;
	vector <unsigned> allowed_chunk_sizes;
	vector <vector <double> > distances;
	vector <double> distances_of_sample;
	vector <double> distances_of_last_group;
	vector <double> distances_of_non_assigned;
	vector <unsigned> not_assigned_yet;
	

	deactivate_display();
	
// Determine number the sizes of the cells
	
	allowed_chunk_sizes.assign(chunks, chunk_size_aligned);
	allowed_chunk_sizes[chunks-1] = size() - (chunks-1) * chunk_size_aligned;


// Create overlapping cover and pick the centers that are most crowded. Then compute
// the distances to these centers, and adjust the radius for the last group.
	
	centers = create_cover_subset_info_by_kNN(allowed_chunk_sizes[0], 1, false, radii, 0);
	merge_sort_up(radii, centers);
	
	distances.resize(size());
	distances_of_last_group.resize(size());
	for (i=0; i<size(); i++)
	{
		distances[i].resize(chunks);
		for (j=0; j<chunks; j++)
			distances[i][j] = squared_distance(sample_list[i], sample_list[centers[j]]);
		distances_of_last_group[i] = distances[i][chunks-1];
	}

	get_k_smallest(distances_of_last_group, allowed_chunk_sizes[chunks-1]-1);
	radii[chunks-1] = distances_of_last_group[allowed_chunk_sizes[chunks-1]-1];

	
// Now we assign each sample that lies in at least one specified ball to the corresponding center.
// If there are two or more such centers, we pick the closest one. 
	
	groups.assign(size(), 0);
	
	not_assigned_yet.reserve(size());
	groups_sizes.assign(chunks, 0);
	for (i=0; i<size(); i++)
	{
		distances_of_sample = distances[i];
		for (j=0; j<chunks; j++)
			if (distances_of_sample[j] > radii[j])
				distances_of_sample[j] = numeric_limits<double>::max();

		k = argmin(distances_of_sample);
		if ((distances_of_sample[k] <= radii[k]) and (groups_sizes[k] < allowed_chunk_sizes[k]))
		{
			groups[i] = k+1;
			groups_sizes[k]++;
		}
		else
			not_assigned_yet.push_back(i);
	}
	
	
// Next we need to deal with the samples that are not in any of the radii ...
// We do this greedily: for the first center we order all remaining samples
// by their distance to this center and then assign those to the first center,
// for which the first center is the nearest center. Then we repeat this for the 
// second center, where we allow assignment to the first two centers as long as 
// the cells are not full yet, and so on ...
	
	for (j=0; j<chunks; j++)
	{
		distances_of_non_assigned.resize(not_assigned_yet.size());
		for (i=0; i<not_assigned_yet.size(); i++)
			distances_of_non_assigned[i] = distances[not_assigned_yet[i]][j];
		merge_sort_up(distances_of_non_assigned, not_assigned_yet);
		
		i = 0;
		while ((groups_sizes[j] < allowed_chunk_sizes[j]) and (i < not_assigned_yet.size()))
		{
			distances_of_sample = distances[not_assigned_yet[i]];
			for (l=0; l<chunks; l++)
				if (groups_sizes[l] == allowed_chunk_sizes[l])
					distances_of_sample[l] = numeric_limits<double>::max();
			
			k = argmin(distances_of_sample);
			if (k <= j)
			{
				groups[not_assigned_yet[i]] = k+1;
				groups_sizes[k]++;
			}
			i++;
		}
		
		not_assigned_yet.clear();
		distances_of_non_assigned.clear();
		for (i=0; i<size(); i++)
			if (groups[i] == 0)
				not_assigned_yet.push_back(i);
	}
	
	
// Still there may be some samples left. We assign them to the nearest
// center whose cell is not full, yet.

	for (i=0; i<not_assigned_yet.size(); i++)
	{
		distances_of_sample = distances[not_assigned_yet[i]];
		for (l=0; l<chunks; l++)
			if (groups_sizes[l] == allowed_chunk_sizes[l])
				distances_of_sample[l] = numeric_limits<double>::max();
			
		k = argmin(distances_of_sample);
		groups[not_assigned_yet[i]] = k+1;
		groups_sizes[k]++;
	}

	
	groups_sizes.assign(chunks, 0);
	permutation.resize(size());
	for (i=0; i<size(); i++)
	{
		permutation[(groups[i]-1) * chunk_size_aligned + groups_sizes[groups[i]-1]] = i;
		groups_sizes[groups[i]-1]++;
	}

	apply_permutation(sample_list, permutation);
	
	reactivate_display();
}

//**********************************************************************************************************************************


Tsubset_info Tdataset::create_subset_info_with_label(double label) const
{
	unsigned i;
	Tsubset_info subset_info;

	for (i=0; i<size(); i++)
		if (sample_list[i]->label == label)
			subset_info.push_back(i);

	return subset_info;
}


//**********************************************************************************************************************************


double* Tdataset::convert_to_array(unsigned start_index, unsigned end_index) const
{
	unsigned i;
	unsigned length;
	unsigned dim_max;
	double* array;

	if (start_index > end_index)
		flush_exit(ERROR_DATA_STRUCTURE, "Cannot convert described part of dataset to array");

	else if (start_index == end_index)
		return NULL;

	length = end_index - start_index;
	dim_max = dim();
	array = new double[length * dim_max];

	
// Make sure the array contains zero coordinates for samples, whose dimension is smaller than dim().
	
	for (i=0; i<length * dim_max; i++)
		array[i] = 0.0;

	
// Now copy the samples into the array
	
	for(i=0;i<length;i++)
		sample_list[start_index + i]->get_x_part(&array[i * dim_max]);

	return array;
}



//**********************************************************************************************************************************


unsigned Tdataset::required_memory_on_GPU(unsigned start_index, unsigned end_index) const
{
	return (end_index - start_index) * dim();
}


//**********************************************************************************************************************************


bool Tdataset::operator == (const Tdataset& dataset) const
{
	unsigned i;
	bool equal;


	if (size() != dataset.size())
		return false;

	equal = true;
	i = 0;
	while ((i < size()) and (equal == true))
	{
		equal = (*sample_list[i] == *dataset.sample_list[i]);
		i++;
	}
	return equal;
}


//**********************************************************************************************************************************

void Tdataset::copy(const Tdataset& dataset)
{
	bool owns_samples_tmp;
	
	owns_samples_tmp = owns_samples;
	clear();
	
	data_size = dataset.data_size;
	sample_list = dataset.sample_list;
	
	if (owns_samples_tmp == true)
		enforce_ownership();
	owns_samples = owns_samples_tmp;
	
	original_label1 = dataset.original_label1;
	original_label2 = dataset.original_label2;
	original_most_frequent_label = dataset.original_most_frequent_label;
}


#endif


