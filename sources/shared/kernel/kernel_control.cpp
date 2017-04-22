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


#if !defined (KERNEL_CONTROL_CPP)
	#define KERNEL_CONTROL_CPP


#include "sources/shared/kernel/kernel_control.h"




#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/random_subsets.h"
#include "sources/shared/basic_functions/random_generator.h"
#include "sources/shared/basic_functions/basic_file_functions.h"




//**********************************************************************************************************************************


Tkernel_control::Tkernel_control()
{
	clear();
};




//**********************************************************************************************************************************


void Tkernel_control::read_from_file(FILE *fp)
{
	file_read(fp, kernel_type);
	file_read(fp, memory_model_kernel);
	file_read(fp, memory_model_pre_kernel);
	file_read(fp, cache_size);
	file_read(fp, pre_cache_size);
	file_read(fp, kNNs);
	file_read(fp, hierarchical_kernel_control_read_filename);

}

//**********************************************************************************************************************************


void Tkernel_control::write_to_file(FILE *fp) const
{
	file_write(fp, kernel_type);
	file_write(fp, memory_model_kernel);
	file_write(fp, memory_model_pre_kernel);
	file_write(fp, cache_size);
	file_write(fp, pre_cache_size);
	file_write(fp, kNNs);
	file_write_eol(fp);
	file_write(fp, hierarchical_kernel_control_read_filename);
	file_write_eol(fp);
}



//**********************************************************************************************************************************


void Tkernel_control::read_hierarchical_kernel_info_from_file()
{
	unsigned l;
	unsigned nodes;
	FILE *fpd;
	
	
	if (hierarchical_kernel_control_read_filename.size() > 0)
	{
		flush_info(INFO_3, "\nReading hierarchical kernel information from file %s.", hierarchical_kernel_control_read_filename.c_str());
		fpd = open_file(hierarchical_kernel_control_read_filename, "r");
		
		file_read(fpd, kernel_type);
		file_read(fpd, full_kernel_type);
		file_read(fpd, hierarchical_weights_squared);
		file_read(fpd, full_dim);  

		file_read(fpd, nodes);
		hierarchical_gammas.resize(nodes);
		for (l=0; l<hierarchical_gammas.size(); l++)
			file_read(fpd, hierarchical_gammas[l]);
        
		file_read(fpd, hierarchical_coordinates);
		close_file(fpd);
		
		hierarchical_coordinates_exanded = false;
		make_consistent();
		
		flush_info(INFO_3, " Check sum is %1.4f", get_hierarchical_weight_square_sum());
	}
}

//**********************************************************************************************************************************


void Tkernel_control::write_hierarchical_kernel_info_to_file()
{
	unsigned l;
	FILE *fpd;
	

	if (hierarchical_kernel_control_write_filename.size() > 0)
	{
		flush_info(INFO_3, "\nWriting hierarchical kernel information with check sum %1.4f to file %s.", get_hierarchical_weight_square_sum(), hierarchical_kernel_control_write_filename.c_str());
		fpd = open_file(hierarchical_kernel_control_write_filename, "w");

		file_write(fpd, kernel_type);
		file_write(fpd, full_kernel_type);
		file_write(fpd, hierarchical_weights_squared, "%3.15f ", "");
		file_write(fpd, full_dim, "\n");  
		
		file_write(fpd, unsigned(hierarchical_gammas.size()), "\n");
		for (l=0; l<hierarchical_gammas.size(); l++)
			file_write(fpd, hierarchical_gammas[l], "%3.15f ", "");

		file_write(fpd, hierarchical_coordinates, " ");
		close_file(fpd);
	}
}



//**********************************************************************************************************************************


void Tkernel_control::init_random_hierarchical_weights(unsigned dim, unsigned nodes, double saturation, bool balanced, int random_seed, unsigned extra_seed)
{
	unsigned i;
	unsigned j;
	unsigned l;
	unsigned li;
	unsigned disjoint_start;
	unsigned disjoint_stop;
	unsigned full_disjoint_start;
	unsigned full_disjoint_stop;
	unsigned current_dim;
	unsigned full_current_dim;
	unsigned rest_current_dim;
	vector <unsigned> random_perm;
	vector <unsigned> random_perm_rest;
	vector <unsigned> random_rest_coordinates;
	double coordinates_per_node;

	
	if (nodes == 1)
		saturation = 1.0;
	
	full_dim = unsigned(double(dim) * saturation);
	coordinates_per_node = double(full_dim) / double(nodes);
	
	if (coordinates_per_node < 1.0)
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "The specified saturation %1.2f, the dimension %d of the data, and the number of\nnodes %d lead to %1.2f coordinates per node (less than one coordinate per node).", saturation, dim, nodes, coordinates_per_node);
	
	if (coordinates_per_node > double(dim))
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "The specified saturation %1.2f, the dimension %d of the data, and the number of\nnodes %d lead to %1.2f coordinates per node (identical coordinates per node).", saturation, dim, nodes, coordinates_per_node);	
	
	
	hierarchical_gammas.resize(nodes);
	hierarchical_weights_squared.assign(nodes, 1.0 / nodes);
	hierarchical_coordinates.resize(nodes);
	make_consistent();
	
	init_random_generator(random_seed, extra_seed);
	random_perm = random_permutation(dim, random_seed, extra_seed);

	if (balanced == true)
	{
		for (l=0; l<nodes; l++)
		{
			disjoint_start = ((full_dim * l) / nodes) % dim;
			disjoint_stop = ((full_dim * (l+1)) / nodes) % dim;
			
			if (disjoint_start < disjoint_stop)
			{
				current_dim = disjoint_stop - disjoint_start;
				hierarchical_coordinates[l].resize(current_dim);
				for (li=0; li<current_dim; li++)
					hierarchical_coordinates[l][li] = random_perm[disjoint_start + li];
			}
			else
			{
				current_dim = (disjoint_stop + dim) - disjoint_start;
				hierarchical_coordinates[l].resize(current_dim);
				
				random_shuffle(random_perm, 0, disjoint_stop, random_seed, extra_seed + l);
				for (li=0; li<current_dim; li++)
					hierarchical_coordinates[l][li] = random_perm[(disjoint_start + li) % dim];
				
				random_shuffle(random_perm, disjoint_stop, random_perm.size(), random_seed, extra_seed + l * nodes + 2);
			}
			sort_up(hierarchical_coordinates[l]);
			hierarchical_gammas[l].assign(current_dim, 1.0);
		}
	}
	else
	{
		for (l=0; l<nodes; l++)
		{
			disjoint_start = unsigned(double(l * dim) / double(nodes)); 
			disjoint_stop = unsigned(double((l+1) * dim) / double(nodes)); 
			current_dim = disjoint_stop - disjoint_start;
			
			full_disjoint_start = unsigned(double(l * full_dim) / double(nodes)); 
			full_disjoint_stop = unsigned(double((l+1) * full_dim) / double(nodes)); 
			full_current_dim = full_disjoint_stop - full_disjoint_start;
			hierarchical_coordinates[l].resize(full_current_dim);
			
			for (li=0; li<current_dim; li++)
				hierarchical_coordinates[l][li] = random_perm[disjoint_start + li];
			
			random_perm_rest.resize(int(dim) - int(current_dim));
			j = 0;
			for (i=0; i<dim; i++)
				if ((i < disjoint_start) or (i >= disjoint_stop))
				{
					random_perm_rest[j] = random_perm[i];
					j++;
				}
			
			rest_current_dim = int(full_current_dim) - int(current_dim); 
			random_rest_coordinates = random_subset(random_perm_rest, rest_current_dim, random_seed, extra_seed + l);
			
			for (li=0; li<rest_current_dim; li++)
				hierarchical_coordinates[l][current_dim +li] = random_rest_coordinates[li];
			
			sort_up(hierarchical_coordinates[l]);
			hierarchical_gammas[l].assign(full_current_dim, 1.0);
		}
	}
}







//**********************************************************************************************************************************

void Tkernel_control::init_image_hierarchical_weights(unsigned square_x, unsigned square_y, unsigned image_x, unsigned image_y)
{
	unsigned i;
	unsigned x;
	unsigned y;
	unsigned l;
	unsigned nx;
	unsigned ny;
	unsigned nodes;
	unsigned grid_x;
	unsigned grid_y;
	
	
	grid_x = unsigned(ceil(double(image_x) / double(square_x)));
	grid_y = unsigned(ceil(double(image_y) / double(square_y)));
	nodes = grid_x * grid_y;
	full_dim = nodes * square_x * square_y;
	
	hierarchical_weights_squared.assign(nodes, 1.0 / nodes);
	hierarchical_gammas.resize(nodes);
	hierarchical_coordinates.resize(nodes);
	make_consistent();
	
	for (y=0; y<image_y; y++)
	{
		ny = y / square_y;
		for (x=0; x<image_x; x++)
		{
			nx = x / square_x;
			i = y * image_x + x;
			l = ny * grid_x + nx;
			
			hierarchical_coordinates[l].push_back(i);
		}
	}
	
	for (l=0; l<nodes; l++)
		hierarchical_gammas[l].assign(hierarchical_coordinates[l].size(), 1.0);
}




//**********************************************************************************************************************************

void Tkernel_control::change_random_hierarchical_weights(double max_spread, int random_seed, unsigned extra_seed)
{
	unsigned l;
	unsigned li;
	double rand_factor;

	
	init_random_generator(random_seed, extra_seed);
	
	for (l=0; l<hierarchical_coordinates.size(); l++)
	{
		rand_factor = ((1.0 - max_spread) + 2.0 * max_spread * get_uniform_random_number());
		hierarchical_weights_squared[l] = hierarchical_weights_squared[l] * rand_factor * rand_factor; 
		for (li=0; li<hierarchical_coordinates[l].size(); li++)
			hierarchical_gammas[l][li] = hierarchical_gammas[l][li] * ((1.0 - max_spread) + 2.0 * max_spread * get_uniform_random_number()); 
	}
}


//**********************************************************************************************************************************

void Tkernel_control::convert_to_hierarchical_data_set(const Tdataset& dataset, vector <Tdataset>& hierarchical_dataset) const
{
	unsigned i;
	unsigned l;
	

	hierarchical_dataset.resize(dataset.size());
	for (i=0; i<dataset.size(); i++)
	{
		hierarchical_dataset[i].clear();
		hierarchical_dataset[i].enforce_ownership();
		for (l=0; l<hierarchical_coordinates.size(); l++)
			hierarchical_dataset[i].push_back(hierarchical_gammas[l] * dataset.sample(i)->project(hierarchical_coordinates[l]));
	}
}


//**********************************************************************************************************************************

void Tkernel_control::convert_to_hierarchical_GPU_data_set(const vector <Tdataset>& hierarchical_dataset, Tdataset& full_coord_dataset, unsigned start_index, unsigned stop_index) const
{
	unsigned i;
	unsigned j;
	Tsample current_sample;
	vector <double> all_coordinates;
	vector <double> new_coordinates;
	
	
	full_coord_dataset.clear();
	full_coord_dataset.enforce_ownership();
	for (i=start_index; i<stop_index; i++)
	{
		all_coordinates.clear();
		for (j=0; j<hierarchical_coordinates.size(); j++)
		{
			new_coordinates = hierarchical_dataset[i].sample(j)->get_x_part();
			all_coordinates.insert(all_coordinates.end(), new_coordinates.begin(), new_coordinates.end());
		}
		current_sample = Tsample(all_coordinates, 0.0);
		full_coord_dataset.push_back(current_sample);
	}
}

//**********************************************************************************************************************************

unsigned Tkernel_control::get_hierarchical_coordinate_intervals(vector <unsigned>& hierarchical_coordinate_intervals) const
{
	unsigned l;
	unsigned total_number_of_coordinates;
	
	hierarchical_coordinate_intervals.resize(hierarchical_coordinates.size() + 1);
	total_number_of_coordinates = 0;
	for (l=0; l<hierarchical_coordinates.size(); l++)
	{
		hierarchical_coordinate_intervals[l] = total_number_of_coordinates;
		total_number_of_coordinates = total_number_of_coordinates + hierarchical_coordinates[l].size();
	}
	hierarchical_coordinate_intervals[hierarchical_coordinates.size()] = total_number_of_coordinates;	
	
	return total_number_of_coordinates;
}

//**********************************************************************************************************************************


unsigned Tkernel_control::get_total_number_of_hierarchical_coordinates() const
{
	unsigned l;
	unsigned total_number_of_coordinates;
	

	total_number_of_coordinates = 0;
	for (l=0; l<hierarchical_coordinates.size(); l++)
		total_number_of_coordinates = total_number_of_coordinates + hierarchical_coordinates[l].size();
	
	return total_number_of_coordinates;
}




//**********************************************************************************************************************************


unsigned Tkernel_control::get_max_number_hierarchical_coordinates_at_nodes() const
{
	unsigned l;
	unsigned max_number_hierarchical_coordinates;
	
	
	max_number_hierarchical_coordinates = 0.0;
	for (l=0; l<hierarchical_coordinates.size(); l++)
		max_number_hierarchical_coordinates = max(max_number_hierarchical_coordinates, unsigned(hierarchical_coordinates[l].size()));

	return max_number_hierarchical_coordinates;
}


//**********************************************************************************************************************************

double Tkernel_control::get_hierarchical_weight_square_sum() const
{
	return sum(hierarchical_weights_squared);
}



//**********************************************************************************************************************************

void Tkernel_control::expand_hierarchical_coordinates()
{
	unsigned l;
	unsigned li;
	vector <vector <double> > hierarchical_gammas_new;
	vector < vector <unsigned> > hierarchical_coordinates_new;
	
	if (full_dim == 0) 
		flush_exit(ERROR_DATA_MISMATCH, "Cannot expand deep kernel coordinates without knowing true dimension.");
	
	if (hierarchical_coordinates_exanded == false)
	{
		hierarchical_gammas_new.resize(hierarchical_coordinates.size());
		hierarchical_coordinates_new.resize(hierarchical_coordinates.size());
		for (l=0; l<hierarchical_coordinates.size(); l++)
		{
			hierarchical_gammas_new[l].assign(full_dim, 0.0);
			hierarchical_coordinates_new[l] = id_permutation(full_dim);
			for (li=0; li<hierarchical_coordinates[l].size(); li++)
				hierarchical_gammas_new[l][hierarchical_coordinates[l][li]] = hierarchical_gammas[l][li];
		}
		
		hierarchical_gammas = hierarchical_gammas_new;
		hierarchical_coordinates = hierarchical_coordinates_new;
		
		hierarchical_coordinates_exanded = true;
	}
}




//**********************************************************************************************************************************

void Tkernel_control::shrink_hierarchical_coordinates()
{
	unsigned l;
	unsigned li;
	vector <vector <double> > hierarchical_gammas_new;
	vector < vector <unsigned> > hierarchical_coordinates_new;
	
	
	if (hierarchical_coordinates_exanded == true)
	{
		hierarchical_gammas_new.resize(hierarchical_coordinates.size());
		hierarchical_coordinates_new.resize(hierarchical_coordinates.size());
		
		for (l=0; l<hierarchical_coordinates.size(); l++)
		{
			hierarchical_gammas_new[l].reserve(full_dim);
			hierarchical_coordinates_new[l].reserve(full_dim);
			for (li=0; li<hierarchical_coordinates[l].size(); li++)
				if (hierarchical_gammas[l][li] != 0.0)
				{
					hierarchical_gammas_new[l].push_back(hierarchical_gammas[l][li]);
					hierarchical_coordinates_new[l].push_back(li);
				}
		}
		
		hierarchical_gammas = hierarchical_gammas_new;
		hierarchical_coordinates = hierarchical_coordinates_new;
		
		hierarchical_coordinates_exanded = false;
	}
}





//**********************************************************************************************************************************


bool Tkernel_control::is_hierarchical_kernel_type(unsigned kernel_type) const
{
	if ((kernel_type == HIERARCHICAL_GAUSS) or (kernel_type == HETEROGENEOUS_GAUSS))
		return true;
	else
		return false;
}


//**********************************************************************************************************************************


bool Tkernel_control::is_hierarchical_kernel() const
{
	return ((is_hierarchical_kernel_type(kernel_type) == true) or (is_hierarchical_kernel_type(full_kernel_type) == true));
}


//**********************************************************************************************************************************


void Tkernel_control::make_consistent()
{
	unsigned nodes;
	
	if (made_consistent == false)
	{
		made_consistent = true;
		nodes = hierarchical_weights_squared.size();
		orginal_number_of_nodes = nodes;

		if (nodes == 1)
			full_kernel_type = HETEROGENEOUS_GAUSS;
		else if (nodes > 1)
			full_kernel_type = HIERARCHICAL_GAUSS;
	
		if (nodes >= 1)
			kernel_type = GAUSS_RBF;
	}
}



//**********************************************************************************************************************************


void Tkernel_control::clear()
{
	kernel_type = GAUSS_RBF;
	
	#ifdef __MINGW32__
		memory_model_kernel = LINE_BY_LINE;
		memory_model_pre_kernel = LINE_BY_LINE;
	#else
		memory_model_kernel = BLOCK;
		memory_model_pre_kernel = BLOCK;
	#endif
	
	cache_size = 512;
	pre_cache_size = 1024;
	
	kernel_store_on_GPU = false;
	pre_kernel_store_on_GPU = true;
	split_matrix_on_GPU_by_rows = true;
	allowed_percentage_of_GPU_RAM = 0.5;
	
	same_data_sets = false;
	max_col_set_size = 0;
	max_row_set_size = 0;
	
	include_labels = true;
	kNNs = DEFAULT_NN;
	kNN_number_of_chunks = 1;
	
	kernel_type = GAUSS_RBF;
	full_kernel_type = GAUSS_RBF;
	
	hierarchical_kernel_control_read_filename.clear();
	hierarchical_kernel_control_write_filename.clear();
	
	full_dim = 0;
	hierarchical_coordinates_exanded = false;
	
	
	made_consistent = false;
	orginal_number_of_nodes = 0;
}

//**********************************************************************************************************************************


bool Tkernel_control::is_full_matrix_pre_model() const
{
	return ((memory_model_pre_kernel == LINE_BY_LINE) or (memory_model_pre_kernel == BLOCK));
}
		
//**********************************************************************************************************************************

bool Tkernel_control::is_full_matrix_model() const
{
	return ((memory_model_kernel == LINE_BY_LINE) or (memory_model_kernel == BLOCK));
}


#endif
