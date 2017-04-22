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


#if !defined (KERNEL_CONTROL_H)
	#define KERNEL_CONTROL_H


 
#include "sources/shared/basic_types/dataset.h"

 
#include <vector>
#include <string>
#include <cstdio>
using namespace std;


//**********************************************************************************************************************************


const int DEFAULT_NN = -1;
enum KERNEL_TYPES {GAUSS_RBF, POISSON, HETEROGENEOUS_GAUSS, HIERARCHICAL_GAUSS, KERNEL_TYPES_MAX};
enum KERNEL_MEMORY_MODELS {LINE_BY_LINE, BLOCK, CACHE, EMPTY, KERNEL_MEMORY_MODELS_MAX};



//**********************************************************************************************************************************

class Tkernel_control
{
	public:
		Tkernel_control();
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;
		void read_hierarchical_kernel_info_from_file();
		void write_hierarchical_kernel_info_to_file();
		void clear();
		
		
// 	HIERARCHICAL KERNEL DEVELOPMENT
		
		void init_random_hierarchical_weights(unsigned dim, unsigned nodes, double saturation, bool balanced, int random_seed, unsigned extra_seed);
		void init_image_hierarchical_weights(unsigned square_x, unsigned square_y, unsigned image_x, unsigned image_y);
		void change_random_hierarchical_weights(double max_spread, int random_seed, unsigned extra_seed);
		
		
		void convert_to_hierarchical_data_set(const Tdataset& dataset, vector <Tdataset>& hierarchical_dataset) const;
		void convert_to_hierarchical_GPU_data_set(const vector <Tdataset>& hierarchical_dataset, Tdataset& full_coord_dataset, unsigned start_index, unsigned stop_index) const;
		double get_hierarchical_weight_square_sum() const;
		unsigned get_hierarchical_coordinate_intervals(vector <unsigned>& hierarchical_coordinate_intervals) const;
		unsigned get_total_number_of_hierarchical_coordinates() const;
		unsigned get_max_number_hierarchical_coordinates_at_nodes() const;
		void expand_hierarchical_coordinates();
		void shrink_hierarchical_coordinates();
		
		void make_consistent();
		bool is_hierarchical_kernel() const;

		bool is_full_matrix_pre_model() const;
		bool is_full_matrix_model() const;

		unsigned kernel_type;
		unsigned full_kernel_type;

		unsigned memory_model_pre_kernel;
		unsigned memory_model_kernel;
		unsigned cache_size;
		unsigned pre_cache_size;

		bool kernel_store_on_GPU;
		bool pre_kernel_store_on_GPU;
		bool split_matrix_on_GPU_by_rows;
		double allowed_percentage_of_GPU_RAM; 

		bool same_data_sets;
		unsigned max_col_set_size;
		unsigned max_row_set_size;

		bool include_labels;
		int kNNs;
		unsigned kNN_number_of_chunks;
		
		string hierarchical_kernel_control_read_filename;
		string hierarchical_kernel_control_write_filename;

		
// 	HIERARCHICAL KERNEL DEVELOPMENT

		vector <double> hierarchical_weights_squared;
		vector <vector <double> > hierarchical_gammas;
		vector < vector <unsigned> > hierarchical_coordinates;
		
		unsigned full_dim;
		bool hierarchical_coordinates_exanded;
		
	private:
		bool is_hierarchical_kernel_type(unsigned kernel_type) const;
		
		bool made_consistent;
		unsigned orginal_number_of_nodes;
};



//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/kernel/kernel_control.cpp"
#endif


#endif

