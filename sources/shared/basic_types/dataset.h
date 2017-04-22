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


#if !defined (DATASET_H)
	#define DATASET_H

  
   
#include "sources/shared/basic_types/sample.h"


#include <cstdio>
#include <vector>
#include <string>
#include <utility>
#include <limits>
using namespace std;


//**********************************************************************************************************************************

typedef vector <unsigned> Tsubset_info;

Tsubset_info compose(Tsubset_info info1, Tsubset_info info2);


//**********************************************************************************************************************************

enum SCALING_TYPES {VARIANCE, QUANTILE, SCALING_TYPES_MAX};


//**********************************************************************************************************************************

class Tdataset
{
	public:
		Tdataset();
		Tdataset(const Tdataset& dataset);
		Tdataset(const double* data_array, unsigned size, unsigned dim, const double* labels, bool array_transposed = false);
		~Tdataset();

		void clear();
		void push_back(const Tsample& sample);
		void push_back(Tsample* sample);
		void push_back(const Tdataset& dataset);
		
		void read_from_file(string filename);
		void read_from_file(FILE* fpread, unsigned filetype, unsigned size, unsigned dim);
		void write_to_file(string filename) const;
		void write_to_file(FILE* fpwrite, unsigned filetype) const;
		
		void enforce_ownership();
		inline bool has_ownership() const;
		
		inline Tsample* sample(unsigned index) const;

		inline unsigned dim() const;
		inline unsigned size() const;

		bool is_unsupervised_data() const;
		bool is_classification_data() const;

		vector <double> get_labels() const;
		void change_labels(double old_label, double new_label);
		void set_label_of_sample(unsigned index, double new_label); 
		
		void store_original_labels();
		pair <pair <double, double>, double> get_original_labels() const;
		
		
		void compute_scaling(vector <double>& scaling, vector <double>& translate, double tau, unsigned type, bool uniform_scaling, bool scale_to_01) const;
		void apply_scaling(const vector <double>& scaling, const vector <double>& translate);
		
		Tsubset_info create_subset_info_with_label(double label) const;
		void create_subset(Tdataset& data_subset, Tsubset_info subset_info, bool give_ownership = false) const;

		template <typename float_type> float_type* upload_to_GPU(unsigned start_index, unsigned end_index) const;
		unsigned required_memory_on_GPU(unsigned start_index, unsigned end_index) const;
		double* convert_to_array(unsigned start_index, unsigned end_index) const;
		
		Tdataset& operator = (const Tdataset& dataset);
		bool operator == (const Tdataset& dataset) const;
		
		unsigned get_index_of_closest_sample(Tsample* new_sample) const;
		Tsubset_info create_cover_subset_info_by_radius(double radius, int random_seed, unsigned subset_size = 0) const; 
		Tsubset_info create_cover_subset_info_by_kNN(unsigned NNs, int random_seed, bool reduce, vector <double>& radii, unsigned subset_size = 0) const; 
		Tsubset_info create_region_subset_info(unsigned NNs, int random_seed, double max_ignore_factor, vector <double>& radii, unsigned subset_size = 0) const; 

		
		double get_approximate_radius() const;
		
		void sort_by_number();
		void group_spatially(unsigned chunk_size_aligned, unsigned chunks, vector <unsigned>& permutation);
		vector <unsigned> get_sample_numbers() const;
		void check_whether_complete_and_ordered() const;
		

	private:
		friend class Tdataset_info;
		void copy(const Tdataset& dataset);
		inline void check_index(unsigned index) const;
		
		template <typename float_type> float_type* convert_to_GPU_format(unsigned start_index, unsigned end_index) const;
		
		vector <Tsample*> sample_list;
		
		bool owns_samples;
		unsigned data_size;
		
		double original_label1;
		double original_label2;
		double original_most_frequent_label;
};


//**********************************************************************************************************************************



#include "sources/shared/basic_types/dataset.ins.cpp"


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/dataset.cpp"
#endif


#endif
