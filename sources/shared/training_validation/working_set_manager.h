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


#if !defined (WORKING_SET_MANAGER_H)
	#define WORKING_SET_MANAGER_H


 

#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/basic_types/dataset_info.h"
#include "sources/shared/training_validation/working_set_control.h"
 
#include <cstdio>
#include <vector>
using namespace std;



//**********************************************************************************************************************************


class Tvoronoi_by_tree_node
{
	public:
		void clear();
		void read_from_file(FILE* fp);
		void write_to_file(FILE* fp) const;
		void copy(Tvoronoi_by_tree_node* source_node);

		
		Tsubset_info cover;
		Tdataset cover_dataset;
		vector <double> radii;
		vector <int> cell_numbers;
		vector <Tvoronoi_by_tree_node*> child_partitions;
};


//**********************************************************************************************************************************


class Tvoronoi_tree
{
	public:
		Tvoronoi_tree();
		~Tvoronoi_tree();
		Tvoronoi_tree(const Tvoronoi_tree& voronoi_tree);
		
		void clear();
		void read_from_file(FILE* fp);
		void write_to_file(FILE* fp) const;
		Tvoronoi_tree& operator = (const Tvoronoi_tree& voronoi_tree);
		
		unsigned determine_cell_numbers();

		
		Tvoronoi_by_tree_node root_node;
		
	private:
		void copy(const Tvoronoi_tree& source_tree);
		void clear_recursive(Tvoronoi_by_tree_node* current_tree_node);
};


//**********************************************************************************************************************************


class Tworking_set_manager
{
	public:
		Tworking_set_manager();
		~Tworking_set_manager();
		Tworking_set_manager(const Tworking_set_manager& working_set_manager);
		Tworking_set_manager(const Tworking_set_control& working_set_ctrl, const Tdataset& dataset);
		
		void clear();
		void write_to_file(FILE* fp) const;
		void read_from_file(FILE* fp, const Tdataset& dataset);
		void push_back(const Tworking_set_manager& working_set_manager);
		Tworking_set_manager& operator = (const Tworking_set_manager& working_set_manager);
		
		void build_working_set(Tdataset& working_set, unsigned task, unsigned cell);
		void build_working_set(Tdataset& working_set, const Tdataset& data_set, unsigned task, unsigned cell);

		unsigned number_of_tasks() const;
		unsigned number_of_cells(unsigned task) const;
		vector <unsigned> cells_of_sample(Tsample* sample, unsigned task) const;
		void determine_cell_numbers_for_data_set(const Tdataset& data_set, vector <vector <vector <unsigned> > >& cell_numbers) const;
		
		unsigned total_number_of_working_sets() const;
		unsigned working_set_number(unsigned task, unsigned cell) const;
		unsigned average_working_set_size() const;
		bool cells_are_partition() const;
		vector <double> get_squared_radii_of_task(unsigned task) const;
		double get_squared_radius_of_cell(unsigned task, unsigned cell) const;
		
		Tsubset_info cover_of_task(unsigned task) const;
		Tsubset_info working_set_of_task(unsigned task) const;
		Tsubset_info working_set_of_cell(unsigned task, unsigned cell) const;
		unsigned size_of_working_set_of_cell(unsigned task, unsigned cell) const;
		vector <int> get_labels_of_task(unsigned task) const;
		
		inline Tworking_set_control get_working_set_control() const;
		inline void get_timings(double& partition_time, double& cell_assign_time) const;
		
		
	private:
		void copy(const Tworking_set_manager& working_set_manager);
		void check_working_set_method();
		void compute_working_set_numbers();
		void assign_cell(Tdataset working_set, unsigned task);
		void load_dataset(const Tdataset& dataset, bool build_cover);
		
		void push_back(const Tsubset_info& new_task_subset_info);
		void assign(const Tworking_set_control& working_set_ctrl, const Tdataset& dataset);
		
		void change_label_for_classification(Tdataset& working_set, unsigned task);
		
		void check_task(unsigned task) const;
		void check_cell(unsigned task, unsigned cell) const;

		vector <unsigned> create_random_chunk_affiliation(unsigned working_set_size, unsigned number_of_chunks) const;
		vector <unsigned> create_voronoi_subset_affiliation(const Tdataset& working_set, const Tdataset& cover_dataset);
		vector <unsigned> create_voronoi_tree_affiliation(const Tdataset& working_set, unsigned task);
		
		unsigned get_cell_from_tree(Tsample* sample, unsigned task) const;
		void determine_radii_from_tree(unsigned task);
		void assign_cell_recursive(const Tdataset& current_working_set, Tsubset_info current_subset_info, Tvoronoi_by_tree_node* current_tree_node, unsigned current_depth, unsigned NNs);

		
		void assign_from_cell_affiliation(vector <unsigned> cell_affiliation, unsigned task, unsigned number_of_cells);
		void cover_datasets_resize(unsigned new_size);
		

		
		bool partition;
		bool tree_based;

		double partition_time;
		double cell_assign_time;
		
		Tdataset dataset;
		Tdataset_info dataset_info;
		Tworking_set_control working_set_control;
		
		vector <Tsubset_info> covers;
		vector <Tdataset> cover_datasets;
		
		vector < vector <double> > radii;
		
		vector <Tsubset_info> ws_of_task;
		vector < vector <unsigned> > ws_numbers;
		vector < vector <Tsubset_info> > ws_of_task_and_cell;
		
		vector <Tvoronoi_tree> voronoi_trees;
		
		vector <vector <int> > labels_of_tasks;
};


//**********************************************************************************************************************************


#include "sources/shared/training_validation/working_set_manager.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/working_set_manager.cpp"
#endif

#endif
