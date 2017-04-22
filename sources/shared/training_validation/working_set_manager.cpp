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


#if !defined (WORKING_SET_MANAGER_CPP)
	#define WORKING_SET_MANAGER_CPP
 
 
#include "sources/shared/training_validation/working_set_manager.h"



#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/random_subsets.h"
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_types/vector.h"

#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/binding_specifics.h"


#include <stack>

//**********************************************************************************************************************************


void Tvoronoi_by_tree_node::clear()
{
	unsigned i;
	
	cover.clear();
	cover_dataset.clear();
	radii.clear();
	cell_numbers.clear();
	
	for (i=0; i<child_partitions.size(); i++)
		delete child_partitions[i];
	child_partitions.clear();
}


//**********************************************************************************************************************************

void Tvoronoi_by_tree_node::read_from_file(FILE* fp)
{
	file_read(fp, cover);
	file_read(fp, cell_numbers);
	file_read(fp, radii);
}

//**********************************************************************************************************************************

void Tvoronoi_by_tree_node::write_to_file(FILE* fp) const
{
	file_write_eol(fp);
	file_write(fp, cover);
	file_write(fp, cell_numbers);
	file_write(fp, radii, "%3.15f ", "");
}


//**********************************************************************************************************************************

void Tvoronoi_by_tree_node::copy(Tvoronoi_by_tree_node* source_node)
{
	cover = source_node->cover;
	cover_dataset = source_node->cover_dataset;
	cover_dataset.enforce_ownership();
	radii = source_node->radii;
	cell_numbers = source_node->cell_numbers;
}


//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************



Tvoronoi_tree::Tvoronoi_tree()
{
}

//**********************************************************************************************************************************


Tvoronoi_tree::~Tvoronoi_tree()
{
	clear();
}

//**********************************************************************************************************************************


Tvoronoi_tree::Tvoronoi_tree(const Tvoronoi_tree& voronoi_tree)
{
	copy(voronoi_tree);
}

//**********************************************************************************************************************************

Tvoronoi_tree& Tvoronoi_tree::operator = (const Tvoronoi_tree& voronoi_tree)
{
	copy(voronoi_tree);
	return *this;
}

//**********************************************************************************************************************************


void Tvoronoi_tree::clear()
{
	Tvoronoi_by_tree_node* current_tree_node;
	
	current_tree_node = &root_node;
	clear_recursive(current_tree_node);
}


//**********************************************************************************************************************************

void Tvoronoi_tree::read_from_file(FILE* fp)
{
	unsigned c;
	unsigned number_of_childs;
	unsigned child_flag;
	Tvoronoi_by_tree_node* current_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;
	

	stack_of_nodes.push(&root_node);
	while (stack_of_nodes.empty() == false)
	{
		current_node = stack_of_nodes.top();
		stack_of_nodes.pop();
		
		file_read(fp, number_of_childs);
		current_node->child_partitions.resize(number_of_childs);
		
		for (c=0; c<current_node->child_partitions.size(); c++)
		{
			file_read(fp, child_flag);
			if (child_flag == 1)
			{
				current_node->child_partitions[c] = new Tvoronoi_by_tree_node;
				stack_of_nodes.push(current_node->child_partitions[c]);
			}
			else
				current_node->child_partitions[c] = NULL;
		}
		current_node->read_from_file(fp);
	}
}

//**********************************************************************************************************************************

void Tvoronoi_tree::write_to_file(FILE* fp) const
{
	unsigned c;
	Tvoronoi_by_tree_node root_node_copy;
	Tvoronoi_by_tree_node* current_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;
	

	root_node_copy = root_node;
	stack_of_nodes.push(&root_node_copy);
	while (stack_of_nodes.empty() == false)
	{
		current_node = stack_of_nodes.top();
		stack_of_nodes.pop();
		file_write(fp, unsigned(current_node->child_partitions.size()));
		for (c=0; c<current_node->child_partitions.size(); c++)
			if (current_node->child_partitions[c] != NULL)
			{
				stack_of_nodes.push(current_node->child_partitions[c]);
				file_write(fp, 1);
			}
			else
				file_write(fp, 0);
		current_node->write_to_file(fp);
	}
}



//**********************************************************************************************************************************

void Tvoronoi_tree::copy(const Tvoronoi_tree& source_tree)
{
	unsigned c;
	Tvoronoi_by_tree_node* current_node;
	Tvoronoi_by_tree_node* current_source_node;
	Tvoronoi_by_tree_node source_root_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;
	stack <Tvoronoi_by_tree_node*> stack_of_source_nodes;
	
	
	stack_of_nodes.push(&root_node);
	source_root_node = source_tree.root_node;
	stack_of_source_nodes.push(&source_root_node );
	
	while (stack_of_nodes.empty() == false)
	{
		current_node = stack_of_nodes.top();
		current_source_node = stack_of_source_nodes.top();
		stack_of_nodes.pop();
		stack_of_source_nodes.pop();
		
		current_node->child_partitions.resize(current_source_node->child_partitions.size());
		for (c=0; c<current_source_node->child_partitions.size(); c++)
			if (current_source_node->child_partitions[c] != NULL)
			{
				current_node->child_partitions[c] = new Tvoronoi_by_tree_node;
				stack_of_nodes.push(current_node->child_partitions[c]);
				stack_of_source_nodes.push(current_source_node->child_partitions[c]);
			}
			else
				current_node->child_partitions[c] = NULL;
		current_node->copy(current_source_node);
	}
}


//**********************************************************************************************************************************



void Tvoronoi_tree::clear_recursive(Tvoronoi_by_tree_node* current_tree_node)
{
	unsigned c;
	
	for (c=0; c<current_tree_node->child_partitions.size(); c++)
		if (current_tree_node->child_partitions[c] != NULL)
			clear_recursive(current_tree_node->child_partitions[c]);
	current_tree_node->clear();
}



//**********************************************************************************************************************************

unsigned Tvoronoi_tree::determine_cell_numbers()
{
	unsigned c;
	unsigned cell_number;
	Tvoronoi_by_tree_node* current_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;
	

	cell_number = 0;
	stack_of_nodes.push(&root_node);
	while (stack_of_nodes.empty() == false)
	{
		current_node = stack_of_nodes.top();
		stack_of_nodes.pop();
		current_node->cell_numbers.resize(current_node->child_partitions.size());
		for (c=0; c<current_node->child_partitions.size(); c++)
			if (current_node->child_partitions[c] != NULL)
			{
				stack_of_nodes.push(current_node->child_partitions[c]);
				current_node->cell_numbers[c] = -1;
			}
			else
			{
				current_node->cell_numbers[c] = cell_number;
				cell_number++;
			}
	}
	
	flush_info(INFO_1, "\nTree segementation results in %d cells.", cell_number);
	return cell_number;
}


//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************

Tworking_set_manager::Tworking_set_manager()
{
}

//**********************************************************************************************************************************

Tworking_set_manager::~Tworking_set_manager()
{
	clear();
}

//**********************************************************************************************************************************

Tworking_set_manager::Tworking_set_manager(const Tworking_set_manager& working_set_manager)
{
	copy(working_set_manager);
}

//**********************************************************************************************************************************

Tworking_set_manager::Tworking_set_manager(const Tworking_set_control& working_set_ctrl, const Tdataset& dataset)
{
	assign(working_set_ctrl, dataset);
}


//**********************************************************************************************************************************


Tworking_set_manager& Tworking_set_manager::operator = (const Tworking_set_manager& working_set_manager)
{
	copy(working_set_manager);
	return *this;
}

//**********************************************************************************************************************************

void Tworking_set_manager::copy(const Tworking_set_manager& working_set_manager)
{
	unsigned i;
	unsigned t;

	clear();

	partition = working_set_manager.partition;
	tree_based = working_set_manager.tree_based;
	
	partition_time = working_set_manager.partition_time;
	cell_assign_time = working_set_manager.cell_assign_time;
	
	dataset = working_set_manager.dataset;
	dataset.enforce_ownership();
	dataset_info = working_set_manager.dataset_info;
	working_set_control = working_set_manager.working_set_control;
		
	cover_datasets.resize(working_set_manager.cover_datasets.size());
	for (i=0; i<cover_datasets.size(); i++)
	{
		cover_datasets[i] = working_set_manager.cover_datasets[i];
		cover_datasets[i].enforce_ownership();
	}

	covers = working_set_manager.covers;
	ws_of_task = working_set_manager.ws_of_task;
	ws_of_task_and_cell = working_set_manager.ws_of_task_and_cell;

	radii = working_set_manager.radii;
	ws_numbers = working_set_manager.ws_numbers;
	
	voronoi_trees.resize(number_of_tasks());
	for (t=0; t<number_of_tasks(); t++)
		voronoi_trees[t] = working_set_manager.voronoi_trees[t];
	
		
	labels_of_tasks = working_set_manager.labels_of_tasks;
}

//**********************************************************************************************************************************

void Tworking_set_manager::write_to_file(FILE* fp) const
{
	unsigned t;
	
	working_set_control.write_to_file(fp);
	
	file_write(fp, partition);
	file_write(fp, tree_based);
	
	file_write(fp, ws_of_task);
	file_write(fp, ws_of_task_and_cell);
	file_write(fp, covers);
	
	file_write(fp, unsigned(radii.size()));
	for (t=0; t<radii.size(); t++)
		file_write(fp, radii[t], "%3.15f ", "");
	
	for (t=0; t<voronoi_trees.size(); t++)
		voronoi_trees[t].write_to_file(fp);
	
	if (working_set_control.classification == true)
		for (t=0; t<number_of_tasks(); t++)
			file_write(fp, labels_of_tasks[t]);
};

//**********************************************************************************************************************************


void Tworking_set_manager::read_from_file(FILE* fp, const Tdataset& dataset)
{
	unsigned t;
	unsigned number_of_radii;
	
	working_set_control.read_from_file(fp);
	
	file_read(fp, partition);
	file_read(fp, tree_based);
	
	file_read(fp, ws_of_task);
	file_read(fp, ws_of_task_and_cell);
	file_read(fp, covers);
	
	file_read(fp, number_of_radii);
	radii.resize(number_of_radii);
	for (t=0; t<number_of_radii; t++)
		file_read(fp, radii[t]);

	voronoi_trees.resize(number_of_tasks());
	for (t=0; t<voronoi_trees.size(); t++)
		voronoi_trees[t].read_from_file(fp);
	
	if (working_set_control.classification == true)
	{
		labels_of_tasks.resize(number_of_tasks());
		for (t=0; t<number_of_tasks(); t++)
			file_read(fp, labels_of_tasks[t]);
	}
	else
		labels_of_tasks.clear();
	
	load_dataset(dataset, true);
	compute_working_set_numbers();
};


//**********************************************************************************************************************************


void Tworking_set_manager::clear() 
{
	unsigned task;
	
	dataset.clear();
	
	ws_numbers.clear();
	ws_of_task.clear();
	ws_of_task_and_cell.clear();
	
	radii.clear();
	covers.clear();
	cover_datasets.clear();

	for (task=0; task < voronoi_trees.size(); task++)
		voronoi_trees[task].clear();
	voronoi_trees.clear();
	
	labels_of_tasks.clear();
	
	partition = false;
	tree_based = false;
	
	partition_time = 0.0;
	cell_assign_time = 0.0;
}



//**********************************************************************************************************************************


unsigned Tworking_set_manager::number_of_tasks() const
{
	return unsigned(ws_of_task_and_cell.size());
}


//**********************************************************************************************************************************


unsigned Tworking_set_manager::number_of_cells(unsigned task) const
{
	check_task(task);
	return unsigned(ws_of_task_and_cell[task].size());
}

#endif


//**********************************************************************************************************************************


unsigned Tworking_set_manager::total_number_of_working_sets() const
{
	return ws_numbers[number_of_tasks() - 1][number_of_cells(number_of_tasks() - 1) - 1] + 1;
}



//**********************************************************************************************************************************


vector <unsigned> Tworking_set_manager::cells_of_sample(Tsample* sample, unsigned task) const
{
	unsigned i;
	vector <unsigned> cells;
	vector <double> distances;

	check_task(task);
	
	switch (working_set_control.partition_method)
	{
		case RANDOM_CHUNK_BY_SIZE:
			for (i=0; i<number_of_cells(task); i++)
				cells.push_back(i);
			break;
			
		case RANDOM_CHUNK_BY_NUMBER:
			for (i=0; i<number_of_cells(task); i++)
				cells.push_back(i);
			break;
		
		case VORONOI_BY_RADIUS:
			cells.push_back(cover_datasets[task].get_index_of_closest_sample(sample));
			break;
			
		case VORONOI_BY_SIZE:
			cells.push_back(cover_datasets[task].get_index_of_closest_sample(sample));
			break;
			
		case OVERLAP_BY_SIZE:
			for (i=0; i<cover_datasets[task].size(); i++)
				if (squared_distance(cover_datasets[task].sample(i), sample) <= radii[task][i])
				{
					cells.push_back(i);
					distances.push_back(radii[task][i]);
				}
			if (cells.size() == 0)
			{
				cells.push_back(cover_datasets[task].get_index_of_closest_sample(sample));
				distances.push_back(1.0);
			}
			merge_sort_up(distances, cells);
			break;
		
		case VORONOI_TREE_BY_SIZE:
			cells.push_back(get_cell_from_tree(sample, task));
			break;
		default:
			cells.push_back(0);
	}	
	
	return cells;
}


//**********************************************************************************************************************************


void Tworking_set_manager::determine_cell_numbers_for_data_set(const Tdataset& data_set, vector <vector <vector <unsigned> > >& cell_numbers) const
{
	unsigned task;
	unsigned i;
	
	
	cell_numbers.clear();
	cell_numbers.resize(number_of_tasks());

	for (task=0; task<number_of_tasks(); task++)
		for (i=0; i<data_set.size(); i++)
			cell_numbers[task].push_back(cells_of_sample(data_set.sample(i), task));
}

//**********************************************************************************************************************************

Tsubset_info Tworking_set_manager::cover_of_task(unsigned task) const
{
	check_task(task);
	return covers[task];
}


//**********************************************************************************************************************************

Tsubset_info Tworking_set_manager::working_set_of_task(unsigned task) const
{
	check_task(task);
	return ws_of_task[task];
}


//**********************************************************************************************************************************


Tsubset_info Tworking_set_manager::working_set_of_cell(unsigned task, unsigned cell) const
{
	check_cell(task, cell);
	return ws_of_task_and_cell[task][cell];
}

//**********************************************************************************************************************************


unsigned Tworking_set_manager::size_of_working_set_of_cell(unsigned task, unsigned cell) const
{
	check_cell(task, cell);
	return ws_of_task_and_cell[task][cell].size();
}


//**********************************************************************************************************************************

unsigned Tworking_set_manager::working_set_number(unsigned task, unsigned cell) const
{
	check_cell(task, cell);
	return ws_numbers[task][cell];
}


//**********************************************************************************************************************************

unsigned Tworking_set_manager::average_working_set_size() const
{
	unsigned task;
	unsigned cell;
	unsigned total_size;
	
	total_size = 0;
	for (task=0; task<ws_of_task_and_cell.size(); task++)
		for (cell=0; cell<ws_of_task_and_cell[task].size(); cell++)
			total_size = total_size + unsigned(ws_of_task_and_cell[task][cell].size());
		
	return total_size / total_number_of_working_sets();
}

//**********************************************************************************************************************************


vector <double> Tworking_set_manager::get_squared_radii_of_task(unsigned task) const
{
	check_task(task);
	return radii[task];
}

//**********************************************************************************************************************************


double Tworking_set_manager::get_squared_radius_of_cell(unsigned task, unsigned cell) const
{
	check_cell(task, cell);
	
	if (radii[task].size() > 0)
		return radii[task][cell];
	else 
		return -1.0;
}

//**********************************************************************************************************************************



void Tworking_set_manager::check_working_set_method()
{
	if (working_set_control.classification == true)
	{
		if (dataset.is_classification_data() == false)
			flush_exit(ERROR_DATA_MISMATCH, "You have chosen a classification method but the data does not have integer labels.");

		if (dataset_info.label_list.size() > 2) 
		{
			if ((working_set_control.working_set_selection_method != MULTI_CLASS_ALL_VS_ALL) and (working_set_control.working_set_selection_method != MULTI_CLASS_ONE_VS_ALL))
			{
				working_set_control.working_set_selection_method = MULTI_CLASS_ALL_VS_ALL;
				flush_warn(WARN_ALL, "Changing to AvA since data set contains more than 2 labels and multiclass method has not been specified.\n");
			}
		}
		else if ((working_set_control.working_set_selection_method == MULTI_CLASS_ALL_VS_ALL) or (working_set_control.working_set_selection_method == MULTI_CLASS_ONE_VS_ALL))
		{
			working_set_control.working_set_selection_method = FULL_SET;
			flush_warn(WARN_ALL, "Changing to binary classification since dataset contains only 2 labels.\n");
		}
	}
}


//**********************************************************************************************************************************



void Tworking_set_manager::load_dataset(const Tdataset& dataset, bool build_cover)
{
	unsigned c;
	unsigned task;
	Tvoronoi_by_tree_node* current_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;
	
	
	if (dataset.size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to load an empty dataset into a working_set_manager.");
	
	dataset_info = Tdataset_info(dataset, true);
	Tworking_set_manager::dataset = dataset;

	if (build_cover == true)
	{
		if (tree_based == false)
		{
			cover_datasets.resize(number_of_tasks());
			for (task=0; task<number_of_tasks(); task++)
			{
				Tworking_set_manager::dataset.create_subset(cover_datasets[task], covers[task], true);
				cover_datasets[task].enforce_ownership();
			}
		}
		else
		{
			for (task=0; task<number_of_tasks(); task++)
			{
				stack_of_nodes.push(&(voronoi_trees[task].root_node));
				while (stack_of_nodes.empty() == false)
				{
					current_node = stack_of_nodes.top();
					stack_of_nodes.pop();
					
					for (c=0; c<current_node->child_partitions.size(); c++)
						if (current_node->child_partitions[c] != NULL)
							stack_of_nodes.push(current_node->child_partitions[c]);
					Tworking_set_manager::dataset.create_subset(current_node->cover_dataset, current_node->cover, true);
				}
			}
		}
	}
}


//**********************************************************************************************************************************



void Tworking_set_manager::assign(const Tworking_set_control& working_set_ctrl, const Tdataset& dataset)
{
	unsigned i;
	unsigned j;
	Tsubset_info subset_info1;
	Tsubset_info subset_info2;


	clear();
	load_dataset(dataset, false);
	working_set_control = working_set_ctrl;
	working_set_control.number_of_tasks = 0;
	check_working_set_method();

	switch (working_set_control.working_set_selection_method)
	{
		case FULL_SET:
			push_back(id_permutation(dataset_info.size));
			break;
			
		case MULTI_CLASS_ALL_VS_ALL:
			for (i=0; i<dataset_info.label_list.size(); i++)
				for (j=i+1; j<dataset_info.label_list.size(); j++)
				{
					subset_info1 = dataset.create_subset_info_with_label(dataset_info.label_list[i]);
					subset_info2 = dataset.create_subset_info_with_label(dataset_info.label_list[j]);
					subset_info1.insert(subset_info1.end(), subset_info2.begin(), subset_info2.end());
					push_back(subset_info1);
				}
			break;

		case MULTI_CLASS_ONE_VS_ALL:
			for (i=0; i<dataset_info.label_list.size(); i++)
			{
				subset_info1 = dataset.create_subset_info_with_label(dataset_info.label_list[i]);
				for (j=0; j<dataset_info.label_list.size(); j++)
					if (i != j)
					{
						subset_info2 = dataset.create_subset_info_with_label(dataset_info.label_list[j]);
						subset_info1.insert(subset_info1.end(), subset_info2.begin(), subset_info2.end());
					}
				push_back(subset_info1);
			}
			break;
			
		case BOOT_STRAP:
			for (i=0; i<working_set_ctrl.number_of_tasks; i++)
				push_back(random_multiset(id_permutation(dataset.size()), working_set_control.size_of_tasks, working_set_control.random_seed, i));
			break;
	}
}




//**********************************************************************************************************************************


void Tworking_set_manager::push_back(const Tsubset_info& new_task_subset_info)
{
	unsigned i;
	unsigned task;
	Tdataset working_set;
	Tdataset_info dataset_info;


	if (dataset.size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Working set manager cannot push a new working set at its back without\n       having loaded a data set.");

	task = ws_of_task_and_cell.size();
	ws_of_task.push_back(new_task_subset_info);
	working_set_control.number_of_tasks = unsigned(ws_of_task_and_cell.size()) + 1;
	ws_of_task_and_cell.resize(working_set_control.number_of_tasks);

	radii.resize(working_set_control.number_of_tasks);
	covers.resize(working_set_control.number_of_tasks);
	cover_datasets_resize(working_set_control.number_of_tasks);
	voronoi_trees.resize(working_set_control.number_of_tasks);
	

	for (i=0; i<cover_datasets.size(); i++)
		cover_datasets[i].enforce_ownership();

	dataset.create_subset(working_set, new_task_subset_info, true);
	
	if (working_set_control.classification == true)
	{
		labels_of_tasks.resize(working_set_control.number_of_tasks);
		dataset_info = Tdataset_info(working_set, true);
		
		if (dataset_info.label_list.size() == 0)
			flush_exit(ERROR_DATA_STRUCTURE, "Working set manager cannot push a new working set for classification\nat its back that has no labels.");
		
		if (dataset_info.label_list.size() == 1)
			dataset_info.label_list.push_back(dataset_info.label_list[0]);
		
		labels_of_tasks[task] = dataset_info.label_list;
	}
	else 
		labels_of_tasks.clear();

	assign_cell(working_set, working_set_control.number_of_tasks - 1);
	
	compute_working_set_numbers();
}


//**********************************************************************************************************************************


void Tworking_set_manager::cover_datasets_resize(unsigned new_size)
{
	unsigned i;
	unsigned old_size;
	vector <Tdataset> cover_datasets_tmp;
	
	
	old_size = cover_datasets.size();
	cover_datasets_tmp.resize(old_size);
	for (i=0; i<old_size; i++)
	{
		cover_datasets_tmp[i] = cover_datasets[i];
		cover_datasets_tmp[i].enforce_ownership();
	}

	cover_datasets.clear();
	cover_datasets.resize(new_size);
	for (i=0; i<old_size; i++)
	{
		cover_datasets[i] = cover_datasets_tmp[i];
		cover_datasets[i].enforce_ownership();
	}
}




//**********************************************************************************************************************************



void Tworking_set_manager::push_back(const Tworking_set_manager& working_set_manager)
{
	unsigned i;
	unsigned old_size;
	

	if (number_of_tasks() == 0)
		copy(working_set_manager);
	else
	{
		if (not(dataset == working_set_manager.dataset))
			flush_exit(ERROR_DATA_MISMATCH, "Trying to combine two working set managers that have different data sets.");

		
		// Push cover_dataset of working_set_manager object to current object.
		// Since calling <vector> functions creates temporary objects of type Tdataset
		// that loose ownership, the copying mechanism needs to be implemented 'by hand'. 
		
		old_size = cover_datasets.size();
		cover_datasets_resize(old_size + working_set_manager.cover_datasets.size());
		for (i=0; i<working_set_manager.cover_datasets.size(); i++)
		{
			cover_datasets[old_size + i] = working_set_manager.cover_datasets[i];
			cover_datasets[old_size + i].enforce_ownership();
		}
		
		
		// Now we can push back the rest of the data structures.
		
		covers.insert(covers.end(), working_set_manager.covers.begin(), working_set_manager.covers.end());

		ws_of_task.insert(ws_of_task.end(), working_set_manager.ws_of_task.begin(), working_set_manager.ws_of_task.end());
		ws_of_task_and_cell.insert(ws_of_task_and_cell.end(), working_set_manager.ws_of_task_and_cell.begin(), working_set_manager.ws_of_task_and_cell.end());

		radii.insert(radii.end(), working_set_manager.radii.begin(), working_set_manager.radii.end());
		
		voronoi_trees.insert(voronoi_trees.end(), working_set_manager.voronoi_trees.begin(), working_set_manager.voronoi_trees.end());
		
		if (working_set_control.classification == true)
			labels_of_tasks.insert(labels_of_tasks.end(), working_set_manager.labels_of_tasks.begin(), working_set_manager.labels_of_tasks.end());
		else
			labels_of_tasks.clear();
		
		compute_working_set_numbers();
	}
}



//**********************************************************************************************************************************



void Tworking_set_manager::assign_cell(Tdataset working_set, unsigned task)
{
	unsigned i;
	unsigned j;
	unsigned number_of_cells;
	Tdataset empty_data_set;
	Tdataset working_subset;
	Tsubset_info subset_info;
	vector <double> radii_tmp;
	Tsubset_info covers_tmp;
	vector <unsigned> cell_affiliation;
	bool assigned_to_cell;
	unsigned closest_cell;
	double distance;
	double closest_distance_to_cell;
	double reduction_fraction;
	double reduced_NNs;
	
	flush_info(INFO_1, "\nAssigning samples to cells for task %d.", task);
	
	check_task(task);
	ws_of_task_and_cell[task].clear();
	switch (working_set_control.partition_method)
	{
		case NO_PARTITION:
			partition_time = get_wall_time_difference(partition_time);
			covers[task].push_back(0);
			cover_datasets[task] = empty_data_set;
			partition_time = get_wall_time_difference(partition_time);
			
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			ws_of_task_and_cell[task].push_back(ws_of_task[task]);
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = true;
			tree_based = false;
			break;
			
		case RANDOM_CHUNK_BY_SIZE:
			partition_time = get_wall_time_difference(partition_time);
			covers[task].push_back(0);
			cover_datasets[task] = empty_data_set;
			working_set_control.number_of_cells = unsigned(double(working_set.size())/double(working_set_control.size_of_cells)) + 1;
			partition_time = get_wall_time_difference(partition_time);
			
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			cell_affiliation = create_random_chunk_affiliation(working_set.size(), working_set_control.number_of_cells);
			assign_from_cell_affiliation(cell_affiliation, task, working_set_control.number_of_cells);
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = true;
			tree_based = false;
			break;

		case RANDOM_CHUNK_BY_NUMBER:
			partition_time = get_wall_time_difference(partition_time);
			covers[task].push_back(0);
			cover_datasets[task] = empty_data_set;
			partition_time = get_wall_time_difference(partition_time);
			
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			cell_affiliation = create_random_chunk_affiliation(working_set.size(), working_set_control.number_of_cells);
			assign_from_cell_affiliation(cell_affiliation, task, working_set_control.number_of_cells);
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = true;
			tree_based = false;
			break;

		case VORONOI_BY_RADIUS:
			partition_time = get_wall_time_difference(partition_time);
			covers[task] = working_set.create_cover_subset_info_by_radius(working_set_control.radius, working_set_control.random_seed, working_set_control.size_of_dataset_to_find_partition);
			working_set.create_subset(cover_datasets[task], covers[task], true);

			for (i=0; i<covers[task].size(); i++)
				covers[task][i] = ws_of_task[task][covers[task][i]]; 
			partition_time = get_wall_time_difference(partition_time);

			cell_assign_time = get_wall_time_difference(cell_assign_time);
			cell_affiliation = create_voronoi_subset_affiliation(working_set, cover_datasets[task]);
			assign_from_cell_affiliation(cell_affiliation, task, unsigned(covers[task].size()));
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = true;
			tree_based = false;
			break;
			
		case VORONOI_BY_SIZE:
			partition_time = get_wall_time_difference(partition_time);
			covers[task] = working_set.create_cover_subset_info_by_kNN(working_set_control.size_of_cells, working_set_control.random_seed, working_set_control.reduce_covers, radii[task], working_set_control.size_of_dataset_to_find_partition);
			working_set.create_subset(cover_datasets[task], covers[task], true);

			for (i=0; i<covers[task].size(); i++)
				covers[task][i] = ws_of_task[task][covers[task][i]]; 
			partition_time = get_wall_time_difference(partition_time);
			
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			cell_affiliation = create_voronoi_subset_affiliation(working_set, cover_datasets[task]);
			assign_from_cell_affiliation(cell_affiliation, task, unsigned(covers[task].size()));
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = true;
			tree_based = false;
			break;
			
		case OVERLAP_BY_SIZE:
			partition_time = get_wall_time_difference(partition_time);
			radii[task].clear();
			covers[task].clear();
			
			for (i=0; i<working_set_control.number_of_covers; i++)
			{
				covers_tmp = working_set.create_region_subset_info(working_set_control.size_of_cells, working_set_control.random_seed+i, working_set_control.max_ignore_factor, radii_tmp, working_set_control.size_of_dataset_to_find_partition);
				
				radii[task].insert(radii[task].end(), radii_tmp.begin(), radii_tmp.end());
				covers[task].insert(covers[task].end(), covers_tmp.begin(), covers_tmp.end());
			}

			working_set.create_subset(cover_datasets[task], covers[task], true);

			ws_of_task_and_cell[task].clear();
			ws_of_task_and_cell[task].resize(unsigned(covers[task].size()));
			for (i=0; i<covers[task].size(); i++)
				covers[task][i] = ws_of_task[task][covers[task][i]]; 
			partition_time = get_wall_time_difference(partition_time);
			
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			for (j=0; j<working_set.size(); j++)
			{
				assigned_to_cell = false;
				closest_cell = 0;
				closest_distance_to_cell = numeric_limits<double>::max();

				for (i=0; i<covers[task].size(); i++)
				{
					distance = squared_distance(cover_datasets[task].sample(i), working_set.sample(j));
					if (distance <= radii[task][i])
					{
						assigned_to_cell = true;
						ws_of_task_and_cell[task][i].push_back(ws_of_task[task][j]);
					}
					if (distance < closest_distance_to_cell)
					{
						closest_cell = i;
						closest_distance_to_cell = distance;
					}
					if (j % 10000 == 0)
						check_for_user_interrupt();
				}
				if (assigned_to_cell == false)
					ws_of_task_and_cell[task][closest_cell].push_back(ws_of_task[task][j]);
			}
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = false;
			tree_based = false;
			break;
			
		case VORONOI_TREE_BY_SIZE:
			partition_time = get_wall_time_difference(partition_time);
			voronoi_trees[task].clear();
			
			subset_info = id_permutation(working_set.size());
			subset_info = random_subset(subset_info, min(working_set.size(), working_set_control.size_of_dataset_to_find_partition), working_set_control.random_seed+task);
			sort_up(subset_info);
			
			reduction_fraction = double(subset_info.size()) / double(working_set.size());
			reduced_NNs = unsigned(reduction_fraction * double(working_set_control.size_of_cells));
			
			working_set.create_subset(working_subset, subset_info);

			for (i=0; i<subset_info.size(); i++)
				subset_info[i] = ws_of_task[task][subset_info[i]];

			assign_cell_recursive(working_subset, subset_info, &voronoi_trees[task].root_node, 1, reduced_NNs);

			number_of_cells = voronoi_trees[task].determine_cell_numbers();
			determine_radii_from_tree(task);
			partition_time = get_wall_time_difference(partition_time);
			
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			cell_affiliation = create_voronoi_tree_affiliation(working_set, task);
			assign_from_cell_affiliation(cell_affiliation, task, number_of_cells);
			cell_assign_time = get_wall_time_difference(cell_assign_time);
			
			partition = true;
			tree_based = true;
			break;
	}
}



//**********************************************************************************************************************************

void Tworking_set_manager::assign_cell_recursive(const Tdataset& current_working_set, Tsubset_info current_subset_info, Tvoronoi_by_tree_node* current_tree_node, unsigned current_depth, unsigned NNs)
{
	unsigned c;
	unsigned i;
	unsigned current_target_cell_size;
	vector <unsigned> cell_affiliation;
	Tdataset new_working_subset;
	Tsubset_info new_subset_info;


	if ((working_set_control.max_tree_depth == 0) or (current_depth < working_set_control.max_tree_depth))
		current_target_cell_size = double(current_working_set.size()) / working_set_control.tree_reduction_factor;
	else
		current_target_cell_size = NNs;
	
	if ((working_set_control.max_theoretical_node_width > 0) and (current_working_set.size() < working_set_control.max_theoretical_node_width * NNs))
		current_target_cell_size = NNs;

	flush_info(INFO_2, "\nSplitting a working set of size %d at a node of depth %d.", current_working_set.size(), current_depth);

// 	deactivate_display();
	current_tree_node->cover = current_working_set.create_cover_subset_info_by_kNN(current_target_cell_size, working_set_control.random_seed, working_set_control.reduce_covers, current_tree_node->radii, working_set_control.size_of_dataset_to_find_partition);
// 	reactivate_display();
	
	current_working_set.create_subset(current_tree_node->cover_dataset, current_tree_node->cover, true);
	
	for (c=0; c<current_tree_node->cover.size(); c++)
		current_tree_node->cover[c] = current_subset_info[current_tree_node->cover[c]]; 

	cell_affiliation = create_voronoi_subset_affiliation(current_working_set, current_tree_node->cover_dataset);
	
	current_tree_node->child_partitions.resize(current_tree_node->cover.size());
	for (c=0; c<current_tree_node->cover.size(); c++)
	{
		new_working_subset.clear();
		new_subset_info.clear();
		for (i=0; i<cell_affiliation.size(); i++)
			if (cell_affiliation[i] == c)
			{
				new_working_subset.push_back(current_working_set.sample(i));
				new_subset_info.push_back(current_subset_info[i]);
			}

		if (new_working_subset.size() > NNs)
		{
			current_tree_node->child_partitions[c] = new Tvoronoi_by_tree_node;
			assign_cell_recursive(new_working_subset, new_subset_info, current_tree_node->child_partitions[c], current_depth + 1, NNs);
		}
		else
			current_tree_node->child_partitions[c] = NULL;
	}
}


//**********************************************************************************************************************************

vector <unsigned> Tworking_set_manager::create_voronoi_tree_affiliation(const Tdataset& working_set, unsigned task)
{
	unsigned i;
	vector <unsigned> cell_affiliation;


	cell_affiliation.resize(working_set.size());
	for (i=0; i<working_set.size(); i++)
	{
		if (i % 10000 == 0)
			check_for_user_interrupt();

		cell_affiliation[i] = get_cell_from_tree(working_set.sample(i), task);
	}
		
	return cell_affiliation;
}


//**********************************************************************************************************************************

unsigned Tworking_set_manager::get_cell_from_tree(Tsample* sample, unsigned task) const
{
	unsigned c;
	unsigned cell;
	bool cell_found;
	Tvoronoi_by_tree_node root_node;
	Tvoronoi_by_tree_node* current_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;

	
	cell = 0;
	cell_found = false;
	root_node = voronoi_trees[task].root_node;
	stack_of_nodes.push(&(root_node));

	while ((stack_of_nodes.empty() == false) and (cell_found == false))
	{
		current_node = stack_of_nodes.top();
		stack_of_nodes.pop();
		
		c = current_node->cover_dataset.get_index_of_closest_sample(sample);
		if (current_node->child_partitions[c] != NULL)
			stack_of_nodes.push(current_node->child_partitions[c]);
		else
		{
			cell_found = true;
			cell = current_node->cell_numbers[c];
		}
	}

	return cell;
}


//**********************************************************************************************************************************



void Tworking_set_manager::determine_radii_from_tree(unsigned task)
{
	unsigned c;
	Tvoronoi_by_tree_node* current_node;
	stack <Tvoronoi_by_tree_node*> stack_of_nodes;

	
	radii[task].clear();
	stack_of_nodes.push(&(voronoi_trees[task].root_node));
	while (stack_of_nodes.empty() == false)
	{
		current_node = stack_of_nodes.top();
		stack_of_nodes.pop();
		
		for (c=0; c<current_node->child_partitions.size(); c++)
			if (current_node->child_partitions[c] != NULL)
				stack_of_nodes.push(current_node->child_partitions[c]);
			else
				radii[task].push_back(current_node->radii[c]);
	}
}

//**********************************************************************************************************************************



void Tworking_set_manager::assign_from_cell_affiliation(vector <unsigned> cell_affiliation, unsigned task, unsigned number_of_cells)
{
	unsigned i;


	ws_of_task_and_cell[task].clear();
	ws_of_task_and_cell[task].resize(number_of_cells);
	
	if (cell_affiliation.size() != ws_of_task[task].size())
		flush_exit(ERROR_DATA_STRUCTURE, "Cell affiliation size %d does not match ws size %d of task %d.", cell_affiliation.size(), ws_of_task[task].size(), task);

	for (i=0; i<cell_affiliation.size(); i++)
		ws_of_task_and_cell[task][cell_affiliation[i]].push_back(ws_of_task[task][i]);
}



//**********************************************************************************************************************************



void Tworking_set_manager::compute_working_set_numbers()
{
	unsigned i;
	unsigned task;
	unsigned cell;
	
	
	i = 0;
	ws_numbers.clear();
	ws_numbers.resize(number_of_tasks());
	
	for (task=0; task<number_of_tasks(); task++)
		for (cell=0; cell<number_of_cells(task); cell++)
		{
			ws_numbers[task].push_back(i);
			i++;
		}
}



//**********************************************************************************************************************************

vector <unsigned> Tworking_set_manager::create_random_chunk_affiliation(unsigned working_set_size, unsigned number_of_chunks) const
{
	unsigned i;
	vector <unsigned> permutation;
	vector <unsigned> cell_affiliation;

	permutation = random_permutation(working_set_size, working_set_control.random_seed);
	cell_affiliation.resize(working_set_size);

	for (i=0; i<working_set_size; i++)
		cell_affiliation[permutation[i]] = (i % number_of_chunks);

	return cell_affiliation;
}


//**********************************************************************************************************************************


vector <unsigned> Tworking_set_manager::create_voronoi_subset_affiliation(const Tdataset& working_set, const Tdataset& cover_dataset) 
{
	unsigned i;
	vector <unsigned> cell_affiliation;

	cell_affiliation.resize(working_set.size());
	for (i=0; i<working_set.size(); i++)
	{
		if (i % 10000 == 0)
			check_for_user_interrupt();
		cell_affiliation[i] = cover_dataset.get_index_of_closest_sample(working_set.sample(i));
	}
		
	return cell_affiliation;
}




//**********************************************************************************************************************************


void Tworking_set_manager::build_working_set(Tdataset& working_set, unsigned task, unsigned cell)
{
	unsigned i;

	check_task(task);
	check_cell(task, cell);
	
	working_set.clear();
	working_set.enforce_ownership();

	for (i=0; i<ws_of_task_and_cell[task][cell].size(); i++)
		working_set.push_back(dataset.sample(ws_of_task_and_cell[task][cell][i]));

	if (working_set_control.classification == true)
		change_label_for_classification(working_set, task);
}



//**********************************************************************************************************************************


void Tworking_set_manager::build_working_set(Tdataset& working_set, const Tdataset& data_set, unsigned task, unsigned cell)
{
	unsigned i;
	unsigned c;
	vector <vector <vector <unsigned> > > cell_numbers;
	
	
	working_set.clear();
	working_set.enforce_ownership();
	
	determine_cell_numbers_for_data_set(data_set, cell_numbers);

	for (i=0; i<data_set.size(); i++)
		for (c=0; c<cell_numbers[task][i].size(); c++)
			if (cell_numbers[task][i][c] == cell)
				working_set.push_back(data_set.sample(i));
			
	if (working_set_control.classification == true)
		change_label_for_classification(working_set, task);
}

//**********************************************************************************************************************************


void Tworking_set_manager::check_task(unsigned task) const
{
	if (task >= ws_of_task_and_cell.size())
		flush_exit(ERROR_DATA_STRUCTURE, "Tried to access task %d in a working_set_manager that only has %d tasks.", task, ws_of_task_and_cell.size());
}

//**********************************************************************************************************************************


void Tworking_set_manager::check_cell(unsigned task, unsigned cell) const
{
	check_task(task);
	if (cell >= ws_of_task_and_cell[task].size())
		flush_exit(ERROR_DATA_STRUCTURE, "Tried to access cell %d of task %d in a working_set_manager that only has %d cells.", cell, task, ws_of_task_and_cell[task].size());
}



//**********************************************************************************************************************************


vector <int> Tworking_set_manager::get_labels_of_task(unsigned task) const
{
	check_task(task);
	return labels_of_tasks[task];
}

//**********************************************************************************************************************************


void Tworking_set_manager::change_label_for_classification(Tdataset& working_set, unsigned task)
{
	unsigned i;
	vector <int> label_list;
	

	check_task(task);
	label_list = get_labels_of_task(task);

	switch (working_set_control.working_set_selection_method)
	{
		case MULTI_CLASS_ALL_VS_ALL:
			working_set.store_original_labels();
			flush_info(INFO_1, "\nChanging labels %d and %d to -1 and 1 for multi-class type AvA.", label_list[0], label_list[1]);
			for (i=0; i<working_set.size(); i++)
				if (working_set.sample(i)->label == double(label_list[0]))
					working_set.set_label_of_sample(i, -1.0);
				else
					working_set.set_label_of_sample(i, 1.0);
		break;
		
		case MULTI_CLASS_ONE_VS_ALL:
			flush_info(INFO_1, "\nChanging label %d and %d other labels to 1 and -1 for multi-class type OvA.", label_list[task], label_list.size()-1);
			for (i=0; i<working_set.size(); i++)
				if (working_set.sample(i)->label == double(label_list[task]))
					working_set.set_label_of_sample(i, 1.0);
				else
					working_set.set_label_of_sample(i, -1.0);
			working_set.store_original_labels();
		break;
		
		default:
			working_set.store_original_labels();
						
			if ((label_list[0] != -1.0) or (label_list[1] != 1.0))
			{
				flush_info(INFO_1, "\nChanging labels %d and %d to -1 and 1 for binary classification.", label_list[0], label_list[1]);
				for (i=0; i<working_set.size(); i++)
					if (working_set.sample(i)->label == double(label_list[0]))
						working_set.set_label_of_sample(i, -1.0);
					else
						working_set.set_label_of_sample(i, 1.0);
			}
		break;
	}
}


//**********************************************************************************************************************************


bool Tworking_set_manager::cells_are_partition() const
{
	return partition;
}




