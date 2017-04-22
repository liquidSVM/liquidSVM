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


#if !defined (FOLD_MANAGER_CPP)
	#define FOLD_MANAGER_CPP


#include "sources/shared/training_validation/fold_manager.h"

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/random_subsets.h"

#include "sources/shared/basic_types/dataset_info.h"

#include <cmath>


//**********************************************************************************************************************************


Tfold_manager::Tfold_manager()
{
	dataset.enforce_ownership();
}

//**********************************************************************************************************************************


Tfold_manager::Tfold_manager(const Tfold_manager& fold_manager)
{
	copy(fold_manager);
}

//**********************************************************************************************************************************


Tfold_manager::Tfold_manager(Tfold_control fold_control, const Tdataset& dataset)
{ 
	load_dataset(dataset);
	
	if (fold_control.train_fraction == FRACTION_NOT_ASSIGNED)
		fold_control.train_fraction = 1.0;
	Tfold_manager::fold_control = fold_control;
	
	switch (fold_control.kind)
	{
		case ALTERNATING:
			create_folds_alternating();
			break;

		case BLOCKS:
			create_folds_block();
			break;

		case RANDOM:
			create_folds_random();
			break;

		case STRATIFIED:
			create_folds_stratified_random();
			break;

		case RANDOM_SUBSET:
			create_folds_subset(fold_control.negative_fraction);
			break;
			
		default:
			flush_exit(ERROR_DATA_MISMATCH, "Error assigning folds for fold type %d", fold_control.kind);
			break;
	}
}


//**********************************************************************************************************************************


Tfold_manager::~Tfold_manager()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tfold_manager of size %d.", size());
	clear();
}


//**********************************************************************************************************************************


void Tfold_manager::clear()
{
	dataset.clear();
	dataset.enforce_ownership();
	fold_affiliation.clear();
}

//**********************************************************************************************************************************

void Tfold_manager::trivialize()
{
	if (size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to trivialize an empty Tfold_manager.");
	
	fold_control.kind = BLOCKS;
	fold_control.number = 1;
	fold_control.train_fraction = 1.0;
	
	create_folds_block();
}


//**********************************************************************************************************************************


unsigned Tfold_manager::size() const
{
	return unsigned(fold_affiliation.size());
}

//**********************************************************************************************************************************


unsigned Tfold_manager::folds() const
{
	return fold_control.number;
}

//**********************************************************************************************************************************


void Tfold_manager::read_from_file(FILE *fp, const Tdataset& dataset)
{
	unsigned i;
	unsigned dataset_size;

	
	load_dataset(dataset);
	
	fold_control.read_from_file(fp);
	file_read(fp, dataset_size);
	
	if (dataset_size != size())
		flush_exit(ERROR_DATA_MISMATCH, "Size %d of loaded fold information does not match dataset size %d.", dataset_size, size());
	
	for (i=0;i<dataset_size;i++)
		file_read(fp, fold_affiliation[i]);
}


//**********************************************************************************************************************************


void Tfold_manager::write_to_file(FILE *fp) const
{
	fold_control.write_to_file(fp);
	file_write(fp, fold_affiliation);
}

//**********************************************************************************************************************************


void Tfold_manager::build_train_and_val_set(unsigned fold, Tdataset& training_set, Tdataset& validation_set) const
{
	unsigned i;

	training_set.clear();
	validation_set.clear();

	if (size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to assign folds in an empty Tfold_manager.");

	if (fold_control.train_fraction == 1.0)
	{
		if (fold_control.number == 1)
			fold = 2;
		
		for (i=0; i<size(); i++)
			if (fold_affiliation[i] == fold)
				validation_set.push_back(dataset.sample(i));
			else
				training_set.push_back(dataset.sample(i));
	}
	else
	{
		for (i=0; i<size(); i++)
			if (fold_affiliation[i] == fold)
				training_set.push_back(dataset.sample(i));
			else if (fold_affiliation[i] == fold_control.number + 1)
				validation_set.push_back(dataset.sample(i));
	}
}

//**********************************************************************************************************************************


Tsubset_info Tfold_manager::get_train_set_info(unsigned fold) const
{
	unsigned i;
	Tsubset_info train_set_info;

	
	if (size() == 0)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to get train_set_info from an empty Tfold_manager.");

	if (fold_control.train_fraction == 1.0)
	{
		if (fold_control.number == 1)
			fold = 2;
		
		for (i=0; i<size(); i++)
			if (fold_affiliation[i] != fold)
				train_set_info.push_back(i);
	}
	else
	{
		for (i=0; i<size(); i++)
			if (fold_affiliation[i] == fold)
				train_set_info.push_back(i);
	}	
	
	return train_set_info;
}


//**********************************************************************************************************************************


unsigned Tfold_manager::fold_size(unsigned fold) const
{
	unsigned i;
	unsigned count;

	count = 0;
	for (i=0;i<size();i++)
		if (fold_affiliation[i] == fold)
			count++;

	return count;
}


//**********************************************************************************************************************************


unsigned Tfold_manager::max_train_size() const
{
	if (fold_control.number == 1)
		return fold_size(1);
	else if (fold_control.train_fraction < 1.0)
		return max_fold_size();
	else
		return size() - min_fold_size();
}


//**********************************************************************************************************************************


unsigned Tfold_manager::max_val_size() const
{
	if ((fold_control.number == 1) or (fold_control.train_fraction < 1.0))
		return fold_size(fold_control.number + 1);
	else
		return max_fold_size();
}

//**********************************************************************************************************************************


void Tfold_manager::copy(const Tfold_manager& fold_manager)
{
	clear();

	dataset = fold_manager.dataset;
	fold_control = fold_manager.fold_control;
	fold_affiliation = fold_manager.fold_affiliation;
}



//**********************************************************************************************************************************


Tfold_manager& Tfold_manager::operator = (const Tfold_manager& fold_manager)
{
	copy(fold_manager);
	return *this;
}

//**********************************************************************************************************************************


void Tfold_manager::load_dataset(const Tdataset& dataset)
{
	clear();

	Tfold_manager::dataset = dataset;
	fold_affiliation.resize(dataset.size());
}

//**********************************************************************************************************************************


void Tfold_manager::create_folds_alternating()
{
	unsigned i;
	unsigned train_size;


	fold_affiliation.assign(size(), fold_control.number + 1);

	train_size = unsigned(double(size()) * fold_control.train_fraction);
	for (i=0; i<train_size; i++)
		fold_affiliation[i] = (i % fold_control.number) + 1;
}



//**********************************************************************************************************************************


void Tfold_manager::create_folds_block()
{
	unsigned i;
	double block_size;

	fold_affiliation.assign(size(), fold_control.number + 1);
	block_size = unsigned(ceil(double(size()) * fold_control.train_fraction / double(fold_control.number)));

	for (i=0; i<unsigned(double(size()) * fold_control.train_fraction); i++)
		fold_affiliation[i] = unsigned((double(i) / block_size)) + 1;
}


//**********************************************************************************************************************************

void Tfold_manager::create_folds_random()
{
	unsigned i;
	unsigned train_size;
	vector <unsigned> permutation;
	
	
	fold_affiliation.assign(size(), fold_control.number + 1);
	permutation = random_permutation(size(), fold_control.random_seed);

	train_size = unsigned(double(size()) * fold_control.train_fraction);
	for (i=0; i<train_size; i++)
		fold_affiliation[permutation[i]] = (i % fold_control.number) + 1;
}


//**********************************************************************************************************************************


void Tfold_manager::create_folds_stratified_random()
{
	unsigned i;
	unsigned l;
	unsigned train_size;
	Tdataset_info data_set_info;
	Tsubset_info subset_info;
	vector <unsigned> permutation;
	

	if (dataset.is_classification_data() == false)
		flush_exit(ERROR_DATA_MISMATCH, "Stratified folds can only be created for classification data.");
	
	data_set_info = Tdataset_info(dataset, true);
	fold_affiliation.assign(size(), fold_control.number + 1);

	for (l=0; l<data_set_info.label_list.size(); l++)
	{
		subset_info = dataset.create_subset_info_with_label(data_set_info.label_list[l]);
		permutation = random_permutation(unsigned(subset_info.size()), fold_control.random_seed);
		train_size = unsigned(double(subset_info.size()) * fold_control.train_fraction);
		for (i=0; i < train_size; i++)
			fold_affiliation[subset_info[permutation[i]]] = (i % fold_control.number) + 1;
	}
}



//**********************************************************************************************************************************


void Tfold_manager::create_folds_subset(double negative_fraction)
{
	unsigned i;
	double prod;
	Tdataset_info data_set_info;
	unsigned requested_subset_size;
	Tsubset_info subset_info;
	vector <unsigned> permutation;

	
// Make sure that the data set has at most the labels -1 and 1
	
	data_set_info = Tdataset_info(dataset, true);
	if (data_set_info.label_list.size() > 2)
		flush_exit(ERROR_DATA_MISMATCH, "Dataset has more than 2 labels, which is not allowed in\nTfold_manager::create_folds_subset(...).");
	if (data_set_info.kind != CLASSIFICATION)
		flush_exit(ERROR_DATA_MISMATCH, "Dataset is not of binary classication type, which is needed in\nTfold_manager::create_folds_subset(...).");
	
	prod = 1.0;
	for (i=0; i<data_set_info.label_list.size(); i++)
		prod = prod * data_set_info.label_list[i];
	if (abs(prod) != 1.0)
		flush_exit(ERROR_DATA_MISMATCH, "Dataset does not have labels +-1, which is needed in\nTfold_manager::create_folds_subset(...).");
		
	
// Create fold_affiliation for negative samples
	
	fold_affiliation.assign(size(), fold_control.number + 1);

	subset_info = dataset.create_subset_info_with_label(-1.0);
	requested_subset_size = unsigned(negative_fraction * fold_control.train_fraction * double(size()));
	if (requested_subset_size > subset_info.size())
		flush_exit(ERROR_DATA_MISMATCH, "%d samples with negative label needed to create the requested folds,\nbut only %d such samples are available.", int(requested_subset_size), int(subset_info.size()));

	permutation = random_permutation(unsigned(subset_info.size()), fold_control.random_seed, 0);
	for (i= 0; i < requested_subset_size; i++)
		fold_affiliation[subset_info[permutation[i]]] =   1 + (i % fold_control.number);

	
// Create fold_affiliation for positive samples.	

	subset_info = dataset.create_subset_info_with_label(1.0);
	requested_subset_size = unsigned((1.0 - negative_fraction) * fold_control.train_fraction * double(size()));
	if (requested_subset_size > subset_info.size())
		flush_exit(ERROR_DATA_MISMATCH, "%d samples with positive label needed to create the requested folds,\nbut only %d such samples are available", int(requested_subset_size), int(subset_info.size()));

	permutation = random_permutation(unsigned(subset_info.size()), fold_control.random_seed, 1);
	for (i= 0; i < requested_subset_size; i++)
		fold_affiliation[subset_info[permutation[i]]] =   1 + (i % fold_control.number);
}


//**********************************************************************************************************************************


unsigned Tfold_manager::min_fold_size() const
{
	unsigned f;
	unsigned size_of_fold;
	unsigned min_size;

	min_size = fold_size(1);
	for (f=2; f<=fold_control.number; f++)
	{
		size_of_fold = fold_size(f);
		if (size_of_fold < min_size)
			min_size = size_of_fold;
	}
	return min_size;
}


//**********************************************************************************************************************************


unsigned Tfold_manager::max_fold_size() const
{
	unsigned f;
	unsigned size_of_fold;
	unsigned max_size;

	max_size = fold_size(1);
	for (f=2; f<=fold_control.number; f++)
	{
		size_of_fold = fold_size(f);
		if (size_of_fold > max_size)
			max_size = size_of_fold;
	}
	return max_size;
}

#endif



