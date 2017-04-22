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


#if !defined (FOLD_MANAGER_H)
	#define FOLD_MANAGER_H
 


#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/training_validation/fold_control.h"

 
#include <cstdio>
#include <vector>
using namespace std;


//**********************************************************************************************************************************

class Tfold_manager
{
	public:
		Tfold_manager();
		Tfold_manager(const Tfold_manager& fold_manager);
		Tfold_manager(Tfold_control fold_control, const Tdataset& dataset);
		~Tfold_manager();
		
		void clear();
		void trivialize();
		unsigned size() const;
		unsigned folds() const;
		void read_from_file(FILE *fp, const Tdataset& dataset);
		void write_to_file(FILE *fp) const;
		Tfold_manager& operator = (const Tfold_manager& fold_manager);

		
		void build_train_and_val_set(unsigned fold, Tdataset& training_set, Tdataset& validation_set) const;

		unsigned max_val_size() const;
		unsigned max_train_size() const;
		unsigned fold_size(unsigned fold) const;
		Tsubset_info get_train_set_info(unsigned fold) const;

		
	private:
		void copy(const Tfold_manager& fold_manager);
		void load_dataset(const Tdataset& dataset);
		
		unsigned min_fold_size() const;
		unsigned max_fold_size() const;
		
		void create_folds_alternating();
		void create_folds_block();
		void create_folds_random();
		void create_folds_stratified_random();
		void create_folds_subset(double negative_fraction);

		
		Tdataset dataset;
		Tfold_control fold_control;
		vector <unsigned> fold_affiliation;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/fold_manager.cpp"
#endif


#endif
