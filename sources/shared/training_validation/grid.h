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


#if !defined (GRID_TMP_H)
	#define GRID_TMP_H
 

 

#include "sources/shared/basic_types/dataset_info.h"
#include "sources/shared/training_validation/grid_control.h"


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> class Tgrid
{
	public:
		Tgrid();
		Tgrid(const Tgrid& grid);
		~Tgrid();
		
		void clear();
		unsigned size() const;
		void resize(const Tgrid_control& grid_control);
		void reduce_gammas(vector <unsigned> gamma_index_list);
		void reduce_weights(vector <unsigned> weight_index_list);
		void reduce_lambdas(vector <unsigned> lambda_index_list);
		
		void write_to_file(FILE* fpinfo, FILE* fpsol = NULL) const;
		void read_from_file(FILE* fpinfo, FILE* fpsol = NULL, unsigned weight_no = 0, unsigned max_weight_no = 0);

		Ttrain_val_info_type summarize() const;
		Tgrid operator + (const Tgrid& grid);
		Tgrid& operator = (const Tgrid& grid);
		
		void apply_mask(const Ttrain_val_info_type& mask);
		void get_entry_with_best_val_error(unsigned& best_ig, unsigned& best_iw, unsigned& best_il);
		void get_entry_with_best_npl_error(int npl_class, double constraint, unsigned& best_ig, unsigned& best_iw, unsigned& best_il);



		vector < vector < vector <Tsolution_type> > > solution;
		vector < vector < vector <Ttrain_val_info_type> > > train_val_info;

	protected:
		void copy(const Tgrid& grid);
		void get_best_entry(unsigned& best_ig, unsigned& best_iw, unsigned& best_il);
};


template <typename Tgrid_type> Tgrid_type operator * (double scalar, Tgrid_type grid);

//**********************************************************************************************************************************


#include "sources/shared/training_validation/grid.ins.cpp"


#endif
