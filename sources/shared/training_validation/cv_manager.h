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


#if !defined (CV_MANAGER_H)
	#define CV_MANAGER_H

 

#include "sources/shared/solver/solver.h"
#include "sources/shared/solver/solver_control.h"
#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/training_validation/grid.h"
#include "sources/shared/training_validation/cv_control.h"
#include "sources/shared/training_validation/fold_manager.h"
#include "sources/shared/system_support/thread_manager_active.h"

 
//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type, class Tsolver_control_type, class Tsolver_type> class Tcv_manager: public Tthread_manager_active
{
	public:
		~Tcv_manager();
		
		virtual void clear_threads();
		
		void train_all_folds(Tcv_control cv_control, const Tsolver_control_type& solver_ctrl, vector < Tgrid<Tsolution_type, Ttrain_val_info_type> >& grids);
		void select_all_folds(Tcv_control& cv_control, const Tsolver_control_type& solver_ctrl, vector < Tgrid<Tsolution_type, Ttrain_val_info_type> >& grids, vector <Tsolution_type>& solutions, vector <Ttrain_val_info_type>& select_val_info);
		
		Tsubset_info get_train_set_info(unsigned fold) const;
		
		
		
		unsigned hit_smallest_gamma;
		unsigned hit_largest_gamma;
		
		unsigned hit_smallest_weight;
		unsigned hit_largest_weight;
		
		unsigned hit_smallest_lambda;
		unsigned hit_largest_lambda;

	private:
		inline void create_solver();
		virtual void thread_entry();
		
		void train_on_grid(Tgrid<Tsolution_type, Ttrain_val_info_type>& grid);
		
		void select_on_grid(unsigned fold);
		void resize_grid_for_select(unsigned fold);
		
		
		unsigned grid_train_method;
		Tcv_control cv_control;
		Tsolver_control_type solver_control;

		vector < Tgrid <Tsolution_type, Ttrain_val_info_type> > grids;
		
		Tkernel val_kernel;
		Tkernel train_kernel;
		
		Tsolver_type* solver;
		vector <Tsolution_type> solutions;
		vector <Ttrain_val_info_type> select_val_info;
		
		Tdataset training_set;
		Tdataset validation_set;
		
		vector <vector <unsigned> > permutations;
		
		unsigned assumed_best_ig;
		int assumed_ig_search_direction;
		vector <unsigned> best_ig_count;
};


//**********************************************************************************************************************************


#include "sources/shared/training_validation/cv_manager.ins.cpp"

 


#endif
