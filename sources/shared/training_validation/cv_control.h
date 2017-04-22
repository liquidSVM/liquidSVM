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


#if !defined (CV_CONTROL_H)
	#define CV_CONTROL_H



#include "sources/shared/training_validation/grid.h"
#include "sources/shared/training_validation/fold_manager.h"


//**********************************************************************************************************************************


enum SELECT_METHODS {SELECT_ON_ENTIRE_TRAIN_SET, SELECT_ON_EACH_FOLD, SELECT_METHODS_MAX};


//**********************************************************************************************************************************

class Tcv_control
{
	public:
		Tcv_control();
		~Tcv_control();
		
		bool full_search;
		unsigned max_number_of_increases;
		unsigned max_number_of_worse_gammas;
		
		bool use_stored_solution;
		unsigned select_method;
		
		bool npl;
		int npl_class;
		double npl_constraint;
		
		unsigned weight_number;
		
		Tfold_manager fold_manager;
		Tgrid_control grid_control;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/cv_control.cpp"
#endif

#endif
