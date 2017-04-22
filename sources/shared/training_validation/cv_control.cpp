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


#if !defined (CV_CONTROL_CPP)
	#define CV_CONTROL_CPP


#include "sources/shared/training_validation/cv_control.h"



//**********************************************************************************************************************************

Tcv_control::Tcv_control()
{
	use_stored_solution = false;
	select_method = SELECT_ON_EACH_FOLD;
	
	npl = false;
	npl_class = -1;
	npl_constraint = 1.0;
	weight_number = 0;
	
	full_search = false;
	max_number_of_increases = 3;
	max_number_of_worse_gammas = 3;
}


//**********************************************************************************************************************************

Tcv_control::~Tcv_control()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tcv_control.");
}


#endif
