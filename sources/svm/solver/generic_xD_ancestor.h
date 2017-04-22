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


#if !defined (ANCESTOR_XD_SVM_H) 
	#define ANCESTOR_XD_SVM_H


//**********************************************************************************************************************************

class Tsvm_xD_ancestor
{
	public:
		strict_inline__ void order_indices(unsigned& index_1, double& gain_1, unsigned& index_2, double& gain_2);
		strict_inline__ void get_index_with_better_gain_simdd(simdd__& best_index_simdd, simdd__& best_gain_simdd, simdd__ index_simdd, simdd__ gain_simdd);
		
		strict_inline__ simdd__ update_2gradients_simdd(simdd__ gradient_simdd, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ kernel_1_simdd, simdd__ kernel_2_simdd);
	
};



//**********************************************************************************************************************************


#include "sources/svm/solver/generic_xD_ancestor.ins.cpp"


#endif
