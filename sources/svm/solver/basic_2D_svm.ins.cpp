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



//**********************************************************************************************************************************


inline void Tbasic_2D_svm::get_index_with_better_gain_simdd(simdd__& best_index_simdd, simdd__& best_gain_simdd, simdd__ index_simdd, simdd__ gain_simdd)
{
	best_index_simdd = seq_argmax_simdd(best_index_simdd, best_gain_simdd, index_simdd, gain_simdd);
	best_gain_simdd = max_simdd(best_gain_simdd, gain_simdd);
}


//**********************************************************************************************************************************


inline void Tbasic_2D_svm::order_indices(unsigned& index_1, double& gain_1, unsigned& index_2, double& gain_2)
{
	if (gain_2 > gain_1)
	{
		swap(gain_1, gain_2);
		swap(index_1, index_2);
	}
}


//**********************************************************************************************************************************


inline simdd__ Tbasic_2D_svm::update_2gradients_simdd(simdd__ gradient_simdd, simdd__ delta_1_simdd, simdd__ delta_2_simdd, simdd__ kernel_1_simdd, simdd__ kernel_2_simdd)
{
	simdd__ grad_simdd;
	
	grad_simdd = fuse_mult_add_simdd(delta_1_simdd, kernel_1_simdd, gradient_simdd);
	return fuse_mult_add_simdd(delta_2_simdd, kernel_2_simdd, grad_simdd);
}


