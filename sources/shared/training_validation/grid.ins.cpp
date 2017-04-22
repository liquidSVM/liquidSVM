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


#include "sources/shared/training_validation/grid.h"

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/training_validation/train_val_info.h"

#include <limits>
using namespace::std;



//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type> 
Tgrid<Tsolution_type, Ttrain_val_info_type>::Tgrid()
{
}


//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type> 
Tgrid<Tsolution_type, Ttrain_val_info_type>::Tgrid(const Tgrid& grid)
{
	copy(grid);
}

//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type> 
Tgrid<Tsolution_type, Ttrain_val_info_type>::~Tgrid()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tgrid of size %d.", size());
	clear();
}

//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type> 
unsigned Tgrid<Tsolution_type, Ttrain_val_info_type>::size() const
{
	unsigned iw;
	unsigned ig;
	unsigned sum;
	
	sum = 0;
	for (ig=0;ig<train_val_info.size();ig++)
		for (iw=0;iw<train_val_info[ig].size();iw++)
			sum = sum + train_val_info[ig][iw].size();
	
	return sum;
}


//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::write_to_file(FILE* fpinfo, FILE* fpsol) const
{
	unsigned iw;
	unsigned ig;
	unsigned il;

	for (ig=0;ig<train_val_info.size();ig++)
		for (iw=0;iw<train_val_info[ig].size();iw++)
			for (il=0;il<train_val_info[ig][iw].size();il++)
			{
				train_val_info[ig][iw][il].write_to_file(fpinfo);
				solution[ig][iw][il].write_to_file(fpsol);
			}
}

//**********************************************************************************************************************************

template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::read_from_file(FILE* fpinfo, FILE* fpsol, unsigned weight_no, unsigned max_weight_no)
{
	unsigned iw;
	unsigned ig;
	unsigned il;
	unsigned weight_size_tmp;
	Ttrain_val_info_type train_val_info_tmp;
	Tsolution_type solution_tmp;
	

	for (ig=0;ig<train_val_info.size();ig++)
	{
		if (weight_no > 0)
			weight_size_tmp = max_weight_no;
		else
			weight_size_tmp = train_val_info[ig].size();
	
		for (iw=0;iw<weight_size_tmp;iw++)
			for (il=0;il<train_val_info[ig][iw].size();il++)
			{
				if (weight_no == 0)
				{
					train_val_info[ig][iw][il].read_from_file(fpinfo);
					solution[ig][iw][il].read_from_file(fpsol);
				}
				else if ((iw + 1) == weight_no)
				{
					train_val_info[ig][0][il].read_from_file(fpinfo);
					solution[ig][0][il].read_from_file(fpsol);
				}
				else
				{
					train_val_info_tmp.read_from_file(fpinfo);
					solution_tmp.read_from_file(fpsol);
				}
			}
	}
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
Tgrid<Tsolution_type, Ttrain_val_info_type>& Tgrid<Tsolution_type, Ttrain_val_info_type>::operator = (const Tgrid& grid)
{
	copy(grid);
	return *this;
}

//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
Tgrid<Tsolution_type, Ttrain_val_info_type> Tgrid<Tsolution_type, Ttrain_val_info_type>::operator + (const Tgrid& grid)
{
	Tgrid result;
	unsigned iw;
	unsigned ig;
	unsigned il;

	result.train_val_info.resize(min(train_val_info.size(), grid.train_val_info.size()));
	result.solution.resize(min(train_val_info.size(), grid.train_val_info.size()));
	for (ig=0;ig<train_val_info.size();ig++)
	{
		result.train_val_info[ig].resize(min(train_val_info[ig].size(), grid.train_val_info[ig].size()));
		result.solution[ig].resize(min(train_val_info[ig].size(), grid.train_val_info[ig].size()));
		for (iw=0;iw<train_val_info[ig].size();iw++)
		{
			result.train_val_info[ig][iw].resize(min(train_val_info[ig][iw].size(), grid.train_val_info[ig][iw].size()));
			result.solution[ig][iw].resize(min(train_val_info[ig][iw].size(), grid.train_val_info[ig][iw].size()));
			for (il=0;il<train_val_info[ig][iw].size();il++)
				result.train_val_info[ig][iw][il] = train_val_info[ig][iw][il] + grid.train_val_info[ig][iw][il];
		}
	}

	return Tgrid(result);
}

//**********************************************************************************************************************************


template <typename Tgrid_type> Tgrid_type operator * (double scalar, Tgrid_type grid)
{
	Tgrid_type result;
	unsigned iw;
	unsigned ig;
	unsigned il;

	result.train_val_info.resize(grid.train_val_info.size());
	result.solution.resize(grid.train_val_info.size());
	for (ig=0;ig<grid.train_val_info.size();ig++)
	{
		result.train_val_info[ig].resize(grid.train_val_info[ig].size());
		result.solution[ig].resize(grid.train_val_info[ig].size());
		for (iw=0;iw<grid.train_val_info[ig].size();iw++)
		{
			result.train_val_info[ig][iw].resize(grid.train_val_info[ig][iw].size());
			result.solution[ig][iw].resize(grid.train_val_info[ig][iw].size());
			for (il=0;il<grid.train_val_info[ig][iw].size();il++)
				result.train_val_info[ig][iw][il] = scalar * grid.train_val_info[ig][iw][il];
		}
	}

	return Tgrid_type(result);
}



//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::resize(const Tgrid_control& grid_control)
{
	unsigned ig;
	unsigned iw;
	unsigned il;

	
	train_val_info.resize(grid_control.gamma_size);
	solution.resize(grid_control.gamma_size);
	for (ig=0;ig<grid_control.gamma_size;ig++)
	{
		train_val_info[ig].resize(grid_control.weight_size);
		solution[ig].resize(grid_control.weight_size);
		for (iw=0;iw<grid_control.weight_size;iw++)
		{
			train_val_info[ig][iw].resize(grid_control.lambda_size);
			solution[ig][iw].resize(grid_control.lambda_size);
			
			for (il=0;il<grid_control.lambda_size;il++)
			{
				train_val_info[ig][iw][il].gamma = grid_control.compute_gamma(ig);
				grid_control.compute_weights(train_val_info[ig][iw][il].neg_weight, train_val_info[ig][iw][il].pos_weight, iw);
				train_val_info[ig][iw][il].lambda = grid_control.compute_lambda(il);
			}
		}
	}
}

//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::reduce_gammas(vector <unsigned> gamma_index_list)
{
	unsigned i;
	unsigned ig;
	Tgrid grid_new;
	
	
	grid_new.solution.resize(gamma_index_list.size());
	grid_new.train_val_info.resize(gamma_index_list.size());
	
	for (i=0;i<gamma_index_list.size();i++)
	{
		ig = gamma_index_list[i];
		grid_new.solution[i] = solution[ig];
		grid_new.train_val_info[i] = train_val_info[ig];
	}
	
	copy(grid_new);
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::reduce_weights(vector <unsigned> weight_index_list)
{
	unsigned i;
	unsigned ig;
	unsigned iw;
	Tgrid grid_new;
	
	
	grid_new.solution.resize(train_val_info.size());
	grid_new.train_val_info.resize(train_val_info.size());
	
	for (ig=0;ig<train_val_info.size();ig++)
	{
		grid_new.solution[ig].resize(weight_index_list.size());
		grid_new.train_val_info[ig].resize(weight_index_list.size());

		for (i=0;i<weight_index_list.size();i++)
		{
			iw = weight_index_list[i];
			grid_new.solution[ig][i] = solution[ig][iw];
			grid_new.train_val_info[ig][i] = train_val_info[ig][iw];
		}
	}
	copy(grid_new);
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::reduce_lambdas(vector <unsigned> lambda_index_list)
{
	unsigned i;
	unsigned ig;
	unsigned iw;
	unsigned il;
	Tgrid grid_new;
	
	
	grid_new.solution.resize(train_val_info.size());
	grid_new.train_val_info.resize(train_val_info.size());
	
	for (ig=0;ig<train_val_info.size();ig++)
	{
		grid_new.solution[ig].resize(train_val_info[ig].size());
		grid_new.train_val_info[ig].resize(train_val_info[ig].size());
		
		for (iw=0;iw<train_val_info[ig].size();iw++)
		{
			grid_new.solution[ig][iw].resize(lambda_index_list.size());
			grid_new.train_val_info[ig][iw].resize(lambda_index_list.size());
			
			for (i=0;i<lambda_index_list.size();i++)
			{
				il = lambda_index_list[i];
				grid_new.solution[ig][iw][i] = solution[ig][iw][il];
				grid_new.train_val_info[ig][iw][i] = train_val_info[ig][iw][il];
			}
		}
	}
	
	copy(grid_new);
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::clear()
{
	train_val_info.clear();
	solution.clear();
}

//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
Ttrain_val_info_type Tgrid<Tsolution_type, Ttrain_val_info_type>::summarize() const
{
	unsigned ig;
	unsigned iw;
	unsigned il;
	Ttrain_val_info_type result;

	for (ig=0;ig<train_val_info.size();ig++)
		for (iw=0;iw<train_val_info[ig].size();iw++)
			for (il=0;il<train_val_info[ig][iw].size();il++)
				result = result + train_val_info[ig][iw][il];

	return result;
}


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::get_entry_with_best_val_error(unsigned& best_ig, unsigned& best_iw, unsigned& best_il)
{
	Ttrain_val_info_type mask;
	Tgrid grid_tmp;


	grid_tmp = *this;
	mask = Ttrain_val_info_type(IGNORE_VALUE);
	mask.val_error = numeric_limits<double>::max();
	grid_tmp.apply_mask(mask);
	grid_tmp.get_best_entry(best_ig, best_iw, best_il);
};


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::get_entry_with_best_npl_error(int npl_class, double constraint, unsigned& best_ig, unsigned& best_iw, unsigned& best_il)
{
	Ttrain_val_info_type mask;
	Ttrain_val_info_type train_val_info_tmp;
	Tgrid grid_tmp;


// 	Check whether there is at least one entry satisfying the NPL constraint
// 	To this end, we first find the smallest (neg/pos)-validation error.

	grid_tmp = *this;
	mask = Ttrain_val_info_type(IGNORE_VALUE);
	if (npl_class == 1)
		mask.pos_val_error = numeric_limits<double>::max( );
	else
		mask.neg_val_error = numeric_limits<double>::max( );
	grid_tmp.apply_mask(mask);
	grid_tmp.get_best_entry(best_ig, best_iw, best_il);
	train_val_info_tmp = train_val_info[best_ig][best_iw][best_il];


// 	Now we check whether the smallest (neg/pos)-validation error satisfies
// 	the NPL constraint. If it does not, we need to adjust the NPL constraint.

	if (npl_class == 1)
	{
		if (train_val_info_tmp.pos_val_error < constraint)
			mask.pos_val_error = constraint;
		else
			mask.pos_val_error = train_val_info_tmp.pos_val_error;

		mask.neg_val_error = numeric_limits<double>::max( );
	}
	else
	{
		if (train_val_info_tmp.neg_val_error < constraint)
			mask.neg_val_error = constraint;
		else
			mask.neg_val_error = train_val_info_tmp.neg_val_error;

		mask.pos_val_error = numeric_limits<double>::max( );
	}

// 	Now we sort out the entries that do not satisfy the (adjusted) NPL constraint

	grid_tmp = *this;
	grid_tmp.apply_mask(mask);

//	Next, we make sure that the validation error the constraint is put on, does
//	not influence the ordering of the entries.

	if (npl_class == 1)
		mask.pos_val_error = IGNORE_VALUE;
	else
		mask.neg_val_error = IGNORE_VALUE;
	grid_tmp.apply_mask(mask);

// 	Finally, we look for the best entry and copy it from the untouched grid,
// 	so that modified values do not appear

	grid_tmp.get_best_entry(best_ig, best_iw, best_il);
	train_val_info_tmp = train_val_info[best_ig][best_iw][best_il];
};


//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::apply_mask(const Ttrain_val_info_type& mask)
{
	unsigned ig;
	unsigned iw;
	unsigned il;

	for (ig=0;ig<train_val_info.size();ig++)
		for (iw=0;iw<train_val_info[ig].size();iw++)
			for (il=0;il<train_val_info[ig][iw].size();il++)
				train_val_info[ig][iw][il].apply_mask(mask);
}

//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::get_best_entry(unsigned& best_ig, unsigned& best_iw, unsigned& best_il)
{
	unsigned ig;
	unsigned iw;
	unsigned il;
	Ttrain_val_info_type best_entry;

	best_ig = 0;
	best_iw = 0;
	best_il = 0;
	best_entry = Ttrain_val_info_type(WORST_VALUES);

	for (ig=0;ig<train_val_info.size();ig++)
		for (iw=0;iw<train_val_info[ig].size();iw++)
			for (il=0;il<train_val_info[ig][iw].size();il++)
				if (train_val_info[ig][iw][il] < best_entry)
				{
					best_entry = train_val_info[ig][iw][il];
					best_ig = ig;
					best_iw = iw;
					best_il = il;
				}
}



//**********************************************************************************************************************************


template <class Tsolution_type, class Ttrain_val_info_type> 
void Tgrid<Tsolution_type, Ttrain_val_info_type>::copy(const Tgrid& grid)
{
	clear();
	
	solution = grid.solution;
	train_val_info = grid.train_val_info;
}


