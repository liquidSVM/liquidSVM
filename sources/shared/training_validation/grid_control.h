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


#if !defined (GRID_CONTROL_H)
	#define GRID_CONTROL_H



 
#include "sources/shared/training_validation/fold_control.h"

#include <cstdio>
#include <vector>
using namespace std;


//**********************************************************************************************************************************


class Tgrid_control
{
	public:
		Tgrid_control();
		
		void write_to_file(FILE *fp);
		void read_from_file(FILE *fp);
		
		void scale_endpoints(Tfold_control fold_control, unsigned data_size, unsigned average_data_size, unsigned data_dim); 
		
		double compute_gamma(unsigned position) const;
		void compute_weights(double& neg_weight, double& pos_weight, unsigned position) const;
		double compute_lambda(unsigned position) const;
		
		
		unsigned gamma_size;
		double max_gamma;
		double min_gamma;
		double min_gamma_unscaled;
		double max_gamma_unscaled;
		bool scale_gamma;
		double spatial_scale_factor;
		vector <double> gammas;
		
		unsigned weight_size;
		bool swap_weights;
		bool geometric_weights;
		double max_weight;
		double min_weight;
		vector <double> weights;
		
		unsigned lambda_size;
		double max_lambda;
		double min_lambda;
		double min_lambda_unscaled;
		bool scale_lambda;
		bool interpret_as_C;
		vector <double> lambdas;
	
	private:
		double compute_geometric_intermediate_value(double a, double b, unsigned size, int position) const;
		
		unsigned approx_train_size;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/grid_control.cpp"
#endif

#endif
