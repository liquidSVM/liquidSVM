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


#if !defined (GRID_CONTROL_CPP)
	#define GRID_CONTROL_CPP

 
 
#include "sources/shared/training_validation/grid_control.h"


#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_functions/basic_file_functions.h" 


#include <cmath>


//**********************************************************************************************************************************

Tgrid_control::Tgrid_control()
{
	gamma_size = 10;
	max_gamma = 5.0;
	min_gamma = 0.2;
	min_gamma_unscaled = 0.2;
	max_gamma_unscaled = 5.0;
	scale_gamma = true;
	spatial_scale_factor = 1.0;
	
	weight_size = 1;
	swap_weights = false;
	geometric_weights = false;
	max_weight = 1.0;
	min_weight = 1.0;
	
	lambda_size = 10;
	max_lambda = 0.1;
	min_lambda = 0.001;
	min_lambda_unscaled = 0.001;
	scale_lambda = true;
	interpret_as_C = false;
};


//**********************************************************************************************************************************

void Tgrid_control::write_to_file(FILE *fp)
{
	file_write(fp, lambda_size);
	file_write(fp, min_lambda_unscaled);
	file_write(fp, max_lambda);
	file_write(fp, lambdas);

	file_write(fp, gamma_size);
	file_write(fp, min_gamma_unscaled);
	file_write(fp, max_gamma_unscaled);
	file_write(fp, gammas);
	
	file_write(fp, weight_size);
	file_write(fp, swap_weights);
	file_write(fp, geometric_weights);
	file_write(fp, min_weight);
	file_write(fp, max_weight);
	file_write(fp, weights);

	file_write(fp, scale_lambda);
	file_write(fp, interpret_as_C);
	file_write(fp, scale_gamma);
	
	file_write_eol(fp);
};

//**********************************************************************************************************************************


void Tgrid_control::read_from_file(FILE *fp)
{
	file_read(fp, lambda_size);
	file_read(fp, min_lambda_unscaled);
	file_read(fp, max_lambda);
	file_read(fp, lambdas);

	file_read(fp, gamma_size);
	file_read(fp, min_gamma_unscaled);
	file_read(fp, max_gamma_unscaled);
	file_read(fp, gammas);

	file_read(fp, weight_size);
	file_read(fp, swap_weights);
	file_read(fp, geometric_weights);
	file_read(fp, min_weight);
	file_read(fp, max_weight);
	file_read(fp, weights);

	file_read(fp, scale_lambda);
	file_read(fp, interpret_as_C);
	file_read(fp, scale_gamma);
}


//**********************************************************************************************************************************


void Tgrid_control::scale_endpoints(Tfold_control fold_control, unsigned data_size, unsigned average_data_size, unsigned data_dim)
{
	double base;
	
	if (fold_control.number > 1)
		approx_train_size = unsigned(fold_control.train_fraction * ((double(fold_control.number) - 1.0)  / double(fold_control.number)) * double(average_data_size));
	else
		approx_train_size = unsigned(fold_control.train_fraction * double(average_data_size));
	
	if (scale_gamma == true)
	{
		base = 1.0 + 5.0 / double(approx_train_size);
		min_gamma = min_gamma_unscaled * pow(double(approx_train_size), -1.0/double(data_dim));
		max_gamma = max_gamma_unscaled * pow(base, double(data_dim));
	}
	else
	{
		min_gamma = min_gamma_unscaled;
		max_gamma = max_gamma_unscaled;
	}

	
// 	Recompute approx_train_size, since the used values differ for lambda and gamma.
	
	if (fold_control.number > 1)
		approx_train_size = unsigned(fold_control.train_fraction * ((double(fold_control.number) - 1.0)  / double(fold_control.number)) * double(data_size));
	else
		approx_train_size = unsigned(fold_control.train_fraction * double(data_size));

	if (scale_lambda == true)
		min_lambda = min_lambda_unscaled/(approx_train_size);
	else 
		min_lambda = min_lambda_unscaled;
	
	
	if (interpret_as_C == false)
		sort_down(lambdas);
	else
		sort_up(lambdas);
}



//**********************************************************************************************************************************


double Tgrid_control::compute_gamma(unsigned position) const
{
	double geometric_step;
	int geometric_position_translation;
	
	
	if (gammas.size() > 0)
	{
		if (position >= gammas.size())
			flush_exit(ERROR_DATA_MISMATCH, "Trying to access gamma number %d but there are only %d gammas.", position, gammas.size());
		
		return gammas[position];
	}

	if ((spatial_scale_factor == 1.0) or (scale_gamma == false))
		return compute_geometric_intermediate_value(max_gamma, min_gamma, gamma_size, int(position));

	geometric_step = compute_geometric_intermediate_value(max_gamma, min_gamma, gamma_size, 1) / max_gamma;
	geometric_position_translation = int(floor(log(sqrt(spatial_scale_factor)) / log(geometric_step)));

	return compute_geometric_intermediate_value(max_gamma, min_gamma, gamma_size, int(position) + geometric_position_translation);
}


//**********************************************************************************************************************************


void Tgrid_control::compute_weights(double& neg_weight, double& pos_weight, unsigned position) const
{
	if (weights.size() > 0)
	{
		if (position >= weights.size())
			flush_exit(ERROR_DATA_MISMATCH, "Trying to access weight number %d but there are only %d weights.", position, weights.size());
		
		pos_weight = weights[position];
		neg_weight = 1.0 - pos_weight;
	}
	else if (weight_size == 1)
	{
		neg_weight = min_weight;
		pos_weight = max_weight;
	}
	else if (geometric_weights == false)
	{
		pos_weight = min_weight + double(position)/(double(weight_size) - 1.0) * (max_weight - min_weight);
		neg_weight = 1.0 - pos_weight;
	}
	else 
	{
		pos_weight = compute_geometric_intermediate_value(min_weight, max_weight, weight_size, position);
		neg_weight = 1.0 - pos_weight;
	}
	
	if ((swap_weights == true) and (weight_size > 1))
		swap(pos_weight, neg_weight);
}

//**********************************************************************************************************************************


double Tgrid_control::compute_lambda(unsigned position) const
{
	double lambda;
	
	if (lambdas.size() > 0)
	{
		if (position >= lambdas.size())
			flush_exit(ERROR_DATA_MISMATCH, "Trying to access lambda number %d but there are only %d lambdas.", position, lambdas.size());
		
		lambda = lambdas[position];
	}	
	else
		lambda = compute_geometric_intermediate_value(max_lambda, min_lambda, lambda_size, position);
	
	if (interpret_as_C == false)
		return lambda;
	else
		return 1.0/(2.0 * lambda * double(approx_train_size));
}


//**********************************************************************************************************************************


double Tgrid_control::compute_geometric_intermediate_value(double a, double b, unsigned size, int position) const
{
	if (size < 2)
		return b;
	else
		return a * pow((b/a), double(position)/(double(size) - 1.0));
}

#endif

