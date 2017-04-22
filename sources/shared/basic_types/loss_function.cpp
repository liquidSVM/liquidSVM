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


#if !defined (LOSS_FUNCTION_CPP) 
	#define LOSS_FUNCTION_CPP

 
 
#include "sources/shared/basic_types/loss_function.h"


#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"

#include <cmath>
#include <limits>
using namespace std;




//**********************************************************************************************************************************

Tloss_control::Tloss_control()
{
	type = CLASSIFICATION_LOSS;
	clipp_value = -1.0;
	
	neg_weight = 1.0;
	pos_weight = 1.0;
	
	yp = 1.0;
	ym = -1.0;
};


//**********************************************************************************************************************************


void Tloss_control::read_from_file(FILE *fp)
{
	file_read(fp, type);
	file_read(fp, neg_weight);
	file_read(fp, pos_weight);
}

//**********************************************************************************************************************************


void Tloss_control::write_to_file(FILE *fp) const
{
	file_write(fp, type);
	file_write(fp, neg_weight);
	file_write(fp, pos_weight);
	file_write_eol(fp);
}


//**********************************************************************************************************************************

Tloss_function::Tloss_function()
{
};


//**********************************************************************************************************************************


Tloss_function::Tloss_function(Tloss_control loss_control)
{
	type = loss_control.type;
	
	neg_weight = loss_control.neg_weight;
	pos_weight = loss_control.pos_weight;

	yp = loss_control.yp;
	ym = loss_control.ym;
	
	check_integrity();
}


//**********************************************************************************************************************************

unsigned Tloss_function::get_type() const
{
	return type;
};

//**********************************************************************************************************************************

void Tloss_function::set_weights(double neg_weight, double pos_weight)
{
	Tloss_function::neg_weight = neg_weight;
	Tloss_function::pos_weight = pos_weight;

	check_integrity();
};


//**********************************************************************************************************************************

void Tloss_function::set_clipp_value(double clipp_value)
{
	Tloss_function::clipp_value = clipp_value;
	
	check_integrity();
}


//**********************************************************************************************************************************

double Tloss_function::evaluate(double y, double t) const
{
	double a;
	double b;
	double ct;
	
	if (clipp_value > 0.0)
		ct = max( -clipp_value, min( clipp_value, t));
	else
		ct = t;
	
	switch(type)
	{
		case CLASSIFICATION_LOSS:
			a = 2.0 / (yp - ym);
			b = - (yp + ym) / (yp - ym);
			return ( (y == yp)?
					pos_weight * classification_loss(a * y + b, a * ct + b):
					neg_weight * classification_loss(a * y + b, a * ct + b)
				);
				
		case MULTI_CLASS_LOSS:
			return ( (abs(y - ct) < 0.5)?
					0.0:
					1.0
				);
				
		case LEAST_SQUARES_LOSS:
			return (y - ct) * (y - ct);
			
		case WEIGHTED_LEAST_SQUARES_LOSS:	
			return ( ((y - ct) < 0.0)?
					neg_weight * (ct - y) * (ct - y):
					pos_weight * (y - ct) * (y - ct)
				);
			
		case PINBALL_LOSS:	
			return ( ((y - ct) < 0.0)?
					neg_weight * (ct - y):
					pos_weight * (y - ct)
				);
			
		case HINGE_LOSS:
			return max(0.0, 1.0-y*t);
			
// 		Change the next lines according to your needs.
// 		CHANGE_FOR_OWN_SOLVER
		case TEMPLATE_LOSS:
			return 0.0;
			
		default:
			flush_exit(ERROR_UNSPECIFIED, "Specified loss function does not exist.");
			return 0.0;
	}
};

//**********************************************************************************************************************************


void Tloss_function::check_integrity()
{
	if ((type == MULTI_CLASS_LOSS) and ((neg_weight != 1.0) or (pos_weight != 1.0)))
		flush_exit(ERROR_DATA_STRUCTURE, "Multi-class loss does not allow weights.");
	
	if ((type == LEAST_SQUARES_LOSS) and ((neg_weight != 1.0) or (pos_weight != 1.0)))
		flush_exit(ERROR_DATA_STRUCTURE, "Unweighted least squares loss does not allow weights.");
	
	if (yp == ym)
		flush_exit(ERROR_DATA_STRUCTURE, "Binary classification loss needs two destinct labels.");
	
// 	Add a check for your loss if necessary.
}

#endif





