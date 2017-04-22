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


#if !defined (LOSS_FUNCTION_H) 
	#define LOSS_FUNCTION_H


   
 
#include <cstdio>
using namespace std;


//**********************************************************************************************************************************
   
// 		CHANGE_FOR_OWN_SOLVER 
enum LOSS_TYPES {CLASSIFICATION_LOSS, MULTI_CLASS_LOSS, LEAST_SQUARES_LOSS, WEIGHTED_LEAST_SQUARES_LOSS, PINBALL_LOSS, HINGE_LOSS, TEMPLATE_LOSS, LOSS_TYPES_MAX};


//**********************************************************************************************************************************


inline double sign(double x);
inline double classification_loss(double y, double t);
inline double neg_classification_loss(double y, double t);
inline double pos_classification_loss(double y, double t);


//**********************************************************************************************************************************


class Tloss_control
{
	public:
		Tloss_control();
		void read_from_file(FILE *fp);
		void write_to_file(FILE *fp) const;
		
		unsigned type;
		double clipp_value;
		double neg_weight;
		double pos_weight;
		
		double yp;
		double ym;
};


//**********************************************************************************************************************************


class Tloss_function: protected Tloss_control
{
	public:
		Tloss_function();
		Tloss_function(Tloss_control loss_control);
		
		unsigned get_type() const;
		
		void set_weights(double neg_weight, double pos_weight);
		void set_clipp_value(double clipp_value);
		double evaluate(double y, double t) const;
		
	private:
		void check_integrity();
};


//**********************************************************************************************************************************

#include "sources/shared/basic_types/loss_function.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/loss_function.cpp"
#endif


#endif





