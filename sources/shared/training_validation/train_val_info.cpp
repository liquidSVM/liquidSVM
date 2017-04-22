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


#if !defined (TRAIN_VAL_INFO_CPP)
	#define TRAIN_VAL_INFO_CPP


#include "sources/shared/training_validation/train_val_info.h"


#include <limits>
using namespace std;

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_functions/extra_string_functions.h"


unsigned weight_display_mode;


//**********************************************************************************************************************************

double Ttrain_val_info::add_error(double error1, double error2)
{
	if ((error1 == NOT_EVALUATED) or (error2 == NOT_EVALUATED))
		return NOT_EVALUATED;
	else if ((error1 == IGNORE_VALUE) or (error2 == IGNORE_VALUE))
		return IGNORE_VALUE;
	else
		return error1 + error2;
}


//**********************************************************************************************************************************

double Ttrain_val_info::multiply_error(double scalar, double error)
{
	if (error == NOT_EVALUATED)
		return NOT_EVALUATED;
	else if (error == IGNORE_VALUE)
		return IGNORE_VALUE;
	else
		return scalar * error;
}


//**********************************************************************************************************************************

Ttrain_val_info::Ttrain_val_info()
{
	if ((weight_display_mode != DISPLAY_WEIGHTS_NO_ERROR) and (weight_display_mode != DISPLAY_WEIGHTS_AND_ERROR))
		weight_display_mode = DISPLAY_NO_WEIGHTS;
	
	gamma = 1.0;
	neg_weight = 1.0;
	pos_weight = 1.0;
	lambda = 1.0;
	clear();
};


//**********************************************************************************************************************************

Ttrain_val_info::Ttrain_val_info(double init_type)
{
	clear();

	if (init_type == IGNORE_VALUE)
		ignore();
	else if (init_type == WORST_VALUES)
	{
		val_error = numeric_limits<double>::max( );
		pos_val_error = numeric_limits<double>::max( );
		neg_val_error = numeric_limits<double>::max( );
	}
};



//**********************************************************************************************************************************

Ttrain_val_info::Ttrain_val_info(const Ttrain_val_info& train_val_info)
{
	copy(train_val_info);
};


//**********************************************************************************************************************************

void Ttrain_val_info::display(unsigned display_mode, unsigned info_level) const
{
	string output;

 	if ((info_mode < INFO_3) and (display_mode == TRAIN_INFO_DISPLAY_FORMAT_REGULAR))
		display_mode = TRAIN_INFO_DISPLAY_FORMAT_SHORT;
	
	output = displaystring(display_mode);

	flush_info(info_level, "\n");
	flush_info(info_level, output.c_str());
}


//**********************************************************************************************************************************

void Ttrain_val_info::read_from_file(FILE* fp)
{
	double dummy_r;

	// Skip grid values to avoid rounding errors

	file_read(fp, dummy_r);
	file_read(fp, dummy_r);
	file_read(fp, dummy_r);
	file_read(fp, dummy_r);

	// The rest is read from the file

	file_read(fp, train_error);
	file_read(fp, neg_train_error);
	file_read(fp, pos_train_error);
	file_read(fp, val_error);
	file_read(fp, neg_val_error);
	file_read(fp, pos_val_error);
	
	file_read(fp, train_pre_build_time);
	file_read(fp, train_build_time);
	file_read(fp, train_build_transfer_time);
	file_read(fp, train_kNN_build_time);
	file_read(fp, train_cache_hits);
	file_read(fp, train_pre_cache_hits);
	
	file_read(fp, val_pre_build_time);
	file_read(fp, val_build_time);
	file_read(fp, val_build_transfer_time);
	
	file_read(fp, init_time);
	file_read(fp, train_time);
	file_read(fp, val_time);
}



//**********************************************************************************************************************************

void Ttrain_val_info::write_to_file(FILE* fp) const
{
	file_write(fp, gamma);
	file_write(fp, neg_weight);
	file_write(fp, pos_weight);
	file_write(fp, lambda);
	
	file_write(fp, train_error);
	file_write(fp, neg_train_error);
	file_write(fp, pos_train_error);
	file_write(fp, val_error);
	file_write(fp, neg_val_error);
	file_write(fp, pos_val_error);
	
	file_write(fp, train_pre_build_time);
	file_write(fp, train_build_time);
	file_write(fp, train_build_transfer_time);
	file_write(fp, train_kNN_build_time);
	file_write(fp, train_cache_hits);
	file_write(fp, train_pre_cache_hits);
	
	file_write(fp, val_pre_build_time);
	file_write(fp, val_build_time);
	file_write(fp, val_build_transfer_time);
	
	file_write(fp, init_time);
	file_write(fp, train_time);
	file_write(fp, val_time);
}



//**********************************************************************************************************************************

void Ttrain_val_info::clear()
{
	train_error = NOT_EVALUATED;
	pos_train_error = NOT_EVALUATED;
	neg_train_error = NOT_EVALUATED;

	val_error = NOT_EVALUATED;
	pos_val_error = NOT_EVALUATED;
	neg_val_error = NOT_EVALUATED;

	train_pre_build_time = 0.0;
	train_build_time = 0.0;
	train_build_transfer_time = 0.0;
	train_kNN_build_time = 0.0;
	train_cache_hits = 0.0;
	train_pre_cache_hits = 0.0;
	
	val_pre_build_time = 0.0;
	val_build_time = 0.0;
	val_build_transfer_time = 0.0;
	
	init_time = 0.0;
	train_time = 0.0;
	val_time = 0.0;
}


//**********************************************************************************************************************************

void Ttrain_val_info::apply_mask(const Ttrain_val_info& mask)
{
	bool result;

	result = apply_mask_entry(val_error, mask.val_error);
	result = result or apply_mask_entry(pos_val_error, mask.pos_val_error);
	result = result or apply_mask_entry(neg_val_error, mask.neg_val_error);

	if (result == true)
		ignore();
}


//**********************************************************************************************************************************


Ttrain_val_info Ttrain_val_info::operator = (const Ttrain_val_info& train_val_info)
{
	copy(train_val_info);
	return *this;
}

//**********************************************************************************************************************************


Ttrain_val_info Ttrain_val_info::operator + (const Ttrain_val_info& train_val_info)
{
	Ttrain_val_info result;

	result.gamma = gamma;
	result.neg_weight = neg_weight;
	result.pos_weight = pos_weight;
	result.lambda = lambda;

	result.train_error = add_error(train_error, train_val_info.train_error);
	result.pos_train_error = add_error(pos_train_error, train_val_info.pos_train_error);
	result.neg_train_error = add_error(neg_train_error, train_val_info.neg_train_error);

	result.val_error = add_error(val_error, train_val_info.val_error);
	result.pos_val_error = add_error(pos_val_error, train_val_info.pos_val_error);
	result.neg_val_error = add_error(neg_val_error, train_val_info.neg_val_error);

	result.train_pre_build_time = train_pre_build_time + train_val_info.train_pre_build_time;
	result.train_build_time = train_build_time + train_val_info.train_build_time;
	result.train_build_transfer_time = train_build_transfer_time + train_val_info.train_build_transfer_time;
	result.train_kNN_build_time = train_kNN_build_time + train_val_info.train_kNN_build_time;
	result.train_cache_hits = train_cache_hits + train_val_info.train_cache_hits;
	result.train_pre_cache_hits = train_pre_cache_hits + train_val_info.train_pre_cache_hits;
	
	result.val_pre_build_time = val_pre_build_time + train_val_info.val_pre_build_time;
	result.val_build_time = val_build_time + train_val_info.val_build_time;
	result.val_build_transfer_time = val_build_transfer_time + train_val_info.val_build_transfer_time;
	
	result.init_time = init_time + train_val_info.init_time;
	result.train_time = train_time + train_val_info.train_time;
	result.val_time = val_time + train_val_info.val_time;

	return Ttrain_val_info(result);
}


//**********************************************************************************************************************************

Ttrain_val_info operator * (double scalar, Ttrain_val_info train_val_info)
{
	Ttrain_val_info result;

	result.gamma = train_val_info.gamma;
	result.neg_weight = train_val_info.neg_weight;
	result.pos_weight = train_val_info.pos_weight;
	result.lambda = train_val_info.lambda;

	result.train_error = result.multiply_error(scalar, train_val_info.train_error);
	result.pos_train_error = result.multiply_error(scalar, train_val_info.pos_train_error);
	result.neg_train_error = result.multiply_error(scalar, train_val_info.neg_train_error);

	result.val_error = result.multiply_error(scalar, train_val_info.val_error);
	result.pos_val_error = result.multiply_error(scalar, train_val_info.pos_val_error);
	result.neg_val_error = result.multiply_error(scalar, train_val_info.neg_val_error);

	result.train_pre_build_time = scalar * train_val_info.train_pre_build_time;
	result.train_build_time = scalar * train_val_info.train_build_time;
	result.train_build_transfer_time = scalar * train_val_info.train_build_transfer_time;
	result.train_kNN_build_time = scalar * train_val_info.train_kNN_build_time;
	result.train_cache_hits = scalar * train_val_info.train_cache_hits;
	result.train_pre_cache_hits = scalar * train_val_info.train_pre_cache_hits;
	
	result.val_pre_build_time = scalar * train_val_info.val_pre_build_time;
	result.val_build_time = scalar * train_val_info.val_build_time;
	result.val_build_transfer_time = scalar * train_val_info.val_build_transfer_time;
	
	result.init_time = scalar * train_val_info.init_time;
	result.train_time = scalar * train_val_info.train_time;
	result.val_time = scalar * train_val_info.val_time;

	return Ttrain_val_info(result);
}




//**********************************************************************************************************************************

bool Ttrain_val_info::operator == (const Ttrain_val_info& train_val_info)
{
	bool result;

	result = equal(val_error, train_val_info.val_error);
	result = result and equal(pos_val_error, train_val_info.pos_val_error);
	result = result and equal(neg_val_error, train_val_info.neg_val_error);

	return result;
}


//**********************************************************************************************************************************


bool Ttrain_val_info::operator < (const Ttrain_val_info& train_val_info)
{
	if (not equal(val_error, train_val_info.val_error))
		return less(val_error, train_val_info.val_error);

	if (not equal(pos_val_error, train_val_info.pos_val_error))
		return less(pos_val_error, train_val_info.pos_val_error);

	if (not equal(neg_val_error, train_val_info.neg_val_error))
		return less(neg_val_error, train_val_info.neg_val_error);

	else return false;
}



//**********************************************************************************************************************************

void Ttrain_val_info::ignore()
{
	val_error = IGNORE_VALUE;
	pos_val_error = IGNORE_VALUE;
	neg_val_error = IGNORE_VALUE;
}

//**********************************************************************************************************************************

void Ttrain_val_info::copy(const Ttrain_val_info& train_val_info)
{
	gamma = train_val_info.gamma;
	neg_weight = train_val_info.neg_weight;
	pos_weight = train_val_info.pos_weight;
	lambda = train_val_info.lambda;

	train_error = train_val_info.train_error;
	pos_train_error = train_val_info.pos_train_error;
	neg_train_error = train_val_info.neg_train_error;

	val_error = train_val_info.val_error;
	pos_val_error = train_val_info.pos_val_error;
	neg_val_error = train_val_info.neg_val_error;

	train_pre_build_time = train_val_info.train_pre_build_time;
	train_build_time = train_val_info.train_build_time;
	train_build_transfer_time = train_val_info.train_build_transfer_time;
	train_kNN_build_time = train_val_info.train_kNN_build_time;
	train_cache_hits = train_val_info.train_cache_hits;
	train_pre_cache_hits = train_val_info.train_pre_cache_hits;
	
	val_pre_build_time = train_val_info.val_pre_build_time;
	val_build_time = train_val_info.val_build_time;
	val_build_transfer_time = train_val_info.val_build_transfer_time;
	
	init_time = train_val_info.init_time;
	train_time = train_val_info.train_time;
	val_time = train_val_info.val_time;
};



//**********************************************************************************************************************************

double Ttrain_val_info::full_kernel_time() const
{
	return train_build_time + train_pre_build_time + train_build_transfer_time + train_kNN_build_time + val_build_time + val_pre_build_time + val_build_transfer_time;
}


//**********************************************************************************************************************************

bool Ttrain_val_info::equal(double x, double y)
{
	if (x == NOT_EVALUATED)
	{
		if (y == NOT_EVALUATED)
			return true;
		else
			return false;
	}

	if (x == IGNORE_VALUE)
	{
		if (y == NOT_EVALUATED)
			return false;
		else
			return true;
	}

	if (y == NOT_EVALUATED)
		return false;
	else if (y == IGNORE_VALUE)
		return true;
	else
		return (x==y);
}

//**********************************************************************************************************************************

bool Ttrain_val_info::less(double x, double y)
{
	if (x == NOT_EVALUATED)
		return false;

	if (x == IGNORE_VALUE)
	{
		if (y == NOT_EVALUATED)
			return true;
		else
			return false;
	}

	if (y == NOT_EVALUATED)
		return true;
	else if (y == IGNORE_VALUE)
		return false;
	else
		return (x<y);
}


//**********************************************************************************************************************************


string Ttrain_val_info::displaystring(unsigned display_mode) const
{
	string output;

	output = displaystring_parameters(display_mode);
	output = output + displaystring_train_error(display_mode); 
	output = output + displaystring_val_error(display_mode); 
	output = output + displaystring_kernel(display_mode); 
	output = output + displaystring_time(display_mode);
	
	return output;
}


//**********************************************************************************************************************************


string Ttrain_val_info::displaystring_parameters(unsigned display_mode) const
{
	string output;

	if (display_mode != TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED)
	{
		output = "g: " + number_to_string(gamma, 2);
		if (weight_display_mode != DISPLAY_NO_WEIGHTS)
		{
			output = output + "  nw: " + number_to_string(neg_weight, 2);
			output = output + "  pw: " + number_to_string(pos_weight, 2);
		}
		output = output + "  l: " + number_to_string(lambda, 2);
	}
	return output;
}


//**********************************************************************************************************************************


string Ttrain_val_info::displaystring_train_error(unsigned display_mode) const
{
	string output;

	if (display_mode != TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED)
	{
		if (weight_display_mode == DISPLAY_WEIGHTS_AND_ERROR)
		{
			output = output + "   nte: " + pos_number_to_string(neg_train_error, 3);
			output = output + "  pte: " + pos_number_to_string(pos_train_error, 3);
		}
		else
			output = "   te: " + pos_number_to_string(train_error, 3);
	}
	return output;
}


//**********************************************************************************************************************************


string Ttrain_val_info::displaystring_val_error(unsigned display_mode) const
{
	string output;
	
	if (display_mode != TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED)
	{
		if (weight_display_mode == DISPLAY_WEIGHTS_AND_ERROR)
		{
			output = output + "   nve: " + pos_number_to_string(neg_val_error, 3);
			output = output + "  pve: " + pos_number_to_string(pos_val_error, 3);
		}
		else
			output = "  ve: " + pos_number_to_string(val_error, 3);
	}
	return output;
}

//**********************************************************************************************************************************


string Ttrain_val_info::displaystring_kernel(unsigned display_mode) const
{
	string output;

	if (display_mode == TRAIN_INFO_DISPLAY_FORMAT_SHORT)
		output = "   kt: " + pos_number_to_string(train_pre_build_time + train_build_time + train_build_transfer_time + train_kNN_build_time + val_pre_build_time + val_build_time + val_build_transfer_time, 2);
	else
	{
		if (display_mode != TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED)
			output = "   ";
		output = output + "tpt: " + pos_number_to_string(train_pre_build_time, 2);
		output = output + "  tbt: " + pos_number_to_string(train_build_time, 2);
		if (train_build_transfer_time > 0.0)
			output = output + "  ttt: " + pos_number_to_string(train_build_transfer_time, 2);
		output = output + "  tnt: " + pos_number_to_string(train_kNN_build_time, 2);
		
		output = output + "   vpt: " + pos_number_to_string(val_pre_build_time, 2);
		output = output + "  vbt: " + pos_number_to_string(val_build_time, 2);
		if (train_build_transfer_time > 0.0)
			output = output + "  vtt: " + pos_number_to_string(val_build_transfer_time, 2);
	}
	return output;

}



//**********************************************************************************************************************************


string Ttrain_val_info::displaystring_time(unsigned display_mode) const
{
	string output;
	
	if (display_mode == TRAIN_INFO_DISPLAY_FORMAT_SHORT)
		output = "  tvt: " + pos_number_to_string(init_time + train_time + val_time, 3);
	else
	{
		output = "   it: " + pos_number_to_string(init_time, 3);
		output = output + "  tt: " + pos_number_to_string(train_time, 3);
		output = output + "  vt: " + pos_number_to_string(val_time, 3);
	}
	return output;
}

#endif


