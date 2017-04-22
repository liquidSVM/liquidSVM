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


#if !defined (SVM_TRAIN_VAL_INFO_CPP)
	#define SVM_TRAIN_VAL_INFO_CPP




#include "sources/svm/training_validation/svm_train_val_info.h"


#include <limits>


#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_functions/extra_string_functions.h"
#include "sources/shared/basic_functions/flush_print.h"


//**********************************************************************************************************************************

Tsvm_train_val_info::Tsvm_train_val_info()
{
	lambda = 1.0;
	gamma = 1.0;
	neg_weight = 1.0;
	pos_weight = 1.0;
	
	train_iterations = 0;
	gradient_updates = 0;
	tries_2D = 0;
	tries_4D = 0;
	hits_2D = 0;
	hits_4D = 0;
	
	sync_time = 0.0;
	inner_loop_time = 0.0;
	opt_4D_time = 0.0;

	clear();
};

//**********************************************************************************************************************************

Tsvm_train_val_info::~Tsvm_train_val_info()
{
	flush_info(INFO_VERY_PEDANTIC_DEBUG, "\nDestroying an object of type Tsvm_train_val_info.");
} 

//**********************************************************************************************************************************

Tsvm_train_val_info::Tsvm_train_val_info(double init)
{
	clear();

	if (init == IGNORE_VALUE)
		ignore();
	else if (init == WORST_VALUES)
	{
		val_error = numeric_limits<double>::max( );
		pos_val_error = numeric_limits<double>::max( );
		neg_val_error = numeric_limits<double>::max( );

		SVs = numeric_limits<int>::max( );
		train_iterations = numeric_limits<int>::max( );
	}
};


//**********************************************************************************************************************************

Tsvm_train_val_info::Tsvm_train_val_info(const Ttrain_val_info& train_val_info)
{
	clear();
	Ttrain_val_info::copy(train_val_info);
};


//**********************************************************************************************************************************

Tsvm_train_val_info::Tsvm_train_val_info(const Tsvm_train_val_info& train_val_svm_info)
{
	copy(train_val_svm_info);
};


//**********************************************************************************************************************************

void Tsvm_train_val_info::display(unsigned display_mode, unsigned info_level) const
{
	string output;


 	if ((info_mode < INFO_3) and (display_mode == TRAIN_INFO_DISPLAY_FORMAT_REGULAR))
		display_mode = TRAIN_INFO_DISPLAY_FORMAT_SHORT;

	output = displaystring(display_mode);
	if ((display_mode == TRAIN_INFO_DISPLAY_FORMAT_REGULAR) or (display_mode == TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED))
	{
		output = output + "   ii: " + pos_number_to_string(init_iterations, 7);
		output = output + "  ti: " + pos_number_to_string(train_iterations, 7);
		output = output + "  tu: " + pos_number_to_string(gradient_updates, 7);
		output = output + "  vi: " + pos_number_to_string(val_iterations, 5);
		if (display_mode == TRAIN_INFO_DISPLAY_FORMAT_REGULAR)
		{
			output = output + "   SV: " + pos_number_to_string(SVs, 5);
			if (bSVs >= 0)
				output = output + "  bSV: " + pos_number_to_string(bSVs, 5);
		}
		output = output + "   h2D: " + pos_number_to_string(double(hits_2D)/double(tries_2D), 3);
		if (tries_4D >= 0)
		{
			output = output + "  h4D: " + pos_number_to_string(double(hits_4D)/double(tries_4D), 3);
			output = output + "  ti4D: " + pos_number_to_string(train_iterations_4D, 7);
		}
		
		if  (sync_time > 0.0)
		{
			output = output + "  syt: " + pos_number_to_string(sync_time, 4);
	    	output = output + "  ilt: " + pos_number_to_string(inner_loop_time, 4);
	       	output = output + "  opt: " + pos_number_to_string(opt_4D_time, 4);
		}
	}

	flush_info(info_level, "\n");
	flush_info(info_level, output.c_str());
}


//**********************************************************************************************************************************

void Tsvm_train_val_info::read_from_file(FILE* fp)
{
	Ttrain_val_info::read_from_file(fp);

	file_read(fp, init_iterations);
	file_read(fp, train_iterations);
	file_read(fp, gradient_updates);
	file_read(fp, val_iterations);
	file_read(fp, SVs);
	file_read(fp, bSVs);
	file_read(fp, tries_2D);
	file_read(fp, hits_2D);
	file_read(fp, tries_4D);
	file_read(fp, hits_4D);
	file_read(fp, train_iterations_4D);
	
	file_read(fp, sync_time);
	file_read(fp, inner_loop_time);
	file_read(fp, opt_4D_time);
}



//**********************************************************************************************************************************

void Tsvm_train_val_info::write_to_file(FILE* fp) const
{
	Ttrain_val_info::write_to_file(fp);

	file_write(fp, init_iterations);
	file_write(fp, train_iterations);
	file_write(fp, gradient_updates);
	file_write(fp, val_iterations);
	file_write(fp, SVs);
	file_write(fp, bSVs);
	file_write(fp, tries_2D);
	file_write(fp, hits_2D);
	file_write(fp, tries_4D);
	file_write(fp, hits_4D);
	file_write(fp, train_iterations_4D);
	
	file_write(fp, sync_time);
	file_write(fp, inner_loop_time);
	file_write(fp, opt_4D_time);
	
	file_write_eol(fp);
}



//**********************************************************************************************************************************

void Tsvm_train_val_info::clear()
{
	Ttrain_val_info::clear();

	init_iterations = 0;
	train_iterations = 0;
	gradient_updates = 0;
	val_iterations = 0;

	SVs = 0;
	bSVs = -1;

	tries_2D = 0;
	hits_2D = 0;
	tries_4D = -1;
	hits_4D = 0;
	train_iterations_4D = 0;
}


//**********************************************************************************************************************************

void Tsvm_train_val_info::apply_mask(const Tsvm_train_val_info& mask)
{
	bool result;

	result = apply_mask_entry(val_error, mask.val_error);
	result = result or apply_mask_entry(pos_val_error, mask.pos_val_error);
	result = result or apply_mask_entry(neg_val_error, mask.neg_val_error);
	result = result or apply_mask_entry(SVs, mask.SVs);
	result = result or apply_mask_entry(train_iterations, mask.train_iterations);

	if (result == true)
		ignore();
}


//**********************************************************************************************************************************


Tsvm_train_val_info Tsvm_train_val_info::operator = (const Tsvm_train_val_info& train_val_svm_info)
{
	copy(train_val_svm_info);
	return *this;
}

//**********************************************************************************************************************************


Tsvm_train_val_info Tsvm_train_val_info::operator + (const Tsvm_train_val_info& train_val_svm_info)
{
	Ttrain_val_info result_base;
	Tsvm_train_val_info result;

	result_base = *this;
	result_base = result_base + train_val_svm_info;
	result = Tsvm_train_val_info(result_base);

	result.init_iterations = init_iterations + train_val_svm_info.init_iterations;
	result.train_iterations = train_iterations + train_val_svm_info.train_iterations;
	result.gradient_updates = gradient_updates + train_val_svm_info.gradient_updates;
	result.val_iterations = val_iterations + train_val_svm_info.val_iterations;

	result.SVs = SVs + train_val_svm_info.SVs;
	result.bSVs = bSVs + train_val_svm_info.bSVs;

	result.tries_2D = tries_2D + train_val_svm_info.tries_2D;
	result.hits_2D = hits_2D + train_val_svm_info.hits_2D;
	result.tries_4D = tries_4D + train_val_svm_info.tries_4D;
	result.hits_4D = hits_4D + train_val_svm_info.hits_4D;
	result.train_iterations_4D = train_iterations_4D + train_val_svm_info.train_iterations_4D;
	
	result.sync_time = sync_time + train_val_svm_info.sync_time;
	result.inner_loop_time = inner_loop_time + train_val_svm_info.inner_loop_time;
	result.opt_4D_time = opt_4D_time + train_val_svm_info.opt_4D_time;

	return Tsvm_train_val_info(result);
}


//**********************************************************************************************************************************

Tsvm_train_val_info operator * (double scalar, Tsvm_train_val_info train_val_info)
{
	Ttrain_val_info result_base;
	Tsvm_train_val_info result;

	result_base = train_val_info;
	result_base = scalar * result_base;
	result = Tsvm_train_val_info(result_base);

	result.init_iterations = int(scalar * double(train_val_info.init_iterations));
	result.train_iterations = int(scalar * double(train_val_info.train_iterations));
	result.gradient_updates = int(scalar * double(train_val_info.gradient_updates));
	result.val_iterations = int(scalar * double(train_val_info.val_iterations));

	result.SVs = int(scalar * double(train_val_info.SVs));
	result.bSVs = int(scalar * double(train_val_info.bSVs));

	result.tries_2D = int(scalar * double(train_val_info.tries_2D));
	result.hits_2D = int(scalar * double(train_val_info.hits_2D));
	result.tries_4D = int(scalar * double(train_val_info.tries_4D));
	result.hits_4D = int(scalar * double(train_val_info.hits_4D));
	result.train_iterations_4D = int(scalar * double(train_val_info.train_iterations_4D));

	result.sync_time = scalar * train_val_info.sync_time;
	result.inner_loop_time = scalar * train_val_info.inner_loop_time;
	result.opt_4D_time = scalar * train_val_info.opt_4D_time;
	
	return Tsvm_train_val_info(result);
}


//**********************************************************************************************************************************

bool Tsvm_train_val_info::operator == (const Tsvm_train_val_info& train_val_svm_info)
{
	bool result;

	result = equal(val_error, train_val_svm_info.val_error);
	result = result and equal(pos_val_error, train_val_svm_info.pos_val_error);
	result = result and equal(neg_val_error, train_val_svm_info.neg_val_error);
	result = result and equal(SVs, train_val_svm_info.SVs);
	result = result and equal(train_iterations, train_val_svm_info.train_iterations);

	return result;
}


//**********************************************************************************************************************************


bool Tsvm_train_val_info::operator < (const Tsvm_train_val_info& train_val_svm_info)
{
	if (not equal(val_error, train_val_svm_info.val_error))
		return less(val_error, train_val_svm_info.val_error);

	if (not equal(pos_val_error, train_val_svm_info.pos_val_error))
		return less(pos_val_error, train_val_svm_info.pos_val_error);

	if (not equal(neg_val_error, train_val_svm_info.neg_val_error))
		return less(neg_val_error, train_val_svm_info.neg_val_error);

	if (not equal(SVs, train_val_svm_info.SVs))
		return less(SVs, train_val_svm_info.SVs);

	if (not equal(train_iterations, train_val_svm_info.train_iterations))
		return less(train_iterations, train_val_svm_info.train_iterations);
	else return false;
}



//**********************************************************************************************************************************

void Tsvm_train_val_info::ignore()
{
	Ttrain_val_info::ignore();

	SVs = int(IGNORE_VALUE);
	train_iterations = int(IGNORE_VALUE);
}

//**********************************************************************************************************************************

void Tsvm_train_val_info::copy(const Tsvm_train_val_info& train_val_svm_info)
{
	Ttrain_val_info::copy(train_val_svm_info);

	init_iterations = train_val_svm_info.init_iterations;
	train_iterations = train_val_svm_info.train_iterations;
	gradient_updates = train_val_svm_info.gradient_updates;
	val_iterations = train_val_svm_info.val_iterations;

	SVs = train_val_svm_info.SVs;
	bSVs = train_val_svm_info.bSVs;

	tries_2D = train_val_svm_info.tries_2D;
	hits_2D = train_val_svm_info.hits_2D;
	tries_4D = train_val_svm_info.tries_4D;
	hits_4D = train_val_svm_info.hits_4D;
	train_iterations_4D = train_val_svm_info.train_iterations_4D;
	
	
	sync_time = train_val_svm_info.sync_time;
	inner_loop_time = train_val_svm_info.inner_loop_time;
	opt_4D_time = train_val_svm_info.opt_4D_time;
};

#endif




