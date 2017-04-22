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


#if !defined (TRAIN_VAL_INFO_H)
	#define TRAIN_VAL_INFO_H


 
 
#include "sources/shared/basic_functions/extra_string_functions.h" 
 
#include <cstdio>
#include <string>
using namespace std;


//**********************************************************************************************************************************


const double IGNORE_VALUE = -2.0;
const double NOT_EVALUATED = -1.0;
const double WORST_VALUES = 0.0;


enum DISPLAY_MODE_TYPES {TRAIN_INFO_DISPLAY_FORMAT_SHORT, TRAIN_INFO_DISPLAY_FORMAT_REGULAR, TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED};
enum WEIGHT_DISPLAY_MODE_TYPES {DISPLAY_NO_WEIGHTS, DISPLAY_WEIGHTS_NO_ERROR, DISPLAY_WEIGHTS_AND_ERROR};

extern unsigned weight_display_mode;


//**********************************************************************************************************************************

class Ttrain_val_info
{
	public:
		Ttrain_val_info();
		Ttrain_val_info(double init_type);
		Ttrain_val_info(const Ttrain_val_info& train_val_info);
		
		void read_from_file(FILE* fp);
		void write_to_file(FILE* fp) const;
		void display(unsigned display_mode, unsigned info_level) const;
		double full_kernel_time() const;
		
		void clear();
		void apply_mask(const Ttrain_val_info& mask);

		Ttrain_val_info operator = (const Ttrain_val_info& train_val_info);
		Ttrain_val_info operator + (const Ttrain_val_info& train_val_info);
		friend Ttrain_val_info operator * (double scalar, Ttrain_val_info train_val_info);

		bool operator == (const Ttrain_val_info& train_val_info);
		bool operator < (const Ttrain_val_info& train_val_info);

		double gamma;
		double neg_weight;
		double pos_weight;
		double lambda;
		
		double train_error;
		double neg_train_error;
		double pos_train_error;
		
		double val_error;
		double neg_val_error;
		double pos_val_error;
		
		double train_build_time;
		double train_pre_build_time;
		double train_build_transfer_time;
		double train_kNN_build_time;
		double train_cache_hits;
		double train_pre_cache_hits;
		double val_build_time;
		double val_pre_build_time;
		double val_build_transfer_time;
		
		double init_time;
		double train_time;
		double val_time;

	protected:
		void ignore();
		void copy(const Ttrain_val_info& train_val_info);
		bool equal(double x, double y);
		bool less(double x, double y);
		double add_error(double error1, double error2);
		double multiply_error(double scalar, double error);
		template <typename Template_type> bool apply_mask_entry(Template_type& entry, Template_type mask);
		
		string displaystring(unsigned display_mode) const;
		string displaystring_parameters(unsigned display_mode) const;
		string displaystring_train_error(unsigned display_mode) const;
		string displaystring_val_error(unsigned display_mode) const;
		string displaystring_kernel(unsigned display_mode) const;
		string displaystring_time(unsigned display_mode) const;
};


//**********************************************************************************************************************************


#include "sources/shared/training_validation/train_val_info.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/training_validation/train_val_info.cpp"
#endif

#endif
