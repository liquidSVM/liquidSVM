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


#if !defined (SVM_TRAIN_VAL_INFO_H)
	#define SVM_TRAIN_VAL_INFO_H


 
 
#include "sources/shared/training_validation/train_val_info.h"



//**********************************************************************************************************************************


class Tsvm_train_val_info: public Ttrain_val_info
{
	public:
		Tsvm_train_val_info();
		~Tsvm_train_val_info();
		Tsvm_train_val_info(double init);
		Tsvm_train_val_info(const Ttrain_val_info& train_val_info);
		Tsvm_train_val_info(const Tsvm_train_val_info& train_val_svm_info);
		
		void read_from_file(FILE* fp);
		void write_to_file(FILE* fp) const;
		void display(unsigned mode, unsigned info_level) const;

		void clear();
		void apply_mask(const Tsvm_train_val_info& mask);

		Tsvm_train_val_info operator = (const Tsvm_train_val_info& train_val_svm_info);
		Tsvm_train_val_info operator + (const Tsvm_train_val_info& train_val_svm_info);
		friend Tsvm_train_val_info operator * (double scalar, Tsvm_train_val_info train_val_info);

		bool operator == (const Tsvm_train_val_info& train_val_svm_info);
		bool operator < (const Tsvm_train_val_info& train_val_svm_info);

		
		unsigned init_iterations;
		int train_iterations;
		unsigned gradient_updates;
		unsigned val_iterations;
		
		int SVs;
		int bSVs;

		unsigned tries_2D;
		unsigned hits_2D;
		int tries_4D;
		unsigned hits_4D;
		unsigned train_iterations_4D;
		
		double sync_time;
		double inner_loop_time;
		double opt_4D_time;

	private:
		void ignore();
		void copy(const Tsvm_train_val_info& train_val_svm_info);
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/training_validation/svm_train_val_info.cpp"
#endif




#endif
