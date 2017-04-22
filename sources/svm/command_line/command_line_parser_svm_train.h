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


#if !defined (COMMAND_LINE_PARSER_SVM_TRAIN_H)
	#define COMMAND_LINE_PARSER_SVM_TRAIN_H


#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/command_line/command_line_parser.h"
#include "sources/shared/training_validation/fold_control.h"
#include "sources/shared/training_validation/grid.h"
#include "sources/shared/training_validation/working_set_control.h"


#include "sources/svm/training_validation/svm_manager.h"



//**********************************************************************************************************************************


class Tcommand_line_parser_svm_train : public Tcommand_line_parser
{
	public:
		Tcommand_line_parser_svm_train();
		void parse(Ttrain_control& train_control, bool read_filenames);

		string train_filename;

	protected:
		void make_consistent(Ttrain_control& train_control);
		virtual void exit_with_help();
		virtual void display_help(unsigned error_code);
		
		
	private:
		bool clipping_set;
		bool init_set;
};


//**********************************************************************************************************************************


#ifndef COMPILE_SEPERATELY__
	#include "sources/svm/command_line/command_line_parser_svm_train.cpp"
#endif

#endif

