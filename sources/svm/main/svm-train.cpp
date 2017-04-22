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



#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/system_support/timing.h"

#include "sources/svm/command_line/command_line_parser_svm_train.h"
#include "sources/svm/training_validation/svm_manager.h"



//**********************************************************************************************************************************

int main(int argc, char** argv)
{
	double full_time;
	double file_time;
	
	Tdataset data_set;
	
	Tcommand_line_parser_svm_train command_line_parser;
	
	Ttrain_control train_control;
	Tsvm_full_train_info svm_full_train_info;
	
	Tsvm_manager svm_manager;


//-------- Read from command line and prepare strucures ----------------------------------------------------------

	full_time = get_wall_time_difference();

	command_line_parser.setup(argc, argv);
	command_line_parser.parse(train_control, true);

	train_control.store_logs_internally = false;
	
	
//-------- Load data --------------------------------------------------------------------------------------------

	file_time = get_process_time_difference();
	data_set.read_from_file(command_line_parser.train_filename);
	file_time = get_process_time_difference(file_time);
	
	
//-------- Train ----------------------------------------------------------------------------------------------

	svm_manager.load(data_set);
	svm_manager.train(train_control, svm_full_train_info);
	
	
//----------------------- Final duties -----------------------------------------------------------------------------

	svm_full_train_info.train_val_info_log.display(TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED, INFO_1);
 	full_time = get_wall_time_difference(full_time);

	flush_info(INFO_1,"\n\n%4.2f seconds used to run svm-train.", full_time);
	flush_info(INFO_1,"\n%4.2f seconds used for training.", svm_full_train_info.train_time);
	flush_info(INFO_1,"\n%4.2f seconds used for file operations.", file_time + svm_full_train_info.file_time);

	command_line_parser.copyright();

	flush_info(INFO_1,"\n\n");
}

