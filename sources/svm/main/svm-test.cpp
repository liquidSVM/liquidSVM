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


#include <string.h>
#include <cstdlib>


#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/parallel_control.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_types/dataset_info.h"
#include "sources/shared/training_validation/working_set_manager.h"


#include "sources/svm/training_validation/svm_train_val_info.h"
#include "sources/svm/decision_function/svm_decision_function_manager.h"
#include "sources/svm/command_line/command_line_parser_svm_test.h"




#include "sources/svm/training_validation/svm_manager.h"


//**********************************************************************************************************************************

int main(int argc, char** argv)
{
	Tcommand_line_parser_svm_test command_line_parser;

	Tdataset training_set;
	Tdataset test_set;

	Tdataset_info testset_info;

	Tsvm_full_test_info test_info;
	Tsvm_manager svm_manager;
	
	vector <double> predictions;

	double detection_rate;
	double false_alarm_rate;
	double full_time;

	FILE* fplogwrite;
	FILE* fpresultwrite;

	unsigned i;
	unsigned j;
	unsigned task_offset;


//--------- Read command line ---------------------------------------------------------------------------- 

	full_time = get_wall_time_difference();
	command_line_parser.setup(argc, argv);
	command_line_parser.parse(true);
	
	
//------- Load training set, the solutions & their performance, and the test data ------------------------

	test_info.file_time = get_process_time_difference();
	training_set.read_from_file(command_line_parser.train_filename);
	test_set.read_from_file(command_line_parser.test_filename);
	test_info.file_time = get_process_time_difference(test_info.file_time);
		
	if (training_set.dim() != test_set.dim())
		flush_exit(1, "Training and testing data have different dimensions.");
	
	testset_info = Tdataset_info(test_set, true);
	if ((test_set.is_classification_data() == false) or (testset_info.label_list.size() > 2))
		command_line_parser.display_roc_style_errors = false;

	
//--------- Test the decision functions -------------------------------------------------------------------

	svm_manager.load(training_set);
	svm_manager.test(test_set, command_line_parser.test_control, test_info);

	
//--------- Write obtained test results to file -----------------------------------------------------------

	test_info.file_time = get_process_time_difference(test_info.file_time);
	if (command_line_parser.result_filename != "")
	{
		fpresultwrite = open_file(command_line_parser.result_filename, "w");
		for(i=0; i<test_set.size(); i++)
		{
			predictions = svm_manager.get_predictions_for_test_sample(i);
			
			file_write(fpresultwrite, predictions[0], "%5.7g", "");
			for(j=1; j<predictions.size(); j++)
				file_write(fpresultwrite, predictions[j], ", %5.7g", "");
			file_write_eol(fpresultwrite);
		}
		close_file(fpresultwrite);
	}

	fplogwrite = open_file(command_line_parser.test_control.write_log_test_filename, "a");
	
	
	if (test_info.number_of_tasks == test_info.number_of_all_tasks)
		task_offset = 1;
	else
		task_offset = 0;
	
	for(j=0;j<test_info.train_val_info.size();j++)
	{
		test_info.train_val_info[j].val_time = test_info.test_time;
		test_info.train_val_info[j].write_to_file(fplogwrite);
		
		if (get_filetype(command_line_parser.test_filename) != NLA)
		{
			if (command_line_parser.display_roc_style_errors == false)
				flush_info(INFO_1,"\nTask %d: Test error %1.4f.",   j + task_offset, test_info.train_val_info[j].val_error);
			else
			{
				if (command_line_parser.test_control.vote_control.npl_class == -1)
				{
					detection_rate = 1.0 - test_info.train_val_info[j].pos_val_error;
					false_alarm_rate = test_info.train_val_info[j].neg_val_error;
				}
				else
				{
					detection_rate = 1.0 - test_info.train_val_info[j].neg_val_error;
					false_alarm_rate = test_info.train_val_info[j].pos_val_error;
				}
				flush_info(INFO_1,"\nTask %d: Detection Rate %1.4f.  False Alarm Rate %1.4f.", j + task_offset, detection_rate, false_alarm_rate);
			}
		}
	}
	close_file(fplogwrite);
	test_info.file_time = get_process_time_difference(test_info.file_time);


//----------------------- Final duties -----------------------------------------------------------------------------

	full_time = get_wall_time_difference(full_time);
	flush_info(INFO_1,"\n\n%4.2f seconds used to run svm-test.", full_time);
	flush_info(INFO_1,"\n%4.2f seconds used for testing.", test_info.test_time);
	flush_info(INFO_1,"\n%4.2f seconds used for file operations.", test_info.file_time);

	command_line_parser.copyright();

	flush_info(INFO_1,"\n\n");
}

