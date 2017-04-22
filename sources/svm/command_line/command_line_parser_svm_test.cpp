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


#if !defined (COMMAND_LINE_PARSER_SVM_TEST_CPP)
	#define COMMAND_LINE_PARSER_SVM_TEST_CPP


#include "sources/svm/command_line/command_line_parser_svm_test.h"


#include "sources/shared/basic_functions/basic_file_functions.h"



#include <string.h>

//**********************************************************************************************************************************


unsigned const ERROR_clp_svm_test_v = 71;
unsigned const ERROR_clp_svm_test_o = 72;


//**********************************************************************************************************************************



Tcommand_line_parser_svm_test::Tcommand_line_parser_svm_test()
{
	result_flag = true;
	display_roc_style_errors = false;
	command_name = "svm-test";
};



//**********************************************************************************************************************************


void Tcommand_line_parser_svm_test::display_help(unsigned error_code)
{
	Tvote_control vote_control;
	
	Tcommand_line_parser::display_help(error_code);
	
	if (error_code == ERROR_clp_svm_test_v)
	{
		display_separator("-v <weighted> <scenario> [<npl_class>]");
		flush_info(INFO_1, 
		"Sets the weighted vote method to combine decision functions from different\n"
		"folds. If <weighted> = 1, then weights are computed with the help of the\n"
		"validation error, otherwise, equal weights are used. In the classification\n"
		"scenario, the decision function values are first transformed to -1 and +1,\n"
		"before a weighted vote is performed, in the regression scenario, the bare\n"
		"function values are used in the vote. In the weighted NPL scenario, the weights\n"
		"are computed according to the validation error on the samples with label\n"
		"<npl_class>, the rest is like in the classification scenario.\n"
		"<npl_class> can only be set for the NPL scenario.\n");
		display_specifics();
		flush_info(INFO_1, 
		"<scenario> = %d  =>   classification\n"
		"<scenario> = %d  =>   regression\n"
		"<scenario> = %d  =>   NPL\n", VOTE_CLASSIFICATION, VOTE_REGRESSION, VOTE_NPL);
		display_ranges();
		flush_info(INFO_1, "<weighted>: 0 or 1\n");
		flush_info(INFO_1, "<scenario>: integer between %d and %d\n", 0, VOTE_SCENARIOS_MAX-1);
		flush_info(INFO_1, "<npl_class>: -1 or 1\n");
		display_defaults();
		flush_info(INFO_1, "<weighted> = %d\n", vote_control.weighted_folds);
		flush_info(INFO_1, "<scenario> = %d\n", vote_control.scenario);
		flush_info(INFO_1, "<npl_class> = %d\n", vote_control.npl_class);
	}
	
	if (error_code == ERROR_clp_svm_test_v)
	{
		display_separator("-o <display_roc_style>");
		flush_info(INFO_1, 
		"Sets a flag that decides, wheather classification errors are displayed by\n"
		"true positive and false positives.\n");
		display_ranges();
		flush_info(INFO_1, "<display_roc_style>: 0 or 1\n");
		display_defaults();
		flush_info(INFO_1, "<display_roc_style>: Depends on option -v\n");
	}
}


//**********************************************************************************************************************************


void Tcommand_line_parser_svm_test::exit_with_help()
{
	flush_info(INFO_SILENCE, 
	"\n\nsvm-test [options] <trainfile> <solfile> <testfile> <logfile> [<resultfile>] [<summary_log_file>]\n"
	"\nsvm-test reads the SVM decision functions produced by svm-select from <solfile>\n"
	"and their support vectors from <trainfile>. For each task recorded in <solfile>\n"
	"it then produces a weighted predictor generated from the decision functions of\n"
	"the task. These predictors are applied to the samples of <testfile>. Their\n"
	"performance is recorded in <logfile> and their predictions are saved in the\n"
	"optional <resultfile>.\n"
	"\nAllowed extensions:\n"
	"<trainfile>:  .csv, .lsv, and .uci\n"
	"<solfile>:    .sol\n"
	"<logfile>:    .log\n"
	"<testfile>:   .csv, .lsv, .uci, and .nla\n"
	"<resultfile>: unspecified\n");

	if (full_help == false)
		flush_info(INFO_SILENCE, "\nOptions:");
	display_help(ERROR_clp_gen_d);
	display_help(ERROR_clp_gen_GPU);
	display_help(ERROR_clp_gen_h);
	display_help(ERROR_clp_svm_test_o);
	display_help(ERROR_clp_gen_L);
	display_help(ERROR_clp_gen_T);
	display_help(ERROR_clp_svm_test_v);

	flush_info(INFO_SILENCE,"\n\n");
	copyright();
	flush_exit(ERROR_SILENT, "");
}


//**********************************************************************************************************************************


void Tcommand_line_parser_svm_test::make_consistent()
{
	Tcommand_line_parser::make_consistent();
	
	test_control.loss_control = get_loss_control();
	test_control.parallel_control = get_parallel_control();
	test_control.vote_control.loss_weights_are_set = not loss_weights_are_set();
	
	if (test_control.vote_control.scenario == VOTE_NPL)
		display_roc_style_errors = true;
	if (test_control.vote_control.scenario == VOTE_REGRESSION)
		display_roc_style_errors = false;
};

//**********************************************************************************************************************************


void Tcommand_line_parser_svm_test::parse(bool read_filenames)
{
	bool v_is_set;
	
	v_is_set = false;
	check_parameter_list_size();
	for(current_position=1; current_position<parameter_list_size; current_position++)
		if (Tcommand_line_parser::parse("-d-h-GPU-L-T") == false)
		{
			if (parameter_list[current_position][0] != '-') 
				break;
			if (string(parameter_list[current_position]).size() > 2)
				Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			
			switch(parameter_list[current_position][1])
			{
				case 'o':
					display_roc_style_errors = get_next_bool(ERROR_clp_svm_test_o);
					if ((v_is_set == false) or (test_control.vote_control.scenario != VOTE_NPL))
						test_control.vote_control.npl_class = -1;
					break;
				case 'v':
					test_control.vote_control.weighted_folds = get_next_bool(ERROR_clp_svm_test_v);
					test_control.vote_control.scenario = get_next_enum(ERROR_clp_svm_test_v, 0, VOTE_SCENARIOS_MAX-1);
					if ((test_control.vote_control.scenario == VOTE_NPL) and (next_parameter_is_number() == true))
						test_control.vote_control.npl_class = get_next_class(ERROR_clp_svm_test_v);
					v_is_set = true;
					break;
				default:
					Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			}
		}
	
	// Read filenames
	
	if (read_filenames == true)
	{
		train_filename = get_next_labeled_data_filename(ERROR_clp_gen_missing_train_file_name);
		test_control.read_sol_select_filename = get_next_solution_filename(ERROR_clp_gen_missing_sol_file_name);
		test_filename = get_next_data_filename(ERROR_clp_gen_missing_test_file_name);	
		test_control.write_log_test_filename = get_next_log_filename(ERROR_clp_gen_missing_log_file_name);

		if (current_position < parameter_list_size)
			result_filename = get_next_filename();
		else 
			result_flag = false;
		
		if (current_position < parameter_list_size)
			test_control.summary_log_filename = get_next_log_filename();
	}
		
	make_consistent();
};

#endif




