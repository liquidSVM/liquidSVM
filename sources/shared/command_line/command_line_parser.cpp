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


#if !defined (COMMAND_LINE_PARSER_CPP)
	#define COMMAND_LINE_PARSER_CPP


#include "sources/shared/command_line/command_line_parser.h"



#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/system_support/thread_manager.h"


#include <cstdio>
#include <cstdlib>
#include <stdarg.h>


#ifdef  COMPILE_WITH_CUDA__
	#include <cuda_runtime.h>
#endif


//**********************************************************************************************************************************


Tcommand_line_parser::Tcommand_line_parser()
{
	random_seed = -1;
	
	full_help = false;
	loss_set = false;
	loss_weights_set = false;
}

//**********************************************************************************************************************************


void Tcommand_line_parser::copyright() const
{
	#ifdef __DEMOSVM__
		flush_info("\n\nCopyright 2015, 2016, 2017 Ingo Steinwart\n\n");
		flush_info("%s is part of liquidSVM.\n\n", command_name.c_str());
		flush_info("liquidSVM is free software: you can redistribute it\n"
						"and/or modify it under the terms of the GNU Affero\n"
						"General Public License as published by the Free Software\n"
						"Foundation, either version 3 of the License, or (at your\n"
						"option) any later version.\n\n"

						"liquidSVM is distributed in the hope that it will be\n"
						"useful, but WITHOUT ANY WARRANTY; without even the implied\n"
						"warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR\n"
						"PURPOSE. See the GNU Affero General Public License for more\n"
						"details.\n\n"

						"You should have received a copy of the GNU Affero General\n"
						"Public License along with liquidSVM. If not, see\n"
						"<http://www.gnu.org/licenses/>.\n\n");
	#endif
}


//**********************************************************************************************************************************


void Tcommand_line_parser::demoversion() const
{
	flush_info("\n\nI am sorry, you are using a demo version in which some\n"
	"of the chosen options are disabled. Good-Bye!\n\n");

	copyright();
	flush_exit(ERROR_COMMAND_LINE, "");
}


//**********************************************************************************************************************************


void Tcommand_line_parser::setup(int argc, char** argv)
{
	parameter_list = argv;
	parameter_list_size = argc;
}


//**********************************************************************************************************************************

bool  Tcommand_line_parser::loss_weights_are_set() const
{
	return loss_weights_set;
}

//**********************************************************************************************************************************


int Tcommand_line_parser::get_random_seed() const
{
	return random_seed;
}

//**********************************************************************************************************************************


Tloss_control Tcommand_line_parser::get_loss_control() const
{
	return loss_ctrl;
}

//**********************************************************************************************************************************


Tparallel_control Tcommand_line_parser::get_parallel_control() const
{
	return parallel_ctrl;
}



//**********************************************************************************************************************************


void Tcommand_line_parser::exit_with_help(unsigned error_code)
{
	unsigned i;
	unsigned option_position;
	
	info_mode = INFO_1;
	
	if (current_position < parameter_list_size)
	{	
		if (error_code != ERROR_clp_gen_unknown_option)
			option_position = get_current_option_position();
		else
			option_position = current_position;
		
		flush_info("\n\nThe command line parser of %s detected a problem with the following\noption:\n\n", command_name.c_str());
		for(i=max(unsigned(1), option_position); i<current_position; i++)
			flush_info("%s ", parameter_list[i]);
		if (parameter_is_option(current_position) == false)
			flush_info("%s ", parameter_list[current_position]);
		flush_info("\n");
		
		if ((error_code != ERROR_clp_gen_unknown) and (error_code != ERROR_clp_gen_unknown_option))
			flush_info("\nThe correct usage of this option is:\n");
	}
	
	this->display_help(error_code);
	flush_info("\n\n");
	flush_exit(ERROR_SILENT, "");
}


//**********************************************************************************************************************************


void Tcommand_line_parser::exit_with_help_for_inconsistent_values(unsigned error_code1, unsigned error_code2)
{
	info_mode = INFO_1;
	
	flush_info("\n\nThe command line consistency check of %s detected a problem with\ninconsistent values for the following options:\n\n", command_name.c_str());
	
	Tcommand_line_parser::display_help(error_code1);
	display_help(error_code1);
	
	Tcommand_line_parser::display_help(error_code2);
	display_help(error_code2);
	
	flush_info("\n\n");
	flush_exit(ERROR_COMMAND_LINE, "");
}

//**********************************************************************************************************************************


void Tcommand_line_parser::display_separator(string option)
{
	if (full_help == true)
		flush_info(INFO_1, "\n--------------------------------------------------------------------------------\n");
	flush_info(INFO_SILENCE, "\n%s", option.c_str());

	flush_info(INFO_1, "\n\n");
}


//**********************************************************************************************************************************


void Tcommand_line_parser::display_separator(string option, const char* message_format,...)
{
	if (full_help == true)
		flush_info(INFO_1, "\n--------------------------------------------------------------------------------\n");
	
	if (flush_info_will_show(INFO_1) == false)
		flush_info(INFO_SILENCE, "\n%s", option.c_str());
	else
	{
		flush_info(INFO_SILENCE, "\n");
		VPRINTF(message_format);
	}	
	flush_info(INFO_1, "\n\n");
}

//**********************************************************************************************************************************


void Tcommand_line_parser::display_specifics()
{
	flush_info(INFO_1, "\nMeaning of specific values:\n");
}

//**********************************************************************************************************************************


void Tcommand_line_parser::display_ranges()
{
	flush_info(INFO_1, "\nAllowed values:\n");
}

//**********************************************************************************************************************************


void Tcommand_line_parser::display_defaults()
{
	flush_info(INFO_1, "\nDefault values:\n");
}

//**********************************************************************************************************************************


void Tcommand_line_parser::display_help(unsigned error_code)
{
	Tloss_control loss_control;
	Tthread_manager_base thread_manager;
	
	
	if (error_code == ERROR_clp_gen_unknown)
		flush_info(INFO_SILENCE, "\n"
		"An unknown error occurred while reading the %d-th token. Use option -h\n"
		"to analyze manually!\n", current_position);
	
	if (error_code == ERROR_clp_gen_unknown_option)
		flush_info(INFO_SILENCE, "\n"
		"The option %s does not exist. Use option -h to see all available options.\n", parameter_list[current_position]);
	
	if (error_code == ERROR_clp_gen_d)
	{
		display_separator("-d <level>");
		flush_info(INFO_1, 
		"Controls the amount of information displayed, where larger values lead to more\n"
		"information.\n");
		display_ranges();
		flush_info(INFO_1, "<level>: integer between %d and %d\n", INFO_SILENCE, INFO_LEVELS_MAX - 1);
		display_defaults();
		flush_info(INFO_1, "<level> = 1\n");
	}
	
	if (error_code == ERROR_clp_gen_GPU)
	{
		display_separator("-GPU <use_gpus> [<GPU_offset>]");
		flush_info(INFO_1, 
		"Flag controlling whether the GPU support is used. If <use_gpus> = 1, then each\n"
		"CPU thread gets a thread on a GPU. In the case of multiple GPUs, these threads\n"
		"are uniformly distributed among the available GPUs. The optional <GPU_offset>\n"
		"is added to the CPU thread number before the GPU is added before distributing\n"
		"the threads to the GPUs. This makes it possible to avoid that two or more\n"
		"independent processes use the same GPU, if more than one GPU is available.\n"
		);
		display_ranges();
		flush_info(INFO_1, "<use_gpus>:   bool\n");
		flush_info(INFO_1, "<use_offset>: non-negative integer.\n");
		display_defaults();
		flush_info(INFO_1, "<gpus>       = 0\n");
		flush_info(INFO_1, "<gpu_offset> = 0\n");

		#ifndef COMPILE_WITH_CUDA__
			flush_info(INFO_1, "\nUnfortunately, this option is not activated for the binaries you are currently\n"
			"using. Install CUDA and recompile to activate this option.\n");
		#endif
	}
	
	if (error_code == ERROR_clp_gen_h)
	{
		display_separator("-h [<level>]");
		flush_info(INFO_1, "Displays all help messages.\n");
		display_specifics();
		flush_info(INFO_1, "<level> = 0  =>  short help messages\n");
		flush_info(INFO_1, "<level> = 1  =>  detailed help messages\n");
		display_ranges();
		flush_info(INFO_1, "<level>: 0 or 1\n", numeric_limits<int>::max());
		display_defaults();
		flush_info(INFO_1, "<level> = 0\n");
	}
	
	// 		CHANGE_FOR_OWN_SOLVER
	if (error_code == ERROR_clp_gen_L)
	{
		display_separator("-L <loss> [<neg_weight> <pos_weight>]");
		flush_info(INFO_1, 
		"Sets the loss that is used to compute empirical errors. The optional weights can\n"
		"only be set, if <loss> specifies a loss that has weights.\n");
		display_specifics();
		flush_info(INFO_1, 
		"<loss> = %d  =>   binary classification loss\n"
		"<loss> = %d  =>   multiclass class\n"
		"<loss> = %d  =>   least squares loss\n"
		"<loss> = %d  =>   weighted least squares loss\n"
		"<loss> = %d  =>   your own template loss\n", CLASSIFICATION_LOSS, MULTI_CLASS_LOSS, LEAST_SQUARES_LOSS, WEIGHTED_LEAST_SQUARES_LOSS, TEMPLATE_LOSS);
		display_ranges();
		flush_info(INFO_1, "<loss>: integer between %d and %d\n", CLASSIFICATION_LOSS, LEAST_SQUARES_LOSS);
		flush_info(INFO_1, "<neg_weight>: float > 0.0\n");
		flush_info(INFO_1, "<pos_weight>: float > 0.0\n");
		display_defaults();
		flush_info(INFO_1, "<loss> = %d\n", loss_control.type);
		flush_info(INFO_1, "<neg_weight> = %1.1f\n", loss_control.neg_weight);
		flush_info(INFO_1, "<pos_weight> = %1.1f\n", loss_control.pos_weight);
	}
	
	if (error_code == ERROR_clp_gen_r)
	{
		display_separator("-r <seed>");
		flush_info(INFO_1, "Initializes the random number generator with <seed>.\n");
		display_specifics();
		flush_info(INFO_1, "<seed> = -1  =>  a random seed based on the internal timer is used\n");
		display_ranges();
		flush_info(INFO_1, "<seed>: integer between -1 and %d\n", numeric_limits<int>::max());
		display_defaults();
		flush_info(INFO_1, "<seed> = -1\n");
	}
	
	if (error_code == ERROR_clp_gen_T)
	{
		display_separator("-T <threads> [<thread_id_offset>]");
		flush_info(INFO_1, "Sets the number of threads that are going to be used. Each thread is\n"
		"assigned to a logical processor on the system, so that the number of\n"
		"allowed threads is bounded by the number of logical processors. On\n"
		"systems with activated hyperthreading each physical core runs one thread,\n");
		flush_info(INFO_1, "if <threads> does not exceed the number of physical cores. Since hyper-\n"
		"threads on the same core share resources, using more threads than cores\n"
		"does usually not increase the performance significantly, and may even\n"
		"decrease it. The optional <thread_id_offset> is added before distributing\n"
		"the threads to the cores. This makes it possible to avoid that two or more\n"
		"independent processes use the same physical cores.\n"
		"Example: To run 2 processes with 3 threads each on a 6-core system call\n"
		"the first process with -T 3 0 and the second one with -T 3 3 .\n"
		);
		display_specifics();
		flush_info(INFO_1, 
		"<threads> =  0   =>   %d threads are used (all physical cores run one thread)\n"
		"<threads> = -1   =>   %d threads are used (all but one of the physical cores\n"
		"                                          run one thread)\n",
				 thread_manager.get_number_of_physical_cores(), thread_manager.get_number_of_physical_cores() - 1);
		display_ranges();
		flush_info(INFO_1, "<threads>:          integer between -1 and %d\n", thread_manager.get_number_of_logical_processors());
		flush_info(INFO_1, "<thread_id_offset>: integer between  0 and %d\n", thread_manager.get_number_of_logical_processors());
		display_defaults();
		flush_info(INFO_1, "<threads>          = 0\n");
		flush_info(INFO_1, "<thread_id_offset> = 0\n");
	}
	
	if ((error_code >= ERROR_clp_gen_missing_data_file_name) and (error_code <=  ERROR_clp_gen_missing_sol_file_name))
	{
		flush_info("\n\nThe command line parser of %s detected the following problem:\n", command_name.c_str());

		if (error_code == ERROR_clp_gen_missing_data_file_name)
			flush_info(INFO_SILENCE, "\nMissing filename for data set.\n");
		
		if (error_code == ERROR_clp_gen_missing_train_file_name)
			flush_info(INFO_SILENCE, "\nMissing filename for training data set.\n");
		
		if (error_code == ERROR_clp_gen_missing_test_file_name)
			flush_info(INFO_SILENCE, "\nMissing filename for test data set.\n");
		
		if (error_code == ERROR_clp_gen_missing_log_file_name)
			flush_info(INFO_SILENCE, "\nMissing filename for log file.\n");
		
		if (error_code == ERROR_clp_gen_missing_sol_file_name)
			flush_info(INFO_SILENCE, "\nMissing filename for solution file.\n");
	}
}


//**********************************************************************************************************************************


void Tcommand_line_parser::make_consistent()
{
	#ifndef COMPILE_WITH_CUDA__
		parallel_ctrl.GPUs = 0;
	#endif
}

//**********************************************************************************************************************************


bool Tcommand_line_parser::parse(string activated_options)
{
	string current_token;
	Tthread_manager_base thread_manager;

	
	activated_options = activated_options + '-';
	current_token = string(parameter_list[current_position]);
	
	if (activated_options.find(current_token + '-') != string::npos)
	{
		if (current_token == "-d")
			info_mode = get_next_enum(ERROR_clp_gen_d, INFO_SILENCE, INFO_LEVELS_MAX - 1);
		
		if (current_token == "-h")
		{
			if (next_parameter_is_number() == true)
				info_mode = get_next_enum(ERROR_clp_gen_h, INFO_SILENCE, INFO_1);
			else
				info_mode = INFO_SILENCE;
			if (info_mode == INFO_1)
				full_help = true;
			this->exit_with_help();
		}
		
		if (current_token == "-GPU")
		{
			parallel_ctrl.GPUs = get_next_number(ERROR_clp_gen_GPU, 0, 1);
			if (next_parameter_is_number() == true)
				parallel_ctrl.GPU_number_offset = get_next_number(ERROR_clp_gen_GPU, 0);
		}
		
		if (current_token == "-L")
		{
			loss_ctrl.type = get_next_enum(ERROR_clp_gen_L, CLASSIFICATION_LOSS, LOSS_TYPES_MAX-1);
			if (((loss_ctrl.type == CLASSIFICATION_LOSS) or (loss_ctrl.type == WEIGHTED_LEAST_SQUARES_LOSS) or (loss_ctrl.type == PINBALL_LOSS)) and (next_parameter_is_number() == true))
			{
				loss_ctrl.neg_weight = get_next_number(ERROR_clp_gen_L, 0.0);
				loss_ctrl.pos_weight = get_next_number(ERROR_clp_gen_L, 0.0);
				loss_weights_set = true;
			}
			loss_set = true;
		}
		
		if (current_token == "-r")
			random_seed = get_next_number(ERROR_clp_gen_r, -1);
		
		if (current_token == "-T") 
		{
			parallel_ctrl.requested_team_size = get_next_number(ERROR_clp_gen_T, -1, int(thread_manager.get_number_of_logical_processors()));
			if (next_parameter_is_number() == true)
				parallel_ctrl.core_number_offset = get_next_number(ERROR_clp_gen_T, 0, int(thread_manager.get_number_of_logical_processors()));
		}
		
		return true;
	}	
	else
		return false;
}


//**********************************************************************************************************************************


void Tcommand_line_parser::check_parameter_list_size()
{
	if (parameter_list_size == 1)
	{
		info_mode = INFO_SILENCE;
		exit_with_help();
	}
}


//**********************************************************************************************************************************

void Tcommand_line_parser::check_parameter_position(unsigned error_code)
{
	current_position++;
	if (current_position >= parameter_list_size)
		this->exit_with_help(error_code);
}

//**********************************************************************************************************************************


bool Tcommand_line_parser::get_next_bool(unsigned error_code)
{
	return (get_next_number(error_code, 0, 1) > 0);
}



//**********************************************************************************************************************************


bool Tcommand_line_parser::next_parameter_equals(char character)
{
	bool result;
	
	current_position++;
	if (current_position < parameter_list_size)
		result = (parameter_list[current_position][0] == character);
	else
		result = false;
	current_position--;
	
	return result;
}


//**********************************************************************************************************************************


unsigned Tcommand_line_parser::get_next_class(unsigned error_code)
{
	unsigned label;
	
	label = get_next_number(error_code, -1, 1);
	if (label == 0)
		this->exit_with_help(error_code);
	
	return label;
}


//**********************************************************************************************************************************


unsigned Tcommand_line_parser::get_next_enum(unsigned error_code, unsigned min, unsigned max)
{
	return get_next_number(error_code, min, max);
}



//**********************************************************************************************************************************


string Tcommand_line_parser::get_next_string(unsigned error_code)
{
	string return_string;
	
	check_parameter_position(error_code);	
	return_string = string(parameter_list[current_position]);

	return return_string;
}



//**********************************************************************************************************************************

bool Tcommand_line_parser::next_parameter_is_number()
{
	bool result;
	
	current_position++;
	if (current_position < parameter_list_size)
		result = (is_integer(parameter_list[current_position]) or is_real(parameter_list[current_position]));
	else
		result = false;
	current_position--;
	
	return result;
}

//**********************************************************************************************************************************

bool Tcommand_line_parser::parameter_is_option(unsigned position)
{
	if (position < parameter_list_size)
		return ((parameter_list[position][0] == '-') and (is_integer(parameter_list[position]) == false) and (is_real(parameter_list[position]) == false));
	else
		return false;
}

//**********************************************************************************************************************************

unsigned Tcommand_line_parser::get_current_option_position()
{
	unsigned position;
	
	position = current_position - 1;
	while ((position > 0) and (parameter_is_option(position) == false))
		position--;

	return position;
}


//**********************************************************************************************************************************


string Tcommand_line_parser::get_next_filename(unsigned error_code)
{
	string filename;
	
	if (current_position >= parameter_list_size)
		exit_with_help(error_code);
		
	filename = string(parameter_list[current_position]);
	current_position++;

	return filename;
}


//**********************************************************************************************************************************


string Tcommand_line_parser::get_next_data_filename(unsigned error_code)
{
	string filename;
	
	filename = get_next_filename(error_code);
	check_data_filename(filename);
	
	return filename;
}


//**********************************************************************************************************************************


string Tcommand_line_parser::get_next_aux_filename(unsigned error_code)
{
	string filename;
	
	filename = get_next_filename(error_code);
	check_aux_filename(filename);
	
	return filename;
}

//**********************************************************************************************************************************


string Tcommand_line_parser::get_next_labeled_data_filename(unsigned error_code)
{
	string filename;
	
	filename = get_next_filename(error_code);
	check_labeled_data_filename(filename);
	
	return filename;
}


//**********************************************************************************************************************************

string Tcommand_line_parser::get_next_unlabeled_data_filename(unsigned error_code)
{
	string filename;
	
	filename = get_next_filename(error_code);
	check_unlabeled_data_filename(filename);
	
	return filename;
}

//**********************************************************************************************************************************

string Tcommand_line_parser::get_next_log_filename(unsigned error_code)
{
	string filename;
	
	filename = get_next_filename(error_code);
	check_log_filename(filename);
	
	return filename;
}

//**********************************************************************************************************************************

string Tcommand_line_parser::get_next_solution_filename(unsigned error_code)
{
	string filename;
	
	filename = get_next_filename(error_code);
	check_solution_filename(filename);
	
	return filename;
}

#endif
