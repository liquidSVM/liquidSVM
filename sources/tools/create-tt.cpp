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



#include "../shared/system_support/os_specifics.h"
#include "../shared/basic_functions/flush_print.h"
#include "../shared/basic_types/dataset_info.h"
#include "../shared/system_support/timing.h"
#include "../shared/command_line/command_line_parser.h"
#include "../shared/training_validation/fold_manager.h"



//**********************************************************************************************************************************


unsigned const ERROR_clp_ttt_f = 120;
unsigned const ERROR_clp_ttt_F = 121;
unsigned const ERROR_clp_ttt_t = 122;


//**********************************************************************************************************************************


class Tcommand_line_parser_create_tt: public Tcommand_line_parser
{
	public:
		Tcommand_line_parser_create_tt();
		void parse();

		
		Tfold_control fold_control;
		
		double val_fraction;
		unsigned train_size;
		unsigned val_size;
		unsigned test_size;
		string filename;

	protected:
		void exit_with_help();
		void display_help(unsigned error_code);
};


//**********************************************************************************************************************************


Tcommand_line_parser_create_tt::Tcommand_line_parser_create_tt()
{
	fold_control.number = 1;
	fold_control.kind = STRATIFIED;
	fold_control.train_fraction = 0.7;
	val_fraction = 0.0;
	
	train_size = 0;
	val_size = 0;
	test_size = 0;
	
	command_name = "create-tt";
};

//**********************************************************************************************************************************


void Tcommand_line_parser_create_tt::parse()
{
	check_parameter_list_size();
	for(current_position=1; current_position<parameter_list_size; current_position++)
		if (Tcommand_line_parser::parse("-h-r-d") == false)
		{
			if(parameter_list[current_position][0] != '-') 
				break;
			if (string(parameter_list[current_position]).size() > 2)
				Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			
			switch(parameter_list[current_position][1])
			{
				case 'f':
					fold_control.train_fraction = get_next_number_no_limits(ERROR_clp_ttt_f, 0.0, 1.0);
					if (next_parameter_is_number() == true)
						val_fraction = get_next_number(ERROR_clp_ttt_f, 0.0, 1.0 - fold_control.train_fraction);
					break;
				case 'F':
					train_size = get_next_number_no_limits(ERROR_clp_ttt_F, 0);
					if (next_parameter_is_number() == true)
						val_size = get_next_number(ERROR_clp_ttt_F, 0);
					if (next_parameter_is_number() == true)
						test_size = get_next_number_no_limits(ERROR_clp_ttt_F, 0.0);
					break;
				case 't':
					fold_control.kind = get_next_enum(ERROR_clp_ttt_t, BLOCKS, STRATIFIED);
					break;
				default:
					Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			}
		}
		
	fold_control.random_seed = random_seed;
	filename = get_next_data_filename(ERROR_clp_gen_missing_data_file_name);
};


//**********************************************************************************************************************************


void Tcommand_line_parser_create_tt::exit_with_help()
{
	flush_info(INFO_SILENCE,
	"\n\ncreate-tt [options] <data_file> \n"
	"\ncreate-tt creates a training, test, and optionally also a validation, data set" 
	"\nfrom <data_file> and writes these data sets into 2 or 3 files. These have the"
	"\nthe same name as <data_file> extended by .train', '.test', and '.val',"
	"\nrespectively. The extension of <data_file> is inherited.\n"
	"\nAllowed extensions:\n"
		"<data_file>:  .csv, .lsv, .uci, and .nla\n");

	if (full_help == false)
		flush_info(INFO_SILENCE, "\nOptions:");
	display_help(ERROR_clp_gen_d);
	display_help(ERROR_clp_ttt_f);
	display_help(ERROR_clp_ttt_F);
	display_help(ERROR_clp_gen_h);
	display_help(ERROR_clp_gen_r);
	display_help(ERROR_clp_ttt_t);
	
	flush_info(INFO_SILENCE,"\n\n");
	copyright();
	flush_exit(ERROR_SILENT, "");
};


//**********************************************************************************************************************************


void Tcommand_line_parser_create_tt::display_help(unsigned error_code)
{
	Tfold_control fch;
	
	Tcommand_line_parser::display_help(error_code);
	
	if (error_code == ERROR_clp_ttt_f)
	{
		display_separator("-f <train_fraction> [<val_fraction>]");
		flush_info(INFO_1, 
		"Sets the fraction of samples that will be written into the training and the\n"
		"optional validation file. The remaining part is written into the test file.\n");

		display_ranges();
		flush_info(INFO_1, "<train_fraction>:  real number strictly between 0.0 and 1.0\n");
		flush_info(INFO_1, "<val_fraction>:    real number between 0.0 and 1.0 - <train_fraction>\n");
		
		display_defaults();
		flush_info(INFO_1, "<train_fraction> = 0.7\n");
		flush_info(INFO_1, "<val_fraction>   = 0.0\n");
	}
	
	if (error_code == ERROR_clp_ttt_F)
	{
		display_separator("-F <train_size> [<val_size>] [<test_size>]");
		flush_info(INFO_1, 
		"Sets the number of samples that will be written into the training, the optional\n"
		"validation, and the test file. If <test_size> is not set, all remaining samples\n"
		"are written into the test file.\n");

		display_ranges();
		flush_info(INFO_1, "<train_size>:  integer > 0\n");
		flush_info(INFO_1, "<val_size>:    unsigned integer\n");
		flush_info(INFO_1, "<test_size>:   integer > 0\n");
		
		display_defaults();
		flush_info(INFO_1, "<train_size> = 0\n");
		flush_info(INFO_1, "<val_size>   = 0\n");
		flush_info(INFO_1, "<test_size>  = 0\n");
	}
		
	if (error_code == ERROR_clp_ttt_t)
	{
		display_separator("-t <method>");
		flush_info(INFO_1, 
		"Selects the method for creating the files. Warning: stratified sampling may lead to adjusted\n"
		"file sizes!\n");
		display_specifics();
		flush_info(INFO_1, "<method> = %d  =>  both files are a contiguous block\n", BLOCKS);
		flush_info(INFO_1, "<method> = %d  =>  alternating fold assignmend (-f is ignored)\n", ALTERNATING);
		flush_info(INFO_1, "<method> = %d  =>  random\n", RANDOM);
		flush_info(INFO_1, "<method> = %d  =>  stratified random\n", STRATIFIED);

		display_ranges();
		flush_info(INFO_1, "<method>:  integer between %d and %d\n", FROM_FILE+1, FOLD_CREATION_TYPES_MAX-2);
		
		display_defaults();
		flush_info(INFO_1, "<method> = %d\n", fch.kind);
	}
}


//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************


int main(int argc, char **argv)
{
	Tcommand_line_parser_create_tt command_line_parser;

	Tdataset data_set;
	Tdataset val_data_set;
	Tdataset test_data_set;
	Tdataset rest_data_set;
	Tdataset training_set;
	Tdataset val_set;
	Tdataset test_set;
	Tdataset_info data_set_info;
	
	Tfold_manager fold_manager;
	Tfold_manager fold_manager_val;
	
	string extension;
	string train_filename;
	string test_filename;
	string val_filename;

	double write_time;
	double read_time;
	double full_time;

	
// Read command line

	full_time = get_wall_time_difference();
	
	command_line_parser.setup(argc, argv);
	command_line_parser.parse();


// Load data set

	read_time = get_process_time_difference();
	data_set.read_from_file(command_line_parser.filename);
	read_time = get_process_time_difference(read_time);
	

// Build train and test set
	
	data_set_info = Tdataset_info(data_set, true);
	if ((data_set_info.kind != CLASSIFICATION) and (command_line_parser.fold_control.kind == STRATIFIED))
		command_line_parser.fold_control.kind = RANDOM;
	
	if (command_line_parser.train_size > 0)
	{
		if (command_line_parser.train_size + command_line_parser.val_size > data_set_info.size)
			flush_exit(ERROR_DATA_MISMATCH, "The data set only contains %d samples.\n", data_set_info.size);
		
		command_line_parser.fold_control.train_fraction = double(command_line_parser.train_size) / double(data_set_info.size);
		command_line_parser.val_fraction = double(command_line_parser.val_size) / double(data_set_info.size);
	}
	

	fold_manager = Tfold_manager(command_line_parser.fold_control, data_set);
	fold_manager.build_train_and_val_set(1, training_set, test_set);

	if (command_line_parser.val_fraction > 0.0)
	{
		val_data_set = test_set;
		command_line_parser.fold_control.train_fraction = command_line_parser.val_fraction / (1.0 - command_line_parser.fold_control.train_fraction);

		fold_manager_val = Tfold_manager(command_line_parser.fold_control, val_data_set);
		fold_manager_val.build_train_and_val_set(1, val_set, test_set);
		val_set.enforce_ownership();
		test_set.enforce_ownership();
	}
	
	if ((command_line_parser.test_size > 0) and (command_line_parser.test_size < test_set.size()))
	{
		test_data_set = test_set;

		command_line_parser.fold_control.train_fraction = double(command_line_parser.test_size ) / double(test_set.size());

		fold_manager_val = Tfold_manager(command_line_parser.fold_control, test_data_set);
		fold_manager_val.build_train_and_val_set(1, test_set, rest_data_set);
	}


// Write to file
	
	extension = command_line_parser.filename.substr(command_line_parser.filename.length() - 4, command_line_parser.filename.length());
	train_filename = command_line_parser.filename.substr(0, command_line_parser.filename.length() - 4) + ".train" + extension;
	test_filename = command_line_parser.filename.substr(0, command_line_parser.filename.length() - 4) + ".test" + extension;
	val_filename = command_line_parser.filename.substr(0, command_line_parser.filename.length() - 4) + ".val" + extension;
	
	write_time = get_process_time_difference();
	training_set.write_to_file(train_filename);
	if (val_set.size() > 0)
		val_set.write_to_file(val_filename);
	test_set.write_to_file(test_filename);
	write_time = get_process_time_difference(write_time);
	

	// Clean up

 	full_time = get_wall_time_difference(full_time);

	flush_info(INFO_1,"\n\n%4.2f seconds used to run create-tt.", full_time);
	flush_info(INFO_1,"\n%4.2f seconds used for read from file operations.", read_time);
	flush_info(INFO_1,"\n%4.2f seconds used for write to file operations.", write_time);


	command_line_parser.copyright();
	
	flush_info(INFO_1,"\n\n");
}


