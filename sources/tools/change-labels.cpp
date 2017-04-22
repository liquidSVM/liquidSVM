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
#include "../shared/basic_types/vector.h"
#include "../shared/system_support/timing.h"
#include "../shared/command_line/command_line_parser.h"



//**********************************************************************************************************************************


unsigned const ERROR_clp_tcl_c = 100;
unsigned const ERROR_clp_tcl_D = 101;
unsigned const ERROR_clp_tcl_f = 102;
unsigned const ERROR_clp_tcl_s = 103;


enum REPLACE_MODES {REPLACE_LABEL, DIFF_OF_LABEL, REPLACE_MODES_MAX};

//**********************************************************************************************************************************


class Tcommand_line_parser_change_labels: public Tcommand_line_parser
{
	public:
		Tcommand_line_parser_change_labels();
		void parse();
		
		double old_label;
		double new_label;
		double scale_factor;
		bool scaling_flag;
		vector <double> thresholds;
		
		unsigned label_replace_mode;
		
		string label_filename;
		string read_filename;
		string write_filename;

	protected:
		void exit_with_help();
		void display_help(unsigned error_code);
};


//**********************************************************************************************************************************

Tcommand_line_parser_change_labels::Tcommand_line_parser_change_labels()
{
	command_name = "change-labels";
	label_replace_mode = REPLACE_LABEL;
	
	scaling_flag = false;
	scale_factor = 0.0;
};

//**********************************************************************************************************************************

void Tcommand_line_parser_change_labels::parse()
{
	check_parameter_list_size();
	for(current_position=1; current_position<parameter_list_size; current_position++)
		if (Tcommand_line_parser::parse("-d-h") == false)
		{
			if(parameter_list[current_position][0] != '-') 
				break;
			if (string(parameter_list[current_position]).size() > 2)
				Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);

			switch(parameter_list[current_position][1])
			{
				case 'c':
					old_label = get_next_number(ERROR_clp_tcl_c, -numeric_limits<double>::max(), numeric_limits<double>::max());
					new_label = get_next_number(ERROR_clp_tcl_c, -numeric_limits<double>::max(), numeric_limits<double>::max());
					break;

				case 'D':
					thresholds = Tcommand_line_parser_change_labels::get_next_list(ERROR_clp_tcl_D, -std::numeric_limits<double>::max());
					sort_up(thresholds);
					break;
					
				case 'f':
					label_replace_mode = get_next_enum(ERROR_clp_tcl_f, REPLACE_LABEL, REPLACE_MODES_MAX-1);
					current_position++;
					label_filename = get_next_labeled_data_filename(ERROR_clp_tcl_f);
					current_position--;
					break;
					
				case 's':
					scale_factor = get_next_number(ERROR_clp_tcl_s, -numeric_limits<double>::max(), numeric_limits<double>::max());
					scaling_flag = true;
					break;
				
				default:
					Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			}
		}
		
	read_filename = get_next_labeled_data_filename(ERROR_clp_gen_missing_data_file_name);
	if (current_position < parameter_list_size)
		write_filename = get_next_labeled_data_filename(ERROR_clp_gen_missing_data_file_name);
	else
		write_filename = read_filename;
};


//**********************************************************************************************************************************

void Tcommand_line_parser_change_labels::exit_with_help()
{
	flush_info(INFO_SILENCE,
	"\n\nchange-labels [options] <read_data_file> [<write_data_file>]\n"
	"\nChanges the labels of the data set contained in <read_data_file> and saves\n"
	"the results to <write_data_file>. If the latter is missing, the result is\n"
	"written to <read_data_file> instead\n"
	"\nAllowed extensions:\n"
		"<x_data_file>:  .csv, .lsv, and .uci\n");

	if (full_help == false)
		flush_info(INFO_SILENCE, "\nOptions:");
	display_help(ERROR_clp_tcl_c);
	display_help(ERROR_clp_gen_d);
	display_help(ERROR_clp_tcl_D);
	display_help(ERROR_clp_tcl_f);
	display_help(ERROR_clp_gen_h);
	display_help(ERROR_clp_tcl_s);
	
	flush_info(INFO_SILENCE,"\n\n");
	copyright();
	flush_exit(ERROR_SILENT, "");
};


//**********************************************************************************************************************************


void Tcommand_line_parser_change_labels::display_help(unsigned error_code)
{
	Tcommand_line_parser::display_help(error_code);
	
	if (error_code == ERROR_clp_tcl_c)
	{
		display_separator("-c <old_label> <new_label>");
		flush_info(INFO_1, 
		"Change <old_label> to <new_label>.\n");

		display_ranges();
		flush_info(INFO_1, "<old_label>:  real number\n");
		flush_info(INFO_1, "<new_label>:  real number\n");
	}
	
	
	if (error_code == ERROR_clp_tcl_D)
	{
		display_separator("-D [ <threshold list> ]");
		flush_info(INFO_1, 
		"Discretizes the labels according to the list of threshold values.\n");

		display_ranges();
		flush_info(INFO_1, "<threshold>:  real number\n");
	}

	if (error_code == ERROR_clp_tcl_f)
	{
		display_separator("-f <replace_mode> <label_data_file>");
		flush_info(INFO_1, 
		"Takes the labels from <label_data_file> to modify the labels of <read_data_file>\n"
		"in a way described by <replace_mode>.\n");

		display_specifics();
		flush_info(INFO_1, "<replace_mode> = %d  =>  Simply replace the labels.\n", REPLACE_LABEL);
		flush_info(INFO_1, "<replace_mode> = %d  =>  If at least one file contains non-classification data,\n"
		"                        the simple difference is taken, otherwise a new label\n"
		"                        is created for the all samples with different labels.\n", DIFF_OF_LABEL);
		
		display_ranges();
		flush_info(INFO_1, "<replace_mode>:  either %d or %d\n", REPLACE_LABEL, DIFF_OF_LABEL);
	}
	
	
	if (error_code == ERROR_clp_tcl_s)
	{
		display_separator("-s <factor> ");
		flush_info(INFO_1, 
		"Multiplies all labels by <factor>. This operation is performed after all\n"
		"other modifications.\n");

		display_ranges();
		flush_info(INFO_1, "<factor>:  real number\n");
	}
}

//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************


int main(int argc, char **argv)
{
	unsigned i;
	unsigned l;
	Tcommand_line_parser_change_labels command_line_parser;
	Tdataset data_set;
	Tdataset label_data_set;
	Tdataset_info data_set_info;
	Tdataset_info label_data_set_info;
	double file_time;
	double full_time;

	bool simple_diff;
	double new_label;

// Read command line
	
	full_time = get_wall_time_difference();

	command_line_parser.setup(argc, argv);
	command_line_parser.parse();


// Load data set

	file_time = get_process_time_difference();
	data_set.read_from_file(command_line_parser.read_filename);
	if (command_line_parser.label_filename.size() > 0)
	{
		label_data_set.read_from_file(command_line_parser.label_filename);
		if (label_data_set.size() != data_set.size())
			flush_exit(ERROR_DATA_MISMATCH, "File file %s contains %d samples but the file %s contains %d labels.", command_line_parser.read_filename.c_str(), data_set.size(), command_line_parser.label_filename.c_str(), label_data_set.size());
	}
	file_time = get_process_time_difference(file_time);
	flush_info(INFO_1,"\n%d samples read from file %s", data_set.size(), command_line_parser.read_filename.c_str());
	
	if (command_line_parser.label_filename.size() > 0)
	{
		flush_info(INFO_1,"\n%d labels read from file %s", label_data_set.size(), command_line_parser.label_filename.c_str());
	
		data_set_info = Tdataset_info(data_set, true);
		label_data_set_info = Tdataset_info(label_data_set, true);
		
		new_label = data_set_info.min_label - 1.0;
		simple_diff = not(data_set.is_classification_data() and label_data_set.is_classification_data());
		
		for (i=0; i<data_set.size(); i++)
			if (command_line_parser.label_replace_mode == REPLACE_LABEL)
				data_set.sample(i)->label = label_data_set.sample(i)->label;
			else if (simple_diff == true)
				data_set.sample(i)->label = data_set.sample(i)->label - label_data_set.sample(i)->label;
			else if (data_set.sample(i)->label != label_data_set.sample(i)->label)
				data_set.sample(i)->label = new_label;
	}
	else if (command_line_parser.thresholds.size() > 0)
	{
		flush_info(INFO_1, "\nDiscretizing labels to values 0, ..., %d.", command_line_parser.thresholds.size());
		
		command_line_parser.thresholds.push_back(std::numeric_limits<double>::max());
		for (i=0; i<data_set.size(); i++)
		{
			l = 0;
			while ((data_set.sample(i)->label > command_line_parser.thresholds[l]) and (l + 1 < command_line_parser.thresholds.size()))
				l++;
			data_set.sample(i)->label = double(l);
		}
			
	}
	else
		data_set.change_labels(command_line_parser.old_label, command_line_parser.new_label);
	
	if (command_line_parser.scaling_flag == true)
		for (i=0; i<data_set.size(); i++)
				data_set.sample(i)->label = command_line_parser.scale_factor * data_set.sample(i)->label;
	
	
	file_time = get_process_time_difference(file_time);
	data_set.write_to_file(command_line_parser.write_filename);
	file_time = get_process_time_difference(file_time);
	flush_info(INFO_1,"\n%d samples written to file %s", data_set.size(), command_line_parser.write_filename.c_str());


// Clean up

 	full_time = get_wall_time_difference(full_time);

	flush_info(INFO_1,"\n\n%4.2f seconds used to run data-stats.", full_time);
	flush_info(INFO_1,"\n%4.2f seconds used for file operations.", file_time);

	command_line_parser.copyright();
	
	flush_info(INFO_1,"\n\n");
}


