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


#if !defined (COMMAND_LINE_PARSER_H)
	#define COMMAND_LINE_PARSER_H

 

#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/system_support/parallel_control.h"


#include <string>
#include <vector>
using namespace std; 
 
 
//**********************************************************************************************************************************


unsigned const ERROR_clp_gen_unknown = 0;
unsigned const ERROR_clp_gen_unknown_option = 1;
unsigned const ERROR_clp_gen_h = 2;
unsigned const ERROR_clp_gen_d = 3;
unsigned const ERROR_clp_gen_GPU = 4;
unsigned const ERROR_clp_gen_L = 5;
unsigned const ERROR_clp_gen_r = 6;
unsigned const ERROR_clp_gen_T = 7;


unsigned const ERROR_clp_gen_missing_data_file_name = 20;
unsigned const ERROR_clp_gen_missing_train_file_name = 21;
unsigned const ERROR_clp_gen_missing_test_file_name = 22;
unsigned const ERROR_clp_gen_missing_log_file_name = 23;
unsigned const ERROR_clp_gen_missing_sol_file_name = 24;


//**********************************************************************************************************************************


template <typename Template_type> Template_type get_limits_max();


//**********************************************************************************************************************************


class Tcommand_line_parser
{
	public:
		Tcommand_line_parser();
		~Tcommand_line_parser(){};
		
		void setup(int argc, char** argv);
		bool parse(string activated_options);
		
		void copyright() const;
		void demoversion() const;
		
		int get_random_seed() const;
		bool loss_weights_are_set() const;
		Tloss_control get_loss_control() const;
		Tparallel_control get_parallel_control() const;
		

	protected:
		void make_consistent();
		virtual void exit_with_help(){};
		virtual void display_help(unsigned error_code);
		
		void exit_with_help(unsigned error_code);
		void exit_with_help_for_inconsistent_values(unsigned error_code1, unsigned error_code2);
		
		void display_separator(string option);
		void display_separator(string option, const char* message_format,...);
		void display_specifics();
		void display_ranges();
		void display_defaults();
		
		bool get_next_bool(unsigned error_code);
		unsigned get_next_class(unsigned error_code);
		unsigned get_next_enum(unsigned error_code, unsigned min, unsigned max);
		string get_next_string(unsigned error_code);
		

		template <typename Template_type> Template_type get_next_number(unsigned error_code, Template_type min, Template_type max = get_limits_max<Template_type>());
		template <typename Template_type> Template_type get_next_number_no_limits(unsigned error_code, Template_type min, Template_type max = get_limits_max<Template_type>());
		template <typename Template_type> Template_type get_next_number_no_lower_limits(unsigned error_code, Template_type min, Template_type max = get_limits_max<Template_type>());
		template <typename Template_type> Template_type get_next_number_no_upper_limits(unsigned error_code, Template_type min, Template_type max = get_limits_max<Template_type>());
		
		template <typename Template_type> vector <Template_type> get_next_list(unsigned error_code, Template_type min, Template_type max = get_limits_max<Template_type>());
		
		string get_next_filename(unsigned error_code = ERROR_clp_gen_unknown);
		string get_next_data_filename(unsigned error_code = ERROR_clp_gen_unknown);
		string get_next_labeled_data_filename(unsigned error_code = ERROR_clp_gen_unknown);
		string get_next_unlabeled_data_filename(unsigned error_code = ERROR_clp_gen_unknown);
		string get_next_log_filename(unsigned error_code = ERROR_clp_gen_unknown);
		string get_next_aux_filename(unsigned error_code);
		string get_next_solution_filename(unsigned error_code = ERROR_clp_gen_unknown);
		
		
		void check_parameter_list_size();
		bool next_parameter_is_number();
		bool next_parameter_equals(char character);
		

		int random_seed;
		Tparallel_control parallel_ctrl;
		Tloss_control loss_ctrl;
		
		
		char** parameter_list;
		unsigned parameter_list_size;
		unsigned current_position;
		
		bool full_help;
		
		bool loss_set;
		bool loss_weights_set;
		
		string command_name;
		
	private:
		void check_parameter_position(unsigned error_code = ERROR_clp_gen_unknown);
		unsigned get_current_option_position();
		bool parameter_is_option(unsigned position);
};


//**********************************************************************************************************************************


#include "sources/shared/command_line/command_line_parser.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/command_line/command_line_parser.cpp"
#endif


#endif
