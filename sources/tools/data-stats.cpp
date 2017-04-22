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

#include "../shared/basic_types/vector.h"

#include <cmath>


//**********************************************************************************************************************************


unsigned const ERROR_clp_tds_t = 130;
unsigned const ERROR_clp_tds_T = 131;


//**********************************************************************************************************************************


class Tcommand_line_parser_data_stats: public Tcommand_line_parser
{
	public:
		Tcommand_line_parser_data_stats();
		void parse(string& filename);


		double tau;
		double label_tau;

	protected:
		void exit_with_help();
		void display_help(unsigned error_code);
};


//**********************************************************************************************************************************


Tcommand_line_parser_data_stats::Tcommand_line_parser_data_stats()
{
	tau = 0.0005;
	label_tau = 0.005;
	command_name = "data-stats";
};

//**********************************************************************************************************************************


void Tcommand_line_parser_data_stats::parse(string& filename)
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
				case 't':
					tau = get_next_number(ERROR_clp_tds_t, 0.0, 0.5);
					break;
				case 'T':
					label_tau = get_next_number_no_limits(ERROR_clp_tds_T, 0.0, 0.5);
					break;
				default:
					Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			}
		}
	filename = get_next_data_filename(ERROR_clp_gen_missing_data_file_name);
};


//**********************************************************************************************************************************


void Tcommand_line_parser_data_stats::exit_with_help()
{
	flush_info(INFO_SILENCE,
	"\n\nUsage:\ndata-stats [options] <data_file> \n"
	"\ndata-stats displays some statistics about the data contained in <data_file>.\n"
	"\nAllowed extensions:\n"
		"<data_file>:  .csv, .lsv, .nla, .wsv, and .uci\n");

	if (full_help == false)
		flush_info(INFO_SILENCE, "\nOptions:");
	display_help(ERROR_clp_gen_d);
	display_help(ERROR_clp_gen_h);
	display_help(ERROR_clp_tds_t);
	display_help(ERROR_clp_tds_T);
	flush_info(INFO_SILENCE,"\n\n");
	copyright();
	flush_exit(ERROR_SILENT, "");
};


//**********************************************************************************************************************************


void Tcommand_line_parser_data_stats::display_help(unsigned error_code)
{
	Tcommand_line_parser_data_stats parser_dummy;
	Tcommand_line_parser::display_help(error_code);
	
	if (error_code == ERROR_clp_tds_t)
	{
		display_separator("-t <tau>");
		flush_info(INFO_1, 
		"Determines which tau and 1-tau quantiles are considered.\n");

		display_ranges();
		flush_info(INFO_1, "<tau>: real number in [0.0, 0.5]\n");

		display_defaults();
		flush_info(INFO_1, "<tau> = %1.4f\n", parser_dummy.tau);
	}
	
	if (error_code == ERROR_clp_tds_T)
	{
		display_separator("-T <tau>");
		flush_info(INFO_1, 
		"Determines which tau and 1-tau quantiles are considered for the labels.\n");

		display_ranges();
		flush_info(INFO_1, "<tau>: real number in ]0.0, 0.5[\n");

		display_defaults();
		flush_info(INFO_1, "<tau> = %1.3f\n", parser_dummy.label_tau);
	}
}



//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************


int main(int argc, char **argv)
{
	string filename;
	Tcommand_line_parser_data_stats command_line_parser;

	Tdataset data_set;
	Tdataset_info data_set_info;

	unsigned i;
	double file_time;
	double full_time;

	
// Read command line
	
	full_time = get_wall_time_difference();

	command_line_parser.setup(argc, argv);
	command_line_parser.parse(filename);


// Load data set

	file_time = get_process_time_difference();
	data_set.read_from_file(filename);
	file_time = get_process_time_difference(file_time);


// Get data set statistics and display them

	data_set_info = Tdataset_info(data_set, false, command_line_parser.tau, command_line_parser.label_tau);

	flush_info("\nDataset dim:          %4d", data_set_info.dim);
	flush_info("\nDataset size:     %8d\n", data_set_info.size);

	switch(data_set_info.kind)
	{
		case CLASSIFICATION:
			flush_info(INFO_1, "\nNumber of labels:     %d", data_set_info.label_list.size());
			flush_info(INFO_1, "\nMost frequent label:  %d", data_set_info.label_list[data_set_info.most_frequent_label_number]);
			for (i=0; i<data_set_info.label_list.size(); i++)
				flush_info(INFO_1, "\nLabel %2d  occurs %d times (%1.4f)", data_set_info.label_list[i], data_set_info.label_count[i], float(data_set_info.label_count[i])/float(data_set_info.size));
			break;

		case REGRESSION:
			flush_info(INFO_1, "\nMedian label:          %3.3f", data_set_info.median_label);
			flush_info(INFO_1, "\nNaive MAE:             %3.3f", data_set_info.abs_label_error);
			flush_info(INFO_1, "\nMean label:            %3.3f", data_set_info.mean_label);
			flush_info(INFO_1, "\nNaive MSE:             %3.3f", data_set_info.square_label_error);
			flush_info(INFO_1, "\nSmallest label:        %3.3f", data_set_info.min_label);
			flush_info(INFO_1, "\n%0.2f label quantile:  %3.3f", command_line_parser.label_tau, data_set_info.lower_label_quantile);
			flush_info(INFO_1, "\n%0.2f label quantile:  %3.3f", 1.0 - command_line_parser.label_tau, data_set_info.upper_label_quantile);
			flush_info(INFO_1, "\nLargest label:         %3.3f", data_set_info.max_label);
			break;
	}
	
	flush_info(INFO_1, "\n\nNumber of coordinates with integral values:                %4d", data_set_info.list_of_categorial_coordinates.size());
	
	if (command_line_parser.tau >= 0.0)
	{
		flush_info(INFO_1, "\nNumber of coordinates with positive quantile distance:   %6d  (%1.4f)", data_set_info.coordinates_with_positive_quantile_distance.size(), double(data_set_info.coordinates_with_positive_quantile_distance.size())/double(data_set_info.dim));
		flush_info(INFO_1, "\nNumber of samples inside quantile box:                   %6d  (%1.4f)", int(data_set_info.size) - int(data_set_info.sample_indices_outside_quantile_box.size()), 1.0 - double(data_set_info.sample_indices_outside_quantile_box.size())/double(data_set_info.size));
		
		flush_info(INFO_2, "\n");
		for (i=0; i<data_set_info.dim; i++)
			flush_info(INFO_2, "\nCoordinate %d:   min = %.3f   %.4f-quantile = %.3f   average = %.3f   %.4f-quantile = %.3f   max = %.3f   std var = %.3f", i+1, data_set_info.minima[i], command_line_parser.tau, data_set_info.lower_quantiles[i], data_set_info.means[i], 1.0 - command_line_parser.tau, data_set_info.upper_quantiles[i], data_set_info.maxima[i], sqrt(data_set_info.variances[i]));
	}

	// Clean up

 	full_time = get_wall_time_difference(full_time);

	flush_info(INFO_1, "\n\n%4.2f seconds used to run data-stats.", full_time);
	flush_info(INFO_1, "\n%4.2f seconds used for file operations.", file_time);

	command_line_parser.copyright();
	
	flush_info("\n\n");
}


