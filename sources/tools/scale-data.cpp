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
#include "../shared/basic_functions/random_subsets.h"

#include "../shared/basic_types/vector.h"
#include "../shared/basic_types/dataset_info.h"

#include "../shared/system_support/timing.h"
#include "../shared/command_line/command_line_parser.h"


#include <cmath>


//**********************************************************************************************************************************

// enum SCALING_TYPES {VARIANCE, QUANTILE, SCALING_TYPES_MAX};
enum SCALING_LABEL_TYPES {DO_NOT_SCALE_LABELS, SCALE_LABELS, SCALE_LABELS_IF_REGRESSION, SCALING_LABEL_TYPES_MAX};


unsigned const ERROR_clp_tsc_c = 30;
unsigned const ERROR_clp_tsc_C = 31;
unsigned const ERROR_clp_tsc_l = 32;
unsigned const ERROR_clp_tsc_L = 33;
unsigned const ERROR_clp_tsc_s = 34;
unsigned const ERROR_clp_tsc_S = 35;
unsigned const ERROR_clp_tsc_u = 36;
unsigned const ERROR_clp_tsc_v = 37;


//**********************************************************************************************************************************


class Tcommand_line_parser_scale_data: public Tcommand_line_parser
{
	public:
		Tcommand_line_parser_scale_data();
		void parse();

		
		
		unsigned type;
		
		bool scale;
		bool scale_to_01;
		unsigned scale_labels;
		bool uniform_scaling;
		
		bool eliminate_coordinates;
		vector <unsigned> coordinates_to_be_kept;
		vector <unsigned> categorial_coordinates_to_be_ordered;
		
		double tau;
		
		bool load_scaling_info;
		bool save_scaling_info;

		string aux_filename;
		string in_filename;
		string out_filename;

	protected:
		void exit_with_help();
		void display_help(unsigned error_code);
};


//**********************************************************************************************************************************


Tcommand_line_parser_scale_data::Tcommand_line_parser_scale_data()
{
	tau = 0.0;
	type = QUANTILE;
	scale = true;
	scale_to_01 = true;
	scale_labels = DO_NOT_SCALE_LABELS;
	uniform_scaling = false;
	eliminate_coordinates = false;
	
	load_scaling_info = false;
	save_scaling_info = false;
	
	command_name = "scale-data";
};

//**********************************************************************************************************************************


void Tcommand_line_parser_scale_data::parse()
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
					categorial_coordinates_to_be_ordered = Tcommand_line_parser_scale_data::get_next_list(ERROR_clp_tsc_c, unsigned(0));
					sort_up(categorial_coordinates_to_be_ordered);
					break;
				case 'C':
					coordinates_to_be_kept = Tcommand_line_parser_scale_data::get_next_list(ERROR_clp_tsc_C, unsigned(0));
					sort_up(coordinates_to_be_kept);
					break;
				case 'l':
					scale_labels = get_next_enum(ERROR_clp_tsc_l, DO_NOT_SCALE_LABELS, SCALE_LABELS_IF_REGRESSION);
					break;
				case 'L':
					if (save_scaling_info == true)
						Tcommand_line_parser::exit_with_help(ERROR_clp_tsc_L);
					aux_filename = get_next_string(ERROR_clp_tsc_L);
					load_scaling_info = true;
					break;
				case 's':
					type = QUANTILE;
					tau = get_next_number(ERROR_clp_tsc_s, -1.0, 0.5);
					if (tau < 0.0)
					{
						tau = 0.0;
						scale = false;
					}
					if (next_parameter_is_number() == true)
						eliminate_coordinates = get_next_bool(ERROR_clp_tsc_s);
					if (next_parameter_is_number() == true)
						scale_to_01 = get_next_bool(ERROR_clp_tsc_s);
					break;
				case 'S':
					if (load_scaling_info == true)
						Tcommand_line_parser::exit_with_help(ERROR_clp_tsc_S);
					aux_filename = get_next_string(ERROR_clp_tsc_S);
					save_scaling_info = true;
					break;
				case 'u':
						uniform_scaling = get_next_bool(ERROR_clp_tsc_u);
					break;
				case 'v':
					type = VARIANCE;
					eliminate_coordinates = get_next_bool(ERROR_clp_tsc_v);
					break;
				default:
					Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			}
		}
	in_filename = get_next_data_filename(ERROR_clp_gen_missing_data_file_name);
	out_filename = get_next_data_filename(ERROR_clp_gen_missing_data_file_name);
};


//**********************************************************************************************************************************


void Tcommand_line_parser_scale_data::exit_with_help()
{
	flush_info(INFO_SILENCE,
	"\n\nUsage:\nscale-data [options] <input_data_file> <output_data_file>\n"
	"\nscale-data manipulates the data in <inpt_data_file> according to the options and" 
	"\nsaves the result in <output_data_file>.\n"
	"\nAllowed extensions:\n"
		"<data_file>:  .csv, .lsv, .nla, .wsv, and .uci\n");

	if (full_help == false)
		flush_info(INFO_SILENCE, "\nOptions:");
	
	display_help(ERROR_clp_tsc_c);
	display_help(ERROR_clp_tsc_C);
	display_help(ERROR_clp_gen_d);
	display_help(ERROR_clp_gen_h);
	display_help(ERROR_clp_tsc_l);
	display_help(ERROR_clp_tsc_L);
	display_help(ERROR_clp_tsc_s);
	display_help(ERROR_clp_tsc_S);
	display_help(ERROR_clp_tsc_u);
	display_help(ERROR_clp_tsc_v);
	
	flush_info(INFO_SILENCE,"\n\n");
	copyright();
	flush_exit(ERROR_SILENT, "");
};


//**********************************************************************************************************************************


void Tcommand_line_parser_scale_data::display_help(unsigned error_code)
{
	Tcommand_line_parser_scale_data parser_dummy;
	if ((error_code != ERROR_clp_tsc_C) and (error_code != ERROR_clp_gen_L))
		Tcommand_line_parser::display_help(error_code);


	if (error_code == ERROR_clp_tsc_c)
	{
		display_separator("-c [ <column-list> ]");
		flush_info(INFO_1, 
		"Checks, whether the columns listed in <column-list> have only integral values\n"
		"and orders the values of the columns that do by the average label one observes.\n"
		"This function may be useful for categorial values converted by R into integral\n"
		"values, since this conversion is usually by alphabetical order, i.e. quite\n"
		"arbitrary.\n");
	}
	
	if (error_code == ERROR_clp_tsc_C)
	{
		display_separator("-C [ <column-list> ]");
		flush_info(INFO_1, 
		"Only the columns listed in <column-list> are kept, all others are removed.\n");
	}	
	
	if (error_code == ERROR_clp_tsc_l)
	{
		display_separator("-l <scale_labels>");
		flush_info(INFO_1, 
		"Determines, whether the labels are scaled.\n");
		display_specifics();
		flush_info(INFO_1, "<scale_labels> = %d  =>  Do not scale labels.\n", DO_NOT_SCALE_LABELS);
		flush_info(INFO_1, "<scale_labels> = %d  =>  Scale labels to [-1,1]\n", SCALE_LABELS);
		flush_info(INFO_1, "<scale_labels> = %d  =>  Only scale labels to [-1,1], if the data is of\n"
		"                        regression type.\n", SCALE_LABELS_IF_REGRESSION);
		
		display_ranges();
		flush_info(INFO_1, "<scale_labels>:         integral in [0, 2]\n");
	
		display_defaults();
		flush_info(INFO_1, "<scale_labels> =        %d\n", parser_dummy.scale_labels);
	}
	
	if (error_code == ERROR_clp_tsc_L)
	{
		display_separator("-L <filename>");
		flush_info(INFO_1, 
		"Loads the transformations that should be applied to the data from the\n"
		"file <filename>.\n");
	}
	
	if (error_code == ERROR_clp_tsc_s)
	{
		display_separator("-s <tau> [<eliminate>] [<scale_to_01>]");
		flush_info(INFO_1, 
		"Every sample within the <tau>-quantile box is scaled to the unit ball [-1,1]^d\n"
		"or [0,1]^d. If a negative <tau> is specified, the scaling is deactivated.\n\n");

		display_specifics();
		flush_info(INFO_1, "<eliminate>        Flag indicating whether coordinates with zero\n" 
		                   "                   quantile distance should be omitted.\n");
		flush_info(INFO_1, "<scale_to_01>      Flag indicating whether data should be scaled\n" 
		                   "                   to [0,1]^d.");
		
		display_ranges();
		flush_info(INFO_1, "<tau>:             real number in [-1.0, 0.5]\n");
		flush_info(INFO_1, "<eliminate>:       bool\n");
		flush_info(INFO_1, "<scale_to_01>:     bool\n");
		
		display_defaults();
		flush_info(INFO_1, "<eliminate>        = %d\n", int(parser_dummy.eliminate_coordinates));
		flush_info(INFO_1, "<scale_to_01>      = %d\n", int(parser_dummy.scale_to_01));
	}
	
	if (error_code == ERROR_clp_tsc_S)
	{
		display_separator("-S <filename>");
		flush_info(INFO_1, 
		"Saves the transformations applied to the data to file <filename>.\n");
	}
	
	if (error_code == ERROR_clp_tsc_u)
	{
		display_separator("-u <uniform>");
		flush_info(INFO_1, 
		"Modifies the scaling of -q and -v: If <uniform> is true, then all coordinates\n"
		"are scaled by the same factor determined by either the maximal inter-quantile\n"
		"distance or variance, where the maximum is taken over all coordinates. This\n"
		"effecitvely keeps the distances of the sample points");

		display_ranges();
		flush_info(INFO_1, "<uniform>: bool\n");

		display_defaults();
		flush_info(INFO_1, "<uniform> = %d\n", int(parser_dummy.uniform_scaling));
	}	
	
	
	if (error_code == ERROR_clp_tsc_v)
	{
		display_separator("-v <eliminate>");
		flush_info(INFO_1, 
		"The scaled data has zero mean and variance one. If <eliminate> is true\n"
		"coordinates with zero variance are omitted.\n");

		display_ranges();
		flush_info(INFO_1, "<eliminate>: bool\n");

		display_defaults();
		flush_info(INFO_1, "<eliminate> = %d\n", int(parser_dummy.eliminate_coordinates));
	}
}



//**********************************************************************************************************************************
//**********************************************************************************************************************************
//**********************************************************************************************************************************



int main(int argc, char **argv)
{
	string filename;
	Tcommand_line_parser_scale_data command_line_parser;

	Tsample sample;
	Tdataset data_set;
	Tdataset projected_data_set;
	Tdataset scaled_data_set;
	Tdataset_info data_set_info;

	unsigned c;
	unsigned i;
	unsigned j;
	
	vector <double> labels;
	unsigned coordinate;
	double coordinate_value;
	vector <unsigned> coordinate_value_hits;
	vector <double> average_label_for_coordinate_value;
	vector <unsigned> categorial_coordinates_to_be_changed;
	map <int, unsigned> coordinate_value_position;
	map <int, unsigned> coordinate_translation_map;
	vector <unsigned> founds;
	vector <vector <int> > list_of_sorted_coordinate_values;
	vector <unsigned> kept_coordinates;

	double min_label;
	double max_label;
	double scaling_labels;
	double translate_labels;
	
	vector <double> scaling;
	vector <double> translate;

	

	FILE* fpaux;
	double file_time;
	double full_time;

	
	
// -------------------------------------------------------------------------------------------------------------	
	
// Get ready
	
// -------------------------------------------------------------------------------------------------------------	
	
	
// Read command line
	
	full_time = get_wall_time_difference();

	command_line_parser.setup(argc, argv);
	command_line_parser.parse();


// Load data set and, if applicable, scaling information

	file_time = get_process_time_difference();
	data_set.read_from_file(command_line_parser.in_filename);

	
	if (command_line_parser.load_scaling_info == true)
	{
		fpaux = open_file(command_line_parser.aux_filename, "r");
		
		file_read(fpaux, command_line_parser.type);
		file_read(fpaux, command_line_parser.tau);
		file_read(fpaux, kept_coordinates);
		
		file_read(fpaux, command_line_parser.scale_labels);
		file_read(fpaux, scaling_labels);
		file_read(fpaux, translate_labels);
		
		file_read(fpaux, command_line_parser.scale);
		file_read(fpaux, scaling);
		file_read(fpaux, translate);
		
		file_read(fpaux, categorial_coordinates_to_be_changed);
		file_read(fpaux, list_of_sorted_coordinate_values);

		close_file(fpaux);
	}
	else if (command_line_parser.scale_labels == SCALE_LABELS_IF_REGRESSION)
	{
		if (data_set.is_classification_data() == true)
			command_line_parser.scale_labels = DO_NOT_SCALE_LABELS;
		else
			command_line_parser.scale_labels = SCALE_LABELS;
	}
	file_time = get_process_time_difference(file_time);

	
// -------------------------------------------------------------------------------------------------------------	
	
// Deal with coordinates having categorial values. 

// -------------------------------------------------------------------------------------------------------------	


	labels = data_set.get_labels();
	data_set_info = Tdataset_info(data_set, true, command_line_parser.tau);

	if (command_line_parser.load_scaling_info == false)
		for (c=0; c<data_set_info.list_of_categorial_coordinates.size(); c++)
		{
			coordinate = data_set_info.list_of_categorial_coordinates[c];

			if ((data_set_info.categorial_values_of_coordinates[c].size() > 1) and (find(command_line_parser.categorial_coordinates_to_be_ordered, coordinate).size() > 0))
			{
				categorial_coordinates_to_be_changed.push_back(coordinate);

	//			Compute the average label value for each coordinate value
				
				coordinate_value_position = create_map(data_set_info.categorial_values_of_coordinates[c]);
				coordinate_value_hits.assign(data_set_info.categorial_values_of_coordinates[c].size(), 0);
				average_label_for_coordinate_value.assign(data_set_info.categorial_values_of_coordinates[c].size(), 0.0);

				for (i=0; i<data_set.size(); i++)
				{
					coordinate_value = data_set.sample(i)->coord(coordinate);
					coordinate_value_hits[coordinate_value_position[coordinate_value]]++;

					average_label_for_coordinate_value[coordinate_value_position[coordinate_value]] = average_label_for_coordinate_value[coordinate_value_position[coordinate_value]] + labels[i]; 
				}
				
				for (j=0; j<average_label_for_coordinate_value.size(); j++)
					average_label_for_coordinate_value[j] = average_label_for_coordinate_value[j] / double(coordinate_value_hits[j]);
				

	//			Sort by average label value and store
				
				merge_sort_up(average_label_for_coordinate_value, data_set_info.categorial_values_of_coordinates[c]);
				list_of_sorted_coordinate_values.push_back(data_set_info.categorial_values_of_coordinates[c]);
			}
		}
	else
		for (c=0; c<categorial_coordinates_to_be_changed.size(); c++)
		{
			j = 0;
			founds = find(data_set_info.list_of_categorial_coordinates, categorial_coordinates_to_be_changed[c]);
			if (founds.size() == 0)
				flush_exit(ERROR_DATA_MISMATCH, "Coordinate %d sorted previously is not integral in present data set.", categorial_coordinates_to_be_changed[c]);
			else
				j = founds[0];
			

			for (i=0; i<data_set_info.categorial_values_of_coordinates[j].size(); i++)
				if (find(list_of_sorted_coordinate_values[c], data_set_info.categorial_values_of_coordinates[j][i]).size() == 0)
					list_of_sorted_coordinate_values[c].push_back(data_set_info.categorial_values_of_coordinates[j][i]);
		}

	for (c=0; c<categorial_coordinates_to_be_changed.size(); c++)
	{
		coordinate = categorial_coordinates_to_be_changed[c];
		coordinate_translation_map = create_map(list_of_sorted_coordinate_values[c]);

		flush_info(INFO_2, "\nSorting integral values of coordinate %d.", coordinate);

		for (i=0; i<data_set.size(); i++)
			data_set.sample(i)->change_coord(coordinate, coordinate_translation_map[data_set.sample(i)->coord(coordinate)]);
	}
	if (categorial_coordinates_to_be_changed.size() == 0)
		flush_info(INFO_2, "\nSkipped sorting integral values of coordinates.");
	
// -------------------------------------------------------------------------------------------------------------	
	
// Project the data set

// -------------------------------------------------------------------------------------------------------------	


	if (command_line_parser.load_scaling_info == false)
	{
		kept_coordinates = command_line_parser.coordinates_to_be_kept;
		if (command_line_parser.eliminate_coordinates == true)
		{
			data_set_info = Tdataset_info(data_set, false, command_line_parser.tau);
			if (command_line_parser.type == QUANTILE)
			{
				for (i=0; i<data_set_info.dim; i++)
					if ((data_set_info.upper_quantiles[i] > data_set_info.lower_quantiles[i]) and (find(kept_coordinates, i).size() == 0))
						kept_coordinates.push_back(i);
			}
			else
			{
				for (i=0; i<data_set_info.dim; i++)
					if ((data_set_info.variances[i] != 0.0) and (find(kept_coordinates, i).size() == 0))
						kept_coordinates.push_back(i);
			}
			sort_up(kept_coordinates);
		}
		else if (kept_coordinates.size() == 0)
			kept_coordinates = id_permutation(data_set.dim());
	}
	
	if (kept_coordinates.size() < data_set.dim())
		flush_info(INFO_2, "\nProjecting the data to %d dimensions.", kept_coordinates.size());
	else 
		flush_info(INFO_2, "\nSkipped projecting the data to less dimensions.");

	projected_data_set.enforce_ownership();
	for (i=0; i<data_set.size(); i++)
		projected_data_set.push_back(data_set.sample(i)->project(kept_coordinates));
	
	data_set_info = Tdataset_info(projected_data_set, false, command_line_parser.tau);


// -------------------------------------------------------------------------------------------------------------	
	
// Scale the data

// -------------------------------------------------------------------------------------------------------------	

	
	if (command_line_parser.scale == false)
	{
		scaling.assign(data_set_info.dim, 1.0);
		translate.assign(data_set_info.dim, 0.0);
	}
	else if (command_line_parser.load_scaling_info == false)
	{
		projected_data_set.compute_scaling(scaling, translate, command_line_parser.tau, command_line_parser.type, command_line_parser.uniform_scaling, command_line_parser.scale_to_01);
	}

	if (command_line_parser.scale == true)
		flush_info(INFO_2, "\nScaling and translating the data.");
	else 
		flush_info(INFO_2, "\nSkipped scaling and translating the data.");
	

	scaled_data_set = projected_data_set;
	scaled_data_set.apply_scaling(scaling, translate);


	
// -------------------------------------------------------------------------------------------------------------	
	
// Scale the labels

// -------------------------------------------------------------------------------------------------------------		
	

	if (command_line_parser.load_scaling_info == false)
	{
		if (command_line_parser.scale_labels == DO_NOT_SCALE_LABELS)
		{
			scaling_labels = 1.0;
			translate_labels = 0.0;
		}
		else
		{
			labels = scaled_data_set.get_labels();
			
			min_label = labels[argmin(labels)];
			max_label = labels[argmax(labels)];
			
			if (min_label < max_label)
			{
				scaling_labels = 2.0 / (max_label - min_label);
				translate_labels = 1.0 - scaling_labels * max_label;
			}
			else
			{
				scaling_labels = 1.0;
				translate_labels = - max_label;
			}
		}
	}
	
	if (command_line_parser.scale_labels == SCALE_LABELS)
		flush_info(INFO_2, "\nScaling and translating the labels.");
	else 
		flush_info(INFO_2, "\nSkipped scaling and translating the labels.");
	
	for (i=0; i<labels.size(); i++)
		scaled_data_set.set_label_of_sample(i, scaling_labels * labels[i] + translate_labels); 
	

// -------------------------------------------------------------------------------------------------------------	

// Save data set and, if applicable, the scaling information

// -------------------------------------------------------------------------------------------------------------	
	
	
	file_time = get_process_time_difference(file_time);
	scaled_data_set.write_to_file(command_line_parser.out_filename);
	
	if (command_line_parser.save_scaling_info == true)
	{
		fpaux = open_file(command_line_parser.aux_filename, "w");
		
		file_write(fpaux, command_line_parser.type);
		file_write(fpaux, command_line_parser.tau, "%1.12f ");
		file_write(fpaux, kept_coordinates);
		
		file_write(fpaux, command_line_parser.scale_labels);
		file_write_eol(fpaux);
		file_write(fpaux, scaling_labels, "%5.12f ");
		file_write(fpaux, translate_labels, "%5.12f ");
		
		file_write(fpaux, command_line_parser.scale);
		file_write(fpaux, scaling);
		file_write(fpaux, translate);
		
		file_write(fpaux, categorial_coordinates_to_be_changed);
		file_write(fpaux, list_of_sorted_coordinate_values);

		
		close_file(fpaux);
	}
	
	file_time = get_process_time_difference(file_time);	
	
	
	// Clean up

 	full_time = get_wall_time_difference(full_time);

	flush_info(INFO_1, "\n\n%4.2f seconds used to run data-stats.", full_time);
	flush_info(INFO_1, "\n%4.2f seconds used for file operations.", file_time);

	command_line_parser.copyright();
	
	flush_info("\n\n");
}


