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


#if !defined (COMMAND_LINE_PARSER_SVM_TRAIN_CPP)
	#define COMMAND_LINE_PARSER_SVM_TRAIN_CPP
 
 

#include "sources/svm/command_line/command_line_parser_svm_train.h"


#include "sources/shared/basic_functions/basic_file_functions.h"



//**********************************************************************************************************************************





unsigned const ERROR_clp_svm_train_a = 31;
unsigned const ERROR_clp_svm_train_f = 32;
unsigned const ERROR_clp_svm_train_g = 33;
unsigned const ERROR_clp_svm_train_i = 34;
unsigned const ERROR_clp_svm_train_k = 35;
unsigned const ERROR_clp_svm_train_l = 36;
unsigned const ERROR_clp_svm_train_L = 37;
unsigned const ERROR_clp_svm_train_P = 38;
unsigned const ERROR_clp_svm_train_s = 39;
unsigned const ERROR_clp_svm_train_S = 40;
unsigned const ERROR_clp_svm_train_w = 41;
unsigned const ERROR_clp_svm_train_W = 42;

//**********************************************************************************************************************************


Tcommand_line_parser_svm_train::Tcommand_line_parser_svm_train()
{
	clipping_set = false;
	init_set = false;
	
	command_name = "svm-train";
};



//**********************************************************************************************************************************


void Tcommand_line_parser_svm_train::make_consistent(Ttrain_control& train_control)
{
	Tgrid_control gch_tmp;
	Tcommand_line_parser::make_consistent();
	
	// Copy random_seed to all control structures that need it
	
	train_control.fold_control.random_seed = random_seed;
	train_control.working_set_control.random_seed = random_seed;

	
	// Copy values from loss_ctrl into train_control.solver_control.loss_control and set defaults
	
	if (train_control.solver_control.loss_control.type == MULTI_CLASS_LOSS)
		Tcommand_line_parser::exit_with_help(ERROR_clp_svm_train_L);
	if (loss_set == false)
		switch (train_control.solver_control.solver_type)
		{
			case SVM_LS_2D:
				train_control.solver_control.loss_control.type = LEAST_SQUARES_LOSS;
				break;
			case SVM_EXPECTILE_2D:
				train_control.solver_control.loss_control.type = WEIGHTED_LEAST_SQUARES_LOSS;
				break;
			case SVM_QUANTILE:
				train_control.solver_control.loss_control.type = PINBALL_LOSS;
				break;
			case SVM_LS_PAR:
				train_control.solver_control.loss_control.type = LEAST_SQUARES_LOSS;
				break;			
// 			Change the next lines if necessary
			case SVM_TEMPLATE:
				train_control.solver_control.loss_control.type = TEMPLATE_LOSS;
				break;
			default:
				train_control.solver_control.loss_control.type = CLASSIFICATION_LOSS;
		}
	else
		train_control.solver_control.loss_control = loss_ctrl;
	train_control.solver_control.fixed_loss = loss_weights_set;


	// Set the weights to neutral values if the solver is not of hinge, quantile, or expectile type.
	
	if ( (train_control.solver_control.solver_type != SVM_HINGE_2D) and (train_control.solver_control.solver_type != SVM_HINGE_PAR) and (train_control.solver_control.solver_type != SVM_EXPECTILE_2D) and (train_control.solver_control.solver_type != SVM_QUANTILE))
	{
		train_control.grid_control.min_weight = 1.0;
		train_control.grid_control.max_weight = 1.0;
		train_control.grid_control.weight_size = 1;
		train_control.grid_control.weights.clear();
	}


	// Decide whether data should be treated as classfication or regression data. By default, the loss function determines this ...
	// ... but if the working set generation method is classification based, this default behavior needs to be overwritten.


	if ((train_control.solver_control.loss_control.type == LEAST_SQUARES_LOSS) or (train_control.solver_control.loss_control.type == WEIGHTED_LEAST_SQUARES_LOSS) or (train_control.solver_control.loss_control.type == PINBALL_LOSS))
		train_control.working_set_control.classification = false;
	else
		train_control.working_set_control.classification = true;


	if ((train_control.working_set_control.working_set_selection_method == MULTI_CLASS_ALL_VS_ALL) or (train_control.working_set_control.working_set_selection_method == MULTI_CLASS_ONE_VS_ALL))
		train_control.working_set_control.classification = true;

	
	// Exclude certain features from the demo version

	#ifdef __DEMOSVM__
	  if ( (not((train_control.solver_control.solver_type == SVM_HINGE_2D) or (train_control.solver_control.solver_type == SVM_LS_2D) or (train_control.solver_control.solver_type == KERNEL_RULE))) or (train_control.solver_control.kernel_type != 0) or (train_control.solver_control.npl == true) or (train_control.grid_control.min_weight != 1.0) or (train_control.grid_control.max_weight != 1.0) or (threads > 2))
	    demoversion();
	#endif

	
	// Finally, exclude some cases, which so far have not been implemented.
	
	if (((train_control.solver_control.kernel_control_train.memory_model_pre_kernel == CACHE) or (train_control.solver_control.kernel_control_train.memory_model_kernel == CACHE)) and ((parallel_ctrl.requested_team_size != 1) or (parallel_ctrl.GPUs > 0)))
		flush_exit(ERROR_SILENT, "I am sorry, kernel caching only works with one thread and zero GPUs.");
	
	if ((train_control.solver_control.kernel_control_val.memory_model_pre_kernel == CACHE) or (train_control.solver_control.kernel_control_val.memory_model_kernel == CACHE))
		flush_exit(ERROR_SILENT, "I am sorry, kernel caching does not work for the validation matrix.");
};




//**********************************************************************************************************************************


void Tcommand_line_parser_svm_train::display_help(unsigned error_code)
{
	Tgrid_control gch;
	Tsvm_solver_control sch;
	Tworking_set_control wch;
	Tworking_set_control wch_2;
	Tfold_control fch;
	Ttrain_control tch;


	if (error_code != ERROR_clp_svm_train_L)
		Tcommand_line_parser::display_help(error_code);
	
	
	if (error_code == ERROR_clp_svm_train_a)
	{
		display_separator("-a <adaptive_search> [<max_number_of_increases>] [<max_number_of_worse_gammas>]");
		flush_info(INFO_1, 
		"Specifies if and how an adaptive hyper-parameter search is conducted.\n");
		display_specifics();
		flush_info(INFO_1, "<adaptive_search>:              Flag indicating whether an adaptive search is\n" 
		                   "                                conducted. \n");
		flush_info(INFO_1, "<max_number_of_increases>       Describes how conservative the adaptive search\n"
		                   "                                for lambda is. The larger the value, the more\n" 
						   "                                conservative the search is.\n");
		flush_info(INFO_1, "<max_number_of_worse_gammas>    Describes how conservative the adaptive search\n"
		                   "                                for gamma is. The larger the value, the more\n" 
						   "                                conservative the search is.\n");
		
		display_ranges();
		flush_info(INFO_1, "<adaptive_search>:              bool\n");
		flush_info(INFO_1, "<max_number_of_increases>       integer >= 1\n");
		flush_info(INFO_1, "<max_number_of_worse_gammas>    integer >= 1\n");

		display_defaults();
		flush_info(INFO_1, "<adaptive_search>:              %d\n", not(tch.full_search));
		flush_info(INFO_1, "<max_number_of_increases>       %d\n", tch.max_number_of_increases);
		flush_info(INFO_1, "<max_number_of_worse_gammas>    %d\n", tch.max_number_of_worse_gammas);

	}
	
	if (error_code == ERROR_clp_svm_train_f)
	{
		display_separator("-f <kind> <number> [<train_fraction>] [<neg_fraction>]");
		flush_info(INFO_1, 
		"Selects the fold generation method and the number of folds. If <train_fraction>\n"
		"< 1.0, then the folds for training are generated from a subset with the\n"
		" specified size and the remaining samples are used for validation.");
		display_specifics();
		flush_info(INFO_1, "<kind> = %d  =>  each fold is a contiguous block\n", BLOCKS);
		flush_info(INFO_1, "<kind> = %d  =>  alternating fold assignmend\n", ALTERNATING);
		flush_info(INFO_1, "<kind> = %d  =>  random\n", RANDOM);
		flush_info(INFO_1, "<kind> = %d  =>  stratified random\n", STRATIFIED);
		flush_info(INFO_1, "<kind> = %d  =>  random subset (<train_fraction> and <neg_fraction> required)\n", RANDOM_SUBSET);


		display_ranges();
		flush_info(INFO_1, "<kind>:           integer between %d and %d\n", FROM_FILE+1, FOLD_CREATION_TYPES_MAX-1);
		flush_info(INFO_1, "<number>:         integer >= 1\n");
		flush_info(INFO_1, "<train_fraction>: float > 0.0 and <= 1.0\n");
		flush_info(INFO_1, "<neg_fraction>:   float > 0.0 and < 1.0\n");
		display_defaults();
		flush_info(INFO_1, "<kind>           = %d\n", fch.kind);
		flush_info(INFO_1, "<number>         = %d\n", fch.number);
		flush_info(INFO_1, "<train_fraction> = %1.2f\n", fch.train_fraction);
	}
	
	if (error_code == ERROR_clp_svm_train_g)
	{
		display_separator("-g [<option_1> ...]",
						"-g <size> <min_gamma> <max_gamma> [<scale>]\n"
						"-g <gamma_list>");
		flush_info(INFO_1, 
		"The first variant sets the size <size> of the gamma grid and its endpoints\n"
		"<min_gamma> and <max_gamma>.\n"
		"The second variant uses <gamma_list> for the gamma grid.\n");
		
		display_specifics();
		flush_info(INFO_1, 
		"<scale>       Flag indicating whether <min_gamma> and <max_gamma> are scaled\n"
		"              based on the sample size, the dimension, and the diameter.\n");
		display_ranges();
		flush_info(INFO_1, "<size>:       integer >= 1\n");
		flush_info(INFO_1, "<min_gamma>:  float > 0.0\n");
		flush_info(INFO_1, "<max_gamma>:  float > 0.0\n");
		flush_info(INFO_1, "<scale>:      bool\n");
		display_defaults();
		flush_info(INFO_1, "<size>        = %d\n", gch.gamma_size);
		flush_info(INFO_1, "<min_gamma>   = %1.3f\n", gch.min_gamma_unscaled);
		flush_info(INFO_1, "<max_gamma>   = %1.3f\n", gch.max_gamma_unscaled);
		flush_info(INFO_1, "<scale>       = %d\n", gch.scale_gamma);
	}

	if (error_code == ERROR_clp_svm_train_i)
	{
		display_separator("-i <cold> <warm>");
		flush_info(INFO_1, 
		"Selects the cold and warm start initialization methods of the solver. In\n"
		"general, this option should only be used in particular situations such as the\n"
		"implementation and testing of a new solver or when using the kernel cache.\n");
		display_specifics();
		flush_info(INFO_1, "For values between 0 and 6, both <cold> and <warm> have the same meaning as\n");
		flush_info(INFO_1, "in Steinwart et al, 'Training SVMs without offset', JMLR 2011. These are:\n");
		flush_info(INFO_1, " %d      Sets all coefficients to zero.\n", SOLVER_INIT_ZERO);
		flush_info(INFO_1, " %d      Sets all coefficients to C.\n", SOLVER_INIT_FULL);
		flush_info(INFO_1, " %d      Uses the coefficients of the previous solution.\n", SOLVER_INIT_RECYCLE);
		
		flush_info(INFO_1, " %d      Multiplies all coefficients by C_new/C_old.\n", SOLVER_INIT_EXPAND_UNIFORMLY);
		flush_info(INFO_1, " %d      Multiplies all unbounded SVs by C_new/C_old.\n", SOLVER_INIT_EXPAND);
		flush_info(INFO_1, " %d      Multiplies all coefficients by C_old/C_new.\n", SOLVER_INIT_SHRINK_UNIFORMLY);
		flush_info(INFO_1, " %d      Multiplies all unbounded SVs by C_old/C_new.\n", SOLVER_INIT_SHRINK);
		

		
		
		display_ranges();
		flush_info(INFO_1, "Depends on the solver, but the range of <cold> is always a subset of the range\n");
		flush_info(INFO_1, "of <warm>.\n");
		display_defaults();
		flush_info(INFO_1, "Depending on the solver, the (hopefully) most efficient method is chosen.\n");
	}

	if (error_code == ERROR_clp_svm_train_k)
	{
		display_separator("-k <type> [aux-file] [<Tr_mm_Pr> [<size_P>] <Tr_mm> [<size>] <Va_mm_Pr> <Va_mm>]");
		flush_info(INFO_1, 
		"Selects the type of kernel and optionally the memory model for the kernel matrices.\n");
		display_specifics();
		flush_info(INFO_1, "<type>   = %d  =>   Gaussian RBF\n", GAUSS_RBF);
		flush_info(INFO_1, "<type>   = %d  =>   Poisson\n", POISSON);
		flush_info(INFO_1, "<type>   = %d  =>   Experimental hierarchical Gauss kernel\n", HIERARCHICAL_GAUSS);
		flush_info(INFO_1, "<aux_file>    =>   Name of the file that contains additional information for the\n"
		                   "                   hierarchical Gauss kernel. Only this kernel type requires this option.\n", HIERARCHICAL_GAUSS);
		flush_info(INFO_1, "<X_mm_Y> = %d  =>   not contiguously stored matrix\n", LINE_BY_LINE);
		flush_info(INFO_1, "<X_mm_Y> = %d  =>   contiguously stored matrix\n", BLOCK);
		flush_info(INFO_1, "<X_mm_Y> = %d  =>   cached matrix\n", CACHE);
		flush_info(INFO_1, "<X_mm_Y> = %d  =>   no matrix stored\n", EMPTY);
		flush_info(INFO_1, "<size_Y>      =>   size of kernel cache in MB\n", EMPTY);
		flush_info(INFO_1, "Here, X=Tr stands for the training matrix and X=Va for the validation matrix. In\n");
		flush_info(INFO_1, "both cases, Y=Pr stands for the pre-kernel matrix, which stores the distances\n");
		flush_info(INFO_1, "between the samples. If <Tr_mm_Pr> is set, then the other three flags <X_mm_Y>\n");
		flush_info(INFO_1, "need to be set, too. The values <sizeY> must only be set if a cache is chosen.\n");
		flush_info(INFO_1, "NOTICE: Not all possible combinations are allowed.\n");
		display_ranges();
		flush_info(INFO_1, "<type>:          integer between %d and %d\n", 0, KERNEL_TYPES_MAX-1);
		flush_info(INFO_1, "<X_mm_Y>:        integer between %d and %d\n", 0, KERNEL_MEMORY_MODELS_MAX-1);
		flush_info(INFO_1, "<size_Y>:        integer not smaller than 1\n");
		display_defaults();
		flush_info(INFO_1, "<type>           = %d\n", sch.kernel_control_train.kernel_type);
		flush_info(INFO_1, "<X_mm_Y>         = %d\n", sch.kernel_control_train.memory_model_kernel);
		flush_info(INFO_1, "<size_Y>         = %d\n", sch.kernel_control_train.pre_cache_size);
		flush_info(INFO_1, "<size>           = %d\n", sch.kernel_control_train.cache_size);
	}
	
	if (error_code == ERROR_clp_svm_train_l)
	{
		display_separator("-l [<option_1> ...]",
						"-l <size> <min_lambda> <max_lambda> [<scale>]\n"
						"-l <lambda_list> [<interpret_as_C>]");
		flush_info(INFO_1, 
		"The first variant sets the size <size> of the lambda grid and its endpoints\n"
		"<min_lambda> and <max_lambda>.\n"
		"The second variant uses <lambda_list>, after ordering, for the lambda grid.\n");
		display_specifics();
		flush_info(INFO_1, 
		"<scale>             Flag indicating whether <min_lambda> is internally\n"
		"                    devided by the average number of samples per fold.\n"
		"<interpret_as_C>    Flag indicating whether the lambda list should be\n"
		"                    interpreted as a list of C values\n");
		display_ranges();
		flush_info(INFO_1, "<size>:             integer >= 1\n");
		flush_info(INFO_1, "<min_lambda>:       float > 0.0\n");
		flush_info(INFO_1, "<max_lambda>:       float > 0.0\n");
		flush_info(INFO_1, "<scale>:            bool\n");
		flush_info(INFO_1, "<interpret_as_C>:   bool\n");
		display_defaults();
		flush_info(INFO_1, "<size>              = %d\n", gch.lambda_size);
		flush_info(INFO_1, "<min_lambda>        = %1.3f\n", gch.min_lambda_unscaled);
		flush_info(INFO_1, "<max_lambda>        = %1.3f\n", gch.max_lambda);
		flush_info(INFO_1, "<scale>             = %d\n", gch.scale_lambda);
		flush_info(INFO_1, "<scale>             = %d\n", gch.interpret_as_C);
	}
	
	
	if (error_code == ERROR_clp_svm_train_L)
	{	
		display_separator("-L <loss> [<clipp>] [<neg_weight> <pos_weight>]");
		flush_info(INFO_1, 
		"Sets the loss that is used to compute empirical errors. The optional <clipp> value\n"
		"specifies where the predictions are clipped during validation. The optional weights\n"
		"can only be set if <loss> specifies a loss that has weights.\n");
		display_specifics();
		flush_info(INFO_1, 
		"<loss> = %d  =>   binary classification loss\n"
		"<loss> = %d  =>   least squares loss\n"
		"<loss> = %d  =>   weighted least squares loss\n"
		"<loss> = %d  =>   pinball loss\n"
		"<loss> = %d  =>   hinge loss\n"
		"<loss> = %d  =>   your own template loss\n", CLASSIFICATION_LOSS, LEAST_SQUARES_LOSS, WEIGHTED_LEAST_SQUARES_LOSS, PINBALL_LOSS, HINGE_LOSS, TEMPLATE_LOSS);
		flush_info(INFO_1, 
		"<clipp> = -1.0  =>   clipp at smallest possible value (depends on labels)\n"
		"<clipp> =  0.0  =>   no clipping is applied\n");
		display_ranges();
		flush_info(INFO_1, "<loss>:       values listed above\n");
		flush_info(INFO_1, "<neg_weight>: float >= -1.0\n");
		flush_info(INFO_1, "<neg_weight>: float > 0.0\n");
		flush_info(INFO_1, "<pos_weight>: float > 0.0\n");
		display_defaults();
		flush_info(INFO_1, "<loss>       = native loss of solver chosen by option -S\n");
		flush_info(INFO_1, "<clipp>      = %1.3f\n", sch.loss_control.clipp_value);
		flush_info(INFO_1, "<neg_weight> = <weight1> set by option -W\n");
		flush_info(INFO_1, "<pos_weight> = <weight2> set by option -W\n");
	}

	if (error_code == ERROR_clp_svm_train_P)
	{
		display_separator("-P <type> [<option_1> ...]",
		"-P %d [<size>]\n"
		"-P %d [<number>]\n"
		"-P %d [<radius>] [<subset_size>]\n"
		"-P %d [<size>] [<reduce>] [<subset_size>]\n"
		"-P %d [<size>] [<ignore_fraction>] [<subset_size>] [<covers>]\n"
		"-P %d [<size>] [<reduce>] [<subset_size>] [<covers>] [<shrink_factor>]\n"
		"       [<max_width>] [<max_depth>]",
		RANDOM_CHUNK_BY_SIZE, RANDOM_CHUNK_BY_NUMBER, VORONOI_BY_RADIUS, VORONOI_BY_SIZE, OVERLAP_BY_SIZE, VORONOI_TREE_BY_SIZE);
// 		display_separator("-P <type> [<number> | <radius> | <size> [extra_option] [subset_size]]");
		
		flush_info(INFO_1, 
		"Selects the working set partition method.\n");
		display_specifics();
		flush_info(INFO_1, "<type> = %d  =>  do not split the working sets\n", NO_PARTITION);
		wch_2.set_partition_method_with_defaults(RANDOM_CHUNK_BY_SIZE);
		flush_info(INFO_1, "<type> = %d  =>  split the working sets in random chunks using maximum <size> of\n"
		"                each chunk.\n" 
		"                Default values are:\n"
		"                <size>            = %d\n", RANDOM_CHUNK_BY_SIZE, wch_2.size_of_cells);
		wch_2.set_partition_method_with_defaults(RANDOM_CHUNK_BY_NUMBER);
		flush_info(INFO_1, "<type> = %d  =>  split the working sets in random chunks using <number> of\n"
		"                chunks.\n"
		"                Default values are:\n"
		"                <size> = %d\n", RANDOM_CHUNK_BY_NUMBER, wch_2.number_of_cells);
		wch_2.set_partition_method_with_defaults(VORONOI_BY_RADIUS);
		flush_info(INFO_1, "<type> = %d  =>  split the working sets into Voronoi subsets of radius <radius>.\n"
		"                If [subset_size] is set, a subset of this size is used to faster\n"
		"                create the Voronoi partition. If subset_size == 0, the entire\n"
		"                data set is used, otherwise, the radius is only approximately\n"
		"                ensured.\n"
		"                Default values are:\n"
		"                <radius>          = %1.3f\n"
		"                <subset_size>     = %d\n", VORONOI_BY_RADIUS, wch_2.radius, wch_2.size_of_dataset_to_find_partition);
		wch_2.set_partition_method_with_defaults(VORONOI_BY_SIZE);
		flush_info(INFO_1, "<type> = %d  =>  split the working sets into Voronoi subsets of maximal size\n"
		"                <size>. The optional flag <reduce> controls whether a heuristic\n"
		"                to reduce the number of cells is used. If [subset_size] is set,\n"
		"                a subset of this size is used to faster create the Voronoi\n"
		"                partition. If subset_size == 0, the entire data set is used, \n"
		"                otherwise, the maximal size is only approximately ensured.\n"
		"                Default values are:\n"
		"                <size>            = %d\n"
		"                <reduce>          = %d\n"
		"                <subset_size>     = %d\n", 
		VORONOI_BY_SIZE, wch_2.size_of_cells, wch_2.reduce_covers, wch_2.size_of_dataset_to_find_partition);
		wch_2.set_partition_method_with_defaults(OVERLAP_BY_SIZE);
		flush_info(INFO_1, "<type> = %d  =>  devide the working sets into overlapping regions of maximal\n"
		"                size <size>. The process of creating regions is stopped when\n"
		"                <size> * <ignore_fraction> samples have not been assigned to\n"
		"                a region. These samples will then be assigned to the closest\n"
		"                region. If <subset_size> is set, a subset of this size is\n"
		"                used to find the regions. If subset_size == 0, the entire\n"
		"                data set is used. Finally, <covers> controls the number of\n"
		"                times the process of finding regions is repeated.\n"
		"                Default values are:.\n"
		"                <size>            = %d\n"
		"                <ignore_fraction> = %1.1f\n"
		"                <subset_size>     = %d\n"
		"                <covers>          = %d\n", 
		OVERLAP_BY_SIZE, wch_2.size_of_cells, wch_2.max_ignore_factor, wch_2.size_of_dataset_to_find_partition, wch_2.number_of_covers);

		wch_2.set_partition_method_with_defaults(VORONOI_TREE_BY_SIZE);
		flush_info(INFO_1, "<type> = %d  =>  split the working sets into Voronoi subsets of maximal size\n"
		"                <size>. The optional flag <reduce> controls whether a heuristic\n"
		"                to reduce the number of cells is used. If [subset_size] is set,\n"
		"                a subset of this size is used to faster create the Voronoi\n"
		"                partition. If subset_size == 0, the entire data set is used, \n"
		"                otherwise, the maximal size is only approximately ensured.\n"
		"                Unlike for <type> = %d, the centers for the Voronoi partition are\n" 
		"                found by a recursive tree approach, which in many cases may be\n"
		"                faster. <shrink_factor> describes by which factor the number of\n"
		"                samples should at least be decreased. The recursion is stoppend\n"
		"                when either <max_width> * <size> is greater than the current\n"
		"                working subset or the <max_tree_depth> is reached. For both\n"
		"                parameters, a value of 0 means that the corresponding condition\n"
		"                above is ignored.\n"
		"                Default values (so far, they are only a brave guess) are:\n"
		"                <size>            = %d\n"
		"                <reduce>          = %d\n"
		"                <subset_size>     = %d\n"
		"                <shrink_factor>   = %1.4f\n"
		"                <max_width>       = %d\n"
		"                <max_tree_depth>  = %d\n", 
		VORONOI_TREE_BY_SIZE, VORONOI_BY_SIZE, wch_2.size_of_cells, wch_2.reduce_covers, wch_2.size_of_dataset_to_find_partition, wch_2.tree_reduction_factor, wch_2.max_theoretical_node_width, wch_2.max_tree_depth);
		
		display_ranges();
		flush_info(INFO_1, "<type>:            integer between %d and %d\n", 0, PARTITION_TYPES_MAX-1);
		flush_info(INFO_1, "<size>:            positive integer\n");
		flush_info(INFO_1, "<number>:          positive integer\n");
		flush_info(INFO_1, "<radius>:          positive real\n");
		flush_info(INFO_1, "<subset_size>:     non-negative integer\n");
		flush_info(INFO_1, "<reduce>:          bool\n");
		flush_info(INFO_1, "<covers>:          positive integer\n");
		flush_info(INFO_1, "<shrink_factor>:   real > 1.0\n");
		flush_info(INFO_1, "<max_width>:       non-negative integer\n");
		flush_info(INFO_1, "<max_tree_depth>:  non-negative integer\n");
		
		display_defaults();
		flush_info(INFO_1, "<type>             = %d\n", wch.partition_method);
	}
	
	if (error_code == ERROR_clp_svm_train_s)
	{
		display_separator("-s <clipp> [<stop_eps>]");
		flush_info(INFO_1, 
		"Sets the value at which the loss is clipped in the solver to <value>. The\n"
		"optional parameter <stop_eps> sets the threshold in the stopping criterion\n"
		"of the solver.\n");
		display_specifics();
		flush_info(INFO_1, "<clipp> = %2.1f  =>   Depending on the solver type clipp either at the\n"
		                   "                     smallest possible value (depends on labels), or\n"
						   "                     do not clipp.\n", ADAPTIVE_CLIPPING);
		flush_info(INFO_1, "<clipp> = %1.1f   =>   no clipping is applied\n", NO_CLIPPING);
		display_ranges();
		flush_info(INFO_1, 
		"<clipp>:    %1.1f or float >= 0.0.\n"  
		"            In addition, if <clipp> > 0.0, then <clipp> must not be smaller\n"
		"            than the largest absolute value of the samples.\n", ADAPTIVE_CLIPPING);
		flush_info(INFO_1, 
		"<stop_eps>: float > 0.0\n");
		display_defaults();
		flush_info(INFO_1, "<clipp>     = %1.1f\n", sch.global_clipp_value);
		flush_info(INFO_1, "<stop_eps>  = %0.4f\n", sch.stop_eps);
	}

	// CHANGE_FOR_OWN_SOLVER
	if (error_code == ERROR_clp_svm_train_S)
	{
		display_separator("-S <solver> [<NNs>]");
		flush_info(INFO_1, 
		"Selects the SVM solver <solver> and the number <NNs> of nearest neighbors used in the working\n"
		"set selection strategy (2D-solvers only).\n");

		display_specifics();
		flush_info(INFO_1, "<solver> = %d  =>  kernel rule for classification\n", KERNEL_RULE);
		flush_info(INFO_1, "<solver> = %d  =>  LS-SVM with 2D-solver\n", SVM_LS_2D);
		flush_info(INFO_1, "<solver> = %d  =>  HINGE-SVM with 2D-solver\n", SVM_HINGE_2D);
		flush_info(INFO_1, "<solver> = %d  =>  QUANTILE-SVM with 2D-solver\n", SVM_QUANTILE);
		flush_info(INFO_1, "<solver> = %d  =>  EXPECTILE-SVM with 2D-solver\n", SVM_EXPECTILE_2D);
		flush_info(INFO_1, "<solver> = %d  =>  Your SVM solver implemented in template_svm.*\n", SVM_TEMPLATE);
		#ifdef OWN_DEVELOP__
			flush_info(INFO_1, "<solver> = %d  =>  LS-SVM with parallel-solver\n", SVM_LS_PAR);
			flush_info(INFO_1, "<solver> = %d  =>  HINGE-SVM with parallel-solver\n", SVM_HINGE_PAR);
		#endif
		display_ranges();
		flush_info(INFO_1, "<solver>: integer between %d and %d\n", 0 , SOLVER_TYPES_MAX-1);
		flush_info(INFO_1, "<NNs>:    integer between 0 and 100\n");
		display_defaults();
		flush_info(INFO_1, "<solver> = %d\n", sch.solver_type);
		flush_info(INFO_1, "<NNs>    = depends on the solver\n");
	}

	if (error_code == ERROR_clp_svm_train_w)
	{
		display_separator("-w <neg_weight> <pos_weight>\n"
					      "-w <min_weight> <max_weight> <size> [<geometric> <swap>]\n"
			              "-w <weight_list> [<swap>]");
		flush_info(INFO_1, 
		"Sets values for the weights, solvers should be trained with. For solvers\n"
		"that do not have weights this option is ignored.\n" 
		"The first variants sets a pair of values.\n"
		"The second variant computes a sequence of weights of length <size>.\n"
		"The third variant takes the list of weights.\n");
		display_specifics();
		flush_info(INFO_1, "size> = 1      =>  <weight1> is the negative weight and <weight2> is the\n");
		flush_info(INFO_1, "                   positive weight.\n");
		flush_info(INFO_1, "<size> > 1     =>  <size> many pairs are computed, where the positive\n"
		                   "                   weights are between <min_weight> and <max_weight> and\n"
						   "                   the negative weights are 1 - pos_weight.\n");
		flush_info(INFO_1, "<geometric>        Flag indicating whether the intermediate positive\n"
		                   "                   weights are geometrically or arithmetically distributed.\n");
		flush_info(INFO_1, "<swap>             Flag indicating whether the role of the positive and\n"
		                   "                   negative weights are interchanged.\n");
		display_ranges();
		flush_info(INFO_1, "<... weight ...>:  float > 0.0 and < 1.0\n");
		flush_info(INFO_1, "<size>:      integer > 0\n");
		flush_info(INFO_1, "<geometric>: bool\n");
		flush_info(INFO_1, "<swap>:      bool\n");
		display_defaults();
		flush_info(INFO_1, "<weight1>   = %1.1f\n", gch.min_weight);
		flush_info(INFO_1, "<weight2>   = %1.1f\n", gch.max_weight);
		flush_info(INFO_1, "<size>      = %d\n", gch.weight_size);
		flush_info(INFO_1, "<geometric> = %d\n", gch.geometric_weights);
		flush_info(INFO_1, "<swap>      = %d\n", gch.swap_weights);
	}
	
	if (error_code == ERROR_clp_svm_train_W)
	{
		display_separator("-W <type>");
		flush_info(INFO_1, 
		"Selects the working set selection method.\n");
		display_specifics();
		flush_info(INFO_1, "<type> = %d  =>  take the entire data set\n", FULL_SET);
		flush_info(INFO_1, "<type> = %d  =>  multiclass 'all versus all'\n", MULTI_CLASS_ALL_VS_ALL);
		flush_info(INFO_1, "<type> = %d  =>  multiclass 'one versus all'\n", MULTI_CLASS_ONE_VS_ALL);
		flush_info(INFO_1, "<type> = %d  =>  bootstrap with <number> resamples of size <size>\n", BOOT_STRAP);
		display_ranges();
		flush_info(INFO_1, "<type>: integer between %d and %d\n", 0, WORKING_SET_SELECTION_TYPES_MAX-1);
		display_defaults();
		flush_info(INFO_1, "<type>    = %d\n", wch.working_set_selection_method);
	}
}


//**********************************************************************************************************************************


void Tcommand_line_parser_svm_train::exit_with_help()
{
	flush_info(INFO_SILENCE, 
	"\n\nsvm-train [options] <trainfile> <logfile> [<summary_log_file>] [<solution_file>]\n"
	"\nsvm-train builds several SVM decision functions with the help of the samples in\n"
	"<trainfile>. The collected information including the validation errors are\n"
	"recorded in <logfile> and an additional .aux file. Optionally, the SVM decision\n"
	"functions can be saved in <solution_file>.\n"
	"\nAllowed extensions:\n"
	"<trainfile>:  .csv, .lsv, and .uci\n"
	"<logfile>:    .log\n"
	"<solfile>:    .sol\n");

	if (full_help == false)
		flush_info(INFO_SILENCE, "\nOptions:");
	display_help(ERROR_clp_gen_d);
	display_help(ERROR_clp_svm_train_f);
	display_help(ERROR_clp_svm_train_g);
	display_help(ERROR_clp_gen_GPU);
	display_help(ERROR_clp_gen_h);
	display_help(ERROR_clp_svm_train_i);
	display_help(ERROR_clp_svm_train_k);
	display_help(ERROR_clp_svm_train_l);
	display_help(ERROR_clp_svm_train_L);
	display_help(ERROR_clp_svm_train_P);
	display_help(ERROR_clp_gen_r);
	display_help(ERROR_clp_svm_train_s);
	display_help(ERROR_clp_svm_train_S);
	display_help(ERROR_clp_gen_T);
	display_help(ERROR_clp_svm_train_w);
	display_help(ERROR_clp_svm_train_W);

	flush_info(INFO_SILENCE,"\n\n");
	copyright();
	flush_exit(ERROR_SILENT, "");
}


//**********************************************************************************************************************************


void Tcommand_line_parser_svm_train::parse(Ttrain_control& train_control, bool read_filenames)
{
	check_parameter_list_size();
	for(current_position=1; current_position<parameter_list_size; current_position++)
		if (Tcommand_line_parser::parse("-d-h-GPU-r-T") == false)
		{
			if(parameter_list[current_position][0] != '-') 
				break;
			if (string(parameter_list[current_position]).size() > 2)
				Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			
			switch(parameter_list[current_position][1])
			{
				case 'a':
					train_control.full_search = not get_next_bool(ERROR_clp_svm_train_a);
					if (next_parameter_is_number() == true)
						train_control.max_number_of_increases = get_next_number(ERROR_clp_svm_train_a, 1);
					if (next_parameter_is_number() == true)
						train_control.max_number_of_worse_gammas = get_next_number(ERROR_clp_svm_train_a, 1);
					break;
				case 'f':
					train_control.fold_control.kind = get_next_enum(ERROR_clp_svm_train_f, FROM_FILE+1, FOLD_CREATION_TYPES_MAX-1);
					train_control.fold_control.number = get_next_number(ERROR_clp_svm_train_f, 1);
					if (train_control.fold_control.kind == RANDOM_SUBSET)
					{
						train_control.fold_control.train_fraction = get_next_number_no_lower_limits(ERROR_clp_svm_train_f, 0.0, 1.0);
						train_control.fold_control.negative_fraction = get_next_number_no_limits(ERROR_clp_svm_train_f, 0.0, 1.0);
					}
					else if (next_parameter_is_number() == true)
						train_control.fold_control.train_fraction = get_next_number_no_lower_limits(ERROR_clp_svm_train_f, 0.0, 1.0);
					break;
				case 'g':
					if (next_parameter_equals('[') == true)
					{
						train_control.grid_control.gammas = get_next_list(ERROR_clp_svm_train_g, 0.0);
						if (train_control.grid_control.gammas[argmin(train_control.grid_control.gammas)] <= 0.0)
							Tcommand_line_parser::exit_with_help(ERROR_clp_svm_train_g);
						
						train_control.grid_control.gamma_size = train_control.grid_control.gammas.size();
					}
					else
					{
						train_control.grid_control.gamma_size = get_next_number(ERROR_clp_svm_train_g, 1);
						train_control.grid_control.min_gamma_unscaled = get_next_number_no_lower_limits(ERROR_clp_svm_train_g, 0.0);
						train_control.grid_control.max_gamma_unscaled = get_next_number_no_lower_limits(ERROR_clp_svm_train_g, 0.0);
						
						if (next_parameter_is_number() == true)
							train_control.grid_control.scale_gamma = get_next_bool(ERROR_clp_svm_train_g);
					}
					break;
				case 'i':
					train_control.solver_control.cold_start = get_next_enum(ERROR_clp_svm_train_i, SOLVER_INIT_ZERO, SOLVER_INIT_FULL);
					train_control.solver_control.warm_start = get_next_enum(ERROR_clp_svm_train_i, 0, SOLVER_INIT_TYPES_MAX-1);
					init_set = true;
					break;
				case 'k':
					train_control.solver_control.kernel_control_train.kernel_type = get_next_enum(ERROR_clp_svm_train_k, GAUSS_RBF, KERNEL_TYPES_MAX-1);
					train_control.solver_control.kernel_control_val.kernel_type = train_control.solver_control.kernel_control_train.kernel_type;
					if (train_control.solver_control.kernel_control_train.is_hierarchical_kernel() == true)
					{
						current_position++;
						train_control.solver_control.kernel_control_train.hierarchical_kernel_control_read_filename = get_next_filename(ERROR_clp_svm_train_k);
						current_position--;
						train_control.solver_control.kernel_control_val.hierarchical_kernel_control_read_filename = train_control.solver_control.kernel_control_train.hierarchical_kernel_control_read_filename;
					}
					if (next_parameter_is_number() == true)
					{
						train_control.solver_control.kernel_control_train.memory_model_pre_kernel = get_next_enum(ERROR_clp_svm_train_k, LINE_BY_LINE, KERNEL_MEMORY_MODELS_MAX-1);
						if (train_control.solver_control.kernel_control_train.memory_model_pre_kernel == CACHE)
							train_control.solver_control.kernel_control_train.pre_cache_size = get_next_number_no_upper_limits(ERROR_clp_svm_train_k, 1);
						
						train_control.solver_control.kernel_control_train.memory_model_kernel = get_next_enum(ERROR_clp_svm_train_k, LINE_BY_LINE, KERNEL_MEMORY_MODELS_MAX-1);
						if (train_control.solver_control.kernel_control_train.memory_model_kernel == CACHE)
							train_control.solver_control.kernel_control_train.cache_size = get_next_number_no_upper_limits(ERROR_clp_svm_train_k, 1);
						
						train_control.solver_control.kernel_control_val.memory_model_pre_kernel = get_next_enum(ERROR_clp_svm_train_k, LINE_BY_LINE, KERNEL_MEMORY_MODELS_MAX-1);
						train_control.solver_control.kernel_control_val.memory_model_kernel = get_next_enum(ERROR_clp_svm_train_k, LINE_BY_LINE, KERNEL_MEMORY_MODELS_MAX-1);
					}
					break;
				case 'l':
					if (next_parameter_equals('[') == true)
					{
						train_control.grid_control.lambdas = get_next_list(ERROR_clp_svm_train_l, 0.0);
						if (train_control.grid_control.lambdas[argmin(train_control.grid_control.lambdas)] <= 0.0)
							Tcommand_line_parser::exit_with_help(ERROR_clp_svm_train_l);
						
						train_control.grid_control.lambda_size = train_control.grid_control.lambdas.size();
						
						if (next_parameter_is_number() == true)
							train_control.grid_control.interpret_as_C = get_next_bool(ERROR_clp_svm_train_l);
					}
					else
					{
						train_control.grid_control.lambda_size = get_next_number(ERROR_clp_svm_train_l, 1);
						train_control.grid_control.min_lambda_unscaled = get_next_number_no_lower_limits(ERROR_clp_svm_train_l, 0.0);
						train_control.grid_control.max_lambda = get_next_number_no_lower_limits(ERROR_clp_svm_train_l, 0.0);
						
						if (next_parameter_is_number() == true)
							train_control.grid_control.scale_lambda = get_next_bool(ERROR_clp_svm_train_l);
					}
					break;
				case 'L':
					loss_ctrl.type = get_next_enum(ERROR_clp_svm_train_L, CLASSIFICATION_LOSS, LOSS_TYPES_MAX-1);
					if (next_parameter_is_number() == true)
						loss_ctrl.clipp_value = get_next_number(ERROR_clp_svm_train_L, -1.0);
		
					if (((loss_ctrl.type == CLASSIFICATION_LOSS) or (loss_ctrl.type == WEIGHTED_LEAST_SQUARES_LOSS) or (loss_ctrl.type == PINBALL_LOSS)) and (next_parameter_is_number() == true))
					{
						loss_ctrl.neg_weight = get_next_number(ERROR_clp_svm_train_L, 0.0);
						loss_ctrl.pos_weight = get_next_number(ERROR_clp_svm_train_L, 0.0);
						loss_weights_set = true;
					}
					loss_set = true;
					break;
				case 'P':
					train_control.working_set_control.partition_method = get_next_enum(ERROR_clp_svm_train_P, NO_PARTITION, PARTITION_TYPES_MAX-1);
					train_control.working_set_control.set_partition_method_with_defaults(train_control.working_set_control.partition_method);
					switch(train_control.working_set_control.partition_method)
					{
						case RANDOM_CHUNK_BY_SIZE:
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_cells = get_next_number(ERROR_clp_svm_train_P, 1);
						break;
						
						case RANDOM_CHUNK_BY_NUMBER:
							if (next_parameter_is_number() == true)
								train_control.working_set_control.number_of_cells = get_next_number(ERROR_clp_svm_train_P, 1);
						break;
						
						case VORONOI_BY_RADIUS:
							if (next_parameter_is_number() == true)
								train_control.working_set_control.radius = get_next_number_no_lower_limits(ERROR_clp_svm_train_P, 0.0);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_dataset_to_find_partition = get_next_number(ERROR_clp_svm_train_P, 0);
								
						break;
						
						case VORONOI_BY_SIZE:
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_cells = get_next_number(ERROR_clp_svm_train_P, 1);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.reduce_covers = get_next_bool(ERROR_clp_svm_train_P);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_dataset_to_find_partition = get_next_number(ERROR_clp_svm_train_P, 0);
						break;
						
						case OVERLAP_BY_SIZE:
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_cells = get_next_number(ERROR_clp_svm_train_P, 1);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.max_ignore_factor = get_next_number(ERROR_clp_svm_train_P, 0.0);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_dataset_to_find_partition = get_next_number(ERROR_clp_svm_train_P, 0);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.number_of_covers = get_next_number(ERROR_clp_svm_train_P, 1);
						break;
						
						case VORONOI_TREE_BY_SIZE:
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_cells = get_next_number(ERROR_clp_svm_train_P, 1);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.reduce_covers = get_next_bool(ERROR_clp_svm_train_P);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.size_of_dataset_to_find_partition = get_next_number(ERROR_clp_svm_train_P, 0);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.max_ignore_factor = get_next_number_no_lower_limits(ERROR_clp_svm_train_P, 1.0);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.max_theoretical_node_width = get_next_number(ERROR_clp_svm_train_P, 0);
							if (next_parameter_is_number() == true)
								train_control.working_set_control.max_tree_depth = get_next_number(ERROR_clp_svm_train_P, 0);
						break;	
					}
					break;
				case 's':
					train_control.solver_control.global_clipp_value = get_next_number(ERROR_clp_svm_train_s, -1.0);
					if ((train_control.solver_control.global_clipp_value != ADAPTIVE_CLIPPING) and (train_control.solver_control.global_clipp_value < 0.0))
						Tcommand_line_parser::exit_with_help(ERROR_clp_svm_train_s);
					clipping_set = true;
					if (next_parameter_is_number() == true)
						train_control.solver_control.stop_eps = get_next_number_no_lower_limits(ERROR_clp_svm_train_s, 0.0);
					break;
				// CHANGE_FOR_OWN_SOLVER
				case 'S':
					train_control.solver_control.solver_type = get_next_enum(ERROR_clp_svm_train_S, 0, AVAILABLE_SOLVER_TYPES_MAX-1);
					if (next_parameter_is_number() == true)
					{
						if ((train_control.solver_control.solver_type == SVM_HINGE_2D) or (train_control.solver_control.solver_type == SVM_LS_2D) or (train_control.solver_control.solver_type == SVM_EXPECTILE_2D) or (train_control.solver_control.solver_type == SVM_QUANTILE))
						{
							if (next_parameter_is_number() == true)
								train_control.solver_control.kernel_control_train.kNNs = get_next_number(ERROR_clp_svm_train_S, 0, 100);
							if (train_control.solver_control.kernel_control_train.kNNs > 0)
								train_control.solver_control.wss_method = USE_NNs;
							else
								train_control.solver_control.wss_method = DONT_USE_NNs;
						}
					}
					break;
				case 'w':
					if (next_parameter_equals('[') == true)
					{
						train_control.grid_control.weights = get_next_list(ERROR_clp_svm_train_w, 0.0, 1.0);
						sort_up(train_control.grid_control.weights);
						if ((train_control.grid_control.weights[0] <= 0.0) or (train_control.grid_control.weights.back() >= 1.0))
							Tcommand_line_parser::exit_with_help(ERROR_clp_svm_train_w);
						
						train_control.grid_control.weight_size = train_control.grid_control.weights.size();
						if (next_parameter_is_number() == true)
							train_control.grid_control.swap_weights = get_next_bool(ERROR_clp_svm_train_w);
					}
					else
					{
						train_control.grid_control.min_weight = get_next_number(ERROR_clp_svm_train_w, 0.0, 1.0);
						train_control.grid_control.max_weight = get_next_number(ERROR_clp_svm_train_w, 0.0, 1.0);
						if (next_parameter_is_number() == true)
						{
							train_control.grid_control.weight_size = get_next_number(ERROR_clp_svm_train_w, 1);
							if (next_parameter_is_number() == true)
							{	
								train_control.grid_control.geometric_weights = get_next_bool(ERROR_clp_svm_train_w);
								if (next_parameter_is_number() == true)
									train_control.grid_control.swap_weights = get_next_bool(ERROR_clp_svm_train_w);
							}
						}
					}
					if (train_control.grid_control.weight_size > 1)
						weight_display_mode = DISPLAY_WEIGHTS_AND_ERROR;
					break;
				case 'W':
					train_control.working_set_control.working_set_selection_method = get_next_enum(ERROR_clp_svm_train_W, FULL_SET, WORKING_SET_SELECTION_TYPES_MAX-1);
					if (train_control.working_set_control.working_set_selection_method == BOOT_STRAP)
					{
						train_control.working_set_control.number_of_tasks = get_next_number(ERROR_clp_svm_train_W, 1);
						train_control.working_set_control.size_of_tasks = get_next_number(ERROR_clp_svm_train_W, 1);
					}
					break;
					
				default:
					Tcommand_line_parser::exit_with_help(ERROR_clp_gen_unknown_option);
			}
		}


	// Read filenames

	if (read_filenames == true)
	{
		train_filename = get_next_labeled_data_filename(ERROR_clp_gen_missing_train_file_name);
		train_control.write_log_train_filename = get_next_log_filename(ERROR_clp_gen_missing_log_file_name);
		train_control.write_aux_train_filename = convert_log_to_aux_filename(train_control.write_log_train_filename);
		
		if (current_position < parameter_list_size)
			train_control.summary_log_filename = get_next_log_filename();
		
		if (current_position < parameter_list_size)
			train_control.write_sol_train_filename = get_next_filename(); 
	}
	
	train_control.parallel_control = get_parallel_control();
	make_consistent(train_control);
};

#endif


