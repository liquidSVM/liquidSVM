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




#include "sources/shared/kernel/kernel_functions.h"


//*********************************************************************************************************************************

template <class float_type> void Tsvm_decision_function_manager::setup_GPU(Tsvm_decision_function_GPU_control<float_type>* GPU_control, Tdataset& test_set_chunk, bool sparse_evaluation)
{
	#ifdef  COMPILE_WITH_CUDA__
		unsigned i;
		unsigned j;
		unsigned df;
		unsigned max_size_df;
		unsigned max_size_SV_gamma;
		unsigned max_test_chunk_size;
		unsigned corrected_train_size;
		double bytes_per_test_sample;
		vector <float_type> offsets;
		vector <float_type> clipp_values;
		vector <float_type> scaled_gamma_list;
		vector <unsigned> hierarchical_coordinate_intervals;
		vector <unsigned> inverse_SVs;
		vector <unsigned> sample_no_as_SV;
		vector <float_type> coefficients_on_SVs;
		Tthread_chunk thread_chunk;
		Tdataset SV_set;
		Tdataset full_coord_dataset;
		Tcuda_timer cuda_timer;
		bool keep_test_data_original;
		
		
	// 	Copy basic information 
		

		GPU_control->kernel_type = kernel_type;
		GPU_control->full_kernel_type = full_kernel_type;
		GPU_control->number_of_decision_functions = size();

		GPU_control->test_set_size = test_set.size();
		GPU_control->sparse_evaluation = sparse_evaluation;
		
		keep_test_data_original = GPU_control->keep_test_data;
		GPU_control->keep_test_data = false;
		
		GPU_control->decision_function_chunks = NULL;


	// Copy specific information for hierarchical kernels if needed

		if (hierarchical_kernel_flag == true)
		{
			GPU_control->weights_square_sum = weights_square_sum;
			GPU_control->number_of_nodes = kernel_control.hierarchical_coordinates.size();
			GPU_control->total_number_of_hierarchical_coordinates = kernel_control.get_hierarchical_coordinate_intervals(hierarchical_coordinate_intervals);
			
			cuda_timer.start_timing(test_info.GPU_misc_upload_time);
			my_alloc_GPU(&GPU_control->hierarchical_weights_squared_GPU, GPU_control->number_of_nodes);
			my_alloc_GPU(&GPU_control->hierarchical_coordinate_intervals_GPU, hierarchical_coordinate_intervals.size());
			
			copy_to_GPU(&kernel_control.hierarchical_weights_squared[0], GPU_control->hierarchical_weights_squared_GPU, GPU_control->number_of_nodes);
			copy_to_GPU(&hierarchical_coordinate_intervals[0], GPU_control->hierarchical_coordinate_intervals_GPU, hierarchical_coordinate_intervals.size());
			cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
		}
		
		
	// Copy SV sets to the GPU
		
		get_time_difference(test_info.data_convert_time, test_info.data_convert_time);
		if (hierarchical_kernel_flag == false)
			for(i=0; i<SVs.size(); i++)
				SV_set.push_back(training_set.sample(SVs[i]));
		else
		{
			kernel_control.convert_to_hierarchical_GPU_data_set(hierarchical_training_set, full_coord_dataset, 0, hierarchical_training_set.size());
			for(i=0; i<SVs.size(); i++)
				SV_set.push_back(full_coord_dataset.sample(SVs[i]));
		}
		get_time_difference(test_info.data_convert_time, test_info.data_convert_time);

		GPU_control->dim = SV_set.dim();
		GPU_control->SV_set_size = SVs.size();
		GPU_control->SV_set_mem_size = SV_set.required_memory_on_GPU(0, SV_set.size());
		GPU_control->training_set_size = training_set.size();

		cuda_timer.start_timing(test_info.GPU_data_upload_time);
		GPU_control->SV_set_GPU = SV_set.upload_to_GPU<float_type>(0, GPU_control->SV_set_size);
		cuda_timer.stop_timing(test_info.GPU_data_upload_time);
		
		
	// Copy coefficients and indices from the decision functions to the GPU

		inverse_SVs.assign(training_set.size(), 0);
		for (j=0; j<SVs.size(); j++)
			inverse_SVs[SVs[j]] = j;
		
		
	// This line makes sure that there is no error occurring if a decision function has zero weight but many SVs
		
		if (sparse_evaluation == false)
		{
			max_size_df = SVs.size();
			GPU_control->SVs_of_decision_functions_GPU = NULL;
		}
		else
		{
			max_size_df = size_of_largest_decision_function();
			
			cuda_timer.start_timing(test_info.GPU_misc_upload_time);
			my_alloc_GPU(&GPU_control->SVs_of_decision_functions_GPU, size() * max_size_df);
			cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
		}
		GPU_control->decision_function_max_size = max_size_df;

		cuda_timer.start_timing(test_info.GPU_misc_upload_time);
		my_alloc_GPU(&GPU_control->decision_function_size_GPU, size());
		my_alloc_GPU(&GPU_control->coefficient_GPU, size() * max_size_df);
		cuda_timer.stop_timing(test_info.GPU_misc_upload_time);

		for (df=0; df<size(); df++)
		{
			sample_no_as_SV.resize(decision_functions[df].sample_number.size());
			for (i=0; i<decision_functions[df].sample_number.size(); i++)
				sample_no_as_SV[i] = inverse_SVs[decision_functions[df].sample_number[i]];
			
			if (sparse_evaluation == false)
			{
				coefficients_on_SVs.assign(SVs.size(), 0.0);
				for (i=0; i<decision_functions[df].sample_number.size(); i++)
					coefficients_on_SVs[sample_no_as_SV[i]] = float_type(decision_functions[df].coefficient[i]);

				cuda_timer.start_timing(test_info.GPU_misc_upload_time);
				copy_to_GPU(max_size_df, &(GPU_control->decision_function_size_GPU[df]));
				copy_to_GPU(coefficients_on_SVs, &(GPU_control->coefficient_GPU[df * max_size_df]));
				cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
			}
			else
			{
				cuda_timer.start_timing(test_info.GPU_misc_upload_time);
				copy_to_GPU(unsigned(decision_functions[df].size()), &(GPU_control->decision_function_size_GPU[df]));
				copy_to_GPU(decision_functions[df].coefficient, &(GPU_control->coefficient_GPU[df * max_size_df]));
				copy_to_GPU(sample_no_as_SV, &(GPU_control->SVs_of_decision_functions_GPU[df * max_size_df]));
				cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
			}
		}

		cuda_timer.start_timing(test_info.GPU_misc_upload_time);
		my_alloc_GPU(&GPU_control->SVs_GPU, SVs.size());
		copy_to_GPU(SVs, GPU_control->SVs_GPU);
		cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
		
		
	// Copy gamma information to the GPU
		
		GPU_control->gamma_list_size = gamma_list.size();
		max_size_SV_gamma = size_of_largest_SV_with_gamma();
		GPU_control->SVs_with_gamma_max_size = max_size_SV_gamma;
		
		cuda_timer.start_timing(test_info.GPU_misc_upload_time);
		my_alloc_GPU(&GPU_control->SVs_with_gamma_size_GPU, GPU_control->gamma_list_size);
		my_alloc_GPU(&GPU_control->SVs_with_gamma_GPU, GPU_control->gamma_list_size * max_size_SV_gamma);
		cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
		
		for (i=0; i<GPU_control->gamma_list_size; i++)
		{
			scaled_gamma_list.push_back(float_type(compute_gamma_factor(kernel_type, gamma_list[i])));
			
			sample_no_as_SV.resize(SVs_with_gamma[i].size());
			for (j=0; j<SVs_with_gamma[i].size(); j++)
				sample_no_as_SV[j] = inverse_SVs[SVs_with_gamma[i][j]];

			cuda_timer.start_timing(test_info.GPU_misc_upload_time);
			copy_to_GPU(unsigned(SVs_with_gamma[i].size()), &(GPU_control->SVs_with_gamma_size_GPU[i]));
			copy_to_GPU(sample_no_as_SV, &(GPU_control->SVs_with_gamma_GPU[i * max_size_SV_gamma]));
			cuda_timer.stop_timing(test_info.GPU_misc_upload_time);
		}

		cuda_timer.start_timing(test_info.GPU_misc_upload_time);
		my_alloc_GPU(&GPU_control->gamma_list_GPU, GPU_control->gamma_list_size);
		copy_to_GPU(scaled_gamma_list, GPU_control->gamma_list_GPU);
		
		my_alloc_GPU(&GPU_control->gamma_indices_GPU, gamma_indices.size());
		copy_to_GPU(gamma_indices, GPU_control->gamma_indices_GPU);
		cuda_timer.stop_timing(test_info.GPU_misc_upload_time);


	// Copy clipping values and offsets to GPU;

		offsets.resize(size());
		clipp_values.resize(size());
		for (df=0; df<size(); df++)
		{
			offsets[df] = float_type(decision_functions[df].get_offset());
			clipp_values[df] = float_type(decision_functions[df].get_clipp_value());
		}

		cuda_timer.start_timing(test_info.GPU_misc_upload_time);
		my_alloc_GPU(&GPU_control->offsets, size());
		copy_to_GPU(offsets, GPU_control->offsets);
		
		my_alloc_GPU(&GPU_control->clipp_values, size());
		copy_to_GPU(clipp_values, GPU_control->clipp_values);
		cuda_timer.stop_timing(test_info.GPU_misc_upload_time);

		
	// Prepare kernel rows and evaluations_GPU

		thread_chunk = get_thread_chunk(test_set.size());

		corrected_train_size = unsigned(SVs.size());
		bytes_per_test_sample = (double(corrected_train_size) * (double(gamma_list.size() + 1)) + double(size())) * double(sizeof(float_type));
		max_test_chunk_size = unsigned(ceil(available_memory_on_GPU(0.9) / bytes_per_test_sample));
		GPU_control->number_of_chunks = unsigned(ceil(double(thread_chunk.size) / double(max_test_chunk_size)));
		max_test_chunk_size = unsigned(ceil(double(thread_chunk.size) / double(GPU_control->number_of_chunks)));
		
		GPU_control->start_index = thread_chunk.start_index;
		GPU_control->stop_index = thread_chunk.stop_index;
		GPU_control->max_test_chunk_size = max_test_chunk_size;
	
		cuda_timer.start_timing(test_info.GPU_misc_upload_time);
		my_alloc_GPU(&GPU_control->pre_kernel_GPU, max_test_chunk_size * corrected_train_size);
		my_alloc_GPU(&GPU_control->kernel_GPU, max_test_chunk_size * corrected_train_size * gamma_list.size());
		my_alloc_GPU(&GPU_control->evaluations_GPU, max_test_chunk_size * size());
		cuda_timer.stop_timing(test_info.GPU_misc_upload_time);


	// Provide test set in the correct format
		
		test_set_chunk.enforce_ownership();
		GPU_control->hierarchical_kernel_flag = hierarchical_kernel_flag;
		
		get_time_difference(test_info.data_convert_time, test_info.data_convert_time);
		if (hierarchical_kernel_flag == true)
		{
			kernel_control.convert_to_hierarchical_GPU_data_set(hierarchical_test_set, full_coord_dataset, 0, hierarchical_test_set.size());
			test_set_chunk = full_coord_dataset;
		}
		else
			test_set_chunk = test_set;
		get_time_difference(test_info.data_convert_time, test_info.data_convert_time);
		GPU_control->test_set_mem_size = test_set_chunk.required_memory_on_GPU(0, test_set_chunk.size());
		
		
		GPU_control->keep_test_data = keep_test_data_original;
	#endif
}



//*********************************************************************************************************************************

template <class float_type> void Tsvm_decision_function_manager::clean_GPU(Tsvm_decision_function_GPU_control<float_type>* GPU_control)
{
	my_dealloc_GPU(&GPU_control->decision_function_chunks);
	
	my_dealloc_GPU(&GPU_control->SV_set_GPU);
	
	my_dealloc_GPU(&GPU_control->gamma_list_GPU);
	my_dealloc_GPU(&GPU_control->SVs_with_gamma_GPU);
	my_dealloc_GPU(&GPU_control->SVs_with_gamma_size_GPU);
	my_dealloc_GPU(&GPU_control->gamma_indices_GPU);
	
	my_dealloc_GPU(&GPU_control->decision_function_size_GPU);
	my_dealloc_GPU(&GPU_control->coefficient_GPU);
	my_dealloc_GPU(&GPU_control->SVs_of_decision_functions_GPU);
	my_dealloc_GPU(&GPU_control->SVs_GPU);
	
	my_dealloc_GPU(&GPU_control->offsets);
	my_dealloc_GPU(&GPU_control->clipp_values);
	
	my_dealloc_GPU(&GPU_control->kernel_GPU);
	my_dealloc_GPU(&GPU_control->pre_kernel_GPU);
	my_dealloc_GPU(&GPU_control->evaluations_GPU);
	
	if (hierarchical_kernel_flag == true)
	{
		my_dealloc_GPU(&GPU_control->hierarchical_coordinate_intervals_GPU);
		my_dealloc_GPU(&GPU_control->hierarchical_weights_squared_GPU);
	}
}





