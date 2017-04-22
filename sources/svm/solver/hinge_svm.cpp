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


#if !defined (HINGE_SVM_CPP) 
	#define HINGE_SVM_CPP


#include "sources/svm/solver/hinge_svm.h"


#include "sources/shared/system_support/timing.h"
#include "sources/shared/system_support/simd_basics.h"
#include "sources/shared/system_support/memory_allocation.h"
#include "sources/shared/basic_functions/flush_print.h"



#ifdef  COMPILE_WITH_CUDA__
	#include "sources/shared/system_support/cuda_memory_operations.h"
	#include "sources/shared/system_support/cuda_simple_vector_operations.h"
#endif


//**********************************************************************************************************************************

Thinge_svm::Thinge_svm()
{
	old_alpha_ALGD = NULL;
	old_weights_ALGD = NULL;	
	
	prediction_init_neg_ALGD = NULL;
	prediction_init_pos_ALGD = NULL;
}


//**********************************************************************************************************************************

Thinge_svm::~Thinge_svm()
{
	my_dealloc_ALGD(&old_alpha_ALGD);
	my_dealloc_ALGD(&old_weights_ALGD);
	
	my_dealloc_ALGD(&prediction_init_neg_ALGD);
	my_dealloc_ALGD(&prediction_init_pos_ALGD);
}


//**********************************************************************************************************************************

void Thinge_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	#ifdef  COMPILE_WITH_CUDA__
		prediction_init_neg_GPU.resize(parallel_control.GPUs);
		prediction_init_pos_GPU.resize(parallel_control.GPUs);
	#endif

	Tbasic_svm::reserve(solver_control, parallel_control);
}


//**********************************************************************************************************************************

void Thinge_svm::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	Tbasic_svm::load(training_kernel, validation_kernel);
	if (is_first_team_member() == true)
	{
		my_realloc_ALGD(&old_alpha_ALGD, training_set_size);
		my_realloc_ALGD(&old_weights_ALGD, training_set_size);
		
		my_realloc_ALGD(&prediction_init_neg_ALGD, validation_set_size);
		my_realloc_ALGD(&prediction_init_pos_ALGD, validation_set_size);

		bSV_list.reserve(training_set_size);
		uSV_list.reserve(training_set_size);
		nuSV_list.reserve(training_set_size);
		nbSV_list.reserve(training_set_size);
		
		count_labels(neg_train_size, pos_train_size, training_label_ALGD, training_set_size);
		count_labels(neg_val_size, pos_val_size, validation_label_ALGD, validation_set_size);
	}
	
	if (GPUs > 0)
	{
		#ifdef  COMPILE_WITH_CUDA__
			my_alloc_GPU(&(prediction_init_neg_GPU[get_thread_id()]), kernel_control_GPU[get_thread_id()].col_set_size);
			my_alloc_GPU(&(prediction_init_pos_GPU[get_thread_id()]), kernel_control_GPU[get_thread_id()].col_set_size);
			cudaDeviceSynchronize();
		#endif
	}
}


//**********************************************************************************************************************************

void Thinge_svm::clear()
{
	Tbasic_svm::clear();
	#ifdef  COMPILE_WITH_CUDA__
		if ((GPUs > 0) and (validation_set_size > 0))
		{
			my_dealloc_GPU(&(prediction_init_neg_GPU[get_thread_id()]));
			my_dealloc_GPU(&(prediction_init_pos_GPU[get_thread_id()]));
		}
	#endif
}



//**********************************************************************************************************************************

void Thinge_svm::initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	simdd__ factor_simdd;
	Tthread_chunk thread_chunk;


	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);

	if (C_old == NO_PREVIOUS_LAMBDA)
	{
		for (i=thread_chunk.start_index; i<thread_chunk.stop_index; i++)
			weight_ALGD[i] = C_current * ( (training_label_ALGD[i] < 0.0)? train_val_info.neg_weight : train_val_info.pos_weight);

		for (i=training_set_size;i<training_set_size_aligned;i++)
		{
			weight_ALGD[i] = 0.0;
			old_weights_ALGD[i] = 0.0;
			alpha_ALGD[i] = 0.0;
			old_alpha_ALGD[i] = 0.0;
			gradient_ALGD[i] = 1.0;
		}
	}
	else
	{
		factor_simdd = assign_simdd(C_current / C_old);
		for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
		{
			copy_CL(old_weights_ALGD+i, weight_ALGD+i);
			mult_CL(weight_ALGD+i, factor_simdd, weight_ALGD+i);
		}
	}

	sync_threads();
	switch (init_method)
	{
		case SOLVER_INIT_ZERO:
			zero_box(train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		case SOLVER_INIT_FULL:
			full_box(train_val_info);
			break;
		case SOLVER_INIT_RECYCLE:
			keep_box(train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		case SOLVER_INIT_EXPAND_UNIFORMLY:
			scale_box(C_current / C_old, train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		case SOLVER_INIT_EXPAND:
			expand_box(train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		case SOLVER_INIT_SHRINK_UNIFORMLY:
			scale_box(C_current / C_old, train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		case SOLVER_INIT_SHRINK:
			shrink_box(train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		default:
			flush_exit(1, "Unknown solver initialization method %d for hinge-SVM solver.", init_method);
			break;
	}
	
	solver_clipp_value = 1.0;

	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);
	
	if (is_first_team_member() == true)
		flush_info(INFO_DEBUG, "\nInit method %d. norm_etc = %f, slack_sum = %f, pd_gap = %f, Solver clipping at %f, Validation clipping at %f", init_method, norm_etc_global[0], slack_sum_global[0], primal_dual_gap[0], solver_clipp_value, validation_clipp_value);
}


//**********************************************************************************************************************************

inline void Thinge_svm::zero_box(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	simdd__ slack_sum_simdd;
	unsigned thread_id;
	Tthread_chunk thread_chunk;

	
	init_iterations = 1;
	val_iterations = 0;
	
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	slack_sum_simdd = assign_simdd(0.0);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		assign_CL(alpha_ALGD+i, 0.0);
		assign_CL(gradient_ALGD+i, 1.0);
		cache_prefetch(weight_ALGD+i+32, PREFETCH_L1);
		sum_CL(slack_sum_simdd, weight_ALGD+i);
	}
	slack_sum_local[thread_id] = reduce_sums_simdd(slack_sum_simdd);
	slack_sum_global[thread_id] = reduce_sums(&slack_sum_local[0]);
	
	norm_etc_global[thread_id] = 0.0;
	primal_dual_gap[thread_id] = slack_sum_global[thread_id];
};


//**********************************************************************************************************************************

inline void Thinge_svm::full_box(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	unsigned j;
	simdd__ gradient_simdd;
	simdd__ neg_weight_simdd;
	simdd__ pos_weight_simdd;
	unsigned thread_id;
	Tthread_chunk val_thread_chunk;
	Tthread_chunk train_thread_chunk;
	double* restrict__ kernel_row1_ALGD;

	train_val_info.init_iterations = training_set_size;
	train_val_info.val_iterations = 1;
		
	thread_id = get_thread_id();
	train_thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	if (is_first_team_member() == true)
		solution_old.resize(training_set_size);
	sync_threads();

	norm_etc_local[thread_id] = 0.0;
	slack_sum_local[thread_id] = 0.0;
	for (i=train_thread_chunk.start_index; i<train_thread_chunk.stop_index; i++)
	{
		kernel_row1_ALGD = training_kernel->row(i);

		gradient_simdd = assign_simdd(0.0);
		for (j=0;j+CACHELINE_STEP <= training_set_size_aligned;j+=CACHELINE_STEP)
		{
			cache_prefetch(weight_ALGD+j+32, PREFETCH_L1);
			cache_prefetch(kernel_row1_ALGD+j+32, PREFETCH_NO);
			fuse_mult_sum_CL(weight_ALGD+j, kernel_row1_ALGD+j, gradient_simdd);
		}
		gradient_ALGD[i] = 1.0 - reduce_sums_simdd(gradient_simdd);

		alpha_ALGD[i] = weight_ALGD[i];
		norm_etc_local[thread_id] = norm_etc_local[thread_id] + alpha_ALGD[i] * gradient_ALGD[i];
		slack_sum_local[thread_id] = slack_sum_local[thread_id] + weight_ALGD[i] * clipp_0max(gradient_ALGD[i], 2.0);

		solution_old.index[i] = i;
		solution_old.coefficient[i] = weight_ALGD[i] * training_label_ALGD[i];
	}
	norm_etc_global[thread_id] = reduce_sums(&norm_etc_local[0]);
	slack_sum_global[thread_id] = reduce_sums(&slack_sum_local[0]);

	if (prediction_ALGD != NULL)
	{
		if (GPUs > 0)
		{
			#ifdef  COMPILE_WITH_CUDA__
				init_full_predictions_on_GPU(train_val_info);
				cudaDeviceSynchronize();
			#endif
		}
		else
		{
			val_thread_chunk = get_thread_chunk(validation_set_size, CACHELINE_STEP);

			neg_weight_simdd = mult_simdd(assign_simdd(C_current), assign_simdd(-train_val_info.neg_weight)); 
			pos_weight_simdd = mult_simdd(assign_simdd(C_current), assign_simdd(train_val_info.pos_weight)); 
			for (j=val_thread_chunk.start_index; j+CACHELINE_STEP <= val_thread_chunk.stop_index_aligned; j+=CACHELINE_STEP)
			{
				cache_prefetch(prediction_init_neg_ALGD+j+32, PREFETCH_L1);
				cache_prefetch(prediction_init_pos_ALGD+j+32, PREFETCH_L1);
				for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
					store_simdd(prediction_ALGD+j+i, add_simdd(mult_simdd(neg_weight_simdd, load_simdd(prediction_init_neg_ALGD+j+i)), mult_simdd(pos_weight_simdd, load_simdd(prediction_init_pos_ALGD+j+i))));
			}
		}
	}
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
};



//**********************************************************************************************************************************

inline void Thinge_svm::expand_box(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	unsigned j;
	double ratio;
	Tthread_chunk thread_chunk;
	

	ratio = C_current / C_old;
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	
	if (is_first_team_member() == true)
	{
		bSV_list.clear();
		uSV_list.clear();
		for (i=0;i<training_set_size;i++)
			if (old_weights_ALGD[i]  == alpha_ALGD[i])
				bSV_list.push_back(i);
			else if (alpha_ALGD[i] > 0.0)
				uSV_list.push_back(i);
	}
	sync_threads();
	
	// Quickly update if possible

	if (bSV_list.size() == 0)
	{
		keep_box(init_iterations, val_iterations);
		return;
	}
	else if (bSV_list.size() == training_set_size)
	{
		scale_box(ratio, init_iterations, val_iterations);
		return;
	}

	// Otherwise, do it the hard way ...
	
	if (is_first_team_member() == true)
		for (j=0; j<bSV_list.size(); j++)
			alpha_ALGD[bSV_list[j]] = ratio * alpha_ALGD[bSV_list[j]];
	sync_threads();

	if (uSV_list.size() < bSV_list.size())
	{
		init_iterations = unsigned(uSV_list.size()) + 1;
		for (i=thread_chunk.start_index; i<thread_chunk.stop_index_aligned; i++)
			gradient_ALGD[i] = 1.0 - ratio + ratio * gradient_ALGD[i];
		for (j=0; j<uSV_list.size(); j++)
			add_to_gradient(assign_simdd((ratio - 1.0) * alpha_ALGD[uSV_list[j]]), training_kernel->row(uSV_list[j]));
		scale_predictions(ratio);
	}
	else
	{
		init_iterations = unsigned(bSV_list.size());
		for (j=0; j<bSV_list.size(); j++)
			add_to_gradient(assign_simdd((old_weights_ALGD[bSV_list[j]] - weight_ALGD[bSV_list[j]])), training_kernel->row(bSV_list[j]));
	}
	
	compute_gap_from_scratch();
};


//**********************************************************************************************************************************

inline void Thinge_svm::keep_box(unsigned& init_iterations, unsigned& val_iterations)
{
	init_iterations = 1;
	val_iterations = 0;
	
	compute_gap_from_scratch();
};

//**********************************************************************************************************************************

inline void Thinge_svm::shrink_box(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	unsigned j;
	double ratio;
	simdd__ factor_simdd;
	simdd__ addend_simdd;
	Tthread_chunk thread_chunk;


	ratio = C_current / C_old;	
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	if (is_first_team_member() == true)
	{
		bSV_list.clear();
		nuSV_list.clear();
		nbSV_list.clear();
		for (i=0;i<training_set_size;i++)
		{
			old_alpha_ALGD[i] = alpha_ALGD[i];
			if (alpha_ALGD[i] == old_weights_ALGD[i])
				bSV_list.push_back(i);
			else
			{	
				if (alpha_ALGD[i] > weight_ALGD[i])
					nbSV_list.push_back(i);
				else if (alpha_ALGD[i] > 0.0)
					nuSV_list.push_back(i);
			}
		}
		init_iterations = unsigned(nbSV_list.size() + min(bSV_list.size(), nuSV_list.size()));
	}
	sync_threads();
	
	// Quickly update if possible

	if ((bSV_list.size() == 0) and (nbSV_list.size() == 0))
	{
		keep_box(init_iterations, val_iterations);
		return;
	}
	else if ((nbSV_list.size() == 0) and (nuSV_list.size() == 0))
	{
		scale_box(ratio, init_iterations, val_iterations);
		return;
	}
	

	// Otherwise update in the more expensive way ...  
	
	if (is_first_team_member() == true)
		for (i=0; i<bSV_list.size(); i++)
			alpha_ALGD[bSV_list[i]] = weight_ALGD[bSV_list[i]];
	if (is_last_team_member() == true)
		for (j=0; j<nbSV_list.size(); j++)
			alpha_ALGD[nbSV_list[j]] = weight_ALGD[nbSV_list[j]];
	sync_threads();
	
	if (bSV_list.size() > nuSV_list.size())
	{
		addend_simdd = assign_simdd(1.0 - ratio);
		factor_simdd = assign_simdd(ratio);
		for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
		{
			cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
			fuse_mult_add4_CL(gradient_ALGD+i, factor_simdd, gradient_ALGD+i, addend_simdd);
		}
		for (j=0; j<nuSV_list.size(); j++)
			add_to_gradient(assign_simdd(ratio * old_alpha_ALGD[nuSV_list[j]] - alpha_ALGD[nuSV_list[j]]), training_kernel->row(nuSV_list[j]));
		for (j=0; j<nbSV_list.size(); j++)
			add_to_gradient(assign_simdd(ratio * old_alpha_ALGD[nbSV_list[j]] - weight_ALGD[nbSV_list[j]]), training_kernel->row(nbSV_list[j]));
	}
	else
	{
		for (j=0; j<bSV_list.size(); j++)
			add_to_gradient(assign_simdd(old_weights_ALGD[bSV_list[j]] - weight_ALGD[bSV_list[j]]), training_kernel->row(bSV_list[j]));
		for (j=0; j<nbSV_list.size(); j++)
			add_to_gradient(assign_simdd(old_alpha_ALGD[nbSV_list[j]] - weight_ALGD[nbSV_list[j]]), training_kernel->row(nbSV_list[j])); 
	}
	compute_gap_from_scratch();
};


//**********************************************************************************************************************************

inline void Thinge_svm::scale_box(double factor, unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	simdd__ factor_simdd;
	simdd__ addend_simdd;
	simdd__ norm_etc_simdd;
	simdd__ slack_sum_simdd;
	unsigned thread_id;
	Tthread_chunk thread_chunk;

	
	init_iterations = 1;
	val_iterations = 1;
	
	factor_simdd = assign_simdd(factor);
	addend_simdd = assign_simdd(1.0 - factor);
	norm_etc_simdd = assign_simdd(0.0);
	slack_sum_simdd = assign_simdd(0.0);
	
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(alpha_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(weight_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		
		mult_CL(alpha_ALGD+i, factor_simdd, alpha_ALGD+i);
		fuse_mult_add4_CL(gradient_ALGD+i, factor_simdd, gradient_ALGD+i, addend_simdd);
		fuse_mult_sum_CL(alpha_ALGD+i, gradient_ALGD+i, norm_etc_simdd);
		add_to_slack_sum_CL(slack_sum_simdd, gradient_ALGD+i, weight_ALGD+i);
	}
	norm_etc_local[thread_id] = reduce_sums_simdd(norm_etc_simdd);
	slack_sum_local[thread_id] = reduce_sums_simdd(slack_sum_simdd);

	norm_etc_global[thread_id] = reduce_sums(&norm_etc_local[0]);
	slack_sum_global[thread_id] = reduce_sums(&slack_sum_local[0]);
	
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
	
	scale_predictions(factor);
};


//**********************************************************************************************************************************


inline void Thinge_svm::scale_predictions(double factor)
{
	unsigned i;
	simdd__ factor_simdd;
	Tthread_chunk thread_chunk;
	
	if (is_first_team_member() == true)
		for (i=0;i<solution_old.size();i++)
			solution_old.coefficient[i] = factor * solution_old.coefficient[i];
	
	if (prediction_ALGD == NULL)
		return;
		
	if (GPUs > 0)
	{
		#ifdef  COMPILE_WITH_CUDA__
			mult_vector_on_GPU(factor, prediction_GPU[get_thread_id()], kernel_control_GPU[get_thread_id()].col_set_size);
			cudaDeviceSynchronize();
		#endif
	}
	else
	{
		thread_chunk = get_thread_chunk(validation_set_size, CACHELINE_STEP);
		factor_simdd = assign_simdd(factor);
		for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
		{
			cache_prefetch(prediction_ALGD+i, PREFETCH_L1);
			mult_CL(prediction_ALGD+i, factor_simdd, prediction_ALGD+i);
		}
	}
}


//**********************************************************************************************************************************

void Thinge_svm::count_labels(unsigned& neg_sample_no, unsigned& pos_sample_no, double* labels, unsigned size)
{
	unsigned i;
	
	neg_sample_no = 0;
	pos_sample_no = 0;
	for (i=0;i<size;i++)
		if (labels[i] > 0.0)
			pos_sample_no++;
		else
			neg_sample_no++;
}


//**********************************************************************************************************************************

void Thinge_svm::get_train_error(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	double prediction;

	if (is_first_team_member() == true)
	{
		train_val_info.train_error = 0.0;
		train_val_info.pos_train_error = 0.0;
		train_val_info.neg_train_error = 0.0;
	
		for (i=0;i<training_set_size;i++)
		{
			prediction = training_label_ALGD[i] * (1.0 - gradient_ALGD[i]);
			train_val_info.train_error = train_val_info.train_error + loss_function.evaluate(training_label_ALGD[i], prediction);
			train_val_info.neg_train_error = train_val_info.neg_train_error + neg_classification_loss(training_label_ALGD[i], prediction); 
			train_val_info.pos_train_error = train_val_info.pos_train_error + pos_classification_loss(training_label_ALGD[i], prediction); 
		}
		train_val_info.train_error = train_val_info.train_error / double(training_set_size);
		train_val_info.neg_train_error = ( (neg_train_size > 0)? train_val_info.neg_train_error / double(neg_train_size):train_val_info.neg_train_error);
		train_val_info.pos_train_error = ( (pos_train_size > 0)? train_val_info.pos_train_error / double(pos_train_size):train_val_info.pos_train_error);
	}
}

//**********************************************************************************************************************************

void Thinge_svm::get_val_error(Tsvm_train_val_info& train_val_info)
{
	unsigned j;
	
	Tbasic_svm::get_val_error(train_val_info);
	if (is_first_team_member() == true)
	{
		train_val_info.pos_val_error = 0.0;
		train_val_info.neg_val_error = 0.0;
		
		for (j=0;j<validation_set_size;j++)
		{
			train_val_info.neg_val_error = train_val_info.neg_val_error + neg_classification_loss(validation_label_ALGD[j], prediction_ALGD[j] + offset);
			train_val_info.pos_val_error = train_val_info.pos_val_error + pos_classification_loss(validation_label_ALGD[j], prediction_ALGD[j] + offset);
		}
		train_val_info.neg_val_error = ( (neg_val_size > 0)? train_val_info.neg_val_error / double(neg_val_size):train_val_info.neg_train_error);
		train_val_info.pos_val_error = ( (pos_val_size > 0)? train_val_info.pos_val_error / double(pos_val_size):train_val_info.pos_train_error);
	}
}


//**********************************************************************************************************************************

void Thinge_svm::initialize_new_weight_and_lambda_line(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	unsigned j;
	Tthread_chunk thread_chunk;
	double* restrict__ kernel_row_ALGD;
	
	
	Tbasic_svm::initialize_new_weight_and_lambda_line(train_val_info);
	sync_threads_and_get_time_difference(train_val_info.val_time, train_val_info.val_time);

	if ((validation_set_size != 0) and (solver_ctrl.cold_start == SOLVER_INIT_FULL))
	{
		if (GPUs > 0)
		{
			#ifdef  COMPILE_WITH_CUDA__
				init_neg_and_pos_predictions_on_GPU(prediction_init_neg_GPU[get_thread_id()], -1.0);
				init_neg_and_pos_predictions_on_GPU(prediction_init_pos_GPU[get_thread_id()], 1.0);
			#endif
		}
		else 
		{
			thread_chunk = get_thread_chunk(validation_set_size, CACHELINE_STEP);
			for (j=thread_chunk.start_index; j+CACHELINE_STEP <= thread_chunk.stop_index_aligned; j+=CACHELINE_STEP)
			{
				assign_CL(prediction_init_neg_ALGD+j, 0.0);
				assign_CL(prediction_init_pos_ALGD+j, 0.0);
			}
			for (i=0;i<training_set_size;i++)
			{
				kernel_row_ALGD = validation_kernel->row(i, thread_chunk.start_index, thread_chunk.stop_index);
				if (training_label_ALGD[i] < 0.0)
					for (j=thread_chunk.start_index; j+CACHELINE_STEP <= thread_chunk.stop_index_aligned; j+=CACHELINE_STEP)
					{
						cache_prefetch(prediction_init_neg_ALGD+j+32, PREFETCH_L1);
						add_CL(prediction_init_neg_ALGD+j, prediction_init_neg_ALGD+j, kernel_row_ALGD+j);
					}
				else
					for (j=thread_chunk.start_index; j+CACHELINE_STEP <= thread_chunk.stop_index_aligned; j+=CACHELINE_STEP)
					{
						cache_prefetch(prediction_init_pos_ALGD+j+32, PREFETCH_L1);
						add_CL(prediction_init_pos_ALGD+j, prediction_init_pos_ALGD+j, kernel_row_ALGD+j);
					}
			}
		}
	}
	sync_threads_and_get_time_difference(train_val_info.val_time, train_val_info.val_time);
}



//**********************************************************************************************************************************

void Thinge_svm::build_solution(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	unsigned iv;
	unsigned size;

	if (is_first_team_member() == true)
	{
		this->build_SV_list(train_val_info);
		size = unsigned(SV_list.size());
		solution_current.resize(size);
		
		for (i=0;i<size;i++)
		{
			iv = SV_list[i];
			solution_current.coefficient[i] = alpha_ALGD[iv] * training_label_ALGD[iv];
			solution_current.index[i] = iv;
		}
	}
}


//**********************************************************************************************************************************

void Thinge_svm::build_bSV_list(Tsvm_train_val_info& train_val_info)
{
	unsigned i;

	if (is_first_team_member() == true)
	{
		bSV_list.clear();
		for (i=0;i<training_set_size;i++)
			if (alpha_ALGD[i] == weight_ALGD[i])
				bSV_list.push_back(i);
		train_val_info.bSVs = unsigned(bSV_list.size());
	}
}


//**********************************************************************************************************************************


#ifndef COMPILE_WITH_CUDA__

void Thinge_svm::init_full_predictions_on_GPU(Tsvm_train_val_info train_val_info){}
void Thinge_svm::init_neg_and_pos_predictions_on_GPU(double* predictions_init_GPU, double sign){}

#endif

#endif

