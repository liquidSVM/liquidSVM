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


#if !defined (QUANTILE_SVM_CPP) 
	#define QUANTILE_SVM_CPP


#include "sources/svm/solver/quantile_svm.h"


#include "sources/shared/system_support/timing.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_types/vector.h"





//**********************************************************************************************************************************

Tquantile_svm::Tquantile_svm()
{
	tau_initialized = false;
	old_alpha_ALGD = NULL;
	training_label_transformed_ALGD = NULL;
}


//**********************************************************************************************************************************

Tquantile_svm::~Tquantile_svm()
{
	my_dealloc_ALGD(&old_alpha_ALGD);
	my_dealloc_ALGD(&training_label_transformed_ALGD);
}

//**********************************************************************************************************************************

void Tquantile_svm::reserve(Tsvm_solver_control& solver_control, const Tparallel_control& parallel_control)
{
	weight_display_mode = DISPLAY_WEIGHTS_NO_ERROR;
	
	solver_control.kernel_control_train.include_labels = false;

	if (solver_control.cold_start == SOLVER_INIT_DEFAULT)
		solver_control.cold_start = SOLVER_INIT_ZERO;
	else if (solver_control.cold_start != SOLVER_INIT_ZERO)
		flush_exit(1, "\nQuantile solver must not be cold started by method %d.\n" 
			"Allowed methods are %d.", solver_control.cold_start, SOLVER_INIT_ZERO);
		
	if (solver_control.warm_start == SOLVER_INIT_DEFAULT)
		solver_control.warm_start = SOLVER_INIT_EXPAND;
	else if ((solver_control.warm_start != SOLVER_INIT_ZERO) and (solver_control.warm_start != SOLVER_INIT_RECYCLE) and (solver_control.warm_start != SOLVER_INIT_EXPAND_UNIFORMLY) and (solver_control.warm_start != SOLVER_INIT_EXPAND) and (solver_control.warm_start != SOLVER_INIT_SHRINK) and (solver_control.warm_start != SOLVER_INIT_SHRINK_UNIFORMLY))
		flush_exit(1, "\nQuantile solver must not be warm started by method %d.\n" 
			"Allowed methods %d, %d, %d, %d, %d, and %d.", solver_control.warm_start, SOLVER_INIT_ZERO, SOLVER_INIT_RECYCLE, SOLVER_INIT_EXPAND_UNIFORMLY, SOLVER_INIT_EXPAND, SOLVER_INIT_SHRINK_UNIFORMLY, SOLVER_INIT_SHRINK);

	
	Tsvm_2D_solver_generic_base_name::reserve(solver_control, parallel_control);
	Tbasic_svm::reserve(solver_control, parallel_control);
}


//**********************************************************************************************************************************

void Tquantile_svm::load(Tkernel* training_kernel, Tkernel* validation_kernel)
{
	Tbasic_svm::load(training_kernel, validation_kernel);
	
	if (is_first_team_member() == true)
	{
		bSV_list.reserve(training_set_size);
		uSV_list.reserve(training_set_size);
		low_SV_list.reserve(training_set_size);
		up_SV_list.reserve(training_set_size);
		lown_SV_list.reserve(training_set_size);
		upn_SV_list.reserve(training_set_size);
		
		my_realloc_ALGD(&old_alpha_ALGD, training_set_size);
		my_realloc_ALGD(&training_label_transformed_ALGD, training_set_size);
	}
}


//**********************************************************************************************************************************

void Tquantile_svm::initialize_new_lambda_line(Tsvm_train_val_info& train_val_info)
{
	tau_initialized = false; 
	Tbasic_svm::initialize_new_lambda_line(train_val_info);
}


//**********************************************************************************************************************************

void Tquantile_svm::get_train_error(Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	double prediction;

	train_val_info.train_error = 0.0;
	for (i=0; i<training_set_size; i++)
	{
		prediction = inverse_transform_label(training_label_transformed_ALGD[i] - gradient_ALGD[i]);
		train_val_info.train_error = train_val_info.train_error + loss_function.evaluate(training_label_ALGD[i], prediction);
	}
	train_val_info.train_error = train_val_info.train_error / double(training_set_size);
}

//**********************************************************************************************************************************

void Tquantile_svm::initialize_solver(unsigned init_method, Tsvm_train_val_info& train_val_info)
{
	unsigned i;
	std::pair<double, double> qt;
	
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);


	for (i=training_set_size;i<training_set_size_aligned;i++)
	{
		alpha_ALGD[i] = 0.0;
		old_alpha_ALGD[i] = 0.0;
		gradient_ALGD[i] = 0.0;
		training_label_ALGD[i] = 0.0;
		training_label_transformed_ALGD[i] = 0.0;
	}
	
	if (is_first_team_member())
	{
		if (tau_initialized == false)
		{
			tau = train_val_info.pos_weight / (train_val_info.neg_weight + train_val_info.pos_weight);
			tau_initialized = true;
			
			if (training_set_size > 0)
			{
				qt = quantile(convert_to_vector(training_label_ALGD, training_set_size), tau);
				label_offset = 0.5 * (qt.first + qt.second);
			}
			else
				label_offset = 0.0;
			
			for (i=0; i<training_set_size; i++)
				training_label_transformed_ALGD[i] = transform_label(training_label_ALGD[i]);
		}

		low_weight_old = low_weight;
		up_weight_old = up_weight;
		
		up_weight = tau * C_current;
		low_weight = - (1.0 - tau) * C_current;
	}
	
	sync_threads();
	switch (init_method)
	{
		case SOLVER_INIT_ZERO:
			init_zero(train_val_info.init_iterations, train_val_info.val_iterations);
			break;
		case SOLVER_INIT_RECYCLE:
			init_keep(train_val_info.init_iterations, train_val_info.val_iterations);
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
			flush_exit(1, "Unknown solver initialization method %d for quantile solver.", init_method);
			break;
	}
		
	sync_threads_and_get_time_difference(train_val_info.init_time, train_val_info.init_time);

	if (solver_ctrl.global_clipp_value == ADAPTIVE_CLIPPING)
	{
		if (classification_data == true) 
			solver_clipp_value = 1.0;
		else
			solver_clipp_value = 0.0;
	}
	else 
		solver_clipp_value = solver_ctrl.clipp_value;

	if (is_first_team_member() == true)
		flush_info(INFO_DEBUG, "\nInit method %d. norm_etc = %f, slack_sum = %f, pd_gap = %f, Solver clipping at %f, Validation clipping at %f", init_method, norm_etc_global[0], slack_sum_global[0], primal_dual_gap[0], solver_clipp_value, validation_clipp_value);
}


//**********************************************************************************************************************************

void Tquantile_svm::init_zero(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	unsigned thread_id;

	
	init_iterations = 1;
	val_iterations = 0;
	
	thread_id = get_thread_id();

	for (i=0; i<training_set_size; i++)
	{
		alpha_ALGD[i] = 0.0;
		gradient_ALGD[i] = training_label_transformed_ALGD[i];
	}
	
	slack_sum_local[thread_id] = compute_slack_sum();
	slack_sum_global[thread_id] = slack_sum_local[thread_id];
	
	norm_etc_local[thread_id] = 0.0;
	norm_etc_global[thread_id] = 0.0;
	primal_dual_gap[thread_id] = slack_sum_global[thread_id];
};


//**********************************************************************************************************************************

void Tquantile_svm::init_keep(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned thread_id;
	
	init_iterations = 0;
	val_iterations = 0;
	
	thread_id = get_thread_id();
	slack_sum_global[thread_id] = compute_slack_sum();
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
};


//**********************************************************************************************************************************

void Tquantile_svm::expand_box(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	unsigned j;
	unsigned thread_id;
	double ratio;
	Tthread_chunk thread_chunk;
	

	thread_id = get_thread_id();
	ratio = C_current / C_old;
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	
	if (is_first_team_member() == true)
	{
		up_SV_list.clear();
		low_SV_list.clear();
		uSV_list.clear();
		for (i=0;i<training_set_size;i++)
			if (alpha_ALGD[i] == low_weight_old) 
				low_SV_list.push_back(i);
			else if (alpha_ALGD[i] == up_weight_old)
				up_SV_list.push_back(i);
			else
				uSV_list.push_back(i);
	}
	sync_threads();

	
	// Quickly update if possible
	
	if ((low_SV_list.size() == 0) and (up_SV_list.size() == 0))
	{
		init_keep(init_iterations, val_iterations);
		return;
	}
	else if (low_SV_list.size() + up_SV_list.size() == training_set_size)
	{
		scale_box(ratio, init_iterations, val_iterations);
		return;
	}

	// Otherwise, do it the hard way ...

	if (is_first_team_member() == true)
	{
		for (j=0; j<up_SV_list.size(); j++)
			alpha_ALGD[up_SV_list[j]] = ratio * alpha_ALGD[up_SV_list[j]];
		for (j=0; j<low_SV_list.size(); j++)
			alpha_ALGD[low_SV_list[j]] = ratio * alpha_ALGD[low_SV_list[j]];
	}
	sync_threads();

	if (uSV_list.size() < low_SV_list.size() + up_SV_list.size())
	{
		if (is_first_team_member() == true)
			init_iterations = unsigned(uSV_list.size()) + 1;
		for (i=thread_chunk.start_index; i<thread_chunk.stop_index_aligned; i++)
			gradient_ALGD[i] = (1.0 - ratio) * training_label_transformed_ALGD[i] + ratio * gradient_ALGD[i];
		for (j=0; j<uSV_list.size(); j++)
			add_to_gradient(assign_simdd((ratio - 1.0) * alpha_ALGD[uSV_list[j]]), training_kernel->row(uSV_list[j]));
	}
	else
	{
		if (is_first_team_member() == true)
			init_iterations = unsigned(low_SV_list.size()) + unsigned(up_SV_list.size());
		for (j=0; j<low_SV_list.size(); j++)
			add_to_gradient(assign_simdd(low_weight_old - low_weight), training_kernel->row(low_SV_list[j]));

		for (j=0; j<up_SV_list.size(); j++)
			add_to_gradient(assign_simdd(up_weight_old - up_weight), training_kernel->row(up_SV_list[j]));
	}
	
	compute_norm_etc();
	slack_sum_local[thread_id] = compute_slack_sum();
	slack_sum_global[thread_id] = slack_sum_local[thread_id];
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
};


//**********************************************************************************************************************************

inline void Tquantile_svm::shrink_box(unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	unsigned j;
	double ratio;
	unsigned thread_id;
	simdd__ factor_1_simdd;
	simdd__ factor_2_simdd;
	Tthread_chunk thread_chunk;

	
	ratio = C_current / C_old;
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);

	if (is_first_team_member() == true)
	{
		up_SV_list.clear();
		low_SV_list.clear();
		upn_SV_list.clear();
		lown_SV_list.clear();
		uSV_list.clear();
		for (i=0;i<training_set_size;i++)
		{
			old_alpha_ALGD[i] = alpha_ALGD[i];
			if (alpha_ALGD[i] < 0.0)
			{
				if (alpha_ALGD[i] == low_weight_old)
					low_SV_list.push_back(i);
				else if (alpha_ALGD[i] <= low_weight)
					lown_SV_list.push_back(i);
				else
					uSV_list.push_back(i);
			}
			else
			{
				if (alpha_ALGD[i] == up_weight_old)
					up_SV_list.push_back(i);
				else if (alpha_ALGD[i] >= up_weight)
					upn_SV_list.push_back(i);
				else
					uSV_list.push_back(i);
			}
		}
	}
	sync_threads();
	
	// Quickly update if possible

	if (uSV_list.size() == training_set_size)
	{
		init_keep(init_iterations, val_iterations);
		return;
	}
	else if (low_SV_list.size() + up_SV_list.size() == training_set_size)
	{
		scale_box(ratio, init_iterations, val_iterations);
		return;
	}
	

	// Otherwise update in the more expensive way ...  
	
	if (is_first_team_member() == true)
	{
		for (i=0; i<low_SV_list.size(); i++)
			alpha_ALGD[low_SV_list[i]] = low_weight;
		for (i=0; i<lown_SV_list.size(); i++)
			alpha_ALGD[lown_SV_list[i]] = low_weight;
	}
	if (is_last_team_member() == true)
	{
		for (i=0; i<up_SV_list.size(); i++)
			alpha_ALGD[up_SV_list[i]] = up_weight;
		for (i=0; i<upn_SV_list.size(); i++)
			alpha_ALGD[upn_SV_list[i]] = up_weight;
	}
	sync_threads();
	

	
	if (uSV_list.size() < low_SV_list.size() + up_SV_list.size())
	{
		factor_1_simdd = assign_simdd(1.0 - ratio);
		factor_2_simdd = assign_simdd(ratio);
		for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
		{
			cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
			cache_prefetch(training_label_transformed_ALGD+i+32, PREFETCH_L1);
			
			fuse_mult_add5_CL(gradient_ALGD+i, factor_1_simdd, training_label_transformed_ALGD+i, factor_2_simdd, gradient_ALGD+i);
		}
	
		for (j=0; j<uSV_list.size(); j++)
			add_to_gradient(assign_simdd(ratio * old_alpha_ALGD[uSV_list[j]] - alpha_ALGD[uSV_list[j]]), training_kernel->row(uSV_list[j]));
		for (j=0; j<lown_SV_list.size(); j++)
			add_to_gradient(assign_simdd(ratio * old_alpha_ALGD[lown_SV_list[j]] - low_weight), training_kernel->row(lown_SV_list[j])); 
		for (j=0; j<upn_SV_list.size(); j++)
			add_to_gradient(assign_simdd(ratio * old_alpha_ALGD[upn_SV_list[j]] - up_weight), training_kernel->row(upn_SV_list[j])); 
		
		if (is_first_team_member() == true)
			init_iterations = uSV_list.size() + lown_SV_list.size() + upn_SV_list.size();
	}
	else
	{
		for (j=0; j<low_SV_list.size(); j++)
			add_to_gradient(assign_simdd(low_weight_old - low_weight), training_kernel->row(low_SV_list[j]));
		for (j=0; j<lown_SV_list.size(); j++)
			add_to_gradient(assign_simdd(old_alpha_ALGD[lown_SV_list[j]] - low_weight), training_kernel->row(lown_SV_list[j])); 
		
		for (j=0; j<up_SV_list.size(); j++)
			add_to_gradient(assign_simdd(up_weight_old - up_weight), training_kernel->row(up_SV_list[j]));
		for (j=0; j<upn_SV_list.size(); j++)
			add_to_gradient(assign_simdd(old_alpha_ALGD[upn_SV_list[j]] - up_weight), training_kernel->row(upn_SV_list[j])); 
		
		if (is_first_team_member() == true)
			init_iterations = low_SV_list.size() + lown_SV_list.size() + up_SV_list.size() + upn_SV_list.size();
	}

	
	slack_sum_global[get_thread_id()] = slack_sum_local[get_thread_id()];
	compute_norm_etc();
	slack_sum_local[thread_id] = compute_slack_sum();
	slack_sum_global[thread_id] = slack_sum_local[thread_id];
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
};


//**********************************************************************************************************************************

inline void Tquantile_svm::compute_norm_etc()
{
	unsigned i;
	simdd__ norm_etc_simdd;
	Tthread_chunk thread_chunk;
	unsigned thread_id;
	
	
	norm_etc_simdd = assign_simdd(0.0);
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(alpha_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		
		fuse_mult_sum_CL(alpha_ALGD+i, gradient_ALGD+i, norm_etc_simdd);
	}
	norm_etc_local[thread_id] = reduce_sums_simdd(norm_etc_simdd);
	norm_etc_global[thread_id] = reduce_sums(&norm_etc_local[0]);
}


//**********************************************************************************************************************************

void Tquantile_svm::scale_box(double factor, unsigned& init_iterations, unsigned& val_iterations)
{
	unsigned i;
	unsigned thread_id;
	simdd__ factor_simdd;
	Tthread_chunk thread_chunk;	

	init_iterations = 1;
	val_iterations = 1;
	
	thread_id = get_thread_id();
	
	if (is_first_team_member() == true)
		for (i=0;i<solution_old.size();i++)
			solution_old.coefficient[i] = factor * solution_old.coefficient[i];
		
	
	thread_chunk = get_thread_chunk(validation_set_size, CACHELINE_STEP);
	factor_simdd = assign_simdd(factor);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(prediction_ALGD+i, PREFETCH_L1);
		mult_CL(prediction_ALGD+i, factor_simdd, prediction_ALGD+i);
	}
	

	norm_etc_local[thread_id] = 0.0;
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	for (i=thread_chunk.start_index; i< thread_chunk.stop_index_aligned; i++)
	{
		alpha_ALGD[i] = factor * alpha_ALGD[i];
		gradient_ALGD[i] = training_label_transformed_ALGD[i] - factor * (training_label_transformed_ALGD[i] - gradient_ALGD[i]); 
		norm_etc_local[thread_id] = norm_etc_local[thread_id] + alpha_ALGD[i] * gradient_ALGD[i];
	}
	sync_threads();
	
	slack_sum_local[thread_id] = compute_slack_sum();
	slack_sum_global[thread_id] = slack_sum_local[thread_id];
	
	norm_etc_global[thread_id] = reduce_sums(&norm_etc_local[0]);
	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
	
};






//**********************************************************************************************************************************

void Tquantile_svm::build_solution(Tsvm_train_val_info& train_val_info)
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
			solution_current.coefficient[i] = label_spread * alpha_ALGD[iv];
			solution_current.index[i] = iv;
		}
		
		offset = label_offset;
	}
}



//**********************************************************************************************************************************

void Tquantile_svm::core_solver(Tsvm_train_val_info& train_val_info)
{
	core_solver_generic_part(train_val_info);
	if (is_first_team_member() == true)
	{	
		MM_CACHELINE_FLUSH(&slack_sum_local[0]);
	}
	
	sync_threads();
	slack_sum_global[get_thread_id()] = slack_sum_local[0];
}







//**********************************************************************************************************************************


inline double Tquantile_svm::compute_slack_sum()
{
	unsigned i;
	unsigned j;
	double pos_slack_sum;
	double neg_slack_sum;
	simdd__ pos_slack_sum_simdd;
	simdd__ neg_slack_sum_simdd;
	simdd__ grad_simdd;
	simdd__ neg_clipp_value_simdd;
	simdd__ pos_clipp_value_simdd;
	simdd__ label_simdd;
	simdd__ clipped_pred_diff_simdd;
	
	
	pos_slack_sum_simdd = assign_simdd(0.0);
	neg_slack_sum_simdd = assign_simdd(0.0);
	
	if (solver_clipp_value == 0.0)
		for (i=0; i+CACHELINE_STEP <= training_set_size_aligned; i+=CACHELINE_STEP)
		{
			cache_prefetch(gradient_ALGD+32+i, PREFETCH_L1);
		
			for(j=i; j<i+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
			{
				grad_simdd = load_simdd(gradient_ALGD+j);
				pos_slack_sum_simdd = add_simdd(pos_slack_sum_simdd, max_simdd(zero_simdd, grad_simdd));
				neg_slack_sum_simdd = add_simdd(neg_slack_sum_simdd, min_simdd(zero_simdd, grad_simdd));
			}
		}
	else
	{
		pos_clipp_value_simdd = assign_simdd(transform_label(solver_clipp_value));
		neg_clipp_value_simdd = assign_simdd(transform_label(-solver_clipp_value));
		for (i=0; i+CACHELINE_STEP <= training_set_size_aligned; i+=CACHELINE_STEP)
		{
			cache_prefetch(gradient_ALGD+32+i, PREFETCH_L1);
			cache_prefetch(training_label_transformed_ALGD+32+i, PREFETCH_L1);
		
			for(j=i; j<i+CACHELINE_STEP; j+=SIMD_WORD_SIZE)
			{
				label_simdd = load_simdd(training_label_transformed_ALGD+j);
				clipped_pred_diff_simdd = sub_simdd(label_simdd, clipp_simdd(sub_simdd(label_simdd, load_simdd(gradient_ALGD+j)), neg_clipp_value_simdd, pos_clipp_value_simdd));
				
				pos_slack_sum_simdd = add_simdd(pos_slack_sum_simdd, max_simdd(zero_simdd, clipped_pred_diff_simdd));
				neg_slack_sum_simdd = add_simdd(neg_slack_sum_simdd, min_simdd(zero_simdd, clipped_pred_diff_simdd));
			}
		}
	}
	pos_slack_sum = reduce_sums_simdd(pos_slack_sum_simdd);
	neg_slack_sum = reduce_sums_simdd(neg_slack_sum_simdd);
	
	return up_weight * pos_slack_sum + low_weight * neg_slack_sum; 
}














#endif







