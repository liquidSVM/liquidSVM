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






//**********************************************************************************************************************************


inline simdd__ Thinge_svm::clipp_02_simdd(simdd__ arg_simdd)
{
	return clipp_0max_simdd(arg_simdd, assign_simdd(2.0));
}


//**********************************************************************************************************************************


inline void Thinge_svm::add_to_slack_sum_CL(simdd__& slack_sum_simdd, double* gradient, double* weight)
{
	unsigned i;
	
	for(i=0; i<CACHELINE_STEP; i+=SIMD_WORD_SIZE)
		slack_sum_simdd = fuse_mult_add_simdd(load_simdd(weight+i), clipp_02_simdd(load_simdd(gradient+i)), slack_sum_simdd);
}


//**********************************************************************************************************************************


inline void Thinge_svm::add_to_gradient(simdd__ factor_simdd, double* restrict__ kernel_row_ALGD)
{
	unsigned i;
	Tthread_chunk thread_chunk;

	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(kernel_row_ALGD+i+32, PREFETCH_NO);
		fuse_mult_add3_CL(factor_simdd, kernel_row_ALGD+i, gradient_ALGD+i);
	}
}



//**********************************************************************************************************************************

inline void Thinge_svm::compute_gap_from_scratch()
{
	unsigned i;
	simdd__ norm_etc_simdd;
	simdd__ slack_sum_simdd;
	Tthread_chunk thread_chunk;
	unsigned thread_id;
	
	
	norm_etc_simdd = assign_simdd(0.0);
	slack_sum_simdd = assign_simdd(0.0);
	thread_id = get_thread_id();
	thread_chunk = get_thread_chunk(training_set_size, CACHELINE_STEP);
	
	for (i=thread_chunk.start_index; i+CACHELINE_STEP <= thread_chunk.stop_index_aligned; i+=CACHELINE_STEP)
	{
		cache_prefetch(alpha_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(weight_ALGD+i+32, PREFETCH_L1);
		cache_prefetch(gradient_ALGD+i+32, PREFETCH_L1);
		fuse_mult_sum_CL(alpha_ALGD+i, gradient_ALGD+i, norm_etc_simdd);
		add_to_slack_sum_CL(slack_sum_simdd, gradient_ALGD+i, weight_ALGD+i);
	}
	norm_etc_local[thread_id] = reduce_sums_simdd(norm_etc_simdd);
	slack_sum_local[thread_id] = reduce_sums_simdd(slack_sum_simdd);

	norm_etc_global[thread_id] = reduce_sums(&norm_etc_local[0]);
	slack_sum_global[thread_id] = reduce_sums(&slack_sum_local[0]);

	primal_dual_gap[thread_id] = slack_sum_global[thread_id] - norm_etc_global[thread_id];
}


//**********************************************************************************************************************************


inline double Thinge_svm::clipp_0max(double x, double max)
{
	x = ( (x > 0.0)? x:0.0);
	return ( (x < max)? x:max);
}
