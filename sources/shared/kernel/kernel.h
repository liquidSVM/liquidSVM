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


#if !defined (KERNEL_H)
	#define KERNEL_H



#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/basic_types/cache_lru.h"
#include "sources/shared/basic_types/small_ordered_index_set.h"
#include "sources/shared/system_support/thread_manager.h"
#include "sources/shared/kernel/kernel_control.h"
#include "sources/shared/kernel/kernel_control_gpu.h"
#include "sources/shared/kernel/kernel_functions.h"


#include <vector>
using namespace std;


//**********************************************************************************************************************************


class Tkernel: public Tthread_manager
{
	public:
		Tkernel();
		~Tkernel();
		
		void clear();
		void reserve(const Tparallel_control& parallel_ctrl, const Tkernel_control& kernel_control);
		
		void load(const Tdataset& row_data_set, const Tdataset& col_data_set, double& build_time, double& transfer_time);
		void assign(double gamma, double& build_time, double& transfer_time, double& kNN_build_time);
		
		unsigned get_row_set_size() const;
		unsigned get_col_set_size() const;
		
		double* restrict__ get_row_labels_ALGD();
		double* restrict__ get_col_labels_ALGD();

		inline double* row(unsigned i);
		inline double* row(unsigned i, unsigned start_column, unsigned end_column);
		inline double entry(unsigned row, unsigned column);
		
		void clear_cache_stats();
		void get_cache_stats(double& pre_cache_hits, double& cache_hits) const;

		inline bool all_kNN_assigned() const;
		inline unsigned get_max_kNNs() const;
		inline Tsubset_info get_kNNs(unsigned i);
		vector <Tsubset_info> get_kNN_list() const;
		Tkernel_control_GPU get_kernel_control_GPU() const;
	
	protected:
		void clear_on_GPU();
		void reserve_on_GPU();
		
	private:
		void pre_assign(double& build_time, double& transfer_time);

		inline double compute_entry(unsigned i, unsigned j);
		inline double compute_entry(unsigned i, unsigned j, double pre_kernel_value);
		void clear_matrix(vector <double*>& rows, unsigned memory_model);
		void reserve_matrix(vector <double*>& rows, unsigned memory_model, bool triangular);
		
		void clear_kNN_list();
		void reserve_kNN_list();
		void assign_kNN_list();
		void find_kNNs(unsigned i, unsigned cache_kernel_row_index);
		
		friend void compute_kernel_on_GPU(const Tkernel& kernel);
		friend void compute_pre_kernel_on_GPU(const Tkernel& kernel);
		
		inline double* pre_row_from_cache(unsigned i);

		void set_remainder_to_zero();
		
		vector <Tsample*> row_data_set;
		vector <Tsample*> col_data_set;
		vector <Tordered_index_set*> kNN_list;

		bool assigned;
		bool all_kNNs_assigned;
		unsigned max_kNNs;
		double gamma_factor;
		double kernel_offset;

		unsigned row_set_size;
		unsigned col_set_size;
		unsigned max_aligned_col_set_size;
		unsigned current_aligned_col_set_size;
		bool remainder_is_zero;
		
		double* restrict__ row_labels_ALGD;
		double* restrict__ col_labels_ALGD;
		double* restrict__ kernel_row_ALGD;
		
		vector <double*> kernel_row;
		vector <double*> pre_kernel_row;
		vector <unsigned> kNNs_found;
		
		Tcache_lru cache;
		Tcache_lru pre_cache;
		
		Tkernel_control kernel_control;

		vector <unsigned> permutated_indices;
		
		vector <Tkernel_control_GPU> kernel_control_GPU;
		#ifdef  COMPILE_WITH_CUDA__
			vector <double*> row_labels_GPU;
			vector <double*> col_labels_GPU;
			vector <double*> matrix_CPU;
			vector <double*> pre_matrix_CPU;
			vector <unsigned*> hierarchical_coordinate_intervals_GPU;
			vector <double*> hierarchical_weights_squared_GPU;
		#endif
			
			
			
// 	HIERARCHICAL KERNEL DEVELOPMENT

		bool hierarchical_kernel_flag;
		vector <Tdataset> hierarchical_row_set;
		vector <Tdataset> hierarchical_col_set;
		double weights_square_sum;
};


//**********************************************************************************************************************************

#include "sources/shared/kernel/kernel.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/kernel/kernel.cpp"
#endif

#endif






