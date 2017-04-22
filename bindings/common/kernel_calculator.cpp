// Copyright 2015-2017 Philipp Thomann
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


#include "kernel_calculator.h"

#if !defined (KERNEL_CALCULATOR_CPP)
	#define KERNEL_CALCULATOR_CPP

Tkernel_calculator::~Tkernel_calculator()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tkernel_calculator.");
	kernel.clear();
}


//**********************************************************************************************************************************

void Tkernel_calculator::clear_threads()
{
	Tthread_manager_active::clear_threads();
	kernel.clear_threads();
}


//**********************************************************************************************************************************


void Tkernel_calculator::calculate(Tkernel_control kernel_ctrl, Tdataset dataset)
{
	kernel_control = kernel_ctrl;
	data_set = dataset;

	kernel_control.kNNs = 0;
	kernel_control.same_data_sets = true;
	kernel_control.include_labels = false;
	kernel_control.max_col_set_size = data_set.size();
	kernel_control.max_row_set_size = data_set.size();
	kernel_control.kernel_store_on_GPU = false;
	kernel_control.pre_kernel_store_on_GPU = false; //true;
	kernel_control.split_matrix_on_GPU_by_rows = false; // true;
	kernel_control.allowed_percentage_of_GPU_RAM = 0.95;// * double(dataset.size()) / double(cv_control.fold_manager.max_train_size() + cv_control.fold_manager.max_val_size());
	kernel_control.read_hierarchical_kernel_info_from_file();
	kernel.reserve(get_parallel_control(), kernel_control);

//	kernel.clear();

	start_threads();


}


void Tkernel_calculator::thread_entry()
{
	unsigned start_index_1;
	unsigned start_index_2;
	unsigned stop_index_1;
	unsigned stop_index_2;



	if (is_first_team_member() == true)
	{
		flush_info(INFO_3,"\n");

		if (order_data == SOLVER_ODER_DATA_SPATIALLY)
		{
			get_aligned_chunk(data_set.size(), 2*get_team_size(), 0, start_index_1, stop_index_1);
			get_aligned_chunk(data_set.size(), 2*get_team_size(), 1, start_index_2, stop_index_2);
			data_set.group_spatially(stop_index_2 - start_index_1, get_team_size(), permutation);
		}
		else
			permutation = id_permutation(data_set.size());
	}
	lazy_sync_threads();


	Ttrain_val_info train_val_info;
	train_val_info.gamma = gamma;

	kernel.load(data_set, data_set, train_val_info.train_pre_build_time, train_val_info.train_build_transfer_time);

	kernel.assign(train_val_info.gamma, train_val_info.train_build_time, train_val_info.train_build_transfer_time, train_val_info.train_kNN_build_time);

}




//**********************************************************************************************************************************


#ifdef COMPILE_KERNEL_CALCULATOR_MAIN__
int main(int argc, char** argv){

	double gamma = 1.0;
	unsigned kernel_type = GAUSS_RBF;

	if(argc > 1)
		gamma = atof(argv[1]);
	if(argc > 2)
		kernel_type = (unsigned)atof(argv[2]);

	printf("Welcome to liquidSVM Kernel Calculation (type=%d, gamma=%f)\n",kernel_type, gamma);

	double trees[31*2] = {8.3,10.3 , 8.6,10.3 , 8.8,10.2 , 10.5,16.4 , 10.7,18.8 , 10.8,19.7 , 11,15.6 , 11,18.2 , 11.1,22.6 , 11.2,19.9 , 11.3,24.2 , 11.4,21 , 11.4,21.4 , 11.7,21.3 , 12,19.1 , 12.9,22.2 , 12.9,33.8 , 13.3,27.4 , 13.7,25.7 , 13.8,24.9 , 14,34.5 , 14.2,31.7 , 14.5,36.3 , 16,38.3 , 16.3,42.6 , 17.3,55.4 , 17.5,55.7 , 17.9,58.3 , 18,51.5 , 18,51 , 20.6,77};
	double iris[150*4] = {5.1,3.5,1.4,0.2 , 4.9,3,1.4,0.2 , 4.7,3.2,1.3,0.2 , 4.6,3.1,1.5,0.2 , 5,3.6,1.4,0.2 , 5.4,3.9,1.7,0.4 , 4.6,3.4,1.4,0.3 , 5,3.4,1.5,0.2 , 4.4,2.9,1.4,0.2 , 4.9,3.1,1.5,0.1 , 5.4,3.7,1.5,0.2 , 4.8,3.4,1.6,0.2 , 4.8,3,1.4,0.1 , 4.3,3,1.1,0.1 , 5.8,4,1.2,0.2 , 5.7,4.4,1.5,0.4 , 5.4,3.9,1.3,0.4 , 5.1,3.5,1.4,0.3 , 5.7,3.8,1.7,0.3 , 5.1,3.8,1.5,0.3 , 5.4,3.4,1.7,0.2 , 5.1,3.7,1.5,0.4 , 4.6,3.6,1,0.2 , 5.1,3.3,1.7,0.5 , 4.8,3.4,1.9,0.2 , 5,3,1.6,0.2 , 5,3.4,1.6,0.4 , 5.2,3.5,1.5,0.2 , 5.2,3.4,1.4,0.2 , 4.7,3.2,1.6,0.2 , 4.8,3.1,1.6,0.2 , 5.4,3.4,1.5,0.4 , 5.2,4.1,1.5,0.1 , 5.5,4.2,1.4,0.2 , 4.9,3.1,1.5,0.2 , 5,3.2,1.2,0.2 , 5.5,3.5,1.3,0.2 , 4.9,3.6,1.4,0.1 , 4.4,3,1.3,0.2 , 5.1,3.4,1.5,0.2 , 5,3.5,1.3,0.3 , 4.5,2.3,1.3,0.3 , 4.4,3.2,1.3,0.2 , 5,3.5,1.6,0.6 , 5.1,3.8,1.9,0.4 , 4.8,3,1.4,0.3 , 5.1,3.8,1.6,0.2 , 4.6,3.2,1.4,0.2 , 5.3,3.7,1.5,0.2 , 5,3.3,1.4,0.2 , 7,3.2,4.7,1.4 , 6.4,3.2,4.5,1.5 , 6.9,3.1,4.9,1.5 , 5.5,2.3,4,1.3 , 6.5,2.8,4.6,1.5 , 5.7,2.8,4.5,1.3 , 6.3,3.3,4.7,1.6 , 4.9,2.4,3.3,1 , 6.6,2.9,4.6,1.3 , 5.2,2.7,3.9,1.4 , 5,2,3.5,1 , 5.9,3,4.2,1.5 , 6,2.2,4,1 , 6.1,2.9,4.7,1.4 , 5.6,2.9,3.6,1.3 , 6.7,3.1,4.4,1.4 , 5.6,3,4.5,1.5 , 5.8,2.7,4.1,1 , 6.2,2.2,4.5,1.5 , 5.6,2.5,3.9,1.1 , 5.9,3.2,4.8,1.8 , 6.1,2.8,4,1.3 , 6.3,2.5,4.9,1.5 , 6.1,2.8,4.7,1.2 , 6.4,2.9,4.3,1.3 , 6.6,3,4.4,1.4 , 6.8,2.8,4.8,1.4 , 6.7,3,5,1.7 , 6,2.9,4.5,1.5 , 5.7,2.6,3.5,1 , 5.5,2.4,3.8,1.1 , 5.5,2.4,3.7,1 , 5.8,2.7,3.9,1.2 , 6,2.7,5.1,1.6 , 5.4,3,4.5,1.5 , 6,3.4,4.5,1.6 , 6.7,3.1,4.7,1.5 , 6.3,2.3,4.4,1.3 , 5.6,3,4.1,1.3 , 5.5,2.5,4,1.3 , 5.5,2.6,4.4,1.2 , 6.1,3,4.6,1.4 , 5.8,2.6,4,1.2 , 5,2.3,3.3,1 , 5.6,2.7,4.2,1.3 , 5.7,3,4.2,1.2 , 5.7,2.9,4.2,1.3 , 6.2,2.9,4.3,1.3 , 5.1,2.5,3,1.1 , 5.7,2.8,4.1,1.3 , 6.3,3.3,6,2.5 , 5.8,2.7,5.1,1.9 , 7.1,3,5.9,2.1 , 6.3,2.9,5.6,1.8 , 6.5,3,5.8,2.2 , 7.6,3,6.6,2.1 , 4.9,2.5,4.5,1.7 , 7.3,2.9,6.3,1.8 , 6.7,2.5,5.8,1.8 , 7.2,3.6,6.1,2.5 , 6.5,3.2,5.1,2 , 6.4,2.7,5.3,1.9 , 6.8,3,5.5,2.1 , 5.7,2.5,5,2 , 5.8,2.8,5.1,2.4 , 6.4,3.2,5.3,2.3 , 6.5,3,5.5,1.8 , 7.7,3.8,6.7,2.2 , 7.7,2.6,6.9,2.3 , 6,2.2,5,1.5 , 6.9,3.2,5.7,2.3 , 5.6,2.8,4.9,2 , 7.7,2.8,6.7,2 , 6.3,2.7,4.9,1.8 , 6.7,3.3,5.7,2.1 , 7.2,3.2,6,1.8 , 6.2,2.8,4.8,1.8 , 6.1,3,4.9,1.8 , 6.4,2.8,5.6,2.1 , 7.2,3,5.8,1.6 , 7.4,2.8,6.1,1.9 , 7.9,3.8,6.4,2 , 6.4,2.8,5.6,2.2 , 6.3,2.8,5.1,1.5 , 6.1,2.6,5.6,1.4 , 7.7,3,6.1,2.3 , 6.3,3.4,5.6,2.4 , 6.4,3.1,5.5,1.8 , 6,3,4.8,1.8 , 6.9,3.1,5.4,2.1 , 6.7,3.1,5.6,2.4 , 6.9,3.1,5.1,2.3 , 5.8,2.7,5.1,1.9 , 6.8,3.2,5.9,2.3 , 6.7,3.3,5.7,2.5 , 6.7,3,5.2,2.3 , 6.3,2.5,5,1.9 , 6.5,3,5.2,2 , 6.2,3.4,5.4,2.3 , 5.9,3,5.1,1.8};
	Tkernel_calculator kernel_calculator;
	kernel_calculator.gamma = gamma;

//	Tdataset data = Tdataset(trees, 31, 2, NULL);
//	Tdataset data = Tdataset(iris, 150, 4, NULL);
	Tdataset data;
	data.read_from_file("data/covtype.10000.train.csv");

	Tkernel_control kernel_control;
	kernel_control.kernel_type = kernel_type;

	kernel_control.memory_model_pre_kernel = BLOCK;
	kernel_control.memory_model_kernel = BLOCK;

	Tparallel_control parallel_ctrl;
	parallel_ctrl.requested_team_size = 4;

	kernel_calculator.reserve_threads(parallel_ctrl);

	kernel_calculator.calculate(kernel_control, data);

	Tkernel *kernel = &kernel_calculator.kernel;

	unsigned n = kernel->get_row_set_size();
	double sum = 0;
	printf("kernel of size n=%d\n",n);
//	if(n>30) n=31;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			sum += kernel->entry(i,j);
			if(i<30 && j<30)
				printf("%.1f,",kernel->entry(i,j));
		}
		if(i<30)
			printf("\n");
	}

	printf("sum=%.2f",sum);
	kernel_calculator.clear_threads();
	return 0;
}
#endif

#endif

