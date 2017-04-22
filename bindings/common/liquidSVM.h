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

#ifndef liquidSVM_h
	#define liquidSVM_h

#ifdef __cplusplus
#include "sources/shared/basic_types/dataset.h"
#include "sources/svm/decision_function/svm_decision_function.h"
#endif

extern "C" {

// The following are used in all the bindings so we have them at a single point (?)
// stages: 0: default_grid; 1: train; 2: select; 3: test; -1: compilation params
extern const char* liquid_svm_default_params(int stage, int solver);

// The bindings-API to liquidSVM
extern int liquid_svm_init(const double* data, const unsigned size, const unsigned dim, const double* labels);
extern double* liquid_svm_train(int cookie, const int argc, char** argv);
extern double* liquid_svm_select(int cookie, const int argc, char** argv);
extern double* liquid_svm_test(int cookie, const int argc, char** argv, const double* test_data, const unsigned test_size, const unsigned dim, const double* labels, double** error_ret);
extern void liquid_svm_clean(int cookie);

extern "C" int liquid_svm_read_solution(int cookie, const char* filename, size_t *len, unsigned char **buffer);
extern "C" void liquid_svm_write_solution(int cookie, const char* filename, size_t len, unsigned char *buffer);

}

#ifdef __cplusplus
extern Tsubset_info liquid_svm_get_cover(int cookie, unsigned task);
extern Tsvm_decision_function liquid_svm_get_solution(int cookie, unsigned task, unsigned cell, unsigned fold);
#endif



#ifndef COMPILE_SEPERATELY__
  #include "./liquidSVM.cpp"
#endif

#endif
