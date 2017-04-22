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


#if !defined (TSAMPLE_H) 
	#define TSAMPLE_H


#include "sources/shared/system_support/os_specifics.h"

#include <cstdio>
#include <vector>
using namespace std;


//**********************************************************************************************************************************


class Tsample
{
	public:
		Tsample();
		Tsample(unsigned sample_type, unsigned dim);
		Tsample(const Tsample* sample);
		Tsample(const Tsample& sample);
		Tsample(const Tsample& sample, unsigned new_sample_type);
		Tsample(const vector <double>& realvector, double label = 0.0);
		Tsample(const double* realvector, unsigned dim, double label = 0.0);
		~Tsample();


		int read_from_file(FILE* fpread, unsigned filetype, unsigned& dim);
		void write_to_file(FILE* fpwrite, unsigned filetype, unsigned dataset_dim) const;

		inline unsigned get_number() const;
		inline unsigned dim() const;
		inline unsigned dim_aligned() const;
		inline unsigned get_internal_representation() const;
		inline double coord(unsigned i) const;
		inline void change_coord(unsigned i, double new_value);
		inline double operator [] (unsigned i) const;
		inline vector <double> get_x_part() const;
		inline void get_x_part(double* memory_location) const;
		
		Tsample& operator = (const Tsample& sample);
		bool operator == (const Tsample& sample) const;
		
		Tsample project(vector <unsigned> kept_coordinates);
		
		friend Tsample operator * (double coefficient, const Tsample& sample);
		friend Tsample operator * (const vector <double>& scaling, const Tsample& sample);
		friend Tsample operator + (const vector <double>& translate, const Tsample& sample2);
		inline friend double operator * (const Tsample& sample1, const Tsample& sample2);
		
		inline double get_2norm2() const;
		inline friend double sup_distance(Tsample* sample1, Tsample* sample2);
		inline friend double squared_distance(Tsample* sample1, Tsample* sample2);
		
		
		double label;
		bool labeled;
		double weight;

 
	private:
		friend class Tdataset;
		
		void create();
		void create(unsigned dim);
		void destroy();
		void copy(const Tsample* sample);
		void alloc_for_csv(unsigned dim);
		
		unsigned get_dim_from_file(FILE* fpread, unsigned filetype, unsigned& dim) const;
		
		inline int check_separator(FILE* fpread, int c) const;
		inline void get_next_nonspace(FILE* fpread, int& c) const;
		inline bool check_end_of_line(int c, unsigned filetype, unsigned position, unsigned dim) const;
		inline void goto_next_line(FILE* fpread) const;
		inline void goto_first_entry(FILE* fpread) const;
		
		
		unsigned number;
		
		
		unsigned sample_type;
		unsigned dimension;
		unsigned dimension_aligned;
		double norm2;
		
		double* restrict__ x_csv;

		vector <double> x_lsv;
		vector <unsigned> index;
		
		bool blocked_destruction;
};


//**********************************************************************************************************************************


#include "sources/shared/basic_types/sample.ins.cpp"

#ifndef COMPILE_SEPERATELY__
	#include "sources/shared/basic_types/sample.cpp"
#endif



#endif


