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




#include "sources/shared/system_support/simd_basics.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"

#ifdef _WIN32
	#if !defined (NOMINMAX) 
		#define NOMINMAX
	#endif
	#include <iso646.h>
#endif

#include <math.h> 


//**********************************************************************************************************************************


inline unsigned Tsample::get_number() const
{
	return number;
}



//**********************************************************************************************************************************


inline unsigned Tsample::get_internal_representation() const
{
	return sample_type;
}


//**********************************************************************************************************************************

inline double Tsample::get_2norm2() const
{
	return norm2;
}

//**********************************************************************************************************************************


inline unsigned Tsample::dim() const
{
	return dimension;
}

//**********************************************************************************************************************************


inline unsigned Tsample::dim_aligned() const
{
	if (sample_type == LSV)
		return dimension;
	else
		return dimension_aligned;
}



//**********************************************************************************************************************************


inline double Tsample::coord(unsigned i) const
{
	unsigned j;
	

	if (sample_type == CSV)
	{
		if (i<dim())
			return x_csv[i];
		else
			return 0.0;
	}
	
	j = 0;
	while ((index[min(j, dim()-1)] < i) and (j < dim()))
		j++;
	
	if (index[min(j, dim()-1)] == i)
		return x_lsv[min(j, dim()-1)];
	else
		return 0.0;
}



//**********************************************************************************************************************************


inline void Tsample::change_coord(unsigned i, double new_value)
{
	unsigned j;
	
	if (sample_type == CSV)
	{
		if (i<dim())
		{
			norm2 = norm2 - x_csv[i] * x_csv[i] + new_value * new_value; 
			x_csv[i] = new_value;
		}
		else
			flush_exit(ERROR_DATA_MISMATCH, "Trying to change coordinate %d of a csv sample of dimension %d", i, dim());
	}
	else
	{
		if (new_value != 0.0)
		{
			j = 0;
			while ((index[min(j, dim()-1)] < i) and (j < dim()))
				j++;

			j = min(j, dim()-1);
			if (index[j] == i)
				x_lsv[j] = new_value;
			else
			{
				index.insert(index.begin() + j, i);
				x_lsv.insert(x_lsv.begin() + j, new_value);
			}
		}
	}

}



//**********************************************************************************************************************************


inline double Tsample::operator [] (unsigned i) const
{
	return coord(i);
}


//**********************************************************************************************************************************


inline vector <double> Tsample::get_x_part() const
{
	vector <double> vector_tmp;
	
	vector_tmp.resize(dim());
	get_x_part(&vector_tmp[0]);
	return vector_tmp;
}

//**********************************************************************************************************************************
		
		
inline void Tsample::get_x_part(double* memory_location) const
{
	unsigned i;
	
	if (sample_type == CSV)
		for (i=0; i<dim(); i++)
			memory_location[i] = x_csv[i];
	else
		for (i=0; i<index.size(); i++)
			memory_location[index[i]] = x_lsv[i];
}



//**********************************************************************************************************************************


inline double operator * (const Tsample& sample1, const Tsample& sample2)
{
	double prod;
	simdd__ prod_simdd;
	unsigned dim;
	unsigned i;
	unsigned j;
	

	if ((sample1.sample_type == CSV) and (sample2.sample_type == CSV))
	{
		dim = min(sample1.dim_aligned(), sample2.dim_aligned());

		prod_simdd = assign_simdd(0.0);
		for (i=0; i+CACHELINE_STEP <=dim; i+=CACHELINE_STEP)
		{
			cache_prefetch(sample1.x_csv+i, PREFETCH_L1);
			fuse_mult_sum_CL(sample1.x_csv+i, sample2.x_csv+i, prod_simdd);
		}
		return reduce_sums_simdd(prod_simdd);
	}
	else if ((sample1.sample_type == LSV) and (sample2.sample_type == LSV))
	{
		prod = 0.0;
		i = 0;
		j = 0;

		while ((i < sample1.x_lsv.size()) and (j < sample2.x_lsv.size()))
		{
			if (sample1.index[i] == sample2.index[j])
			{
				prod = prod + sample1.x_lsv[i] * sample2.x_lsv[j];
				i++;
				j++;
			}
			else if (sample1.index[i] < sample2.index[j])
				i++;
			else
				j++;
		}
	}
	else if ((sample1.sample_type == LSV) and (sample2.sample_type == CSV))
	{
		prod = 0.0;
		for (i=0; i < sample1.x_lsv.size(); i++)
			prod = prod + sample1.x_lsv[i] * sample2.x_csv[sample1.index[i]];
	}
	else
	{
		prod = 0.0;
		for (i=0; i < sample2.x_lsv.size(); i++)
			prod = prod + sample2.x_lsv[i] * sample1.x_csv[sample2.index[i]];
	}
	
	return prod;
};



//**********************************************************************************************************************************

inline double sup_distance(Tsample* sample1, Tsample* sample2)
{
	double distance;
	unsigned dim;
	unsigned i;


	if ((sample1->sample_type != CSV) or (sample2->sample_type != CSV))
		flush_exit(ERROR_DATA_MISMATCH, "The supremum norm is only implemented for samples in internal CSV format.");
	
	if (sample1->dim() != sample2->dim())
		flush_exit(ERROR_DATA_MISMATCH, "The supremum norm cannot be computed for samples with dimensions %d and %d.", sample1->dim(), sample2->dim());
	
	distance = 0.0;
	dim = sample1->dim();
	for (i=0; i<dim; i++)
		distance = max(distance, fabs(sample1->x_csv[i] - sample2->x_csv[i]));
	
	return distance;
};



//**********************************************************************************************************************************

inline double squared_distance(Tsample* sample1, Tsample* sample2)
{
	return ( sample1->get_2norm2() - 2.0 * ((*sample1) * (*sample2)) + sample2->get_2norm2() );
};


