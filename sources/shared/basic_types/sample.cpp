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


#if !defined (TSAMPLE_CPP) 
	#define TSAMPLE_CPP



#include "sources/shared/basic_types/sample.h"


#include "sources/shared/system_support/memory_allocation.h" 

#include "sources/shared/basic_types/vector.h"
#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"


//**********************************************************************************************************************************

Tsample::Tsample() 
{
	create();
	
	labeled = true;
};



//**********************************************************************************************************************************

Tsample::Tsample(unsigned sample_type, unsigned dim)
{
	flush_info(INFO_EXTREMELY_PEDANTIC_DEBUG, "\nCreating an empty sample of type %d and dimension %d.", sample_type, dim);
	
	if (sample_type == CSV)
		create(dim);
	else
		create();
}


//**********************************************************************************************************************************

Tsample::Tsample(const Tsample* sample) 
{
	create();
	copy(sample);
};



//**********************************************************************************************************************************

Tsample::Tsample(const Tsample& sample)
{
    create();
    copy(&sample);
}


//**********************************************************************************************************************************

Tsample::Tsample(const Tsample& sample, unsigned new_sample_type) 
{
	unsigned i;

	flush_info(INFO_EXTREMELY_PEDANTIC_DEBUG, "\nCreating a sample of type %d and dimension %d from sample with number %d.", new_sample_type, sample.dim(), sample.get_number());
	
	create();
	
	if (new_sample_type == sample.sample_type)
		copy(&sample);
	else
	{
		if (new_sample_type == LSV)
		{
			for (i=0;i<sample.dim();i++)
				if (sample.x_csv[i] != 0.0)
				{
					push_back_mem_safe(index, i);
					push_back_mem_safe(x_lsv, sample.x_csv[i]);
				}
		}
		else
		{
			create(sample.dim());
			for (i=0;i<sample.index.size();i++)
				x_csv[sample.index[i]] = sample.x_lsv[i];
		}
		
		label = sample.label;
		dimension = sample.dim();
		weight = sample.weight;
		number = sample.number;
		
		id = sample.id;
		group_id = sample.group_id;
		
		sample_type = new_sample_type;
		norm2 = sample.norm2;
		labeled = sample.labeled;
	}
};



//**********************************************************************************************************************************

Tsample::Tsample(const vector <double>& realvector, double label) 
{	
	unsigned i;

	create(unsigned(realvector.size()));
	Tsample::label = label;

	for (i=0;i<dim();i++)
		x_csv[i] = realvector[i];

	norm2 = ((*this) * (*this)); 
	
	labeled = true;
};



//**********************************************************************************************************************************

Tsample::Tsample(const double* realvector, unsigned dim, double label)
{
	unsigned i;

	create(dim);
	Tsample::label = label;

	for (i=0;i<dim;i++)
		x_csv[i] = realvector[i];

	norm2 = ((*this) * (*this)); 
	
	labeled = true;
}


//**********************************************************************************************************************************

Tsample::~Tsample() 
{
	destroy();
};



//**********************************************************************************************************************************

void Tsample::create() 
{
	sample_type = LSV;
	x_csv = NULL;
	
	label = 0.0;
	labeled = true;
	
	dimension = 0;
	dimension_aligned = 0;
	number = 0;
	norm2 = 0.0;
	
	id = 0;
	group_id = 0;
	weight = 1.0;
	
	blocked_destruction = false;
};


//**********************************************************************************************************************************

void Tsample::create(unsigned dim) 
{	
	unsigned i;

	
	create();
	
	sample_type = CSV;
	my_alloc_ALGD(&x_csv, dim, dimension_aligned);
	
	dimension = dim;
	
	for (i=0; i<dimension_aligned; i++)
		x_csv[i] = 0.0;
	label = 0.0;
};	


//**********************************************************************************************************************************

void Tsample::destroy() 
{
	if (blocked_destruction == true)
		flush_exit(ERROR_DATA_STRUCTURE, "Trying to destroy blocked sample with number %d.", number);
	
	if (dim() > 0)
		flush_info(INFO_EXTREMELY_PEDANTIC_DEBUG, "\nDeleting a sample of type %d, dimension %d, label %1.4f, and number %d.", sample_type, dim(), label, get_number());
	
	my_dealloc_ALGD(&x_csv);
	index.clear();
	x_lsv.clear();
};


//**********************************************************************************************************************************


void Tsample::copy(const Tsample* sample)
{
	unsigned i;
	unsigned size;


	this->destroy();
	flush_info(INFO_EXTREMELY_PEDANTIC_DEBUG, "\nCopying a sample of type %d and dimension %d from a pointer to a sample with number %d and squared norm %f", sample->sample_type, sample->dim(), sample->get_number(), sample->norm2);

	if (sample->sample_type == LSV)
	{
		size = unsigned(sample->index.size());
		index.resize(size);
		x_lsv.resize(size);
		for (i=0;i<size;i++)
		{
			index[i] = sample->index[i];
			x_lsv[i] = sample->x_lsv[i];
		}
		x_csv = NULL;
	}
	else
	{
		create(sample->dim());
		for (i=0; i<dimension_aligned; i++)
			x_csv[i] = sample->x_csv[i];
	}

	label = sample->label;
	labeled = sample->labeled;
	
	dimension = sample->dim();
	
	weight = sample->weight;
	number = sample->number;
	id = sample->id;
	group_id = sample->group_id;
	
	sample_type = sample->sample_type;
	norm2 = sample->norm2;
    
	if (flush_info_will_show(INFO_EXTREMELY_PEDANTIC_DEBUG) == true)
		if (((*this) * (*this)) != norm2)
			flush_warn(INFO_EXTREMELY_PEDANTIC_DEBUG, "Norm of copied sample is %f but it should be %f.", ((*this) * (*this)), norm2);
}


//**********************************************************************************************************************************


Tsample& Tsample::operator = (const Tsample& sample)
{
	destroy();
	create();
	copy(&sample);
	return *this;
}


//**********************************************************************************************************************************


bool Tsample::operator == (const Tsample& sample) const
{
	unsigned i;
	bool equal;
	
	
// 	Check for cheap possible differences first
	
	if (sample_type != sample.sample_type)
		return false;
	
	if (dim() != sample.dim())
		return false;
	
	if (label != sample.label)
		return false;

	if (labeled != sample.labeled)
		return false;
	
	if (sample_type == LSV)
		return ((x_lsv == sample.x_lsv) and (index == sample.index));
	else
	{
// 		If the sample actually use the same memory segment they are equal
		
		if (x_csv == sample.x_csv) 
			return true;
		
		
// 		Otherwise compare coordinate wise ...
		
		flush_info(INFO_VERY_PEDANTIC_DEBUG, "\nComparing two samples of dimension %d coordinate wise.", dim());
		
		equal = true;
		i = 0;
		while ((i < dim()) and (equal == true))
		{
			equal = (x_csv[i] == sample.x_csv[i]);
			i++;
		}
		return equal;
	}
}

//**********************************************************************************************************************************

unsigned Tsample::get_dim_from_file(FILE* fpread, Tsample_file_format sample_file_format, unsigned& dim) const
{
	int c;
	double dummy_x;
	unsigned number_of_extra_positions;
	
	
	rewind(fpread);
	sample_file_format.check_filetype();
	if (sample_file_format.filetype != CSV)
		return ILLEGAL_FILETYPE;
	
	c = getc(fpread);
	if (c == 34)
		file_read_eol(fpread);
	else
		ungetc(c, fpread);
	
	c = getc(fpread);
	if (c == 34)
		goto_first_entry(fpread);
	else
		ungetc(c, fpread);
	
	dim = 0;
	get_next_nonspace(fpread, c);
	do
		if (check_separator(fpread, c) == FILE_OP_OK)
		{
			file_read(fpread, dummy_x);
			dim++;
			get_next_nonspace(fpread, c);
		}
	while (not ((c=='\n') or (c == 13)));
	
	rewind(fpread);
	number_of_extra_positions = sample_file_format.count_extra_positions();

	if (number_of_extra_positions > dim)
	{
		dim = 0;
		return FILE_CORRUPTED;
	}
	else
	{
		dim = dim - number_of_extra_positions;
		return FILE_OP_OK;
	}
}





//**********************************************************************************************************************************

int Tsample::read_from_file(FILE* fpread, Tsample_file_format sample_file_format, unsigned& dim)
{
	unsigned j;
	int c;
	int read_status;
	double dummy_x;
	unsigned dummy_index;
	unsigned old_dummy_index;
	vector <double> io_vector;
	unsigned io_size;
	
	
// Check whether we have reached the end of the file.
// Here we assume that every line must contain data, that is
// we must not have a couple of "\n" at the end of the file
	
	get_next_nonspace(fpread, c);
	if (c == EOF)
		return END_OF_FILE;
	else
		ungetc(c, fpread);

	
// Skip the first column if it is a name and skip the first row if it is a header

	c = getc(fpread);
	if (c == 34)
	{
		goto_first_entry(fpread);
		c = getc(fpread);
		if (c == 34)
		{
			file_read_eol(fpread);
			
			c = getc(fpread);
			if (c == 34)
				goto_first_entry(fpread);
			else
				ungetc(c, fpread);
		}
		else
			ungetc(c, fpread);
	}
	else
		ungetc(c, fpread);


// Once we known that there is something to read we 
// need to make sure we start with a fresh sample
// In addition set labeled flag and read labels for LSV
	
	destroy();
	if (sample_file_format.filetype == CSV)
	{
		io_size = dim + sample_file_format.count_extra_positions();
		io_vector.resize(io_size);
		labeled = (sample_file_format.label_position != 0);
	}
	else
	{
		create();
		io_size = dim;
		labeled = true;
		file_read(fpread, label);
	}
	
// Read the x-part of the file line
	
	j = 0;
	old_dummy_index = 0;
	
	get_next_nonspace(fpread, c);
	if (c == EOF)
		return END_OF_FILE;
	
	do
	{
		read_status = check_separator(fpread, c);
		if (read_status == FILE_OP_OK)
		{
			if (sample_file_format.filetype == LSV)
			{
				file_read(fpread, dummy_index, dummy_x);
				if ((dummy_index <= old_dummy_index) and (old_dummy_index != 0))
					exit_on_file_error(LSV_FILE_CORRUPTED, fpread);
				
// 			LIBSVMs format uses indices from 1 to dim. The next line transforms these to indices from 0 to dim-1.
				
				dummy_index = dummy_index - 1;
				
				push_back_mem_safe(index, dummy_index);
				push_back_mem_safe(x_lsv, dummy_x);
				old_dummy_index = dummy_index;
			}
			else
				file_read(fpread, io_vector[j]);

			j++;
			get_next_nonspace(fpread, c);
		}
	}
	while (not(check_end_of_line(c, sample_file_format.filetype, j, io_size)));

	
// Make sure that the file line was read correctly (including windows conventions).

 	if (((c != '\n') and (c != 13)) or ((j != io_size) and (sample_file_format.filetype != LSV)))
		exit_on_file_error(FILE_CORRUPTED, fpread);
	if (c == 13)
	{
		get_next_nonspace(fpread, c);
		if (c != '\n')
			ungetc(c, fpread);
	}

	
// Set the dimension for filetype LSV
	
	if (sample_file_format.filetype == LSV)
	{
		dim = dummy_index + 1;
		dimension = dummy_index + 1;
	}
	else
		convert_from_io_vector(sample_file_format, io_vector);

	
// Finally, compute norms
	
	norm2 = ((*this) * (*this)); 
	return FILE_OP_OK;
}



//**********************************************************************************************************************************

inline void Tsample::get_next_nonspace(FILE* fpread, int& c) const
{
	do
		c = getc(fpread);
	while (c==32);
}


//**********************************************************************************************************************************

inline int Tsample::check_separator(FILE* fpread, int c) const
{
	if ((c == ',') or (c == ';') or (c == ':'))
		return FILE_OP_OK;
	else if ((c == '+') or (c == '-') or ((c >= '0') and (c <= '9')))
	{
		ungetc(c, fpread);
		return FILE_OP_OK;
	}
	else if ((c == '\n') or (c == 10))
		return END_OF_LINE;
	else
	{
		exit_on_file_error(FILE_CORRUPTED, fpread);
		return FILE_CORRUPTED; // This line will never be reached, but without it, an ugly compilation warning occurs
	}
}


//**********************************************************************************************************************************


inline bool Tsample::check_end_of_line(int c, unsigned filetype, unsigned position, unsigned dim) const
{
// This routine could have been implemented in a simpler form. I decided this way, to be more adaptive to possible future changes
	
	switch(filetype)
	{
		case CSV:
			return ((c == '\n') or (c == 13) or (position >= dim));
		case LSV:
			return ((c == '\n') or (c == 13));
		default:
			return true;
	}
}


//**********************************************************************************************************************************


inline void Tsample::goto_first_entry(FILE* fpread) const
{
	int c;
	
	do
		c = getc(fpread);
	while (c != 34);
	c = getc(fpread);
}



//**********************************************************************************************************************************

void Tsample::write_to_file(FILE* fpwrite, Tsample_file_format sample_file_format) const
{
	unsigned j; 
	Tsample sample_tmp;
	vector <double> io_vector;

	sample_file_format.check_filetype();
	if (sample_file_format.filetype == LSV)
	{
		if (sample_type != LSV)
			sample_tmp = Tsample(*this, LSV);
		else
			sample_tmp = *this;
		
		file_write(fpwrite, label, "%g", " ");
		for (j=0; j<sample_tmp.x_lsv.size(); j++)
			if (sample_tmp.x_lsv[j] != 0.0)
				file_write(fpwrite, sample_tmp.index[j], sample_tmp.x_lsv[j]);
	}
	else
	{
		io_vector = convert_to_io_vector(sample_file_format);
		for (j=0; j+1<io_vector.size(); j++)
			file_write(fpwrite, io_vector[j], "%g, ", "");	
		file_write(fpwrite, io_vector[j], "%g", "");	
	}
	file_write_eol(fpwrite);
}



//**********************************************************************************************************************************

vector <double> Tsample::convert_to_io_vector(Tsample_file_format sample_file_format) const
{
	unsigned i;
	unsigned j;
	unsigned ii;
	unsigned number_of_columns;
	unsigned number_of_extra_positions;
	vector <double> x_part;
	vector <double> io_vector;
	
	
	sample_file_format.compute_full_include_list(sample_file_format.dataset_dim);
	number_of_extra_positions = sample_file_format.count_extra_positions();
	number_of_columns = number_of_extra_positions + sample_file_format.full_include_list.size();

	sample_file_format.compute_extra_position_list(number_of_columns);
		
	x_part = get_x_part();
	for (i=dim(); i<sample_file_format.dataset_dim; i++)
		x_part.push_back(0.0);
	io_vector.resize(number_of_columns);
	
	if (number_of_extra_positions == 0)
		for (i=0; i<sample_file_format.full_include_list.size(); i++)
			io_vector[i] = x_part[sample_file_format.full_include_list[i] - 1];
	else
	{
		number_of_extra_positions--;
		j = 0;
		ii = 0;
		for (i=0; i<io_vector.size(); i++)
			if (i+1 == sample_file_format.extra_position_list[min(j, number_of_extra_positions)])
				j++;
			else
			{
				io_vector[i] = x_part[sample_file_format.full_include_list[ii] - 1];
				ii++;
			}
	}
	
	if (sample_file_format.label_position != 0)
		io_vector[int(sample_file_format.get_true_column(sample_file_format.label_position, number_of_columns)) - 1] = label;

	if (sample_file_format.weight_position != 0)
		io_vector[int(sample_file_format.get_true_column(sample_file_format.weight_position, number_of_columns)) - 1] = weight;

	if (sample_file_format.id_position != 0)
		io_vector[int(sample_file_format.get_true_column(sample_file_format.id_position, number_of_columns)) - 1] = double(id);

	if (sample_file_format.group_id_position != 0)
		io_vector[int(sample_file_format.get_true_column(sample_file_format.group_id_position, number_of_columns)) - 1] = double(group_id);
	
	return io_vector;
}


//**********************************************************************************************************************************

void Tsample::convert_from_io_vector(Tsample_file_format sample_file_format, const vector <double>& io_vector)
{
	unsigned i;
	unsigned ii;
	unsigned j;
	unsigned number_of_columns;
	unsigned dimension_new;
	
	
	destroy();
	number_of_columns = io_vector.size();
	sample_file_format.compute_full_exclude_list(number_of_columns);
	
	if (number_of_columns <= sample_file_format.full_exclude_list.size())
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Cannot create a sample from %d columns with %d extra positions.", number_of_columns, sample_file_format.full_exclude_list.size()); 
	
	dimension_new = number_of_columns - sample_file_format.full_exclude_list.size();
	
	create(dimension_new);
	
	if (sample_file_format.full_exclude_list.size() == 0)
		for (i=0; i<number_of_columns; i++)
			x_csv[i] = io_vector[i];
	else
	{
		j = 0;
		ii = 0;
		for (i=0; i<number_of_columns; i++)
			if (i+1 != sample_file_format.full_exclude_list[j])
			{
				x_csv[ii] = io_vector[i];
				ii++;
			}
			else
				j++;
	}
	
	
	if (sample_file_format.label_position != 0)
	{
		labeled = true;
		label = io_vector[int(sample_file_format.get_true_column(sample_file_format.label_position, number_of_columns)) - 1];
	}
	else
		labeled = false;
	
	if (sample_file_format.weight_position != 0)
		weight = io_vector[int(sample_file_format.get_true_column(sample_file_format.weight_position, number_of_columns)) - 1];
	
	if (weight < 0.0)
		flush_exit(ERROR_DATA_FALLS_OUTSIDE_SAFE_PARAMETERS, "Sample has negative weight %1.3f.", weight);

	if (sample_file_format.id_position != 0)
		id = unsigned(io_vector[int(sample_file_format.get_true_column(sample_file_format.id_position, number_of_columns)) - 1]);

	if (sample_file_format.group_id_position != 0)
		group_id = unsigned(io_vector[int(sample_file_format.get_true_column(sample_file_format.group_id_position, number_of_columns)) - 1]);
}



//**********************************************************************************************************************************


Tsample Tsample::project(vector <unsigned> kept_coordinates)
{
	unsigned i;
	unsigned j;
	double norm;
	unsigned dim_common;
	Tsample return_sample;

	
	if (sample_type == CSV)
	{
		dim_common = 0;
		for (i=0; i<kept_coordinates.size(); i++)
			if (kept_coordinates[i] < dim())
				dim_common++;
			
		return_sample.sample_type = CSV;
		return_sample.dimension = dim_common;
		my_alloc_ALGD(&(return_sample.x_csv), dim_common, return_sample.dimension_aligned);
		
		norm = 0.0;
		for (i=0; i<dim_common; i++)
		{
			return_sample.x_csv[i] = x_csv[kept_coordinates[i]];
			norm = norm + return_sample.x_csv[i] * return_sample.x_csv[i];
		}
		
		for (i=dim_common; i<return_sample.dimension_aligned; i++)
			return_sample.x_csv[i] = 0.0;
	}
	else
	{
		return_sample = Tsample(LSV, 0);
		
		i = 0;
		j = 0;
		norm = 0.0;
		
		while ((i < kept_coordinates.size()) and (j < x_lsv.size()))
		{
			if (kept_coordinates[i] == index[j])
			{
				return_sample.index.push_back(index[j]);
				return_sample.x_lsv.push_back(x_lsv[j]);
				norm = norm + x_lsv[j] * x_lsv[j];

				i++;
				j++;
			}
			else if (kept_coordinates[i] < index[j])
				i++;
			else
				j++;
		}
		
		if (return_sample.index.size() > 0)
			return_sample.dimension = return_sample.index[return_sample.index.size() - 1] + 1;
		else 
			return_sample.dimension = 0;
	}
	
	return_sample.label = label;
	return_sample.norm2 = norm;
	
	return return_sample;
}


//**********************************************************************************************************************************


Tsample operator * (double coefficient, const Tsample& sample)
{
	unsigned i;
	Tsample return_sample;


	if (sample.sample_type == CSV)
	{
		return_sample.sample_type = CSV;
		return_sample.dimension = sample.dim();
		my_alloc_ALGD(&(return_sample.x_csv), sample.dim(), return_sample.dimension_aligned);
		
		for (i=0; i<sample.dim_aligned(); i++)
			return_sample.x_csv[i] = coefficient * sample.x_csv[i];
	}
	else
	{
		return_sample = Tsample(LSV, 0);

		if (coefficient != 0.0)
		{
			for (i=0; i<sample.x_lsv.size(); i++)
			{
				return_sample.index.push_back(sample.index[i]);
				return_sample.x_lsv.push_back(coefficient * sample.x_lsv[i]);
			}
			return_sample.dimension = sample.dim();
		}
		else
			return_sample.dimension = 0;
	}

	return_sample.label = sample.label;
	return_sample.norm2 = coefficient * coefficient * sample.norm2;
	
	return return_sample;
};



//**********************************************************************************************************************************


Tsample operator * (const vector <double>& scaling, const Tsample& sample)
{
	unsigned i;
	unsigned dim_common;
	double norm;
	Tsample return_sample;


	norm = 0.0;
		
	if (sample.sample_type == CSV)
	{
		return_sample.sample_type = CSV;
		return_sample.dimension = sample.dim();
		my_alloc_ALGD(&(return_sample.x_csv), sample.dim(), return_sample.dimension_aligned);
		
		dim_common = min(unsigned(scaling.size()), sample.dim_aligned());

		for (i=0; i<dim_common; i++)
		{
			return_sample.x_csv[i] = scaling[i] * sample.x_csv[i];
			norm = norm + return_sample.x_csv[i] * return_sample.x_csv[i];
		}
		
		for (i=dim_common; i<sample.dim_aligned(); i++)
			return_sample.x_csv[i] = 0.0;
	}
	else
	{
		return_sample = Tsample(LSV, 0);

		for (i=0; i<sample.x_lsv.size(); i++)
			if (sample.index[i] < scaling.size())
			{
				return_sample.index.push_back(sample.index[i]);
				return_sample.x_lsv.push_back(scaling[sample.index[i]] * sample.x_lsv[i]);
				norm = norm + scaling[sample.index[i]] * sample.x_lsv[i] * scaling[sample.index[i]] * sample.x_lsv[i];
			}
			
		if (return_sample.index.size() > 0)
			return_sample.dimension = return_sample.index[return_sample.index.size() - 1] + 1;
		else 
			return_sample.dimension = 0;
	}

	return_sample.norm2 = norm;
	return_sample.label = sample.label;
	return_sample.labeled = sample.labeled;

	return return_sample;
};


//**********************************************************************************************************************************


Tsample operator + (const vector <double>& translate, const Tsample& sample)
{
	unsigned i;
	unsigned dim_common;
	double norm;
	Tsample return_sample;
	
	
	norm = 0.0;
	if (sample.sample_type == CSV)
	{
		return_sample.sample_type = CSV;
		return_sample.dimension = sample.dim();
		my_alloc_ALGD(&(return_sample.x_csv), sample.dim(), return_sample.dimension_aligned);
		
		dim_common = min(unsigned(translate.size()), sample.dim_aligned());
		
		for (i=0; i<dim_common; i++)
		{
			return_sample.x_csv[i] = translate[i] + sample.x_csv[i];
			norm = norm + return_sample.x_csv[i] * return_sample.x_csv[i];
		}
		

		for(i=dim_common; i<return_sample.dimension_aligned; i++)
		{
			return_sample.x_csv[i] = sample.x_csv[i];
			norm = norm + return_sample.x_csv[i] * return_sample.x_csv[i];
		}
	}
	else
	{
		return_sample = Tsample(LSV, 0);
		
		for (i=0; i<sample.x_lsv.size(); i++)
			if (sample.index[i] < translate.size())
			{
				if (translate[sample.index[i]] + sample.x_lsv[i] != 0.0)
				{
					return_sample.index.push_back(sample.index[i]);
					return_sample.x_lsv.push_back(translate[sample.index[i]] + sample.x_lsv[i]);
					norm = norm + (translate[sample.index[i]] + sample.x_lsv[i]) * (translate[sample.index[i]] + sample.x_lsv[i]);
				}
			}
			else
			{
				return_sample.index.push_back(sample.index[i]);
				return_sample.x_lsv.push_back(sample.x_lsv[i]);
				norm = norm + sample.x_lsv[i] * sample.x_lsv[i];
			}
			
		if (return_sample.index.size() > 0)
			return_sample.dimension = return_sample.index[return_sample.index.size() - 1] + 1;
		else 
			return_sample.dimension = 0;
	}

	return_sample.norm2 = norm;
	return_sample.label = sample.label;
	return_sample.labeled = sample.labeled;
	
	return return_sample;
}

#endif

