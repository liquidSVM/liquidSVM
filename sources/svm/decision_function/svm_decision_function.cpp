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


#if !defined (SVM_DECISION_FUNCTION_CPP)
	#define SVM_DECISION_FUNCTION_CPP


#include "sources/svm/decision_function/svm_decision_function.h"

#include "sources/shared/basic_functions/flush_print.h"
#include "sources/shared/basic_functions/basic_file_functions.h"
#include "sources/shared/basic_types/vector.h"


#include "sources/shared/kernel/kernel_functions.h"




//*********************************************************************************************************************************

Tsvm_decision_function::Tsvm_decision_function()
{
	clear();
	
	kernel_type = GAUSS_RBF;
	gamma = 1.0;
}

//*********************************************************************************************************************************

Tsvm_decision_function::~Tsvm_decision_function()
{
	flush_info(INFO_PEDANTIC_DEBUG, "\nDestroying an object of type Tsvm_decision_function of size %d.", size());
	clear();
}


//*********************************************************************************************************************************

Tsvm_decision_function::Tsvm_decision_function(const Tsvm_decision_function& decision_function)
{
	copy(&decision_function);
}


//*********************************************************************************************************************************

Tsvm_decision_function::Tsvm_decision_function(const Tsvm_solution* solution, Tkernel_control kernel_control, const Tsvm_train_val_info& train_val_info, const Tsubset_info& ws_info)
{
	Tsvm_solution::copy(solution);
	sample_number = compose(ws_info, index);
	
	Tsvm_decision_function::kernel_type = kernel_control.kernel_type;
	hierarchical_kernel_control_read_filename = kernel_control.hierarchical_kernel_control_read_filename;
	gamma = train_val_info.gamma;
	
	set_error(train_val_info);
}



//**********************************************************************************************************************************


Tsvm_decision_function operator * (double scalar, const Tsvm_decision_function& decision_function)
{
	unsigned i;
	Tsvm_decision_function new_decision_function;

	new_decision_function = decision_function;
	new_decision_function.offset = scalar * new_decision_function.offset;

	for (i=0; i<new_decision_function.size(); i++)
		new_decision_function.coefficient[i] = scalar * new_decision_function.coefficient[i];
	
	return new_decision_function;
}

//**********************************************************************************************************************************


Tsvm_decision_function Tsvm_decision_function::operator + (const Tsvm_decision_function& decision_function)
{
	unsigned i;
	unsigned new_size;
	vector <double> sum_coefficients;
	Tsvm_decision_function new_decision_function;
	
	
	if (kernel_type != decision_function.kernel_type)
		flush_exit(ERROR_DATA_MISMATCH, "Trying to add two decision functions with kernel types %d and %d.", kernel_type, decision_function.kernel_type);

	if (gamma != decision_function.gamma)
		flush_exit(ERROR_DATA_MISMATCH, "Trying to add two decision functions with kernel widths %1.5f and %1.5d.", gamma, decision_function.gamma);
	

	if (decision_function.size() == 0)
		return *this;
	
	if (size() == 0)
		return decision_function;
	
	
	new_size = max(sample_number[argmax(sample_number)], decision_function.sample_number[argmax(decision_function.sample_number)]) + 1;

	sum_coefficients.assign(new_size, 0.0);
	
	for (i=0; i<size(); i++)
		sum_coefficients[sample_number[i]] = coefficient[i];
			
	for (i=0; i<decision_function.size(); i++)
		sum_coefficients[decision_function.sample_number[i]] += decision_function.coefficient[i];
	
	
	new_decision_function = decision_function;
	new_decision_function.coefficient.clear();
	new_decision_function.index.clear();
	new_decision_function.sample_number.clear();
	
	new_decision_function.coefficient.reserve(new_size);
	new_decision_function.sample_number.reserve(new_size);

	for (i=0; i<new_size; i++)
		if (sum_coefficients[i] != 0.0)
		{
			new_decision_function.coefficient.push_back(sum_coefficients[i]);
			new_decision_function.sample_number.push_back(i);
		}
	new_decision_function.offset = offset + decision_function.offset;
		
	new_decision_function.resize(new_decision_function.sample_number.size());

	return new_decision_function;
}



//**********************************************************************************************************************************


Tsvm_decision_function& Tsvm_decision_function::operator = (const Tsvm_decision_function& decision_function)
{
	copy(&decision_function);
	return *this;
}


//*********************************************************************************************************************************

void Tsvm_decision_function::copy(const Tsvm_decision_function* decision_function)
{
	Tsvm_solution::copy(decision_function);
	Tdecision_function::copy(decision_function);
	
	kernel_type = decision_function->kernel_type;
	hierarchical_kernel_control_read_filename = decision_function->hierarchical_kernel_control_read_filename;
	gamma = decision_function->gamma;
}



//**********************************************************************************************************************************

double Tsvm_decision_function::evaluate(double* kernel_eval, unsigned training_size, unsigned gamma_no, unsigned thread_position)
{
	unsigned i;
	double evaluation;


	evaluation = offset;
	for (i=0; i<size(); i++)
		evaluation = evaluation + coefficient[i] * kernel_eval[thread_position + gamma_no * training_size + sample_number[i]]; 
	
	if (clipp_value > 0.0)
		evaluation = max(-clipp_value, min(clipp_value, evaluation));

	return evaluation;
}

//**********************************************************************************************************************************

double Tsvm_decision_function::evaluate(Tsample* test_sample, const Tdataset& training_set)
{
	unsigned i;
	double evaluation;
	double gamma_factor;

	gamma_factor = compute_gamma_factor(kernel_type, gamma); 
	evaluation = offset;
	for (i=0; i<size(); i++)
		evaluation = evaluation + coefficient[i] * kernel_function(kernel_type, gamma_factor, test_sample, training_set.sample(sample_number[i]));

	if (clipp_value > 0.0)
		evaluation = max(-clipp_value, min(clipp_value, evaluation));

	return evaluation;
	
}


//**********************************************************************************************************************************

void Tsvm_decision_function::write_to_file(FILE* fp) const
{
	if (fp == NULL)
		return;
	
	file_write(fp, kernel_type);
	file_write(fp, gamma);
	file_write(fp, hierarchical_kernel_control_read_filename);

	Tdecision_function::write_to_file(fp);
	Tsvm_solution::write_to_file(fp);
}



//**********************************************************************************************************************************

void Tsvm_decision_function::read_from_file(FILE* fp)
{
	if (fp == NULL)
		return;

	clear();
	file_read(fp, kernel_type);
	file_read(fp, gamma);
	file_read(fp, hierarchical_kernel_control_read_filename);

	Tdecision_function::read_from_file(fp);
	Tsvm_solution::read_from_file(fp);
}


//**********************************************************************************************************************************

void Tsvm_decision_function::set_to_zero()
{
	clear();
}



#ifdef  COMPILE_WITH_CUDA__

//**********************************************************************************************************************************

double Tsvm_decision_function::get_offset() const
{
	return offset;
}
	
//**********************************************************************************************************************************

double Tsvm_decision_function::get_clipp_value() const
{
	return clipp_value;
}

#endif


#endif
