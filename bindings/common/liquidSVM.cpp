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

#ifndef liquidSVM_cpp
	#define liquidSVM_cpp

#include "liquidSVM.h"
#include "./scenario_config.h"

#include "sources/shared/basic_types/dataset.h"
#include "sources/shared/basic_functions/flush_print.h"

#include "sources/svm/training_validation/svm_manager.h"

#include "sources/svm/command_line/command_line_parser_svm_train.h"
#include "sources/svm/command_line/command_line_parser_svm_select.h"
#include "sources/svm/command_line/command_line_parser_svm_test.h"

// in R data is usually stored by columns:
#ifndef __LIQUIDSVM_DATA_BY_COLS
#define __LIQUIDSVM_DATA_BY_COLS false
#endif



//**********************************************************************************************************************************

#ifndef error
#define error(x) flush_info(x);
#endif

#ifndef warning
#define warning(x) flush_info(x);
#endif

#define __DEFAULT_PARAMS_GRID__ "-g 10 .2 5 -l 10 .001 .01 -a 0 3 3"
#define __DEFAULT_PARAMS_TRAIN_MC__ " -r 1 -s 1 0.001 -W 1 -f 4 5"
#define __DEFAULT_PARAMS_TRAIN_LS__ " -r 1 -s -1.0 0.001 -f 3 5"



extern "C" const char* liquid_svm_default_params(int stage, int solver){
	switch(stage){
	case 0:
		return "-g 10 .2 5 -l 10 .001 .01 -a 0 3 3";
		break;
	case 1:
//		char buff[256];
//		strcpy(buff, liquid_svm_default_params(0, solver));
//		if(solver == SVM_HINGE_2D) strcat(buff, " -r 1 -s 1 0.001 -W 1 -f 4 5");
//		else strcat(buff, " -r 1 -s -1.0 0.001 -f 3 5");
//		flush_info(1,"\ndefault_params: %s\n", buff);
//		return buff;
		if(solver == SVM_HINGE_2D) return __DEFAULT_PARAMS_GRID__ __DEFAULT_PARAMS_TRAIN_MC__;
		else return __DEFAULT_PARAMS_GRID__ __DEFAULT_PARAMS_TRAIN_LS__;
		break;
	case 2:
		return "-R 1";
		break;
	case 3:
		if(solver == SVM_HINGE_2D) return "-v 1 0 -L 0";
		else return "-v 1 1 -L 2";
		break;
	case -1:
#if defined(SSE2__) && defined(AVX__)
		return "Compiled with SSE2__ and AVX__";
#elif defined(SSE2__)
		return "Compiled with SSE2__ and no AVX__";
#elif defined(AVX__)
		return "Compiled with no SSE2__ but AVX__";
#else
		return "";
#endif
		break;
	default:
		return "";
	}
}


//**********************************************************************************************************************************


static map<int,Tsvm_manager*> cookies;
static int last_id = 0;
inline Tsvm_manager* getSVMbyCookie(int cookie){
    //flush_info("SVM getting from cookie %d  cookies.size: %d!\n",cookie, cookies.size());
//    map<int,Tsvm_manager*>::iterator it;
//    for(it = cookies.begin(); it != cookies.end(); it++){
//      if(it->first == cookie)
//        return it->second;
//    }
  if(cookies.find(cookie) != cookies.end()){
    return cookies[cookie];
  }else{
    flush_info("SVM not known from cookie %d  cookies.size: %d!\n",cookie, cookies.size());
    throw string("SVM not known");
  }
}

extern "C" int liquid_svm_init(const double* data, const unsigned size, const unsigned dim, const double* labels)
{
	if(size < 1 || dim < 1)
	{
    error("No data provided!\n");
    return -1;
	}
  try{
  Tsvm_manager *SVM = new Tsvm_manager();
		Tdataset data_set = Tdataset(data, size, dim, labels, __LIQUIDSVM_DATA_BY_COLS);
//		for(int i=0; i<8 && i<size; i++){
//		  flush_info(1,"%.2f ",data_set.sample(i)->label);
//		  for(int j=0; j<dim; j++) flush_info(1,",%.2f",data_set.sample(i)->coord(j));
//		  flush_info(1,"\n");
//		}
		SVM->load(data_set);
  
  
    int cookie = ++last_id;
    cookies[cookie] = SVM;
    flush_info(INFO_DEBUG,"\nnew cookie: %d, cookies.size: %d\n",cookie, cookies.size());
    return cookie;
    //getSVMbyCookie(cookie[0]);
  }catch(...){
    error("\nShould not happen!! liquid_svm_init\n");
    return -1;
  }
}

double* convertMatrixToArray(vector < vector<double> > mylog){
	int rows = mylog.size();
	double *ret = NULL;
	if(rows==0){
	  ret = new double[2];
	  ret[0]=ret[1]=0;
	  return ret;
	}
	int cols = mylog[0].size();
	ret = new double[2+rows * cols];
	unsigned k=0;
	ret[k++] = rows;
	ret[k++] = cols;
	for(int i=0; i<rows; i++)
		for(int j=0; j<cols; j++)
			ret[k++] = mylog[i][j];
	return ret;

}

vector<double> convertValInfo(int task, int cell, int fold, Tsvm_train_val_info info)
{
	vector<double> row;
	row.push_back(task);
	row.push_back(cell);
	row.push_back(fold);
	row.push_back(info.gamma);
	row.push_back(info.pos_weight);
	row.push_back(info.lambda);
	row.push_back(info.train_error);
	row.push_back(info.val_error);
	row.push_back(info.init_iterations);
	row.push_back(info.train_iterations);
	row.push_back(info.val_iterations);
	row.push_back(info.init_iterations);
	row.push_back(info.gradient_updates);
	row.push_back(info.SVs);
	return row;
}

extern "C" double* liquid_svm_train(int cookie, const int argc, char** argv)
{
	double* ret = NULL;

	Ttrain_control train_control;

	if(getConfig(cookie)->getI("SCALE")>0){
		train_control.scale_data = true;
		flush_info(1,"Using scaling\n");
	}

	train_control.store_solutions_internally = (getConfig(cookie)->getI("STORE_SOLUTIONS_INTERNALLY",
			getConfig(cookie)->getI("RETRAIN_METHOD", 1))>0);

  Tcommand_line_parser_svm_train command_line_parser;
  try{
	command_line_parser.setup(argc, argv);
	command_line_parser.parse(train_control, false);
  }catch(...){
    flush_info(0,"liquid_svm_select ");
    for(int i=0; i<argc; i++) flush_info(0,"%s ",argv[i]);
    flush_info(0,"\n");
    error("liquid_svm_select problems with command args\n");
    return NULL;
  }

  try{
	Tsvm_manager *SVM = getSVMbyCookie(cookie);
	flush_info(1,"\nWelcome to SVM train (dim=%d size=%d decision_functions=%d cookie=%d)\n",SVM->dim(),SVM->size(),SVM->decision_functions_size(), cookie);
	for(int i=0; i<argc; i++) flush_info(1,"%s ",argv[i]);

	Tsvm_full_train_info svm_full_train_info;

	train_control.store_logs_internally = true;
	
	SVM->train(train_control, svm_full_train_info);

	svm_full_train_info.train_val_info_log.display(TRAIN_INFO_DISPLAY_FORMAT_SUMMARIZED, INFO_1);

	vector <vector <vector < Ttrain_info_grid> > > list_of_grids = SVM->get_list_of_train_info();
	vector < vector<double> > mylog;

	for(unsigned task=0; task<list_of_grids.size(); task++)
	{
		for(unsigned cell=0; cell<list_of_grids[task].size(); cell++)
			for(unsigned fold=0; fold<list_of_grids[task][cell].size(); fold++)
			{
				Ttrain_info_grid train_val_info = list_of_grids[task][cell][fold];
				for (unsigned ig=0;ig<train_val_info.size();ig++)
					for (unsigned iw=0;iw<train_val_info[ig].size();iw++)
						for (unsigned il=0;il<train_val_info[ig][iw].size();il++)
							mylog.push_back(convertValInfo(task, cell, fold, train_val_info[ig][iw][il]));
			}
	}
	ret = convertMatrixToArray(mylog);
	
	flush_info(1, "\n");
  
  }catch(...){
    error("\nShould not happen!! liquid_svm_train\n");
  }
  return ret;
}


//**********************************************************************************************************************************


extern "C" double* liquid_svm_select(int cookie, const int argc, char** argv)
{
  double *ret = NULL;
  Tcommand_line_parser_svm_select command_line_parser;
  try{
	command_line_parser.setup(argc, argv);
	command_line_parser.parse(false);
  }catch(...){
    flush_info(0,"liquid_svm_select ");
    for(int i=0; i<argc; i++) flush_info(0,"%s ",argv[i]);
    flush_info(0,"\n");
    error("liquid_svm_select problems with command args\n");
    return NULL;
  }
	
	command_line_parser.select_control.use_stored_solution
		= (command_line_parser.select_control.select_method==SELECT_ON_EACH_FOLD &&
			getConfig(cookie)->getI("STORE_SOLUTIONS_INTERNALLY", 1) > 0);

  try{
	Tsvm_manager *SVM = getSVMbyCookie(cookie);
	flush_info(1,"\nWelcome to SVM select (dim=%d size=%d decision_functions=%d cookie=%d)\n",SVM->dim(),SVM->size(),SVM->decision_functions_size(), cookie);
	for(int i=0; i<argc; i++) flush_info(1,"%s ",argv[i]);

	Tsvm_full_train_info svm_full_train_info;

  	command_line_parser.select_control.use_stored_logs = true;
	command_line_parser.select_control.append_decision_functions = true;
  	command_line_parser.select_control.store_decision_functions_internally = true;
	
	SVM->select(command_line_parser.select_control, svm_full_train_info);
  
	vector <vector <vector < Tsvm_train_val_info> > > list_of_grids = SVM->get_list_of_select_info();
	vector < vector<double> > mylog;

	for(unsigned task=0; task<list_of_grids.size(); task++)
	{
	  for(unsigned cell=0; cell<list_of_grids[task].size(); cell++)
		  for(unsigned fold=0; fold<list_of_grids[task][cell].size(); fold++)
		  {
			  Tsvm_train_val_info train_val_info = list_of_grids[task][cell][fold];
			  unsigned theTask = task;
			  if((command_line_parser.select_control.weight_number >= 1) and (task==0))
				  theTask = command_line_parser.select_control.weight_number;
			  mylog.push_back(convertValInfo(theTask, cell, fold, train_val_info));
		  }
	}
	ret = convertMatrixToArray(mylog);

	flush_info(1, "\n");
  
  }catch(...){
    error("\nShould not happen!! liquid_svm_select\n");
  }
  
  return ret;
  
}


//**********************************************************************************************************************************


extern "C" double* liquid_svm_test(int cookie, const int argc, char** argv, const double* test_data, const unsigned test_size, const unsigned dim, const double* labels, double** error_ret)
{
  vector <double> predictions;

  Tcommand_line_parser_svm_test command_line_parser;
  try{
	command_line_parser.setup(argc, argv);
	command_line_parser.parse(false);
  }catch(...){
    flush_info(0,"liquid_svm_test ");
    for(int i=0; i<argc; i++) flush_info(0,"%s ",argv[i]);
    flush_info(0,"\n");
    error("liquid_svm_test problems with command args\n");
    return NULL;
  }

  try{

	Tsvm_manager *SVM = getSVMbyCookie(cookie);
	flush_info(1,"\nWelcome to SVM test (using SVM with dim=%d trained on size=%d decision_functions=%d cookie=%d)\n",SVM->dim(),SVM->size(),SVM->decision_functions_size(), cookie);
	for(int i=0; i<argc; i++) flush_info(1,"%s ",argv[i]);

	Tsvm_full_test_info test_info;

	Tdataset test_data_set = Tdataset(test_data, test_size, dim, labels, __LIQUIDSVM_DATA_BY_COLS);
	test_data_set.enforce_ownership();
	
	SVM->test(test_data_set, command_line_parser.test_control, test_info);

	double detection_rate;
	double false_alarm_rate;
	unsigned task_offset;
	// the following is a copy of svm-test.cpp

	if (test_info.number_of_tasks == test_info.number_of_all_tasks)
		task_offset = 1;
	else
		task_offset = 0;

	for(unsigned j=0;j<test_info.train_val_info.size();j++)
	{
	  test_info.train_val_info[j].val_time = test_info.test_time;
	  
	  if (command_line_parser.test_control.vote_control.scenario != VOTE_NPL)
		  flush_info(INFO_1,"\nTask %d: Test error %1.4f.",   j + task_offset, test_info.train_val_info[j].val_error);
	  else
	  {
		  if (command_line_parser.test_control.vote_control.npl_class == -1)
		  {
			  detection_rate = 1.0 - test_info.train_val_info[j].pos_val_error;
			  false_alarm_rate = test_info.train_val_info[j].neg_val_error;
			  
		  }
		  else
		  {
			  detection_rate = 1.0 - test_info.train_val_info[j].neg_val_error;
			  false_alarm_rate = test_info.train_val_info[j].pos_val_error;
		  }
		  flush_info(INFO_1,"\nTask %d: DR %1.4f.  FAR %1.4f.", j + task_offset, detection_rate, false_alarm_rate);
	  }
	}
	// end of copy

	vector < vector<double> > predictions_ret;
	for(unsigned i=0; i<test_size; i++)
		predictions_ret.push_back(SVM->get_predictions_for_test_sample(i));
  

	  // better safe than sorry: the allocated length of error_ret is written in its first argument
/*	int error_ret_len = (int)error_ret[0];
	flush_info(INFO_DEBUG,"error_ret_len=%d\n",error_ret_len);
	unsigned all_tasks = SVM->number_of_all_tasks();
	flush_info(1,"\nall_tasks=%d",all_tasks);
	for(i=0; 3*i<error_ret_len && i<all_tasks && i<test_info.train_val_info.size(); i++){
		//flush_info(1,"Task %d: error=%f\n",i,test_info.train_val_info[i].val_error);
		error_ret[3*i] = test_info.train_val_info[i].val_error;
		error_ret[3*i+1] = test_info.train_val_info[i].pos_val_error;
		error_ret[3*i+2] = test_info.train_val_info[i].neg_val_error;
	}*/
	
	vector < vector<double> > error_matrix;
	for(unsigned task=0; task<test_info.train_val_info.size(); task++)
	{
    vector<double> error_vec;
	  error_vec.push_back(test_info.train_val_info[task].val_error);
	  error_vec.push_back(test_info.train_val_info[task].pos_val_error);
	  error_vec.push_back(test_info.train_val_info[task].neg_val_error);
	  error_matrix.push_back(error_vec);
	}
	error_ret[0] = convertMatrixToArray(error_matrix);
	
	flush_info(1, "\n");
	
	return convertMatrixToArray(predictions_ret);

  }catch(...){
    error("\nShould not happen!! liquid_svm_test\n");
    return NULL;
  }
	
}

//**********************************************************************************************************************************


extern Tsubset_info liquid_svm_get_cover(int cookie, unsigned task)
{
  Tsubset_info info;
  try{

	Tsvm_manager *SVM = getSVMbyCookie(cookie);
	// In display, the tasks are numbered from say 1 to 5, but internally they are from 0 to 4
	info = SVM->get_working_set_manager().cover_of_task(task-1);

  }catch(...){
    error("\nShould not happen!! liquid_svm_test\n");
  }
  return info;
  
}

extern Tsvm_decision_function liquid_svm_get_solution(int cookie, unsigned task, unsigned cell, unsigned fold)
{
  Tsvm_decision_function ret;
  try{

	Tsvm_manager *SVM = getSVMbyCookie(cookie);
	ret = SVM->get_decision_function_manager().get_decision_function(task-1,cell-1,fold-1);

  }catch(...){
    error("\nShould not happen!! liquid_svm_test\n");
  }
  return ret;

}


extern "C" void liquid_svm_write_solution(int cookie, const char* filename, size_t len, unsigned char *buffer)
{
  try{

	Tsvm_manager *SVM = getSVMbyCookie(cookie);
	Tconfig *config = getConfig(cookie);

	FILE* fpsolwrite = open_file(string(filename), "w");

	SVM->write_decision_function_manager_to_file(fpsolwrite);
	config->write_to_file(fpsolwrite);

	if(buffer != NULL){
		file_write(fpsolwrite, (unsigned)len);
		file_write_eol(fpsolwrite);
		unsigned c;
		for(size_t i=0; i<len; i++){
			c = buffer[i];
			putc(c, fpsolwrite);
		}
		file_write_eol(fpsolwrite);
	}else{
		file_write(fpsolwrite, 0);
		file_write_eol(fpsolwrite);
		file_write_eol(fpsolwrite);
	}

	close_file(fpsolwrite);

  }catch(...){
    error("\nShould not happen!! liquid_svm_write_solution\n");
  }

}

extern "C" int liquid_svm_read_solution(int cookie, const char* filename, size_t *len, unsigned char **buffer)
{
  try{

	Tsvm_manager *SVM;
	bool existing = (cookie > 0);
	if(existing){
		SVM = getSVMbyCookie(cookie);
	}else{
		SVM = new Tsvm_manager();
	}

	FILE* fpsolread = open_file(string(filename), "r");

	bool withData = false;
	SVM->read_decision_function_manager_from_file(fpsolread, withData);

	if(!withData && !existing){
		flush_info(INFO_1, "Trying to data from file but it is not there\n");
		close_file(fpsolread);
		return -1;
	}else if(withData && existing){
		flush_info(INFO_1, "Will now read data from solution into SVM that already has data...\n");
	}
	if(!existing){
	    cookie = ++last_id;
	    cookies[cookie] = SVM;
	    flush_info(INFO_DEBUG,"\nnew cookie: %d, cookies.size: %d (created for reading)\n",cookie, cookies.size());
	}

	Tconfig *config = getConfig(cookie);
	config->read_from_file(fpsolread);

	unsigned buffer_len = 0;
	file_read(fpsolread, buffer_len);
	if(len != NULL)
		*len = (size_t)buffer_len;
	if(buffer != NULL){
		if(buffer_len > 0){
			*buffer = new unsigned char[buffer_len];
			unsigned c;
			// read and ignore the newline character given by file_write_eol
			do
				c = getc(fpsolread);
			while (c!='\n');
			for(size_t i=0; i<buffer_len; i++){
				c = getc(fpsolread);
				(*buffer)[i] = c;
			}
		}else{
			*buffer = NULL;
		}
	}

	close_file(fpsolread);

  }catch(...){
    error("\nShould not happen!! liquid_svm_read_solution\n");
  }

  return cookie;

}

//**********************************************************************************************************************************


extern "C" void liquid_svm_clean(int cookie)
{
  if(cookie < 0){
    flush_info(1,"\nNegative cookie (%d) to clean??\n", cookie);
    return;
  }
  try{
    Tsvm_manager *SVM = getSVMbyCookie(cookie);
    if(SVM != NULL){
    	flush_info(2,"\nWelcome to SVM clean (dim=%d size=%d decision_functions=%d cookie=%d)\n",SVM->dim(),SVM->size(),SVM->decision_functions_size(), cookie);
		SVM->clear();
		delete SVM;
		cookies.erase(cookie);
		deleteConfig(cookie);
    }
  }catch(...){
    warning("\nShould not happen!! liquid_svm_R_clean\n");
    return;
  }
}


#endif

