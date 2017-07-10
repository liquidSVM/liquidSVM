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

/*****************************************************
 * Compile using:
mex CXXFLAGS="\$CXXFLAGS -march=native -O3 -std=c++11" -g -I../.. mexliquidSVM.cpp
 *******************************************************/

#include <math.h>
//#include <matrix.h>
#include <mex.h>
#include <stdio.h>
#include <stdarg.h>
#include <string>


#define VPRINTF(message_format, ...) va_list arguments; \
              va_start(arguments, message_format); \
              myMexVPrintf(message_format, arguments); \
              va_end(arguments);

// we need to have a vprintf-analog wrapper around mexPrintf
// taking a va_list argument:
void myMexVPrintf(const char *fmt, va_list args)
{
	int code;
	char buf[2560];
	vsnprintf(buf, (size_t)2560, fmt, args);
	code = mexPrintf("%s", buf);
    
    // the following should flush the buffer??
	// But it only works if called from the main MATLAB-thread, which is
    // not so easy in our case...
//     mexEvalString("drawnow;");
}

void CheckUserInterrupt();
#define COMPILE_FOR_R__

// Undocumented Matlab feature in libut library??
#ifdef DO_MATLAB
extern "C" bool utIsInterruptPending();
extern "C" bool utSetInterruptEnabled(bool);
#endif
#ifdef DO_OCTAVE
#include <signal.h>
extern volatile sig_atomic_t octave_signal_caught;
//#include <octave/oct.h>
#endif

#include "common/liquidSVM.h"

void CheckUserInterrupt(){
#ifdef DO_MATLAB
	if(utIsInterruptPending()){
        flush_info(2,"utIsInterruptPending true");
		throw std::string("Interrupted");
	}
#elif defined(DO_OCTAVE)
	if(octave_signal_caught){
        flush_info(2,"octave_signal_caught");
		throw std::string("Interrupted");
	}

#endif
}



void do_default_params(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  if(nlhs != 1){
      error("default_params returns exactly one argument");
      return;
  }
  if(nrhs != 2){
      error("default_params needs exactly two arguments: stage, solver");
      return;
  }
  int stage = (int) mxGetPr(prhs[0])[0];
  int solver = (int) mxGetPr(prhs[1])[0];
  
  const char *ret = liquid_svm_default_params(stage, solver);
  plhs[0] = mxCreateString(ret);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//////////////////        liquidSVM-test     //////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void do_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  if(nrhs < 2){
	  mexErrMsgTxt("init needs at least two arguments: x,y");
      return;
  }
  if(nrhs > 5){
	  mexErrMsgTxt("init cannot handle more than five arguments: x, y, sampleWeight, groupId, id");
      return;
  }
  //declare variables
  mxArray *a_in_m, *b_in_m, *in_m;
  double *a, *b;
  double *sw=NULL, *gsR=NULL, *idsR=NULL;
  unsigned *gs=NULL, *ids=NULL;
  //associate inputs
  a_in_m = mxDuplicateArray(prhs[0]);
  b_in_m = mxDuplicateArray(prhs[1]);
  //figure out dimensions
  int size = (unsigned) mxGetDimensions(prhs[0])[1];
  int dims = (unsigned) mxGetDimensions(prhs[0])[0];
  
  if(mxGetDimensions(prhs[1])[0] != size || mxGetDimensions(prhs[1])[1] != 1){
	  flush_info("Training labels have not correct dimensions: %d x %d", mxGetDimensions(prhs[1])[0],mxGetDimensions(prhs[1])[1]);
	  mexErrMsgTxt("Training labels have not correct dimensions.");
	  return;
  }

  //flush_info("size %d dims %d\n",size,dims);
  
  a = mxGetPr(a_in_m);
  b = mxGetPr(b_in_m);
  
  int i=2;

  if(nrhs > i){
	if(mxGetDimensions(prhs[i])[1] != 0){
	  if(mxGetDimensions(prhs[i])[0] != size || mxGetDimensions(prhs[1])[1] != 1){
		  flush_info("sampleWeights have not correct dimensions: %d x %d, should be: %d x %d", mxGetDimensions(prhs[i])[0],mxGetDimensions(prhs[i])[1], size, 1);
		  mexErrMsgTxt("sampleWeights labels have not correct dimensions.");
		  return;
	  }
	  in_m = mxDuplicateArray(prhs[i]);
	  sw = mxGetPr(in_m);
	}
  }

  i++;
  if(nrhs > i){
	if(mxGetDimensions(prhs[i])[1] != 0){
	  if(mxGetDimensions(prhs[i])[0] != size || mxGetDimensions(prhs[1])[1] != 1){
		  flush_info("groupIds have not correct dimensions: %d x %d, should be: %d x %d", mxGetDimensions(prhs[i])[0],mxGetDimensions(prhs[i])[1], size, 1);
		  mexErrMsgTxt("groupIds labels have not correct dimensions.");
		  return;
	  }
	  in_m = mxDuplicateArray(prhs[i]);
	  gsR = mxGetPr(in_m);
	  gs = new unsigned[size];
	  for(int j=0; j<size; j++)
		  gs[j] = (unsigned)gsR[j];
	}
  }

  i++;
  if(nrhs > i){
	if(mxGetDimensions(prhs[i])[1] != 0){
	  if(mxGetDimensions(prhs[i])[0] != size || mxGetDimensions(prhs[1])[1] != 1){
		  flush_info("ids have not correct dimensions: %d x %d, should be: %d x %d", mxGetDimensions(prhs[i])[0],mxGetDimensions(prhs[i])[1], size, 1);
		  mexErrMsgTxt("ids labels have not correct dimensions.");
		  return;
	  }
	  in_m = mxDuplicateArray(prhs[i]);
	  idsR = mxGetPr(in_m);
	  ids = new unsigned[size];
	  for(int j=0; j<size; j++)
		  ids[j] = (unsigned)idsR[j];
	}
  }

  int cookie = liquid_svm_init_annotated(a, size, dims, b, sw, gs, ids);
  
  plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
  mxGetPr(plhs[0])[0] = cookie;

  if(gs != NULL)
	  delete[] gs;
  if(ids != NULL)
	  delete[] ids;
}

void do_train(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  info_mode = 1;
  const int argc=nrhs + 1;
  //char* argv[] = { "matlab_liquidSVM", "-d", "1", "-S", "1" };
  char** argv= (char**)mxMalloc(argc * sizeof(char*));
  argv[0] = (char*) "matlab_liquidSVM_train";
  for(int i=0; i<nrhs; i++)
      argv[1+i] = (char*)mxArrayToString(prhs[i]);
  
  double *errors_train = liquid_svm_train(cookie, argc, argv);
  if(errors_train == NULL){
	  mexErrMsgTxt("Train did not complete");
      return;
  }

  mxFree(argv);
  
  mwSize errors_train_rows = (mwSize)errors_train[0];

  //associate outputs
  if(nlhs>=1 && errors_train_rows >0){
      mwSize cols = (mwSize)errors_train[1];
      //mexPrintf("errors train: %d rows %d cols\n",errors_train_rows, cols);
      plhs[0] = mxCreateDoubleMatrix(errors_train_rows,cols,mxREAL);
      double *c = mxGetPr(plhs[0]);
      
      for(int i=0; i<errors_train_rows; i++)
          for(int j=0; j<cols; j++){
              int matl_k = i + errors_train_rows * j;
              int ours_k = i * cols + j;
              c[matl_k] = errors_train[2+ours_k];
          }
  }
  free(errors_train);
  
}

void do_select(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  const int argc=nrhs+1;
  //char* argv[] = { "matlab_liquidSVM", "-d", "1", "-S", "1" };
  char** argv= (char**)mxMalloc(argc * sizeof(char*));
  argv[0] = (char*) "matlab_liquidSVM_select";
  for(int i=0; i<nrhs; i++)
      argv[1+i] = (char*)mxArrayToString(prhs[i]);
  
  double *errors_select = liquid_svm_select(cookie,argc,argv);

  mxFree(argv);
  
  if(errors_select == NULL){
	  mexErrMsgTxt("Select did not complete");
      return;
  }
  
  mwSize errors_select_rows = (mwSize)errors_select[0];

  //associate outputs
  if(nlhs >= 1 && errors_select_rows >0){
      mwSize cols = (mwSize)errors_select[1];
      //mexPrintf("errors select: %d rows %d cols\n",errors_select_rows, cols);
      plhs[0] = mxCreateDoubleMatrix(errors_select_rows,cols,mxREAL);
      double *d = mxGetPr(plhs[0]);

      for(int i=0; i<errors_select_rows; i++)
          for(int j=0; j<cols; j++){
              int matl_k = i + errors_select_rows * j;
              int ours_k = i * cols + j;
              d[matl_k] = errors_select[2+ours_k];
          }
  }

  free(errors_select);
  
}
    
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//////////////////        liquidSVM-test     //////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void do_test(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs < 2){
      return;
  }
  
  //declare variables
  mxArray *a_in_m, *b_in_m;
  double *a, *b;
  //associate inputs
  a_in_m = mxDuplicateArray(prhs[0]);
  b_in_m = mxDuplicateArray(prhs[1]);
  //figure out dimensions
  int size = (unsigned) mxGetDimensions(prhs[0])[1];
  int dims = (unsigned) mxGetDimensions(prhs[0])[0];
  
  //flush_info("size %d dims %d\n",size,dims);
  
  a = mxGetPr(a_in_m); 
  if(mxGetDimensions(prhs[1])[0] > 0)
      b = mxGetPr(b_in_m); 
  else
      b = NULL;
  
  const int argc=1+nrhs-2;
  //char* argv[] = { "matlab_liquidSVM", "-d", "1", "-S", "1" };
  char** argv= (char**)mxMalloc(argc * sizeof(char*));
  argv[0] = (char*) "matlab_liquidSVM";
  for(int i=1; i<argc; i++)
      argv[i] = (char*)mxArrayToString(prhs[2-1+i]);
  
  double *error_dummy_array = NULL;
  double **errors_test = &error_dummy_array;
  
  double *result = liquid_svm_test(cookie, argc, argv, a, size, dims, b, errors_test);
  //mexPrintf("\n");
  
  mxFree(argv);
  if(result == NULL || errors_test==NULL || *errors_test==NULL){
	  mexErrMsgTxt("Training did not complete");
      return;
  }

  mwSize result_rows = (mwSize)result[0];
  mwSize errors_test_rows = (mwSize)(*errors_test)[0];
  
  //associate outputs
  if(nlhs>=1 && result_rows >0){
      mwSize cols = (mwSize)result[1];
      //mexPrintf("result: %d rows %d cols\n",result_rows, cols);
      plhs[0] = mxCreateDoubleMatrix(result_rows,cols,mxREAL);
      double *c = mxGetPr(plhs[0]);
      
      for(int i=0; i<result_rows; i++)
          for(int j=0; j<cols; j++){
              int matl_k = i + result_rows * j;
              int ours_k = i * cols + j;
              c[matl_k] = result[2+ours_k];
          }
  }
  if(nlhs >= 2){
      if(errors_test_rows > 0){
      mwSize cols = (mwSize)(*errors_test)[1];
      //mexPrintf("errors test: %d rows %d cols\n",errors_test_rows, cols);
      plhs[1] = mxCreateDoubleMatrix(errors_test_rows,cols,mxREAL);
      double *d = mxGetPr(plhs[1]);

      for(int i=0; i<errors_test_rows; i++)
          for(int j=0; j<cols; j++){
              int matl_k = i + errors_test_rows * j;
              int ours_k = i * cols + j;
              d[matl_k] = (*errors_test)[2+ours_k];
          }
      }else{
          plhs[1] = mxCreateDoubleMatrix(0,0,mxREAL);
      }
  }

  delete result;
  delete *errors_test;

}

void do_clean(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//////////////////        configuration      //////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void do_get_param(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs != 1 || nlhs != 1)
      return;
  
  const char* name = mxArrayToString(prhs[0]);
  char* value = liquid_svm_get_param(cookie, name);
  plhs[0] = mxCreateString(value);
  free(value);
  
}

void do_set_param(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs != 2 || nlhs != 0)
      return;
  
  const char* name = mxArrayToString(prhs[0]);
  const char* value = mxArrayToString(prhs[1]);
  liquid_svm_set_param(cookie, name, value);
  
}

void do_get_config_line(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs != 1 || nlhs != 1)
      return;
  
  int stage = (int) mxGetPr(prhs[0])[0];
  char* value = liquid_svm_get_config_line(cookie, stage);
  plhs[0] = mxCreateString(value);
  free(value);
}

void do_get_cover(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs != 1 || nlhs != 1)
      return;
  
  int task = (int) mxGetPr(prhs[0])[0];
  Tsubset_info info = liquid_svm_get_cover(cookie, task);
  
  plhs[0] = mxCreateDoubleMatrix(info.size(),1,mxREAL);
  double *c = mxGetPr(plhs[0]);
  for(size_t i=0; i<info.size(); i++)
    c[i] = info[i];
}

void do_get_solution(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    flush_info(1,"getting solution with nlhs=%d nrhs=%d\n",nlhs,nrhs);
  if(nrhs != 3 || nlhs > 3)
      return;
  
  int task = (int) mxGetPr(prhs[0])[0];
  int cell = (int) mxGetPr(prhs[1])[0];
  int fold = (int) mxGetPr(prhs[2])[0];

  Tsvm_decision_function df = liquid_svm_get_solution(cookie, task, cell, fold);
  
  if(nlhs>0){
      flush_info(1,"setting offset\n");
    //plhs[0] = mxCreateDoubleScalar(df.offset);
    plhs[0] = mxCreateDoubleScalar(0);
  }
  if(nlhs>1){
      flush_info(1,"setting samples\n");
      plhs[1] = mxCreateDoubleMatrix(df.sample_number.size(),1,mxREAL);
      double *c = mxGetPr(plhs[1]);
      for(size_t i=0; i<df.sample_number.size(); i++)
        c[i] = df.sample_number[i];
  }
  if(nlhs>2){
      flush_info(1,"setting coeffs\n");
      plhs[2] = mxCreateDoubleMatrix(df.coefficient.size(),1,mxREAL);
      double *c = mxGetPr(plhs[2]);
      for(size_t i=0; i<df.coefficient.size(); i++)
        c[i] = df.coefficient[i];
  }
}

void do_read_solution(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs != 1 || nlhs > 1)
      return;
  
  const char* filename = mxArrayToString(prhs[0]);
  cookie = liquid_svm_read_solution(cookie, filename, NULL, NULL);
  if(nlhs==1)
	  plhs[0] = mxCreateDoubleScalar(cookie);
}

void do_write_solution(int cookie, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if(nrhs != 1 || nlhs != 0)
      return;
  
  const char* filename = mxArrayToString(prhs[0]);
  liquid_svm_write_solution(cookie, filename, 0, NULL);
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  //mexPrintf("nrhs=%d nlhs=%d\n", nrhs, nlhs);
  if(nrhs == 0){
      error("No arguments");
      return;
  }
  int stage = (int) mxGetPr(prhs[0])[0];
  nrhs--;
  prhs++;
  
  int cookie = -1;
  if(stage > 0 && nrhs >= 1){
        cookie = (int) mxGetPr(prhs[0])[0];
        nrhs--;
        prhs++;
  }
  
  // if the following is true we would not do anything at all - helpful to debug Matlab-Interface
  if(false){
      mexPrintf("Would do stage %d at cookie %d ...\n",stage, cookie);
      
      cookie = 123;
      for(int i=0; i<nlhs; i++){
            plhs[i] = mxCreateDoubleMatrix(1,1,mxREAL);
            mxGetPr(plhs[i])[0] = cookie;
      }
      return;
  }
  
#ifdef DO_MATLAB
  bool oldState = utSetInterruptEnabled(false);
#endif
  
  switch(stage){
      case -1:
          do_default_params(nlhs,plhs,nrhs,prhs);
          break;
      case 0:
          do_init(nlhs, plhs, nrhs, prhs);
          break;
      case 1:
          do_train(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 2:
          do_select(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 3:
          do_test(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 10:
          do_get_param(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 11:
          do_set_param(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 12:
          do_get_config_line(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 21:
          do_get_cover(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 22:
          do_get_solution(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 23:
          do_read_solution(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 24:
          do_write_solution(cookie, nlhs, plhs, nrhs, prhs);
          break;
      case 100:
          do_clean(cookie, nlhs, plhs, nrhs, prhs);
          break;
      default:
          error("Stage not known");
  }
#ifdef DO_MATLAB
  utSetInterruptEnabled(oldState);
#endif
}
