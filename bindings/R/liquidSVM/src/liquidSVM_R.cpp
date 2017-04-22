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

// to get Rvprintf loaded in <R.h> we define:
#define R_USE_C99_IN_CXX

#include <R.h>
#ifdef _WIN32
  #undef Realloc
  #undef Free
  #define R_Realloc(p,n,t) (t *) R_chk_realloc( (void *)(p), (size_t)((n) * sizeof(t)) )
#endif

#define VPRINTF(message_format) va_list arguments; \
  								va_start(arguments, message_format); \
									Rvprintf(message_format, arguments); \
									va_end(arguments);


int R_rand(){
  // the follwing uses unif_rand from R.h
  // therefore all bindings (which call this)
  // have to be braced in Get/PutRNDGstate()
  return (int) (unif_rand() * RAND_MAX);
}

#define __RAND__() R_rand();

extern "C" void __NOOP__(int i){ }
#define __SRAND__(i) __NOOP__(i);

// **************************************
// now we include the hole liquidSVM-infrastructure:
// (if COMPILE_SEPERATELY is not defined this also includes all appropriate *.cpp files)
// **************************************

#define COMPILE_FOR_R__
void CheckUserInterrupt();

// in R data is usually stored by columns
#define __LIQUIDSVM_DATA_BY_COLS true

#include "liquidSVM.h"
#include "kernel_calculator.cpp"
#include "scenario_config.h"

#include <Rinternals.h> // On MacOS needs to come after <iostream.h>...


#include <R_ext/GraphicsEngine.h>  // from that we need only: extern int R_interrupts_pending;
void CheckUserInterrupt(){
//  R_CheckUserInterrupt();
     if(R_interrupts_pending){
       throw string("Interrupted");
     }
}

#include <R_ext/Rdynload.h>

// **************************************
//      and now for the bindings
// **************************************


extern "C" {

SEXP liquid_svm_R_default_params(SEXP stage, SEXP solver){
  return mkString(liquid_svm_default_params(INTEGER(stage)[0],INTEGER(solver)[0]));
}
  
SEXP liquid_svm_R_init(SEXP dataR, SEXP labelsR){
  int size = length(labelsR);
  if(size < 1){
    error("No data");
    return R_NilValue;
  }
  int dim = length(dataR) / size;
  if(dim < 1){
    error("No features");
    return R_NilValue;
  }
  
  SEXP cookieR = PROTECT(allocVector(INTSXP, 1));
  double *data, *labels;
  int *cookie;
  data = REAL(dataR);
  labels = REAL(labelsR);
  cookie = INTEGER(cookieR);
  GetRNGstate();
  cookie[0] = liquid_svm_init(data, size, dim, labels);
  PutRNGstate();
  
  UNPROTECT(1);
  return cookieR;
}

SEXP liquid_svm_R_train(SEXP cookieR, SEXP argsR){
  int argc = length(argsR);
  char **argv;
  argv = new char*[argc];
  if(argv == NULL)
    error("Memory allocation fails!\n");
  for(int i = 0; i < argc; i++)
    argv[i] = (char*)CHAR(STRING_ELT(argsR, i));
  
  GetRNGstate();
  double* val_errors = liquid_svm_train(asInteger(cookieR), argc, argv);
  PutRNGstate();

  if(val_errors == NULL){
    return R_NilValue;
  }
  int k=0;
  int rows = (int)val_errors[k++];
  if(rows==0)
    return R_NilValue;
  int cols = val_errors[k++];
  SEXP val_errorsR = PROTECT(allocVector(REALSXP, rows * cols));
  memcpy(REAL(val_errorsR), val_errors + k, rows * cols * sizeof(double));

  delete[] argv;
  delete[] val_errors;
  
  UNPROTECT(1);
  return val_errorsR;
}

extern SEXP liquid_svm_R_select(SEXP cookieR, SEXP argsR){
  int argc = length(argsR);
  char **argv;
  argv = new char*[argc];
  if(argv == NULL)
    error("Memory allocation fails!\n");
  for(int i = 0; i < argc; i++)
    argv[i] = (char*)CHAR(STRING_ELT(argsR, i));

  GetRNGstate();
  double* val_errors = liquid_svm_select(asInteger(cookieR), argc, argv);
  PutRNGstate();
  
  delete[] argv;

  if(val_errors == NULL){
    return ScalarReal(NA_REAL);
  }
  int k=0;
  int rows = (int)val_errors[k++];
  if(rows==0)
    return R_NilValue;
  int cols = val_errors[k++];
  SEXP val_errorsR = PROTECT(allocVector(REALSXP, rows * cols));
  memcpy(REAL(val_errorsR), val_errors + k, rows * cols * sizeof(double));

  delete[] val_errors;
  
  UNPROTECT(1);
  return val_errorsR;
}
extern SEXP liquid_svm_R_test(SEXP cookieR, SEXP argsR, SEXP test_sizeP, SEXP test_dataR, SEXP labelsR){
  int size = asInteger(test_sizeP);
  if(size < 1){
    error("No test data");
    return R_NilValue;
  }
  int dim = length(test_dataR) / size;
  if(dim < 1){
    error("No test features");
    return R_NilValue;
  }
  int argc = length(argsR);
  char **argv;
  argv = new char*[argc];
  if(argv == NULL)
    error("Memory allocation fails!\n");
  for(int i = 0; i < argc; i++)
    argv[i] = (char*)CHAR(STRING_ELT(argsR, i));
  
  double *test_data, *labels;
  test_data = REAL(test_dataR);
  if(length(labelsR)>0){
    labels = REAL(labelsR);
  }else{
    labels = NULL;
  }
  
  double *error_ret = NULL;
  //double **error_ret = &error_dummy_array;

  GetRNGstate();
  double* predictions = liquid_svm_test(asInteger(cookieR), argc, argv, test_data, size, dim, labels, &error_ret);
  PutRNGstate();
  
  if(predictions == NULL){
    return R_NilValue;
  }
  int k=0;
  int rows = (int)predictions[k++];
  if(rows==0)
    return R_NilValue;
  int cols = predictions[k++];
  SEXP predictionsR = PROTECT(allocVector(REALSXP, rows * cols));
  memcpy(REAL(predictionsR), predictions + k, rows * cols * sizeof(double));

  if(error_ret != NULL){
  k=0;
  rows = (int)error_ret[k++];
  if(rows!=0){
  cols = error_ret[k++];
  SEXP error_retR = (allocVector(REALSXP, rows * cols));
  memcpy(REAL(error_retR), error_ret + k, rows * cols * sizeof(double));

  setAttrib(predictionsR, install("error_ret"),error_retR);
  }
  }
  
  delete[] argv;
  delete[] predictions;
  delete[] error_ret;
  
  UNPROTECT(1);
  return predictionsR;
}
extern SEXP liquid_svm_R_clean(SEXP cookieR){
  liquid_svm_clean(asInteger(cookieR));
  return R_NilValue;
}

extern SEXP liquid_svm_R_set_info_mode(SEXP i){ info_mode = asInteger(i); return R_NilValue; }
  
extern SEXP liquid_svm_R_kernel(SEXP dataR, SEXP dimR, SEXP kernelR, SEXP aux_fileR, SEXP gammaR, SEXP threadsR, SEXP GPUsR){

  int dim = *INTEGER(dimR);
  int size = length(dataR)/dim;

  Tkernel_calculator kernel_calculator;
  kernel_calculator.gamma = *REAL(gammaR);
  
  Tdataset data = Tdataset(REAL(dataR), size, dim, NULL);
  
  Tkernel_control kernel_control;
  kernel_control.kernel_type = *INTEGER(kernelR);
  string aux_file(CHAR(STRING_ELT(aux_fileR,0)));
  kernel_control.hierarchical_kernel_control_read_filename = aux_file;
  // TODO activate GPUs
  
  kernel_control.memory_model_pre_kernel = BLOCK;
  kernel_control.memory_model_kernel = BLOCK;
  
  Tparallel_control parallel_ctrl;
  parallel_ctrl.requested_team_size = *INTEGER(threadsR);

  kernel_calculator.reserve_threads(parallel_ctrl);
  
  kernel_calculator.calculate(kernel_control, data);
  
  Tkernel *kernel = &kernel_calculator.kernel;
  
  unsigned n = kernel->get_row_set_size();

  kernel_calculator.clear_threads();

  SEXP retR = PROTECT(allocVector(REALSXP, n * n));
  // TODO is this okay??
  //for(int i=0; i<n; i++)
  //  memcpy(REAL(retR),kernel->row(i),cols * sizeof(double));
  for(unsigned i=0; i<n; i++)
    for(unsigned j=0; j<n; j++)
      REAL(retR)[i*n+j] = kernel->entry(i,j);
  
  UNPROTECT(1);
  return retR;
}

extern SEXP liquid_svm_R_set_param(SEXP cookieR, SEXP nameR, SEXP valueR){
	liquid_svm_set_param(asInteger(cookieR), (char*)CHAR(STRING_ELT(nameR, 0)), (char*)CHAR(STRING_ELT(valueR, 0)));
  return R_NilValue;
}
extern SEXP liquid_svm_R_get_param(SEXP cookieR, SEXP nameR){
  char* value = liquid_svm_get_param(asInteger(cookieR), (char*)CHAR(STRING_ELT(nameR, 0)));
  SEXP ret = mkString(value);
  free(value);
  return ret;
}
extern SEXP liquid_svm_R_get_config_line(SEXP cookieR, SEXP stageR){
  char* line = liquid_svm_get_config_line(asInteger(cookieR), asInteger(stageR));
  SEXP ret = mkString(line);
  free(line);
  return ret;
}
  
  

extern SEXP liquid_svm_R_get_cover(SEXP cookieR, SEXP taskR){
  Tsubset_info info = liquid_svm_get_cover(asInteger(cookieR), asInteger(taskR));
  SEXP retR = PROTECT(allocVector(INTSXP, info.size()));
  for(size_t i=0; i<info.size(); i++)
    INTEGER(retR)[i] = info[i];
  UNPROTECT(1);
  return retR;
}

extern SEXP liquid_svm_R_get_solution(SEXP cookieR, SEXP taskR, SEXP cellR, SEXP foldR){
  
  Tsvm_decision_function df = liquid_svm_get_solution(asInteger(cookieR), asInteger(taskR), asInteger(cellR), asInteger(foldR));
  
  //SEXP offsetR = ScalarReal(df.offset);
  SEXP offsetR = ScalarReal(0);
  
  SEXP svR = PROTECT(allocVector(INTSXP, df.sample_number.size()));
  for(size_t i=0; i<df.sample_number.size(); i++)
    INTEGER(svR)[i] = df.sample_number[i];
  
  SEXP coeffR = PROTECT(allocVector(REALSXP, df.coefficient.size()));
  for(size_t i=0; i<df.coefficient.size(); i++)
    REAL(coeffR)[i] = df.coefficient[i];
  // memcpy(REAL(coeffR), df.coefficients.begin(), df.coefficient.size() * sizeof(double));
  
  const char *names[] = { "offset", "sv", "coeff", "" };
  SEXP retR = PROTECT(mkNamed(VECSXP, names));
  SET_VECTOR_ELT(retR, 0, offsetR);
  SET_VECTOR_ELT(retR, 1, svR);
  SET_VECTOR_ELT(retR, 2, coeffR);
  UNPROTECT(3);
  return retR;
}

extern SEXP liquid_svm_R_write_solution(SEXP cookieR, SEXP filenameR, SEXP further){
  unsigned char *buffer = new unsigned char[length(further)];
  for(size_t i=0; i < (size_t)length(further); i++) buffer[i] = RAW(further)[i];
  //memcpy(buffer, 0, RAW(further), 0, length(further))
  liquid_svm_write_solution(asInteger(cookieR), (char*)CHAR(STRING_ELT(filenameR, 0)),
                            length(further), buffer);
  delete[] buffer;
  return R_NilValue;
}
extern SEXP liquid_svm_R_read_solution(SEXP cookieR, SEXP filenameR){
  unsigned char *buffer = NULL;
  size_t buffer_len = 0;
  int cookie = liquid_svm_read_solution(asInteger(cookieR), (char*)CHAR(STRING_ELT(filenameR, 0)),
                                        &buffer_len, &buffer);
  if(cookie < 0){
	  error("Could not read data");
    if(buffer != NULL)
      delete[] buffer;
	  return R_NilValue;
  }
  SEXP further = R_NilValue;
  if(buffer != NULL){
    further = PROTECT(allocVector(RAWSXP, buffer_len));
    for(size_t i=0; i < (size_t)length(further); i++) RAW(further)[i] = buffer[i];
    //memcpy(RAW(further),0,buffer,0,buffer_len);
    delete[] buffer;
  }
  Tsvm_manager *SVM = getSVMbyCookie(cookie);
  SEXP retR = PROTECT(allocVector(VECSXP, 4));
  SET_VECTOR_ELT(retR, 0, ScalarInteger(cookie));
  SET_VECTOR_ELT(retR, 1, ScalarInteger(SVM->dim()));
  SET_VECTOR_ELT(retR, 2, ScalarInteger(SVM->size()));
  SET_VECTOR_ELT(retR, 3, further);
  UNPROTECT(2);
  return retR;
}

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

static const R_CallMethodDef R_CallDef[] = {
  CALLDEF(liquid_svm_R_default_params,2),
  CALLDEF(liquid_svm_R_init,2),
  CALLDEF(liquid_svm_R_train,2),
  CALLDEF(liquid_svm_R_select,2),
  CALLDEF(liquid_svm_R_test,5),
  CALLDEF(liquid_svm_R_clean,1),
  CALLDEF(liquid_svm_R_set_info_mode,1),
  CALLDEF(liquid_svm_R_kernel,7),
  CALLDEF(liquid_svm_R_set_param,3),
  CALLDEF(liquid_svm_R_get_param,2),
  CALLDEF(liquid_svm_R_get_config_line,2),
  CALLDEF(liquid_svm_R_get_cover,2),
  CALLDEF(liquid_svm_R_get_solution,4),
  CALLDEF(liquid_svm_R_write_solution,3),
  CALLDEF(liquid_svm_R_read_solution,2),
  {NULL, NULL, 0}
};

// static R_NativePrimitiveArgType liquid_svm_set_info_mode_t[] = {
//   REALSXP, INTSXP, STRSXP, LGLSXP
// };
// static const R_CMethodDef R_CDef[] = {
//   {"liquid_svm_set_info_mode", (DL_FUNC) &liquid_svm_set_info_mode, 4, myC_t},
//   {NULL, NULL, 0, NULL}
// };

void R_init_liquidSVM(DllInfo *dll){
  R_registerRoutines(dll, NULL, R_CallDef, NULL, NULL);
  R_useDynamicSymbols(dll, (Rboolean)0);
}
} // ends extern "C"

