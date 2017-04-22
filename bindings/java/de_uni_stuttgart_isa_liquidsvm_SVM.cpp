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


#include <jni.h>

#include "de_uni_stuttgart_isa_liquidsvm_SVM.h"

#define VPRINTF(message_format) va_list arguments; \
va_start(arguments, message_format);               \
vprintf(message_format, arguments);                \
va_end(arguments);                                 \
fflush(stdout);


// liquidSVM uses atof to parse numbers in command-line.
// Usually if a C-Program is run, then LC_ALL is set to "C"
// On the other hand JNI seems to not reset this and hence we have to:
#include <locale.h>

#define BEGIN_LOCALE 	char* old_locale = setlocale(LC_ALL, NULL);\
	setlocale(LC_ALL, "C");

#define END_LOCALE setlocale(LC_ALL, old_locale);


#include "common/liquidSVM.h"
#include "common/scenario_config.h"
#include <stdio.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif


jdoubleArray prepareMatrix(JNIEnv *env, double *arr){
    if(arr == NULL){
      return NULL;
    }
    int k=0;
    int rows = (int)arr[k++];
//    printf("\nSVM-test got: rows=%d\n",rows);
    if(rows==0)
      return NULL;
    int cols = arr[k++];
//    printf("SVM-test got: cols=%d\n",cols);
    jdoubleArray ret = env->NewDoubleArray(rows * cols+2);
    env->SetDoubleArrayRegion(ret, 0, rows*cols+2, arr);
    return ret;
}



/*
  * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_cover_dataset
 * Signature: (II[D)[I
 */
JNIEXPORT jintArray JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1cover_1dataset
  (JNIEnv *env, jclass clzz, jint NNs, jint dim, jdoubleArray data){
	jsize size = env->GetArrayLength(data) / dim;

	jdouble *data_arr = env->GetDoubleArrayElements(data, 0);

	Tdataset dataset = Tdataset(data_arr, size, dim, NULL);
	env->ReleaseDoubleArrayElements(data, data_arr, JNI_ABORT);

	vector <double> radii;
	unsigned RANDOM_SEED = 1;
	Tsubset_info sub = dataset.create_cover_subset_info_by_kNN(NNs, RANDOM_SEED, true, radii, 0);
	printf("\nsub.size %d\n",sub.size());
    jintArray ret = env->NewIntArray(sub.size());
    env->SetIntArrayRegion(ret, 0, sub.size(), (jint*) &sub[0]);

    return ret;
}


/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    default_params
 * Signature: (II)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_default_1params
  (JNIEnv *env, jclass clzz, jint stage, jint solver){

	return env->NewStringUTF(liquid_svm_default_params(stage, solver));
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_init
 * Signature: ([D[D[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1init
	(JNIEnv * env, jclass clzz, jdoubleArray data, jdoubleArray labs){
		jsize size = env->GetArrayLength(labs);
		jsize dim = env->GetArrayLength(data) / size;
		
		jdouble *data_arr = env->GetDoubleArrayElements(data, 0);
		jdouble *labs_arr = env->GetDoubleArrayElements(labs, 0);
		
		
		int cookie = liquid_svm_init(data_arr, size, dim, labs_arr);
		
		
//		printf("C++: releasing elements\n");
		env->ReleaseDoubleArrayElements(data, data_arr, JNI_ABORT);
		env->ReleaseDoubleArrayElements(labs, labs_arr, JNI_ABORT);
		
//		printf("C++: released\n");
		
		return cookie;
	}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_train
 * Signature: (I[Ljava/lang/String;)V
 */
JNIEXPORT jdoubleArray JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1train
  (JNIEnv * env, jclass clzz, jint cookie, jobjectArray argv){
  	jsize argc = env->GetArrayLength(argv);
    char** argv_arr = new char*[argc];
	jstring *argv_strings = new jstring[argc];
	env->EnsureLocalCapacity(argc);
	for(int i=0; i<argc; i++){
		argv_strings[i] = (jstring) env->GetObjectArrayElement(argv, i);
		argv_arr[i] = (char*) env->GetStringUTFChars(argv_strings[i], 0);
	}

	BEGIN_LOCALE
  	double *train_error = liquid_svm_train(cookie, argc, argv_arr);
		// TODO get the train error to Java
    jdoubleArray ret = prepareMatrix(env, train_error);
	delete train_error;
	END_LOCALE
		
    for(int i=0; i<argc; i++){
		env->ReleaseStringUTFChars(argv_strings[i], 0);
		//??? env->ReleaseObjectArrayElement(argv, i);
	}
	delete argv_arr;
	delete argv_strings;

	return ret;
}


/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_select
 * Signature: (I[Ljava/lang/String;)I
 */
JNIEXPORT jdoubleArray JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1select
  (JNIEnv * env, jclass clzz, jint cookie, jobjectArray argv){
    jsize argc = env->GetArrayLength(argv);
    char** argv_arr = new char*[argc];
    jstring *argv_strings = new jstring[argc];
    env->EnsureLocalCapacity(argc);
    for(int i=0; i<argc; i++){
    	argv_strings[i] = (jstring) env->GetObjectArrayElement(argv, i);
    	argv_arr[i] = (char*) env->GetStringUTFChars(argv_strings[i], 0);
    }

    BEGIN_LOCALE
	double* select_error = liquid_svm_select(cookie, argc, argv_arr);
    // TODO get the select error to Java
    jdoubleArray ret = prepareMatrix(env, select_error);
    delete select_error;
    END_LOCALE

	for(int i=0; i<argc; i++){
		env->ReleaseStringUTFChars(argv_strings[i], 0);
		//??? env->ReleaseObjectArrayElement(argv, i);
	}
    delete argv_arr;
    delete argv_strings;

    return ret;
}

void setResultAndError(JNIEnv *env, jobject resAndErr, double *res, double *err){
  	jclass cls = env->GetObjectClass(resAndErr);
    jmethodID mid = env->GetMethodID(cls, "setValues", "([D[D)V");
    if(mid == NULL){
    	printf("mid is null");
    	return;
    }
    jdoubleArray resJ = prepareMatrix(env,res);
    jdoubleArray errJ = prepareMatrix(env,err);
    env->CallVoidMethod(resAndErr, mid, resJ, errJ);
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_test
 * Signature: (I[Ljava/lang/String;[D[DLde/uni_stuttgart/isa/liquidsvm/ResultAndErrors;)Lde/uni_stuttgart/isa/liquidsvm/ResultAndErrors;
 */
JNIEXPORT jobject JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1test
(JNIEnv * env, jclass clzz, jint cookie, jobjectArray argv, jint test_size, jdoubleArray test_data, jdoubleArray labels, jobject resAndErr){
	jsize dim = env->GetArrayLength(test_data) / test_size;

	jdouble *test_arr = env->GetDoubleArrayElements(test_data, 0);
	jdouble *labels_arr = NULL;
	if(labels != NULL && !env->IsSameObject(labels, NULL) && env->GetArrayLength(labels) == test_size)
		labels_arr = env->GetDoubleArrayElements(labels, 0);

	jsize argc = env->GetArrayLength(argv);
	char** argv_arr = new char*[argc];
	jstring *argv_strings = new jstring[argc];
	env->EnsureLocalCapacity(argc);
	for(int i=0; i<argc; i++){
		argv_strings[i] = (jstring) env->GetObjectArrayElement(argv, i);
		argv_arr[i] = (char*) env->GetStringUTFChars(argv_strings[i], 0);
	}

	double *error_dummy_array = NULL;
	double **error_ret = &error_dummy_array;

	BEGIN_LOCALE
	double* predictions = liquid_svm_test(cookie, argc, argv_arr, test_arr, test_size, dim, labels_arr, error_ret);
	END_LOCALE
    
	setResultAndError(env, resAndErr, predictions, *error_ret);

    delete *error_ret;
    delete predictions;

    for(int i=0; i<argc; i++){
		env->ReleaseStringUTFChars(argv_strings[i], 0);
		//??? env->ReleaseObjectArrayElement(argv, i);
	}
	delete argv_arr;
	delete argv_strings;

	env->ReleaseDoubleArrayElements(test_data, test_arr, JNI_ABORT);
	if(labels_arr != NULL)
		env->ReleaseDoubleArrayElements(labels, labels_arr, 0);

    return resAndErr;
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_clean
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1clean
  (JNIEnv * env, jclass clzz, jint cookie){
	  liquid_svm_clean(cookie);
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    set_param
 * Signature: (ILjava/lang/String;Ljava/lang/String;)V
 void liquid_svm_set_param(int cookie, const char* name, const char* value);
 const char* liquid_svm_get_param(int cookie, const char* name);
 const char* liquid_svm_get_config_line(int cookie, int stage);
 */
JNIEXPORT void JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_set_1param
  (JNIEnv *env, jclass clzz, jint cookie, jstring name, jstring value){
	char *name_C = (char*) env->GetStringUTFChars(name, 0);
	char *value_C = (char*) env->GetStringUTFChars(value, 0);

	BEGIN_LOCALE
	liquid_svm_set_param(cookie, name_C, value_C);
	END_LOCALE

	env->ReleaseStringUTFChars(name, 0);
	env->ReleaseStringUTFChars(value, 0);
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    get_param
 * Signature: (ILjava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_get_1param
  (JNIEnv *env, jclass clzz, jint cookie, jstring name){
	char *name_C = (char*) env->GetStringUTFChars(name, 0);
	BEGIN_LOCALE
	char* ret_C = liquid_svm_get_param(cookie, name_C);
	END_LOCALE

	env->ReleaseStringUTFChars(name, 0);
	jstring ret = env->NewStringUTF(ret_C);
	free(ret_C);
	return ret;
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    get_config_line
 * Signature: (II)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_get_1config_1line
  (JNIEnv *env, jclass clzz, jint cookie, jint stage){
	BEGIN_LOCALE
	char* ret_C = liquid_svm_get_config_line(cookie, stage);
	END_LOCALE

	jstring ret = env->NewStringUTF(ret_C);
	free(ret_C);
	return ret;
}



/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_get_cover
 * Signature: (II)[I
 */
JNIEXPORT jintArray JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1get_1cover
  (JNIEnv *env, jclass clzz, jint cookie, jint task){

	Tsubset_info info;
	BEGIN_LOCALE
	info = liquid_svm_get_cover(cookie, task);
	END_LOCALE

    jintArray ret = env->NewIntArray(info.size());
	jint *data_arr = env->GetIntArrayElements(ret, 0);
	for(size_t i=0; i<info.size(); i++)
		data_arr[i] = info[i];
	env->ReleaseIntArrayElements(ret, data_arr, 0);

    return ret;
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_get_solution_svs
 * Signature: (IIII)[I
 */
JNIEXPORT jintArray JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1get_1solution_1svs
  (JNIEnv *env, jclass clzz, jint cookie, jint task, jint cell, jint fold){

	BEGIN_LOCALE
	Tsvm_decision_function df = liquid_svm_get_solution(cookie, task, cell, fold);
	END_LOCALE

    jintArray ret = env->NewIntArray(df.sample_number.size());
	jint *data_arr = env->GetIntArrayElements(ret, 0);
	for(size_t i=0; i<df.sample_number.size(); i++)
		data_arr[i] = df.sample_number[i];
	env->ReleaseIntArrayElements(ret, data_arr, 0);

    return ret;
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_get_solution_coeffs
 * Signature: (IIII)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1get_1solution_1coeffs
  (JNIEnv *env, jclass clzz, jint cookie, jint task, jint cell, jint fold){

	BEGIN_LOCALE
	Tsvm_decision_function df = liquid_svm_get_solution(cookie, task, cell, fold);
	END_LOCALE

    jdoubleArray ret = env->NewDoubleArray(df.coefficient.size());
	jdouble *data_arr = env->GetDoubleArrayElements(ret, 0);
	for(size_t i=0; i<df.coefficient.size(); i++)
		data_arr[i] = df.coefficient[i];
	env->ReleaseDoubleArrayElements(ret, data_arr, 0);

    return ret;
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_write_solution
 * Signature: (ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1write_1solution
  (JNIEnv *env, jclass clzz, jint cookie, jstring filename){

	const char* filenameC = (const char*) env->GetStringUTFChars(filename, 0);
	liquid_svm_write_solution(cookie, filenameC, 0, NULL);
	env->ReleaseStringUTFChars(filename, 0);
}

/*
 * Class:     de_uni_stuttgart_isa_liquidsvm_SVM
 * Method:    svm_read_solution
 * Signature: (ILjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_de_uni_1stuttgart_isa_liquidsvm_SVM_svm_1read_1solution
  (JNIEnv *env, jclass clzz, jint cookie, jstring filename){

	const char* filenameC = (const char*) env->GetStringUTFChars(filename, 0);
	cookie = liquid_svm_read_solution(cookie, filenameC, NULL, NULL);
	env->ReleaseStringUTFChars(filename, 0);
	return cookie;
}




#ifdef __cplusplus
}
#endif

