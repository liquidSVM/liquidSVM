

#include "liquidSVM.h"


// Adapted this from http://stackoverflow.com/questions/1706551/parse-string-into-argv-argc
unsigned makeargs(char *name, char *args, char ***aa) {
    char *buf = (char*)calloc(strlen(args)+1,sizeof(char));
    strcpy(buf, args);
    unsigned c = 2;
    char *delim;
    char **argv = (char**)calloc(c, sizeof (char *));

    argv[0] = name;
    argv[1] = buf;

    while (delim = strchr(argv[c - 1], ' ')) {
        argv = (char**)realloc(argv, (c + 1) * sizeof (char *));
        argv[c] = delim + 1;
        *delim = 0x00;
        c++;
    }

    *aa = argv;

    return c;
}

unsigned getArgs(int cookie, char *name, int stage, char ***aa){
	char *line = (char*) liquid_svm_get_config_line(cookie, stage);
	unsigned ret = makeargs(name, line+1, aa);
	free(line);
	return ret;
}

void freeArgs(int argc, char** argv){
	free(argv[1]); // that was the buf above
	free(argv);
}


int main(int argc_main, char** argv_main){

	double data[] = {8.3,70,8.6,65,8.8,63,10.5,72,10.7,81,10.8,83,11,66,11,75,11.1,80,11.2,75,11.3,79,11.4,76,11.4,76,11.7,69,12,75,12.9,74,12.9,85,13.3,86,13.7,71,13.8,64,14,78,14.2,80,14.5,74,16,72,16.3,77,17.3,81,17.5,82,17.9,80,18,80,18,80,20.6,87};
	double labels[] = {10.3,10.3,10.2,16.4,18.8,19.7,15.6,18.2,22.6,19.9,24.2,21,21.4,21.3,19.1,22.2,33.8,27.4,25.7,24.9,34.5,31.7,36.3,38.3,42.6,55.4,55.7,58.3,51.5,51,77};

	unsigned size = sizeof(labels) / sizeof(double);
	unsigned dim = sizeof(data) / sizeof(labels);

	printf("welcome to commons (size=%d dims=%d)\n", size, dim);

	double test_data[] = {8.3,70,8.6,65,8.8,63,10.5,72,10.7,81,10.8,83,11,66,11,75,11.1,80,11.2,75,11.3,79,11.4,76,11.4,76,11.7,69,12,75,12.9,74,12.9,85,13.3,86,13.7,71,13.8,64,14,78,14.2,80,14.5,74,16,72,16.3,77,17.3,81,17.5,82,17.9,80,18,80,18,80,20.6,87};
	double test_labels[] = {10.3,10.3,10.2,16.4,18.8,19.7,15.6,18.2,22.6,19.9,24.2,21,21.4,21.3,19.1,22.2,33.8,27.4,25.7,24.9,34.5,31.7,36.3,38.3,42.6,55.4,55.7,58.3,51.5,51,77};

	unsigned test_size = sizeof(test_labels) / sizeof(double);

	double* error_ret = NULL;

	char **argv = NULL;
	unsigned argc = 0 ; //sizeof(argv);

	info_mode = INFO_1;

	for(int i=0; i<size; i++){
		if(labels[i]>20)
			labels[i] = 1;
		else
			labels[i] = -1;
	}

	for(int i=0; i<test_size; i++){
		if(test_labels[i]>20)
			test_labels[i] = 1;
		else
			test_labels[i] = -1;
	}


	int cookie = liquid_svm_init(data, size, dim, labels);
	liquid_svm_set_param(cookie, "SCENARIO", "MC");
	liquid_svm_set_param(cookie, "D", "1");
	liquid_svm_set_param(cookie, "THREADS", "1");

	argc = getArgs(cookie, "common", 1, &argv);
	double* train_errs = liquid_svm_train(cookie,argc, argv);
	freeArgs(argc,argv);
	delete[] train_errs;

	argc = getArgs(cookie, "common", 2, &argv);
	double* select_errs = liquid_svm_select(cookie,argc, argv);
	freeArgs(argc,argv);
	delete[] select_errs;

	argc = getArgs(cookie, "common", 3, &argv);
	double* result = liquid_svm_test(cookie, argc, argv, test_data, test_size, dim, test_labels, &error_ret);
	delete[] result;
	delete[] error_ret;
	
	// Doing predict instead of test:
	double* result2 = liquid_svm_test(cookie, argc, argv, test_data, test_size, dim, NULL, &error_ret);
	delete[] result2;
	delete[] error_ret;

	liquid_svm_write_solution(cookie, "test.fsol", 0, NULL);

	liquid_svm_clean(cookie);

	int cookie2 = liquid_svm_read_solution(-1, "test.fsol", NULL, NULL);
	double* result3 = liquid_svm_test(cookie2, argc, argv, test_data, test_size, dim, test_labels, &error_ret);
	delete[] result3;
	delete[] error_ret;
	freeArgs(argc,argv);
	
	liquid_svm_clean(cookie2);

}


