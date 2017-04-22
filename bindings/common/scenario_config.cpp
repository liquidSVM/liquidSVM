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


#ifndef BINDINGS_COMMON_SCENARIO_CONFIG_CPP_
#define BINDINGS_COMMON_SCENARIO_CONFIG_CPP_

#include "./scenario_config.h"
#include "sources/svm/solver/svm_solver_control.h"
#include "sources/shared/training_validation/fold_control.h"
#include "sources/shared/training_validation/cv_control.h"
#include "sources/shared/decision_function/decision_function_manager.h"
#include "sources/shared/basic_types/loss_function.h"
#include "sources/shared/training_validation/working_set_control.h"
#include "sources/shared/basic_functions/flush_print.h"

#include <string>
#include <sstream>
#include <cstdlib>

#define stoi(s) atoi(s.c_str())

template < typename T > std::string to_string( const T& n ){
	std::ostringstream out;
	out << n;
	return out.str();
}

const char* SCENARIO_NAMES_ARR[] = { "MC", "LS", "NPL", "ROC", "QT", "EX", "BS" };

const char* SVM_TYPE_NAMES[] = {"KERNEL_RULE", "SVM_LS_2D", "SVM_HINGE_2D", "SVM_QUANTILE", "SVM_EXPECTILE_2D", "SVM_TEMPLATE", "SOLVER_TYPES_MAX"};
const char* LOSS_TYPE_NAMES[] = {"CLASSIFICATION_LOSS", "MULTI_CLASS_LOSS", "LEAST_SQUARES_LOSS", "WEIGHTED_LEAST_SQUARES_LOSS", "PINBALL_LOSS", "TEMPLATE_LOSS", "LOSS_TYPES_MAX"};
const char* VOTE_SCENARIO_NAMES[] = {"VOTE_CLASSIFICATION", "VOTE_REGRESSION", "VOTE_NPL", "VOTE_SCENARIOS_MAX"};
const char* KERNEL_TYPE_NAMES[] = {"GAUSS_RBF", "POISSON", "HIERARCHICAL_GAUSS", "KERNEL_TYPES_MAX"};
const char* KERNEL_MEMORY_MODEL_NAMES[] = {"LINE_BY_LINE", "BLOCK", "CACHE", "EMPTY", "KERNEL_MEMORY_MODELS_MAX"};
const char* RETRAIN_METHOD_NAMES[] = {"SELECT_ON_ENTIRE_TRAIN_SET", "SELECT_ON_EACH_FOLD", "SELECT_METHODS_MAX"};
const char* FOLDS_KIND_NAMES[] = {"FROM_FILE", "BLOCKS", "ALTERNATING", "RANDOM", "STRATIFIED", "RANDOM_SUBSET", "FOLD_CREATION_TYPES_MAX"};
const char* WS_TYPE_NAMES[] = {"FULL_SET", "MULTI_CLASS_ALL_VS_ALL", "MULTI_CLASS_ONE_VS_ALL", "BOOT_STRAP", "WORKING_SET_SELECTION_TYPES_MAX"};
const char* PARTITION_KIND_NAMES[] = {"NO_PARTITION", "RANDOM_CHUNK_BY_SIZE", "RANDOM_CHUNK_BY_NUMBER", "VORONOI_BY_RADIUS", "VORONOI_BY_SIZE", "OVERLAP_BY_SIZE", "PARTITION_TYPES_MAX"};


map<string, vector<const char*> > ALL_NAMES;
vector<string> SCENARIO_NAMES;

#define TO_VECTOR(x,type) std::vector< type >(x, x+sizeof(x)/sizeof(type))

void init_all_names(){
	if(ALL_NAMES.size() > 0)
		return;
	ALL_NAMES["LOSS_TYPE"] = TO_VECTOR(LOSS_TYPE_NAMES, const char*);
	ALL_NAMES["SVM_TYPE"] = TO_VECTOR(SVM_TYPE_NAMES, const char*);
	ALL_NAMES["VOTE_SCENARIO"] = TO_VECTOR(VOTE_SCENARIO_NAMES, const char*);
	ALL_NAMES["KERNEL"] = TO_VECTOR(KERNEL_TYPE_NAMES, const char*);
	ALL_NAMES["KERNEL_MEMORY_MODEL"] = TO_VECTOR(KERNEL_MEMORY_MODEL_NAMES, const char*);
	ALL_NAMES["RETRAIN_METHOD"] = TO_VECTOR(RETRAIN_METHOD_NAMES, const char*);
	ALL_NAMES["FOLDS_KIND"] = TO_VECTOR(FOLDS_KIND_NAMES, const char*);
	ALL_NAMES["PARTITION_KIND"] = TO_VECTOR(PARTITION_KIND_NAMES, const char*);
	ALL_NAMES["WS_TYPE"] = TO_VECTOR(WS_TYPE_NAMES, const char*);
	SCENARIO_NAMES = std::vector< string >(SCENARIO_NAMES_ARR, SCENARIO_NAMES_ARR+sizeof(SCENARIO_NAMES_ARR)/sizeof(const char*));
}

string parseEnum2(string val, vector<const char*> names){
	// TODO if val has multiple words, only change the first one:
	// "OVERLAP_BY_SIZE 2000 0.5 50000 1" => "5 2000 0.5 50000 1"
//	printf("Trying %s\n",val.c_str());
	int ret = -1;
	const char *valc = val.c_str();
	size_t val_len = val.size();
	for(size_t i=0; i<names.size(); i++){
#ifndef _MSC_VER
		if(strncasecmp(names[i],valc, val_len)==0)
#else
		if(strncmp(names[i],valc, val_len)==0)
#endif
		{
			if(strlen(names[i]) == val_len)
				return to_string(i);
			if(ret==-1)
				ret = i;
			else{
				//error("value is prefix of several labels!");
				//printf("value is prefix of several labels!\n");
				return val;
			}
		}
	}
	if(ret == -1)
		return val;
	return to_string(ret);
}

string parseEnum(string val, string key){
	init_all_names();
	if(ALL_NAMES.find(key)==ALL_NAMES.end())
		return val;
	else
		return parseEnum2(val, ALL_NAMES[key]);
}


Tconfig::Tconfig(){
	VOTE_METHOD = "1";
	clear();
}

void Tconfig::clear(){
	_config.clear();
	set("DISPLAY", 0);
	set("THREADS", 0);
	set("GPUS", 0);
	set("GRID_CHOICE", 0);	// @Ingo: Is this okay as a default?
	set("RANDOM_SEED", 1);

	set("FOLDS_KIND", RANDOM); // from bin/svm-train -f (was 3)
	set("FOLDS", 5); // from bin/svm-train -f
	set("RETRAIN_METHOD", SELECT_ON_EACH_FOLD); // was 1
	set("VOTE_METHOD", 1); // @Ingo rename to weighted_folds?
	set("VOTE_SCENARIO", 1); // @Ingo rename to vote_scenario_regresseion?
	set("VORONOI", NO_PARTITION);

	set("PARTITION_CHOICE", 0);
	set("ADAPTIVITY_CONTROL", 0);

//	set("SCALING_TAU", 0);

//	set("DISPLAY_ROC_STYLE", 0);

#if defined(SSE2__) && defined(AVX__)
	set("COMPILE_INFO", "Compiled with SSE2 and AVX");
#elif defined(SSE2__)
	set("COMPILE_INFO", "Compiled with SSE2 but no AVX");
#elif defined(AVX__)
	set("COMPILE_INFO", "Compiled with no SSE2 but AVX");
#endif

}

void Tconfig::read_from_file(FILE* fpread){
	_config.clear();
	unsigned n=0;
	file_read(fpread, n);
	for(unsigned i=0; i<n; i++){
		string key;
		string value;
		file_read(fpread, key);
		file_read(fpread, value);
		_config[key] = value;
	}
}
void Tconfig::write_to_file(FILE* fpwrite) const{
	file_write(fpwrite, (unsigned)_config.size());
	file_write_eol(fpwrite);
	for(map<string,string>::const_iterator  it = _config.begin(); it != _config.end(); ++it){
		string key = it->first;
		string value = it->second;
		file_write(fpwrite, key);
		file_write(fpwrite, value);
		file_write_eol(fpwrite);
	}
	file_write_eol(fpwrite);
}


void Tconfig::set(const char* name, string value){
	if(getI("CONFIG_DEBUG")>0)
		flush_info("Config: Setting %s = %s\n",name, value.c_str());
	value = parseEnum(value, name);
	_config[name] = value;
	string n = name;
	if(n=="SCENARIO"){
		size_t pos = value.find(" ");
		string scenario = "1";
		string param = "";
		if(pos == string::npos){
			scenario = value;
		}else{
			scenario = value.substr(0,pos);
			param = value.substr(pos+1);
		}
		vector<string>::iterator it = find(SCENARIO_NAMES.begin(), SCENARIO_NAMES.end(),scenario);
		if(it!=SCENARIO_NAMES.end())
			set_scenario(it - SCENARIO_NAMES.begin(), param);
		else
			set_scenario(stoi(scenario), param);
	}else if(n=="D" || n=="d"){
		set("DISPLAY",value);
	}else if(n=="T" || n=="t"){
		set("THREADS",value);
	}else if(n=="GRID_CHOICE"){
		set_grid(stoi(value));
	}else if(n=="ADAPTIVITY_CONTROL"){
		if(value == "1"){
			set("ADAPTIVE_SEARCH", 1);
			set("MAX_LAMBDA_INCREASES", 4);
			set("MAX_NUMBER_OF_WORSE_GAMMAS", 4);
		}else if(value=="2"){
			set("ADAPTIVE_SEARCH", 1);
			set("MAX_LAMBDA_INCREASES", 3);
			set("MAX_NUMBER_OF_WORSE_GAMMAS", 3);
		}else
			set("ADAPTIVE_SEARCH", 0);
	}else if(n=="PARTITION_CHOICE"){
		if(value == "1"){
			set("VORONOI", "1 2000");
		}else if(value=="2"){
			set("VORONOI", "2 10");
		}else if(value=="3"){
			set("VORONOI", "3 1.0 50000");
		}else if(value=="4"){
			set("VORONOI", "4 2000 1 50000");
		}else if(value=="5"){
			set("VORONOI", "5 2000 0.5 50000 1");
		}else if(value=="6"){
			set("VORONOI", "6 2000 1 50000 2.0 20 4");
		}else
			set("VORONOI", NO_PARTITION);
	}
}

void Tconfig::set(const char* name, double value){
	set(name, to_string(value));
}

void Tconfig::set(const char* name, int value){
	set(name, to_string(value));
}

string Tconfig::get(const char* name){
	return _config[name];
}

string Tconfig::get(const char* name, string defaultValue){
	if(has(name))
		return _config[name];
	else
		return defaultValue;
}

int Tconfig::getI(const char* name){
	return stoi(get(name));
}

int Tconfig::getI(const char* name, int defaultValue){
	if(has(name))
		return stoi(get(name));
	else
		return defaultValue;
}

double Tconfig::getD(const char* name){
	return atof(get(name).c_str());
}

double Tconfig::getD(const char* name, double defaultValue){
	if(has(name))
		return atof(get(name).c_str());
	else
		return defaultValue;
}

bool Tconfig::has(const char* name){
	return _config.find(name)!=_config.end();
}

string Tconfig::getPrefixed(const char* name, string prefix){
	if(has(name))
		return prefix + get(name);
	else
		return "";
}

//void Tconfig::set_standard_arguments(string args){
//	if(args.size() == 0)
//		return;
//	size_t pos = 0;
//	pos = args.find(" ", pos);
//	if()
//}


void Tconfig::set_grid(int grid_choice){
	if(grid_choice== -2){
		set("C_VALUES", "0.01 0.1 1 10 100 1000 10000");
		set("GAMMAS", "10.0 5.0 2.0 1.0 0.5 0.25 0.1 0.05");
	}else if(grid_choice== -1){
		set("LAMBDAS", "1.0 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001");
		set("GAMMAS", "10.0 5.0 2.0 1.0 0.5 0.25 0.1 0.05");
	}else{
		if(grid_choice== 0){
			set("MIN_LAMBDA", 0.001);
			set("MAX_LAMBDA", 0.01);
			set("MIN_GAMMA", 0.2);
			set("MAX_GAMMA", 5.0);

			set("LAMBDA_STEPS", 10);
			set("GAMMA_STEPS", 10);
		}else if(grid_choice== 1){
			set("MIN_LAMBDA", 0.0001);
			set("MAX_LAMBDA", 0.01);
			set("MIN_GAMMA", 0.1);
			set("MAX_GAMMA", 10.0);

			set("LAMBDA_STEPS", 15);
			set("GAMMA_STEPS", 15);
		}else{
			set("MIN_LAMBDA", 0.00001);
			set("MAX_LAMBDA", 0.01);
			set("MIN_GAMMA", 0.05);
			set("MAX_GAMMA", 20.0);

			set("LAMBDA_STEPS", 20);
			set("GAMMA_STEPS", 20);
		}
	}
}


void Tconfig::set_scenario(int scenario, string param){
//	printf("Setting scenario %d with param %s\n",scenario, param.c_str());
	switch (scenario) {
		case MC_SVM:
			if(param == "" || param=="0" || param.substr(0,5)=="AvA_h" || param=="AvA"){
				set("WS_TYPE", MULTI_CLASS_ALL_VS_ALL);
				set("SVM_TYPE", SVM_HINGE_2D);
				set("VOTE_SCENARIO", VOTE_CLASSIFICATION);
				set("LOSS_TYPE", CLASSIFICATION_LOSS);
			}else if(param=="1" || param.substr(0,5)=="OvA_l" || param=="OvA"){
				set("WS_TYPE", MULTI_CLASS_ONE_VS_ALL);
				set("SVM_TYPE", SVM_LS_2D);
				set("VOTE_SCENARIO", VOTE_REGRESSION);
				set("LOSS_TYPE", LEAST_SQUARES_LOSS);
			}else if(param=="2" || param.substr(0,5)=="OvA_h"){
				set("WS_TYPE", MULTI_CLASS_ONE_VS_ALL);
				set("SVM_TYPE", SVM_HINGE_2D);
				set("VOTE_SCENARIO", VOTE_REGRESSION);
//				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
				set("LOSS_TYPE", CLASSIFICATION_LOSS);
			}else if(param=="3" || param.substr(0,5)=="AvA_l"){
				set("WS_TYPE", MULTI_CLASS_ALL_VS_ALL);
				set("SVM_TYPE", SVM_LS_2D);
				set("VOTE_SCENARIO", VOTE_REGRESSION);
//				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
				set("LOSS_TYPE", LEAST_SQUARES_LOSS);
			}
			set("FOLDS_KIND", STRATIFIED);
			set("CLIPPING", 1.0);
			break;
		case LS_SVM:
			set("CLIPPING",  -1.0);
			if(param.size()>0)
				set("CLIPPING",  param);

			set("FOLDS_KIND", RANDOM);
			set("SVM_TYPE",  SVM_LS_2D);
			set("VOTE_SCENARIO", VOTE_REGRESSION);
//			set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
			set("LOSS_TYPE", LEAST_SQUARES_LOSS);

			break;
		case NPL_SVM:
			set("VORONOI", 0);
			set("ADAPTIVE_SEARCH", 0);

			{
			string npl_class = "1";
			string npl_constraint = "0.05";

			if(param.size()>0){
				size_t pos = param.find(" ");
				if(pos == string::npos)
					npl_class = param;
				else{
					npl_class = param.substr(0,pos);
					npl_constraint = param.substr(pos+1);
				}
			}

			set("NPL_CLASS", npl_class);
			set("NPL_CONSTRAINT", npl_constraint);

			set("FOLDS_KIND", STRATIFIED);
			set("SVM_TYPE", SVM_HINGE_2D);
			set("LOSS_TYPE", CLASSIFICATION_LOSS);
			set("VOTE_SCENARIO", VOTE_NPL);
//			set("VOTE_TYPE",  VOTE_METHOD + " " + to_string(VOTE_NPL) + " " +npl_class);
			set("CLIPPING", 1);


			set("WEIGHT_STEPS", 10);
			set("MIN_WEIGHT", 0.001);
			set("MAX_WEIGHT", 0.5);
			set("GEO_WEIGHTS", 1);
			//set("NPL_CONSTRAINT_BASE", npl_constraint); // this seems to have no effect

			if(npl_class == "1")
				set("NPL_SWAP", 1);
			else
				set("NPL_SWAP", 0);
			}

			break;
		case ROC_SVM:

			set("FOLDS_KIND", STRATIFIED);
			set("SVM_TYPE", SVM_HINGE_2D);
			set("CLIPPING", 1.0);
			set("VOTE_SCENARIO", VOTE_CLASSIFICATION);
//			set("VOTE_TYPE",  VOTE_METHOD + " " + to_string(VOTE_CLASSIFICATION));
			set("LOSS_TYPE", CLASSIFICATION_LOSS);
			set("DISPLAY_ROC_STYLE", 1);


			set("WEIGHT_STEPS", 9);
			set("MAX_WEIGHT", 0.9);
			set("MIN_WEIGHT", 0.1);
			set("GEO_WEIGHTS", 0);

			break;
		case QT_SVM:
			set("CLIPPING",  -1.0);
			if(param.size()>0)
				set("CLIPPING",  param);

			set("FOLDS_KIND", RANDOM);
			set("SVM_TYPE",  SVM_QUANTILE);
			set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
			set("LOSS_TYPE", PINBALL_LOSS);
			set("WEIGHT_STEPS",5);
			set("WEIGHTS","0.05 0.1 0.5 0.9 0.95");
			break;
		case EX_SVM:
			set("CLIPPING",  -1.0);
			if(param.size()>0)
				set("CLIPPING",  param);

			set("FOLDS_KIND", RANDOM);
			set("SVM_TYPE",  SVM_EXPECTILE_2D);
			set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
			set("LOSS_TYPE", WEIGHTED_LEAST_SQUARES_LOSS);
			set("WEIGHT_STEPS",5);
			set("WEIGHTS","0.05 0.1 0.5 0.9 0.95");
			break;
		case BS_SVM:
			{
			string svm_type = to_string(SVM_HINGE_2D);
			string boot_strap = "5 500";

			if(param.size()>0){
				size_t pos = param.find(" ");
				if(pos == string::npos)
					svm_type = param;
				else{
					svm_type = param.substr(0,pos);
					boot_strap = param.substr(pos+1);
				}
			}

			set("SVM_TYPE", svm_type);
			set("BOOT_STRAP", boot_strap);

			int type = stoi(get("SVM_TYPE"));
			if(type==KERNEL_RULE){
				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_CLASSIFICATION));
				set("LOSS_TYPE", CLASSIFICATION_LOSS);
				set("FOLDS_KIND", STRATIFIED);
			}else if(type==SVM_LS_2D){
				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
				set("LOSS_TYPE", LEAST_SQUARES_LOSS);
				set("FOLDS_KIND", RANDOM);
			}else if(type==SVM_HINGE_2D){
				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_CLASSIFICATION));
				set("LOSS_TYPE", CLASSIFICATION_LOSS);
				set("CLIPPING", 1);
				set("FOLDS_KIND", STRATIFIED);
			}else if(type==SVM_QUANTILE){
				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
				set("LOSS_TYPE", PINBALL_LOSS);
				set("FOLDS_KIND", RANDOM);
			}else if(type==SVM_EXPECTILE_2D){
				set("VOTE_TYPE", VOTE_METHOD + " " + to_string(VOTE_REGRESSION));
				set("LOSS_TYPE", WEIGHTED_LEAST_SQUARES_LOSS);
				set("FOLDS_KIND", RANDOM);
			}
			}

			break;
		default:
			break;
	}
}

const string Tconfig::config_line(int stage){
	string SVM_GLOBAL = ""; // would be " -T 0 -GPU 0 -d 1"
	SVM_GLOBAL += getPrefixed("DISPLAY", " -d ");
	SVM_GLOBAL += getPrefixed("THREADS", " -T ");
	SVM_GLOBAL += getPrefixed("GPUS", " -GPU ");
	if(stage == 0){
		// TODO delete this in future
		return "-g 10 .2 5 -l 10 .001 .01 -a 0 3 3";
	}else if(stage == 1){
		string ret = "";
		ret += getPrefixed("RANDOM_SEED", " -r ");
		ret += getPrefixed("CLIPPING", " -s "); // should be " -s $CLIPPING 0.001" but is default anyway
		ret += getPrefixed("WS_TYPE", " -W ");
		ret += getPrefixed("INIT", " -i ");
		ret += getPrefixed("SVM_TYPE", " -S ");
		ret += getPrefixed("BOOT_STRAP", " -W "+to_string(BOOT_STRAP)+" ");
		ret += getPrefixed("RANDOM_CHUNK_SIZE", " -P "+to_string(RANDOM_CHUNK_BY_SIZE)+" ");
		ret += getPrefixed("RANDOM_CHUNK_NUMBER", " -P "+to_string(RANDOM_CHUNK_BY_NUMBER)+" ");
		ret += getPrefixed("VORONOI", " -P ");
		if(has("FOLDS_KIND")) // in the scripts FOLDS="BLALBLA 5" and number cannot be changed
			ret += " -f " + get("FOLDS_KIND") + getPrefixed("FOLDS", " ");
		if(has("GAMMAS")){
			ret +=  " -g [ " + get("GAMMAS") + " ]";
		}else if(has("GAMMA_STEPS")){
				ret += " -g " + get("GAMMA_STEPS");
				if(has("MIN_GAMMA"))
					ret += " " + get("MIN_GAMMA") + getPrefixed("MAX_GAMMA", " ");
		}else
			ret += " -g 10 0.2 5.0";

		if(has("LAMBDAS")){
			ret +=  " -l [ " + get("LAMBDAS") + " ] 0";
		}else if(has("C_VALUES")){
			ret +=  " -l [ " + get("C_VALUES") + " ] 1";
		}else if(has("LAMBDA_STEPS")){
				ret += " -l " + get("LAMBDA_STEPS");
				if(has("MIN_LAMBDA"))
					ret += " " + get("MIN_LAMBDA") + getPrefixed("MAX_LAMBDA", " ");
		}else
			ret += " -l 10 0.001 0.01";

		if(has("ADAPTIVE_SEARCH")){
				ret += " -a " + get("ADAPTIVE_SEARCH");
				if(has("MAX_LAMBDA_INCREASES"))
					ret += " " +get("MAX_LAMBDA_INCREASES") + getPrefixed("MAX_NUMBER_OF_WORSE_GAMMAS", " ");
		}
		if(has("WEIGHTS"))
			ret += " -w [ " + get("WEIGHTS") + " ]";
		else if(has("MIN_WEIGHT") || has("MAX_WEIGHT") || has("WEIGHT_STEPS")){
			ret += " -w " + get("MIN_WEIGHT", "0.01");
			if(has("MAX_WEIGHT") || has("WEIGHT_STEPS")){
				ret += " " + get("MAX_WEIGHT", "5.0");
				if(has("WEIGHT_STEPS")){
					ret += + " " + get("WEIGHT_STEPS");
					if(has("GEO_WEIGHTS"))
						ret += " " + get("GEO_WEIGHTS") + getPrefixed("NPL_SWAP", " ");
				}
			}
		}
//		ret += kernel_opt;
		ret += getPrefixed("LOSS_TYPE", " -L "); // should be " -L $LOSS -1.0" but is default anyway

		ret += getPrefixed("KERNEL"," -k ");
		ret += getPrefixed("SOLVER_INIT"," -i ");

		ret += SVM_GLOBAL;
		return ret;
	}else if(stage == 2){
		string ret = "";
		ret += getPrefixed("RETRAIN_METHOD", " -R ");
		if(has("NPL_CLASS")){
			ret += " -N " + get("NPL_CLASS") + getPrefixed("NPL_CONSTRAINT", " ");
		}
		ret += getPrefixed("WEIGHT_NUMBER", " -W ");
		ret += getPrefixed("DISPLAY", " -d ");
		return ret;
	}else if(stage == 3){
		string ret = "";
		ret += getPrefixed("LOSS_TYPE", " -L ");
		ret += getPrefixed("DISPLAY_ROC_STYLE", " -o ");

		if(has("VOTE_TYPE"))
			ret += getPrefixed("VOTE_TYPE", " -v ");
		else if(has("VOTE_METHOD")){
				ret += " -v " + get("VOTE_METHOD");
				if(has("VOTE_SCENARIO"))
					ret += " " + get("VOTE_SCENARIO") + getPrefixed("NPL_CLASS", " ");
		}
		ret += SVM_GLOBAL;
		return ret;
	}
	return "";
}

#ifdef COMPILE_SCENARIO_CONFIG_MAIN__
int main(int argc, char** argv){
	Tconfig config;

	printf("MC_SVM\n");
	config.set("SCENARIO", "MC 0 0");
	config.set("KERNEL","POISSON");
	config.set_grid(-1);
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("MC_SVM\n");
	config.set_scenario(MC_SVM, "0 0");
	config.set_grid(-1);
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("LS_SVM\n");
	config.set_scenario(LS_SVM, "");
	config.set_grid(-1);
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("NPL_SVM\n");
	config.set_scenario(NPL_SVM, "");
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("QT_SVM\n");
	config.set_scenario(QT_SVM, "");
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("EX_SVM\n");
	config.set_scenario(EX_SVM, "");
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("BS_SVM\n");
	config.set_scenario(BS_SVM, "");
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();

	printf("ROC_SVM\n");
	config.set_scenario(ROC_SVM, "");
	printf("Commandline  train: %s\n", config.config_line(1).c_str());
	printf("Commandline select: %s\n", config.config_line(2).c_str());
	printf("Commandline   test: %s\n", config.config_line(3).c_str());
	config.clear();
};
#endif


static map<int,Tconfig*> cookies_config;
Tconfig* getConfig(int cookie){
	if(cookies_config.find(cookie) == cookies_config.end()){
		cookies_config[cookie] = new Tconfig();
	}
	return cookies_config[cookie];
};
void deleteConfig(int cookie){
	if(cookies_config.find(cookie) != cookies_config.end()){
		delete cookies_config[cookie];
		cookies_config.erase(cookie);
	}
}

extern "C" void liquid_svm_set_scenario(int cookie, int scenario, string param){
	getConfig(cookie)->set_scenario(scenario, param);
}

extern "C" void liquid_svm_set_param(int cookie, const char* name, const char* value){
	getConfig(cookie)->set(name, value);
}

extern "C" char* liquid_svm_get_param(int cookie, const char* name){
	std::string ret = getConfig(cookie)->get(name);
	const char* value = ret.c_str();
	char *buf = (char*)calloc(strlen(value)+1,sizeof(char));
	strcpy(buf, value);
	return buf;
}

extern "C" char* liquid_svm_get_config_line(int cookie, int stage){
	const std::string ret = getConfig(cookie)->config_line(stage);
	const char* value = ret.c_str();
	char *buf = (char*)calloc(strlen(value)+1,sizeof(char));
	strcpy(buf, value);
	return buf;
}



#endif
