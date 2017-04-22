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

#ifndef BINDINGS_COMMON_SCENARIO_CONFIG_H_
#define BINDINGS_COMMON_SCENARIO_CONFIG_H_

//#include "liquidSVM.h"

#include <string>
#include <map>
#include <vector>

using namespace std;

enum SCENARIO { MC_SVM, LS_SVM, NPL_SVM, ROC_SVM, QT_SVM, EX_SVM, BS_SVM, SCENARIO_MAX };

class Tconfig {
	public:

		Tconfig();

		void read_from_file(FILE* fpread);
		void write_to_file(FILE* fpwrite) const;

		string VOTE_METHOD;

		void set_scenario(int scenario, string param);
		void set_grid(int grid);

		void set(const char* name, string value);

		void set(const char* name, double value);

		void set(const char* name, int value);

		string get(const char* name);
		string get(const char* name, string defaultValue);

		int getI(const char* name);
		double getD(const char* name);

		int getI(const char* name, int defaultValue);
		double getD(const char* name, double defaultValue);

		bool has(const char* name);

		string getPrefixed(const char* name, string prefix);

		void clear();

		const string config_line(int stage);

	private:
		map<string, string> _config;
};


extern "C" {
void liquid_svm_set_param(int cookie, const char* name, const char* value);
char* liquid_svm_get_param(int cookie, const char* name);
char* liquid_svm_get_config_line(int cookie, int stage);
}


#ifndef COMPILE_SEPERATELY__
  #include "./scenario_config.cpp"
#endif


#endif /* BINDINGS_COMMON_SCENARIO_CONFIG_H_ */
