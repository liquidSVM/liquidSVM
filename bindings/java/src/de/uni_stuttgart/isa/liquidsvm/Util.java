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


package de.uni_stuttgart.isa.liquidsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import de.uni_stuttgart.isa.liquidsvm.SVM.LS;
import de.uni_stuttgart.isa.liquidsvm.SVM.MC;
import de.uni_stuttgart.isa.liquidsvm.SVM.QT;

public class Util {
	
	static String sharedLibraryName(){
		String os = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);
		if(os.startsWith("windows")){
			return "liquidsvm.dll";
		}else if(os.startsWith("mac") || os.startsWith("darwin")){
			return "libliquidsvm.jnilib";
		}else{
			return "libliquidsvm.so";
		}
	}
	
	static String LD_LIBRARY_NAME(){
		String os = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);
		if(os.startsWith("windows")){
			return "path=%path%;path\\to\\dir of"+sharedLibraryName()+"\n"
					+ "or copy "+sharedLibraryName()+" to C:\\windows\\system32";
		}else{
			return "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/dir of"+sharedLibraryName();
		}
	}

	
}
