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

import java.io.Serializable;
import java.util.Arrays;

/**
 * This class holds a result matrix and an errors matrix.
 *
 */
public class ResultAndErrors implements Serializable{
	/**
	 * The results of an SVM.test
	 */
	public double[][] result = new double[0][];
	/**
	 * The errors of an SVM.test
	 */
	public double[][] errors = new double[0][];
	public ResultAndErrors() {
	}
	public ResultAndErrors(double[][] result, double[][] errors) {
		super();
		this.result = result;
		this.errors = errors;
	}
	@Override
	public String toString() {
		String res = matrixToString(result);
		String err = matrixToString(errors);
		return "[ Result: " + res +", Errors: " +err+" ]";
	}
	
	public static String matrixToString(double[][] arr){
		if(arr==null)
			return "null";
		StringBuilder ret = new StringBuilder();
		ret.append(arr.length).append("x");
		if(arr.length>0)
			ret.append(arr[0].length).append(" ");
		for(double[] row : arr)
			ret.append(Arrays.toString(row)).append(" ");
		return ret.toString();
	}
	
	private void setValues(double[] res, double[] err){
		result = myconvert(res);
		errors = myconvert(err);
	}
	
	public static double[][] myconvert(double[] res) {
		if(res==null)
			return null;
		if(res.length==0 || res[0]==0)
			return new double[0][];
		int n = (int)res[0];
		int m = (int)res[1];
		double[][] ret = new double[n][m];
		for(int i=0; i<n; i++)
			System.arraycopy(res, 2+i*m, ret[i], 0, m);
		return ret;
	}
	
}
