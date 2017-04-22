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
import java.util.List;

import de.uni_stuttgart.isa.liquidsvm.SVM.LS;
import de.uni_stuttgart.isa.liquidsvm.SVM.MC;
import de.uni_stuttgart.isa.liquidsvm.SVM.QT;

/**
 * Contains training and testing features and labels
 * which are usually read from comma separated files
 * with names <em>filePrefix</em>{@code .train.csv} and
 * <em>filePrefix</em>{@code .test.csv}.
 * @author Philipp Thomann
 *
 */
public class LiquidData {
	
	public static class Data {
		public double[][] x;
		public double[] y;
		public int dim;
		public int size;
		public Data(double[][] x, double[] y, int dim, int size) {
			this.x = x;
			this.y = y;
			this.dim = dim;
			this.size = size;
		}
	}
	
	public int dim;
	public int trainSize;
	
	public String filePrefix;
	public String location;
	
	public Data train;
	public double[][] trainX;
	public double[] trainY;
	public Data test;
	public double[][] testX;
	public double[] testY;

	public LiquidData(String filePrefix) throws IOException{
		this(filePrefix, ",");
	}
	public LiquidData(String filePrefix, String delim) throws IOException{
		this.filePrefix = filePrefix;
		
		double[][] train = readTable(filePrefix+".train.csv", delim);
		double[][] test = readTable(filePrefix+".test.csv", delim);
		
		trainX = extractFeatures(train);
		trainY = extractLabels(train);
		testX = extractFeatures(test);
		testY = extractLabels(test);
		
		
		trainSize = train.length;
		
		if(trainSize > 0 && trainX[0] != null)
			dim = trainX[0].length;
		
		this.train = new Data(trainX, trainY, dim, trainSize); 
		this.test = new Data(testX, testY, dim, test.length); 
	}
	
	public static double[][] readTable(String filename) throws IOException{
		return readTable(new FileReader(filename), ",");
	}
	public static double[][] readTable(String filename, String delim) throws IOException{
		return readTable(new FileReader(filename), delim);
	}
	
	public static double[][] readTable(Reader in, String delim) throws IOException{
		BufferedReader bin = new BufferedReader(in);
		List<double[]> ret = new ArrayList<>();
		String line;
		int dim = -1;
		while( (line=bin.readLine()) != null){
			String[] fields = line.split(delim);
			if(dim < 0)
				dim=fields.length;
			else if(dim != fields.length)
				throw new NumberFormatException("Dimensions are not consistent: dim="+dim+"with delim="+delim+" but got "+line);
			double[] x = new double[fields.length];
			for(int i=0; i<x.length; i++)
				x[i] = Double.parseDouble(fields[i].trim());
			ret.add(x);
		}
		System.out.println("Got data: "+ret.size()+"x"+dim);
		return ret.toArray(new double[1][]);
	}
	
	public static double[] extractLabels(double[][] data){
		double[] ret = new double[data.length];
		for(int i=0; i<ret.length; i++)
			ret[i] = data[i][0];
		return ret;
	}
	
	public static double[][] extractFeatures(double[][] data){
		int dim = data[0].length-1;
		double[][] ret = new double[data.length][dim];
		for(int i=0; i<ret.length; i++)
			System.arraycopy(data[i], 1, ret[i], 0, dim);
		return ret;
	}
	
	public static final double[][] trees = {{8.3,10.3},{8.6,10.3},{8.8,10.2},{10.5,16.4},{10.7,18.8},{10.8,19.7},{11,15.6},{11,18.2},{11.1,22.6},{11.2,19.9},{11.3,24.2},{11.4,21},{11.4,21.4},{11.7,21.3},{12,19.1},{12.9,22.2},{12.9,33.8},{13.3,27.4},{13.7,25.7},{13.8,24.9},{14,34.5},{14.2,31.7},{14.5,36.3},{16,38.3},{16.3,42.6},{17.3,55.4},{17.5,55.7},{17.9,58.3},{18,51.5},{18,51},{20.6,77}};
	public static final double[] trees_labs = {70,65,63,72,81,83,66,75,80,75,79,76,76,69,75,74,85,86,71,64,78,80,74,72,77,81,82,80,80,80,87};
	
	public static final double[][] iris = {{5.1,3.5,1.4,0.2},{4.9,3,1.4,0.2},{4.7,3.2,1.3,0.2},{4.6,3.1,1.5,0.2},{5,3.6,1.4,0.2},{5.4,3.9,1.7,0.4},{4.6,3.4,1.4,0.3},{5,3.4,1.5,0.2},{4.4,2.9,1.4,0.2},{4.9,3.1,1.5,0.1},{5.4,3.7,1.5,0.2},{4.8,3.4,1.6,0.2},{4.8,3,1.4,0.1},{4.3,3,1.1,0.1},{5.8,4,1.2,0.2},{5.7,4.4,1.5,0.4},{5.4,3.9,1.3,0.4},{5.1,3.5,1.4,0.3},{5.7,3.8,1.7,0.3},{5.1,3.8,1.5,0.3},{5.4,3.4,1.7,0.2},{5.1,3.7,1.5,0.4},{4.6,3.6,1,0.2},{5.1,3.3,1.7,0.5},{4.8,3.4,1.9,0.2},{5,3,1.6,0.2},{5,3.4,1.6,0.4},{5.2,3.5,1.5,0.2},{5.2,3.4,1.4,0.2},{4.7,3.2,1.6,0.2},{4.8,3.1,1.6,0.2},{5.4,3.4,1.5,0.4},{5.2,4.1,1.5,0.1},{5.5,4.2,1.4,0.2},{4.9,3.1,1.5,0.2},{5,3.2,1.2,0.2},{5.5,3.5,1.3,0.2},{4.9,3.6,1.4,0.1},{4.4,3,1.3,0.2},{5.1,3.4,1.5,0.2},{5,3.5,1.3,0.3},{4.5,2.3,1.3,0.3},{4.4,3.2,1.3,0.2},{5,3.5,1.6,0.6},{5.1,3.8,1.9,0.4},{4.8,3,1.4,0.3},{5.1,3.8,1.6,0.2},{4.6,3.2,1.4,0.2},{5.3,3.7,1.5,0.2},{5,3.3,1.4,0.2},{7,3.2,4.7,1.4},{6.4,3.2,4.5,1.5},{6.9,3.1,4.9,1.5},{5.5,2.3,4,1.3},{6.5,2.8,4.6,1.5},{5.7,2.8,4.5,1.3},{6.3,3.3,4.7,1.6},{4.9,2.4,3.3,1},{6.6,2.9,4.6,1.3},{5.2,2.7,3.9,1.4},{5,2,3.5,1},{5.9,3,4.2,1.5},{6,2.2,4,1},{6.1,2.9,4.7,1.4},{5.6,2.9,3.6,1.3},{6.7,3.1,4.4,1.4},{5.6,3,4.5,1.5},{5.8,2.7,4.1,1},{6.2,2.2,4.5,1.5},{5.6,2.5,3.9,1.1},{5.9,3.2,4.8,1.8},{6.1,2.8,4,1.3},{6.3,2.5,4.9,1.5},{6.1,2.8,4.7,1.2},{6.4,2.9,4.3,1.3},{6.6,3,4.4,1.4},{6.8,2.8,4.8,1.4},{6.7,3,5,1.7},{6,2.9,4.5,1.5},{5.7,2.6,3.5,1},{5.5,2.4,3.8,1.1},{5.5,2.4,3.7,1},{5.8,2.7,3.9,1.2},{6,2.7,5.1,1.6},{5.4,3,4.5,1.5},{6,3.4,4.5,1.6},{6.7,3.1,4.7,1.5},{6.3,2.3,4.4,1.3},{5.6,3,4.1,1.3},{5.5,2.5,4,1.3},{5.5,2.6,4.4,1.2},{6.1,3,4.6,1.4},{5.8,2.6,4,1.2},{5,2.3,3.3,1},{5.6,2.7,4.2,1.3},{5.7,3,4.2,1.2},{5.7,2.9,4.2,1.3},{6.2,2.9,4.3,1.3},{5.1,2.5,3,1.1},{5.7,2.8,4.1,1.3},{6.3,3.3,6,2.5},{5.8,2.7,5.1,1.9},{7.1,3,5.9,2.1},{6.3,2.9,5.6,1.8},{6.5,3,5.8,2.2},{7.6,3,6.6,2.1},{4.9,2.5,4.5,1.7},{7.3,2.9,6.3,1.8},{6.7,2.5,5.8,1.8},{7.2,3.6,6.1,2.5},{6.5,3.2,5.1,2},{6.4,2.7,5.3,1.9},{6.8,3,5.5,2.1},{5.7,2.5,5,2},{5.8,2.8,5.1,2.4},{6.4,3.2,5.3,2.3},{6.5,3,5.5,1.8},{7.7,3.8,6.7,2.2},{7.7,2.6,6.9,2.3},{6,2.2,5,1.5},{6.9,3.2,5.7,2.3},{5.6,2.8,4.9,2},{7.7,2.8,6.7,2},{6.3,2.7,4.9,1.8},{6.7,3.3,5.7,2.1},{7.2,3.2,6,1.8},{6.2,2.8,4.8,1.8},{6.1,3,4.9,1.8},{6.4,2.8,5.6,2.1},{7.2,3,5.8,1.6},{7.4,2.8,6.1,1.9},{7.9,3.8,6.4,2},{6.4,2.8,5.6,2.2},{6.3,2.8,5.1,1.5},{6.1,2.6,5.6,1.4},{7.7,3,6.1,2.3},{6.3,3.4,5.6,2.4},{6.4,3.1,5.5,1.8},{6,3,4.8,1.8},{6.9,3.1,5.4,2.1},{6.7,3.1,5.6,2.4},{6.9,3.1,5.1,2.3},{5.8,2.7,5.1,1.9},{6.8,3.2,5.9,2.3},{6.7,3.3,5.7,2.5},{6.7,3,5.2,2.3},{6.3,2.5,5,1.9},{6.5,3,5.2,2},{6.2,3.4,5.4,2.3},{5.9,3,5.1,1.8}};
	public static final double[] iris_labs = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};
	
	public static void main(String[] args) {
		
	    String[] argv = { "main.cpp", "-T", "3", "-GPU", "0","-r","1","-s","1","0.001","-W","1","-S","2","-f","4","5","-g","10","0.2","5","-l","10","0.001","0.1","-d","1"};
	    
	    
	    double[] data = { 5.1,4.9,4.7,4.6,5,5.4,4.6,5,4.4,4.9,5.4,4.8,4.8,4.3,5.8,5.7,5.4,5.1,5.7,5.1,5.4,5.1,4.6,5.1,4.8,5,5,5.2,5.2,4.7,4.8,5.4,5.2,5.5,4.9,5,5.5,4.9,4.4,5.1,5,4.5,4.4,5,5.1,4.8,5.1,4.6,5.3,5,7,6.4,6.9,5.5,6.5,5.7,6.3,4.9,6.6,5.2,5,5.9,6,6.1,5.6,6.7,5.6,5.8,6.2,5.6,5.9,6.1,6.3,6.1,6.4,6.6,6.8,6.7,6,5.7,5.5,5.5,5.8,6,5.4,6,6.7,6.3,5.6,5.5,5.5,6.1,5.8,5,5.6,5.7,5.7,6.2,5.1,5.7,6.3,5.8,7.1,6.3,6.5,7.6,4.9,7.3,6.7,7.2,6.5,6.4,6.8,5.7,5.8,6.4,6.5,7.7,7.7,6,6.9,5.6,7.7,6.3,6.7,7.2,6.2,6.1,6.4,7.2,7.4,7.9,6.4,6.3,6.1,7.7,6.3,6.4,6,6.9,6.7,6.9,5.8,6.8,6.7,6.7,6.3,6.5,6.2,5.9,3.5,3,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.7,3.4,3,3,4,4.4,3.9,3.5,3.8,3.8,3.4,3.7,3.6,3.3,3.4,3,3.4,3.5,3.4,3.2,3.1,3.4,4.1,4.2,3.1,3.2,3.5,3.6,3,3.4,3.5,2.3,3.2,3.5,3.8,3,3.8,3.2,3.7,3.3,3.2,3.2,3.1,2.3,2.8,2.8,3.3,2.4,2.9,2.7,2,3,2.2,2.9,2.9,3.1,3,2.7,2.2,2.5,3.2,2.8,2.5,2.8,2.9,3,2.8,3,2.9,2.6,2.4,2.4,2.7,2.7,3,3.4,3.1,2.3,3,2.5,2.6,3,2.6,2.3,2.7,3,2.9,2.9,2.5,2.8,3.3,2.7,3,2.9,3,3,2.5,2.9,2.5,3.6,3.2,2.7,3,2.5,2.8,3.2,3,3.8,2.6,2.2,3.2,2.8,2.8,2.7,3.3,3.2,2.8,3,2.8,3,2.8,3.8,2.8,2.8,2.6,3,3.4,3.1,3,3.1,3.1,3.1,2.7,3.2,3.3,3,2.5,3,3.4,3,1.4,1.4,1.3,1.5,1.4,1.7,1.4,1.5,1.4,1.5,1.5,1.6,1.4,1.1,1.2,1.5,1.3,1.4,1.7,1.5,1.7,1.5,1,1.7,1.9,1.6,1.6,1.5,1.4,1.6,1.6,1.5,1.5,1.4,1.5,1.2,1.3,1.4,1.3,1.5,1.3,1.3,1.3,1.6,1.9,1.4,1.6,1.4,1.5,1.4,4.7,4.5,4.9,4,4.6,4.5,4.7,3.3,4.6,3.9,3.5,4.2,4,4.7,3.6,4.4,4.5,4.1,4.5,3.9,4.8,4,4.9,4.7,4.3,4.4,4.8,5,4.5,3.5,3.8,3.7,3.9,5.1,4.5,4.5,4.7,4.4,4.1,4,4.4,4.6,4,3.3,4.2,4.2,4.2,4.3,3,4.1,6,5.1,5.9,5.6,5.8,6.6,4.5,6.3,5.8,6.1,5.1,5.3,5.5,5,5.1,5.3,5.5,6.7,6.9,5,5.7,4.9,6.7,4.9,5.7,6,4.8,4.9,5.6,5.8,6.1,6.4,5.6,5.1,5.6,6.1,5.6,5.5,4.8,5.4,5.6,5.1,5.1,5.9,5.7,5.2,5,5.2,5.4,5.1,0.2,0.2,0.2,0.2,0.2,0.4,0.3,0.2,0.2,0.1,0.2,0.2,0.1,0.1,0.2,0.4,0.4,0.3,0.3,0.3,0.2,0.4,0.2,0.5,0.2,0.2,0.4,0.2,0.2,0.2,0.2,0.4,0.1,0.2,0.2,0.2,0.2,0.1,0.2,0.2,0.3,0.3,0.2,0.6,0.4,0.3,0.2,0.2,0.2,0.2,1.4,1.5,1.5,1.3,1.5,1.3,1.6,1,1.3,1.4,1,1.5,1,1.4,1.3,1.4,1.5,1,1.5,1.1,1.8,1.3,1.5,1.2,1.3,1.4,1.4,1.7,1.5,1,1.1,1,1.2,1.6,1.5,1.6,1.5,1.3,1.3,1.3,1.2,1.4,1.2,1,1.3,1.2,1.3,1.3,1.1,1.3,2.5,1.9,2.1,1.8,2.2,2.1,1.7,1.8,1.8,2.5,2,1.9,2.1,2,2.4,2.3,1.8,2.2,2.3,1.5,2.3,2,2,1.8,2.1,1.8,1.8,1.8,2.1,1.6,1.9,2,2.2,1.5,1.4,2.3,2.4,1.8,1.8,2.1,2.4,2.3,1.9,2.3,2.5,2.3,1.9,2,2.3,1.8 };
	    double[] labels = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 };
	    
	    double[] predictions_ret = new double[4];
//		svm_all(data, labels, new double[]{1,1,1,1}, predictions_ret, argv);
//		for(double d: predictions_ret){
//			System.out.println("predictions_ret: "+d);
//		}
		
//		System.out.println(Arrays.toString(ls_svm(trees, trees_labs, new double[][]{{1,1}})));
		
//		System.out.println(Arrays.toString(mc_svm(iris, iris_labs, new double[][]{{1,2,3,4},{4,5,6,7}} )));
//		System.out.println(Arrays.toString(mc_svm(iris, iris_labs, iris )));
		
		SVM s = null;
		
		if(args.length==0){
			System.out.println("Welcome to a Demonstration of liquidSVM in Java");
			System.out.println("Library was compiled "+SVM.default_params(-1, 1));
		
//		s = new SVM(Solver.LEAST_SQUARE, trees, trees_labs);
		s = new LS(trees,trees_labs);
		System.out.println("Train Errors: "+s.getTrainErrs());
		System.out.println("Select Errors: "+s.getSelectErrs());
		System.out.println("Predict: " + Arrays.toString(s.predict(trees)));
		System.out.println("Test: " + s.test(trees, new double[trees.length], "-d", "1"));
		
		s = new LS(trees,trees_labs, -1, new Config());
		s.train();
		System.out.println("Train Errors: "+s.getTrainErrs());
		s.select();
		System.out.println("Select Errors: "+s.getSelectErrs());
		System.out.println("Predict: " + Arrays.toString(s.predict(trees)));
		System.out.println("Test: " + s.test(trees, new double[trees.length], "-d", "1"));
		
//		s = new SVM(Solver.MULTI_CLASS, iris, iris_labs);
		s = new MC(iris,iris_labs);
		System.out.println("Predict: " + s.test(iris, iris_labs));
		s = new MC(iris,iris_labs, "OvA");
		System.out.println("Predict: " + s.test(iris, iris_labs));
//		System.out.println("Test: " + Arrays.toString(s.test(trees, trees_labs)));
		
		s = new QT(trees,trees_labs);
		System.out.println("Train Errors: "+s.getTrainErrs());
		System.out.println("Select Errors: "+s.getSelectErrs());
		System.out.println("Predict: " + Arrays.toString(s.predict(trees)));
		System.out.println("Test: " + s.test(trees, new double[trees.length], "-d", "1"));
		}else if(System.getProperties().containsKey("liquidsvm.only.train")){
			String file_prefix = args[0];
			try {
				double[][] train = readTable(new FileReader(file_prefix+".train.csv"), ", ");
				String[] args_rest = new String[args.length-1];
				System.arraycopy(args, 1, args_rest, 0, args_rest.length);
				s = new SVM("LS", extractFeatures(train), extractLabels(train), null);
				s.train(args_rest);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}else{
			String file_prefix = args[0];
			try {
				double[][] train = readTable(new FileReader(file_prefix+".train.csv"), ", ");
				double[][] test = readTable(new FileReader(file_prefix+".test.csv"), ", ");
				String[] args_rest = new String[args.length-1];
				System.arraycopy(args, 1, args_rest, 0, args_rest.length);
				s = new SVM("MC", extractFeatures(train), extractLabels(train), null);
				s.train(args_rest);
				s.select();
				System.out.println(s.test(extractFeatures(test), extractLabels(test)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		if(s != null)
			s.clean();

//		SVM.set_param(s.getCookie(), "SCENARIO", "MC");
//		System.out.println(SVM.get_param(s.getCookie(),"SCENARIO"));
//		System.out.println(SVM.get_config_line(s.getCookie(),1));
//		SVM.set_param(s.getCookie(), "SCENARIO", "LS");
//		System.out.println(SVM.get_param(s.getCookie(),"SCENARIO"));
//		System.out.println(SVM.get_config_line(s.getCookie(),1));
//		SVM.set_param(s.getCookie(), "SCENARIO", "MC OvA");
//		System.out.println(SVM.get_param(s.getCookie(),"SCENARIO"));
//		System.out.println(SVM.get_config_line(s.getCookie(),1));
		
//		System.out.println("Cover of iris: "+Arrays.toString(SVM.cover(50,iris)));
	}
}
