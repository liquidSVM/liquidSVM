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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.Map;
import java.util.Properties;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

/**
 * liquidSVM SVM base class.
 * After initializing using training data
 * and configuring, at some point training has to be performed
 * and selection of the best hyperparameters.
 * Train and select will be done automagically in most cases.
 * 
 * Usually users should use one of the learning scenario subclasses.
 * 
 * @author Philipp Thomann
 *
 */
public class SVM implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	// liquidSVM Thread-Handling is not thread-safe therefore we need
	private static final Object LOCK = new Integer(0);

	public static boolean ENABLE_LOAD_WITHOUT_LIBRARY = false;
	
	private transient int cookie;
	
	private boolean trained = false;
	private boolean selected = false;
	
	private int errors_count;
	
	private double[][] train_errs;
	private double[][] select_errs;
	
	private ResultAndErrors lastResult;

	/**
	 * Returns many information upon {@link #train}ing.
	 * For every point in the hyperparametergrid, every fold,
	 * every cell, and every task there is one line.
	 * 
	 * @return matrix with errors and more information after training.
	 */
	public double[][] getTrainErrs() {
		return train_errs;
	}

	/**
	 * Returns many information upon {@link #select}ing.
	 * For every cell, and every task there is one line.
	 * 
	 * @return matrix with errors and more information after selecting.
	 */
	public double[][] getSelectErrs() {
		return select_errs;
	}

	/**
	 * Keeps the last result and errors around after {@link #test}ing.
	 * 
	 * @return the last result and errors.
	 */
	public ResultAndErrors getLastResult() {
		return lastResult;
	}
	
	
	
	// Make it package private for the moment:
	int getCookie() {
		return cookie;
	}


	/**
	 * For quick demonstration initialize a least squares SVM using the trees dataset.
	 * 
	 * @see LiquidData#trees
	 */
	public SVM(){
		this("LS", LiquidData.trees, LiquidData.trees_labs, null);
	}
	
	private SVM(int cookie){
		this.cookie = cookie;
	}

	/**
	 * Initializes an SVM with training data and optionally configuration options and a scenario.
	 * 
	 * Most users should use rather
	 * {@link LS}, {@link MC}, {@link QT}, {@link EX}, {@link NPL}, or {@link ROC}.
	 * 
	 * After the training data and the configuration are set,
	 * {@link Config#train(boolean)} and {@link Config#select(boolean)}
	 * determine whether {@link #train(String...)} and {@link #select(String...)} should be started.
	 * If {@code config==NULL} they are performed.
	 * 
	 * @param scenario the learning scenario as in the inner classes or {@code NULL}
	 * @param data array of training samples features (X)
	 * @param labels the training labels (Y)
	 * @param config configuration of the SVM. Can be {@code NULL}.
	 * @see LS
	 * @see MC
	 * @see QT
	 * @see EX
	 * @see NPL
	 * @see ROC
	 */
	public SVM(String scenario, double[][] data, double[] labels, Config config){
		this(scenario, data, labels, config, false);
	}
	
	/**
	 * Initializes an SVM with training data and optionally configuration options and a scenario.
	 * 
	 * Most users should use rather
	 * {@link LS}, {@link MC}, {@link QT}, {@link EX}, {@link NPL}, or {@link ROC}.
	 * 
	 * After the training data and the configuration are set,
	 * {@link Config#train(boolean)} and {@link Config#select(boolean)}
	 * determine whether {@link #train(String...)} and {@link #select(String...)} should be started.
	 * If {@code config==NULL} they are performed.
	 * 
	 * @param scenario the learning scenario as in the inner classes or {@code NULL}
	 * @param data the training data.
	 * @param config configuration of the SVM. Can be {@code NULL}.
	 * @see LS
	 * @see MC
	 * @see QT
	 * @see EX
	 * @see NPL
	 * @see ROC
	 */
	public SVM(String scenario, LiquidData.Data data, Config config){
		this(scenario, data.x, data.y, config, false);
	}
	
	/**
	 * Initializes an SVM with training data and optionally configuration options and a scenario
	 * and performs testing.
	 * 
	 * Most users should use rather
	 * {@link LS}, {@link MC}, {@link QT}, {@link EX}, {@link NPL}, or {@link ROC}.
	 * 
	 * After the training data and the configuration are set,
	 * {@link Config#train(boolean)} and {@link Config#select(boolean)}
	 * determine whether {@link #train(String...)} and {@link #select(String...)} should be started.
	 * If {@code config==NULL} they are performed.
	 * 
	 * @param scenario the learning scenario as in the inner classes or {@code NULL}
	 * @param data the training and test data.
	 * @param config configuration of the SVM. Can be {@code NULL}.
	 * @see LS
	 * @see MC
	 * @see QT
	 * @see EX
	 * @see NPL
	 * @see ROC
	 */
	public SVM(String scenario, LiquidData data, Config config){
		this(scenario, data.train.x, data.train.y, config, false);
		// automagical testing:
		test(data.test.x, data.test.y);
	}
	
	
	private SVM(String scenario, double[][] data, double[] labels, Config config, boolean waitAuto){
		double[] dataArr = matrix2array(data);
		if(labels == null)
			throw new NullPointerException("Labels are null");
		if(labels.length != data.length)
			throw new IllegalArgumentException("Data and Labels have to be of the same size");
		cookie = svm_init(dataArr, labels);
		if(getCookie() < 0)
			throw new RuntimeException("Problem in svm_init!");
		setConfig("D",1);
		setConfig("CONFIG_DEBUG",1);
		setConfig("SCENARIO",scenario);
		setConfigAll(config);
		if(!waitAuto)
			doTrainSelect(config);
	}
	
	/**
	 * Performs training on the hyperparameter grid.
	 * {@code argv} can be used by experts in the same way
	 * as in the {@code svm-train} command line program of liquidSVM interface.
	 * Most users should rather use {@link Config}.
	 * 
	 * @param argv further command line arguments (for experts)  
	 * @return the train errors which can also be retrieved afterwards from {@link #getTrainErrs()}
	 */
	public double[][] train(String... argv) {
		train_errs = null;
		synchronized(LOCK){
			argv = configure(1, argv);
			double[] ret = svm_train(getCookie(), argv);
			train_errs = ResultAndErrors.myconvert(ret);
			if(train_errs == null)
				throw new RuntimeException("Problem in train");
			trained = true;
			select_errs = null;
		}
		return train_errs;
	}
	
	/**
	 * Selects the best hyperparameter pair according to validation error.
	 * {@code argv} can be used by experts in the same way
	 * as in the {@code svm-select} command line program of liquidSVM interface.
	 * Most users should rather use {@link Config}.
	 * 
	 * @param argv further command line arguments (for experts)  
	 * @return the select errors which can also be retrieved afterwards from {@link #getSelectErrs()}
	 * @throws IllegalStateException if the model has not been trained
	 */
	public double[][] select(String... argv) {
		if(!trained)
			throw new IllegalStateException("SVM has not been trained yet.");
		synchronized(LOCK){
			argv = configure(2, argv);
			double[] ret = svm_select(getCookie(), argv);
			double[][] select_errs_ret = ResultAndErrors.myconvert(ret);
			if(select_errs_ret == null)
				throw new RuntimeException("Problem in select");
			double[][] select_errs_old = this.select_errs;
			if(select_errs_old != null && select_errs_old.length > 0){
				this.select_errs = new double[select_errs_old.length + select_errs_ret.length][];
				System.arraycopy(select_errs_old, 0, this.select_errs, 0, select_errs_old.length);
				System.arraycopy(select_errs_ret, 0, this.select_errs, select_errs_old.length, select_errs_ret.length);
			}else{
				this.select_errs = select_errs_ret;
			}
			selected = true;
		}
		return select_errs;
	}
	
	/**
	 * Predicts labels for the given {@code test} data.
	 * {@code argv} can be used by experts in the same way
	 * as in the {@code svm-test} command line program of liquidSVM interface.
	 * Most users should rather use {@link Config}.
	 * 
	 * @param test array of test samples features (X)
	 * @param argv further command line arguments (for experts)  
	 * @return array of predictions for test data
	 * @throws IllegalStateException if the model has neither been trained nor selected
	 */
	public double[] predict(double[][] test, String... argv){
		synchronized(LOCK){
			double[] labs = null; // new double[test.length];
			double[][] result = test(test, labs, argv).result;
			double[] ret = new double[test.length];
			for(int i=0; i<ret.length && i<result.length; i++)
				ret[i] = result[i][0];
			return ret;
		}
	}
	
	
	/**
	 * Predicts labels for the given {@code test} data and compares
	 * these to the given {@code labs}.
	 * {@code argv} can be used by experts in the same way
	 * as in the {@code svm-test} command line program of liquidSVM interface.
	 * Most users should rather use {@link Config}.
	 * 
	 * @param test array of test samples features (X)
	 * @param labs array of test labels
	 * @param argv further command line arguments (for experts)  
	 * @return combined predictions and errors
	 * @throws IllegalStateException if the model has neither been trained nor selected
	 */
	public ResultAndErrors test(double[][] test, double[] labs, String... argv){
		if(!selected)
			throw new IllegalStateException("SVM has not been selected yet.");
		synchronized(LOCK){
			argv = configure(3, argv);
			ResultAndErrors predictions = svm_test(getCookie(), argv, test.length, matrix2array(test), labs, new ResultAndErrors());
			if(predictions.result == null)
				throw new RuntimeException("Problem in test");
			this.lastResult = predictions;
			return predictions;
		}
	}
	
	/**
	 * Predicts labels for the given {@code test.x} data and compares
	 * these to the given {@code test.y}.
	 * {@code argv} can be used by experts in the same way
	 * as in the {@code svm-test} command line program of liquidSVM interface.
	 * Most users should rather use {@link Config}.
	 * 
	 * @param test array of test samples
	 * @param argv further command line arguments (for experts)  
	 * @return combined predictions and errors
	 * @throws IllegalStateException if the model has neither been trained nor selected
	 */
	public ResultAndErrors test(LiquidData.Data test, String... argv){
		return this.test(test.x, test.y);
	}
	
	public void clean(){
		if(getCookie() < 0)
			throw new IllegalStateException("SVM cannot be cleaned since there is nothing to clean!");
		svm_clean(getCookie());
	}
	
	@Override
	protected void finalize() throws Throwable {
		//clean();
		if(getCookie() >= 0)
			svm_clean(getCookie());
		super.finalize();
	}
	
	static{
		
		// Maybe at some point use http://fahdshariff.blogspot.de/2011/08/changing-java-library-path-at-runtime.html
		// however that is not so nice, as it uses Reflection on System libraries...
//		System.out.println(System.getProperty("java.library.path"));
//		System.out.println(System.getProperty("user.dir"));
		try{
			System.loadLibrary("liquidsvm");
		}catch(UnsatisfiedLinkError e){
			System.err.println("library liquidsvm could not be loaded. Probably it could not be found.\n\n"
					+ "Add its directory either with:\n    java -Djava.library.path=path/to/dir of "+Util.sharedLibraryName()+"\n"
					+ "or change the environment variable:\n    "+Util.LD_LIBRARY_NAME()+"\n\n"
					+ "Current java.library.path is "+System.getProperty("java.library.path"));
			if(!System.getProperty("enable.load.without.library").equals("true"))
				throw e;
		}
	}

	private native static int[] svm_cover_dataset(int NNs, int dim, double[] data);
	/**
	 * Helper function to calculate Voronoi centers that cover the {@code data} set
	 * s.t. each cell should not have more than {@code NNs} samples.
	 *  
	 * @param NNs biggest size of cells
	 * @param data data set to cover
	 * @return array of center vectors
	 */
	public static double[][] calculateDataCover(int NNs, double[][] data){
		if(data.length == 0)
			return new double[0][];
		int dim = data[0].length;
		if(dim==0)
			return new double[0][0];
		double[] tmp = new double[data.length * dim];
		for(int i=0; i<data.length; i++)
			System.arraycopy(data[i],0,tmp,i*dim, dim);
		
		int[] centers = svm_cover_dataset(NNs, dim, tmp);
		
		double[][] ret = new double[centers.length][];
		for(int i=0; i<ret.length; i++)
			ret[i] = data[centers[i]];
		return ret;
	}

	native static String default_params(int stage, int solver);

	private native static int svm_init(double[] data, double[] labels);
	private native static double[] svm_train(int cookie, String[] argv);
	private native static double[] svm_select(int cookie, String[] argv);
	private native static ResultAndErrors svm_test (int cookie, String[] argv, int test_size, double[] test_data, double[] labels, ResultAndErrors resAndErr);
	private native static void svm_clean(int cookie);
	
	private native static void set_param(int cookie, String name, String value);
	private native static String get_param(int cookie, String name);
	private native static String get_config_line(int cookie, int stage);
	
	/**
	 * Sets configuration directly in the SVM.
	 * It is better to use {@link Config}.
	 * 
	 * @param name
	 * @param value
	 * @return this SVM in order to enable method chaining
	 */
	public SVM setConfig(String name, String value){
		set_param(getCookie(), name.toUpperCase(), value);
		return this;
	}
	/**
	 * Sets configuration directly in the SVM.
	 * It is better to use {@link Config}.
	 * 
	 * @param name
	 * @param value
	 * @return this SVM in order to enable method chaining
	 */
	public SVM setConfig(String name, int value){
		setConfig(name, Integer.toString(value));
		return this;
	}
	/**
	 * Sets configuration directly in the SVM.
	 * It is better to use {@link Config}.
	 * 
	 * @param name
	 * @param value
	 * @return this SVM in order to enable method chaining
	 */
	public SVM setConfig(String name, double value){
		setConfig(name, Double.toString(value));
		return this;
	}
	/**
	 * Sets configuration directly in the SVM.
	 * It is better to use {@link Config}.
	 * The values get joined by spaces to form a string.
	 * 
	 * @param name
	 * @param value
	 * @return this SVM in order to enable method chaining
	 */
	public SVM setConfig(String name, double[] value){
		if(value == null || value.length==0)
			return this;
		StringBuilder s = new StringBuilder(Double.toString(value[0]));
		for(int i=1; i<value.length; i++)
			s.append(" ").append(value[i]);
		setConfig(name, s.toString());
		return this;
	}
	
	/**
	 * Sets all the values in the {@code config} in this SVM.
	 * 
	 * @param config
	 * @return this SVM in order to enable method chaining
	 */
	public SVM setConfigAll(Config config){
		if(config == null)
			return this;
		for(Map.Entry<String, String> kv : config.entries()){
			setConfig(kv.getKey(), kv.getValue());
		}
		return this;
	}
	
	/**
	 * Gets configuration of this SVM
	 * @param name
	 * @return the configuration value of the {@code key} in tihs SVM.
	 */
	public String getConfig(String name){
		return get_param(getCookie(), name.toUpperCase());
	}
	
	private String getConfigLine(int stage){
		return get_config_line(getCookie(), stage);
	}
	
	private static double[] matrix2array(double[][] matrix){
		int n = matrix.length;
		if(n == 0)
			throw new IllegalArgumentException("Matrix with 0 rows not allowed.");
		int dim = matrix[0].length;
		double[] ret = new double[n * dim];
		for(int i=0; i<n; i++)
			for(int j=0; j<dim; j++)
				ret[i*dim + j] = matrix[i][j];
		return ret;
	}
	
	private String[] configure(int stage, String... argv){
		String[] config_line = getConfigLine(stage).trim().split(" ");
		System.out.println(Arrays.toString(config_line));
		String[] ret = new String[1+config_line.length + argv.length];
		ret[0] = "liquidsvm";
		System.arraycopy(config_line, 0, ret, 1, config_line.length);
		System.arraycopy(argv, 0, ret, 1+config_line.length, argv.length);
		System.out.println(Arrays.toString(ret));
		return ret;
	}
	
	public static final int MAX_CPU = Runtime.getRuntime().availableProcessors();

	private void doTrainSelect(Config config){
		if(config == null || config.isAutoTrain()){
			train();
			if(config == null || config.isAutoSelect())
				select();
		}
	}
	
	/**
	 * Has the SVM already been trained.
	 * @return whether the SVM already has been trained.
	 * @see #train(String...)
	 */
	public boolean isTrained(){
		return trained;
	}
	/**
	 * Has the SVM already been selected.
	 * @return whether the SVM already has been selected.
	 * @see #select(String...)
	 */
	public boolean isSelected(){
		return selected;
	}
	
	private static native int[] svm_get_cover(int cookie, int task);
	
	public int[] getCover(int task){
		if(!trained)
			throw new IllegalStateException("SVM has not been trained yet.");
		if(task < 0)
			throw new IllegalArgumentException("task has to be >= 0");
		return svm_get_cover(cookie, task);
	}
	
	private static native int[] svm_get_solution_svs(int cookie, int task, int cell, int fold);
	public int[] getSolutionSVs(int task, int cell, int fold){
		if(!selected)
			throw new IllegalStateException("SVM has not been selected yet.");
		if(task <= 0 || cell <= 0 || fold <= 0)
			throw new IllegalArgumentException("task, cell, and fold have to be >= 1");
		return svm_get_solution_svs(fold, task, cell, fold);
	}
	
	private static native double[] svm_get_solution_coeffs(int cookie, int task, int cell, int fold);
	public double[] getSolutionCoeffs(int task, int cell, int fold){
		if(!selected)
			throw new IllegalStateException("SVM has not been selected yet.");
		return svm_get_solution_coeffs(fold, task, cell, fold);
	}
	
	
	
	
	private static native void svm_write_solution(int cookie, String filename);
	private void writeSVM(String filename){
		if(!(filename.endsWith(".sol") || filename.endsWith(".fsol")))
			throw new IllegalArgumentException("Filename must have extension .fsol or .sol.");
		if(!new File(filename).canWrite())
			throw new IllegalArgumentException("File must be writable.");
		svm_write_solution(cookie, filename);
	}
	
	private static native int svm_read_solution(int cookie, String filename);
	public int readSVM(String filename){
		if(!(filename.endsWith(".sol") || filename.endsWith(".fsol")))
			throw new IllegalArgumentException("Filename must have extension .fsol or .sol.");
		if(!new File(filename).canRead())
			throw new IllegalArgumentException("File must be readable.");
		return this.cookie = svm_read_solution(-1, filename);
	}
	
	private void writeObject(java.io.ObjectOutputStream out) throws IOException{
		File tmp = File.createTempFile("solution", ".fsol");
		tmp.deleteOnExit();
		writeSVM(tmp.getAbsolutePath());
		FileInputStream file = new FileInputStream(tmp);
		try{
			long filelen = tmp.length();
			out.writeLong(filelen);
			byte[] buffer = new byte[10240];
			int len;
			// in the following "&&" it is important that the file.read is performed:
			while( (len=file.read(buffer)) >= 0 && filelen > 0){
				out.write(buffer, 0, len);
				filelen -= len;
			}
			if(filelen > 0)
				throw new IOException("Error copying file to stream: bytes still promised "+filelen+" but stream ended");
			if(len>=0)
				throw new IOException("Error copying file to stream: already read all promised bytes but last read was not end of file stream");
			
			// and now write all the java-fields
			out.defaultWriteObject();
		}finally{
			try{
				file.close();
			}finally{
				tmp.delete();
			}
		}
	}
	private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException{
		File tmp = File.createTempFile("solution", ".fsol");
		tmp.deleteOnExit();
		FileOutputStream file = new FileOutputStream(tmp);
		try{
			byte[] buffer = new byte[10240];
			long filelen = in.readLong();
			while(filelen > 0){
				int len = in.read(buffer, 0, (int)Math.min(buffer.length, filelen));
				if(len < 0)
					throw new IOException("End of stream but still expected "+filelen+" bytes");
				file.write(buffer, 0, len);
				filelen -= len;
			}
			int newCookie = this.readSVM(tmp.getAbsolutePath());
			
			// and now read all the java fields
			in.defaultReadObject();
			
			this.cookie = newCookie;
			this.trained = this.selected = true;
			
		}finally{
			try{
				file.close();
			}finally{
				tmp.delete();
			}
		}
	}
	
	/**
	 * Binary and multi-class learning scenario.
	 * Possible values for {@code mcType} are
	 * {@code AvA, OvA, AvA_hinge, OvA_ls, OvA_hinge, AvA_ls}
	 * encoding whether All-vs-All or One-vs-All should be used 
	 * and whether hinge or least squares loss should be used.
	 * 
	 * @author Philipp Thomann
	 *
	 */
	public static class MC extends SVM {
		
		public MC(LiquidData data){
			super("MC", data, null);
		}
		public MC(LiquidData data, Config config){
			super("MC", data, config);
		}
		public MC(LiquidData data, String mcType){
			this(data, mcType, null);
		}
		public MC(LiquidData data, String mcType, Config config){
			super("MC "+mcType, data, config);
		}
		
		public MC(LiquidData.Data data){
			super("MC", data, null);
		}
		public MC(LiquidData.Data data, Config config){
			super("MC", data, config);
		}
		public MC(LiquidData.Data data, String mcType){
			this(data, mcType, null);
		}
		public MC(LiquidData.Data data, String mcType, Config config){
			super("MC "+mcType, data, config);
		}
		
		public MC(double[][] data, double[] labels){
			super("MC", data, labels, null);
		}
		public MC(double[][] data, double[] labels, Config config){
			super("MC", data, labels, config);
		}
		public MC(double[][] data, double[] labels, String mcType){
			this(data, labels, mcType, null);
		}
		public MC(double[][] data, double[] labels, String mcType, Config config){
			super("MC "+mcType, data, labels, config);
		}
	}

	/**
	 * Neyman-Pearson lemma learning scenario.
	 * For the given constraint base several
	 * constraints around it are considered given as factors
	 * of the original constraint.
	 * 
	 * @author Philipp Thomann
	 *
	 */
	public static class NPL extends SVM {
		
		public static final double[] DEFAULT_CONSTRAINT_FACTORS = new double[]{0.5,2/3,1,1.5,2};
		private double constraint;
		private double[] constraintFactors;
		private double nplClass;
		
		public NPL(LiquidData data, double nplClass){
			this(data, nplClass, 0.05);
		}
		public NPL(LiquidData data, double nplClass, double constraint){
			this(data, nplClass, constraint, DEFAULT_CONSTRAINT_FACTORS , null);
		}
		public NPL(LiquidData data, double nplClass, double constraint, double[] constraintFactors, Config config){
			this(data.train, nplClass, constraint, constraintFactors, config);
			test(data.test);
		}
		
		public NPL(LiquidData.Data data, double nplClass){
			this(data, nplClass, 0.05);
		}
		public NPL(LiquidData.Data data, double nplClass, double constraint){
			this(data, nplClass, constraint, DEFAULT_CONSTRAINT_FACTORS , null);
		}
		public NPL(LiquidData.Data data, double nplClass, double constraint, double[] constraintFactors, Config config){
			this(data.x,data.y, nplClass, constraint, constraintFactors, config);
		}
		
		public NPL(double[][] data, double[] labels, double nplClass){
			this(data, labels, nplClass, 0.05);
		}
		public NPL(double[][] data, double[] labels, double nplClass, double constraint){
			this(data, labels, nplClass, constraint, DEFAULT_CONSTRAINT_FACTORS , null);
		}
		public NPL(double[][] data, double[] labels, double nplClass, double constraint, double[] constraintFactors, Config config){
			super("NPL "+nplClass, data, labels, config, true);
			init(nplClass, constraint, constraintFactors, config);
		}
		
		private void init(double nplClass, double constraint, double[] constraintFactors, Config config) {
			this.nplClass = nplClass;
			this.constraint = constraint;
			this.constraintFactors = constraintFactors;
			super.doTrainSelect(config);
		}
		
		@Override
		public double[][] select(String... argv) {
			double[][] ret = null;
			for(double cf : constraintFactors){
				setConfig("npl_class", nplClass);
				setConfig("npl_constraint",constraint * cf);
				ret = super.select(argv);
			}
			return ret;
		}
	}
	
	/**
	 * Receiver Operating Characteristic curve learning scenario.
	 * {@code weightSteps} many points of the curve will be calculated.
	 * 
	 * @author Philipp Thomann
	 *
	 */
	public static class ROC extends SVM {
		
		private int weightSteps;
		
		public ROC(LiquidData data){
			this(data, 9, null);
		}
		public ROC(LiquidData data, int weightSteps){
			this(data, weightSteps, null);
		}
		public ROC(LiquidData data, int weightSteps, Config config){
			this(data.train, weightSteps, config);
			test(data.test);
		}
		
		public ROC(LiquidData.Data data){
			this(data, 9, null);
		}
		public ROC(LiquidData.Data data, int weightSteps){
			this(data, weightSteps, null);
		}
		public ROC(LiquidData.Data data, int weightSteps, Config config){
			this(data.x,data.y, weightSteps, config);
		}
		
		public ROC(double[][] data, double[] labels){
			this(data, labels, 9, null);
		}
		public ROC(double[][] data, double[] labels, int weightSteps){
			this(data, labels, weightSteps, null);
		}
		public ROC(double[][] data, double[] labels, int weightSteps, Config config){
			super("ROC", data, labels, config, true);
			init(weightSteps, config);
		}
		
		private void init(int weightSteps, Config config) {
			this.weightSteps = weightSteps;
			setConfig("weight_steps", weightSteps);
			super.doTrainSelect(config);
		}
		@Override
		public double[][] select(String... argv) {
			double[][] ret = null;
			for(int i=1; i<=weightSteps; i++){
				setConfig("weight_number", i);
				ret = super.select(argv);
			}
			return ret;
		}
	}
	
	/**
	 * Least squares regression learning scenario.
	 * 
	 * @author Philipp Thomann
	 *
	 */
	public static class LS extends SVM {
		
		public LS(LiquidData data){
			this(data, -1, null);
		}
		public LS(LiquidData data, Config config){
			this(data, -1, config);
		}
		public LS(LiquidData data, double clipping){
			this(data, clipping, null);
		}
		public LS(LiquidData data, double clipping, Config config){
			this(data.train, config);
			test(data.test);
		}
		
		public LS(LiquidData.Data data){
			this(data, -1, null);
		}
		public LS(LiquidData.Data data, Config config){
			this(data, -1, config);
		}
		public LS(LiquidData.Data data, double clipping){
			this(data, clipping, null);
		}
		public LS(LiquidData.Data data, double clipping, Config config){
			this(data.x,data.y, clipping, config);
		}
		
		public LS(double[][] data, double[] labels){
			this(data, labels, -1, null);
		}
		public LS(double[][] data, double[] labels, Config config){
			this(data, labels, -1, config);
		}
		public LS(double[][] data, double[] labels, double clipping){
			this(data, labels, clipping, null);
		}
		public LS(double[][] data, double[] labels, double clipping, Config config){
			super("LS "+clipping, data, labels, config);
		}
	}

	/**
	 * Quantile regression learning scenario.
	 * The weights specify which quantiles should be estimated.
	 * The result of a test will have for every test sample
	 * a vector giving the estimated quantiles at that point.
	 * 
	 * @author Philipp Thomann
	 *
	 */
	public static class QT extends SVM {
		
		public static final double[] DEFAULT_WEIGHTS = new double[] {0.05,0.1,0.5,0.9,0.95};
		
		private double[] weights;
		
		public QT(LiquidData data){
			this(data, DEFAULT_WEIGHTS);
		}
		public QT(LiquidData data, double[] weights){
			this(data, weights, null);
		}
		public QT(LiquidData data, double[] weights, Config config){
			this(data.train, weights, config);
			test(data.test);
		}
		
		public QT(LiquidData.Data data){
			this(data, DEFAULT_WEIGHTS);
		}
		public QT(LiquidData.Data data, double[] weights){
			this(data, weights, null);
		}
		public QT(LiquidData.Data data, double[] weights, Config config){
			this(data.x, data.y, weights, config);
		}
		
		public QT(double[][] data, double[] labels){
			this(data, labels, DEFAULT_WEIGHTS);
		}
		public QT(double[][] data, double[] labels, double[] weights){
			this(data, labels, weights, null);
		}
		public QT(double[][] data, double[] labels, double[] weights, Config config){
			super("QT", data, labels, config, true);
			init(weights, config);
		}
		
		private void init(double[] weights, Config config) {
			this.weights = weights;
			setConfig("weights", weights);
			super.doTrainSelect(config);
		}
		
		@Override
		public double[][] select(String... argv) {
			double[][] ret = null;
			for(int i=1; i<=weights.length; i++){
				setConfig("weight_number", i);
				System.out.println(getConfig("weight_number"));
				ret = super.select(argv);
			}
			return ret;
		}
	}
	
	/**
	 * Expectile regression learning scenario.
	 * The weights specify which expectiles should be estimated.
	 * The result of a test will have for every test sample
	 * a vector giving the estimated expectiles at that point.
	 * 
	 * @author Philipp Thomann
	 *
	 */
	public static class EX extends SVM {
		
		public static final double[] DEFAULT_WEIGHTS = new double[] {0.05,0.1,0.5,0.9,0.95};
		
		private double[] weights;
		
		public EX(LiquidData data){
			this(data, DEFAULT_WEIGHTS, null);
		}
		public EX(LiquidData data, double[] weights){
			this(data, weights, null);
		}
		public EX(LiquidData data, double[] weights, Config config){
			this(data.train, weights, config);
			test(data.test);
		}
		
		public EX(LiquidData.Data data){
			this(data, DEFAULT_WEIGHTS, null);
		}
		public EX(LiquidData.Data data, double[] weights){
			this(data, weights, null);
		}
		public EX(LiquidData.Data data, double[] weights, Config config){
			this(data.x, data.y, weights, config);
		}
		
		public EX(double[][] data, double[] labels){
			this(data, labels, DEFAULT_WEIGHTS, null);
		}
		public EX(double[][] data, double[] labels, double[] weights){
			this(data, labels, weights, null);
		}
		public EX(double[][] data, double[] labels, double[] weights, Config config){
			super("EX", data, labels, config, true);
			init(weights, config);
		}
		
		private void init(double[] weights, Config config) {
			this.weights = weights;
			setConfig("weights", weights);
			super.doTrainSelect(config);
		}
		
		@Override
		public double[][] select(String... argv) {
			double[][] ret = null;
			for(int i=1; i<=weights.length; i++){
				setConfig("weight_number", i);
				ret = super.select(argv);
			}
			return ret;
		}
	}
	
}

