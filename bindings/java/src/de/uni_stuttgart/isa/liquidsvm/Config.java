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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.AbstractMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import de.uni_stuttgart.isa.liquidsvm.SVM.NPL;

/**
 * Represents the configuration of an SVM.
 * 
 * liquidSVM is equipped with experienced default values
 * that should give good results in many cases.
 * However experts can tweak th SVM to their likings.
 * 
 * It has similar interfaces to a {@code Map<String, String>} but it
 * keeps track in which order values were set. 
 * The class uses method chaining to make it easy to make configurations on the fly:
 * <pre>
 * {@code
 * SVM s = new SVM.LS(features, labels, new Config()
 *     .disply(1)
 *     .threads(2)
 *     .useCells()
 *     .folds(10)
 * )
 * }
 * </pre>
 * 
 * @author Philipp Thomann
 *
 */
/**
 * @author Philipp Thomann
 *
 */
public class Config implements java.io.Serializable {
	
	private ArrayList<String> keys = new ArrayList<>();
	private HashMap<String, String> map = new HashMap<String, String>();
	
	private boolean autoTrain = true;
	private boolean autoSelect = true;
	public Config(){}
	public Config(Config other){
		this.keys = new ArrayList<>(other.keys);
		this.map = new HashMap<>(other.map);
		this.autoTrain = other.autoTrain;
		this.autoSelect = other.autoSelect;
	}
	
	/**
	 * Sets the {@code value} for the {@code key} and remembers the order
	 * in which it was added. This order is used in {@link #entries}.
	 * It returns this {@code Config} to enable
	 * method chaining. Many of the other methods use this.
	 * 
	 * @param key
	 * @param value
	 * @return this {@code config}
	 * @see #entries()
	 */
	public Config set(String key, String value) {
		keys.add(key);
		map.put(key, value);
		return this;
	}
	
	public String remove(Object key) {
		keys.remove(key);
		return map.remove(key);
	}

	public String get(String key){
		return map.get(key);
	}

	public boolean has(String key){
		return map.containsKey(key);
	}
	
	/**
	 * Creates a {@code List} that contains all key-value pairs
	 * in the order they were set.
	 * 
	 * @return a list containing all key-value pairs
	 * @see #set(String, String)
	 */
	public List<Map.Entry<String, String>> entries() {
		LinkedList<Map.Entry<String, String>> ret = new LinkedList<>(); 
		for(String s : keys)
			ret.add(new AbstractMap.SimpleEntry<String, String>(s,map.get(s)));
		return ret;
	}
	
	/**
	 * Sets the scenario.
	 * 
	 * @param scenario can be any of {@code "LS", "EX", "QT", "MC", "NPL", "ROC"}
	 * @return this {@code Config}
	 * @see #set(String, String)
	 * @see SVM.LS
	 * @see SVM.MC
	 */
	public Config scenario(String scenario){
		return set("scenario",String.valueOf(scenario));
	}
	
	/**
	 * Sets the display level starting at 0 which means no information.
	 * 
	 * @param d display info level
	 * @return this {@code Config}
	 * @see #set(String, String)
	 */
	public Config display(int d){
		return set("display",String.valueOf(d));
	}
	
	/**
	 * Sets the number of threads to use.
	 * Default 0 means to use all physical cores,
	 * -1 means that all but one of the physical cores are used.s
	 * 
	 * @param d number of threads
	 * @return this {@code Config}
	 * @see #set(String, String)
	 */
	public Config threads(int d){
		return set("threads",String.valueOf(d));
	}
	
	/**
	 * Sets the partition choice. The default 0 means no partitioning.
	 * See <a href="package-summary.html#Cells">Cells</a>.
	 * 
	 * @param d display info level
	 * @return this {@code Config}
	 * @see #set(String, String)
	 */
	public Config partitionChoice(int d){
		return set("partition_choice",String.valueOf(d));
	}
	
	/**
	 * Sets the hyperparameter grid.
	 * See <a href="package-summary.html#hyperparameter-grid">Hyperparameter Grid</a>.
	 * 
	 * @param d
	 * @return this {@code Config}
	 * @see #set(String, String)
	 */
	public Config gridChoice(int d){
		return set("grid_choice",String.valueOf(d));
	}
	
	/**
	 * Activates adaptivity in the hyperparameter grid.
	 * See <a href="package-summary.html#adaptive-grid">Adaptive Grid</a>.
	 * 
	 * @param d
	 * @return this {@code Config}
	 * @see #set(String, String)
	 */
	public Config adaptivityControl(int d){
		return set("adaptivity_control",String.valueOf(d));
	}
	
	public Config randomSeed(int d){
		return set("random_seed",String.valueOf(d));
	}
	
	public Config folds(int d){
		return set("folds",String.valueOf(d));
	}
	
	public Config voronoi(int d){
		return set("voronoi",String.valueOf(d));
	}
	
	public Config scale(boolean d){
		return set("scale",(d)?"1":"0");
	}
	
	public Config useCells(){
		return useCells(true);
	}
	public Config useCells(boolean b){
		if(b)
			return partitionChoice(6);
		else
			return partitionChoice(0);
	}
	
	/**
	 * Specifies whether training should start automagically at the end
	 * of initializing the {@link SVM}.
	 * Default is {@code true}.
	 *  
	 * @param b
	 * @return this {@code Config}
	 * @see SVM#train(String...)
	 */
	public Config train(boolean b){
		autoTrain = b;
		return this;
	}
	
	/**
	 * Specifies whether selecting should start automagically at the end
	 * of initializing the {@link SVM} after the automatic training.
	 * Default is {@code true}.
	 * This only has effect if {@code #isAutoTrain()} is true.
	 *  
	 * @param b
	 * @return this {@code Config}
	 * @see SVM#select(String...)
	 */
	public Config select(boolean b){
		autoSelect = b;
		return this;
	}
	
	public boolean isAutoTrain(){
		return autoTrain;
	}
	
	public boolean isAutoSelect(){
		return autoSelect;
	}
	
}
