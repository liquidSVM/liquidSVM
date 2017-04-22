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


package de.uni_stuttgart.isa.liquidsvm.gui;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import de.uni_stuttgart.isa.liquidsvm.SVM;

public enum CommandLineArgument{
		Display("-d <level>","0:1:4"),
		Folds("-f <kind> <number> [<train_fraction>] [<neg_fraction>]",
				"&1:each fold is a contiguous block&2:alternating fold assignment&*3:random&4:stratified random&5:random subset (<train_fraction> and <neg_fraction> required)", "1:5:"),
		Gamma("-g <size> <min_gamma> <max_gamma> [<scale>]", "1:10:","1.0","5.0"),
		Gpu("-GPU <gpus>","0:0:"),
		Help("-h [<level>]","&*0:short help messages&1:detailed help messages"),
		Initialization("-i <cold> <warm>","0:0:6","0:0:6"),
		Kernel("-k <type> [<tmmp> [sizep] <tmm> [size] <vmmp> <vmm>]","&*0:Gaussian RBF&1:Poisson"),
		Lambda("-l <size> <min_lambda> <max_lambda> [<scale>]","1:10:",null,null,"&0:no scaling, i.e. keep <min_lambda> as its is&*1:devide <min_lambda> by <samples in fold>"),
		Loss("-L <loss> [<neg_weight> <pos_weight>]", "&0:binary classification loss&2:least squares loss&3:weighted least squares loss"),
		Partitioning("-P <type> [<number/radius/size> [subset_size]]",
				"&*0:do not split the working sets&1:split the working sets in random chunks using maximum <size> of each chunk&2:split the working sets in random chunks using <number> of chunks"
				+ "&3:split the working sets by Voronoi subsets using <radius>"
				+ "&4:split the working sets by Voronoi subsets using <size>."
//				+ "If the optional [subset_size] is set, a subset of this size is used to faster create the Voronoi partition. If subset_size == 0, the entire data set is used."
				),
//		Partitioning("-P <type> [<number> | <radius> | <size> [subset_size]]"),
		RandomSeed("-r <seed>","-1:-1:"+Integer.MAX_VALUE),
		Clipp("-s <clipp> [<stop_eps>]","-1.0","0.0010"),
//		Solver("-S <solver> [<NNs>]","&0:kernel rule for classification&1:LS-SVM with 2D-solver&2:HINGE-SVM with 2D-solver&3:HINGE-SVM with 4D-solver&4:EXPECTILE-SVM with 2D-solver"),
		Threads("-T <threads>","-1:0:"+SVM.MAX_CPU),
		Weights("-w <weight1> <weight2> [<weights> <geometric> <class>]"),
		Workingset("-W <type>","&*0:take the entire data set&1:multiclass 'all versus all'&2:multiclass 'one versus all'&3:bootstrap with <number> resamples of size <size>")
		;
		
		private static Set<String> names;
	
		public String character;
		public String paramDescription;
		public List<Param> params = new ArrayList<Param>();
//		private Config(String paramDescription){
//			Config(paramDescription, null, null);
//		}
		private CommandLineArgument(String paramDescription, String... types){
			this.paramDescription = paramDescription;
			
			String[] arr = paramDescription.split(" ");
			character = arr[0].substring(1);

			int optional = 0;
			for (int i = 1; i < arr.length; i++) {
				String s = arr[i];
				String name = s;
				if(s.startsWith("[")){
					name = s.substring(1);
					optional++;
				}
				if(s.endsWith("]"))
					name = name.substring(0,name.length()-1);
				
				// now remove < and >
				if(name.startsWith("<"))
					name = name.substring(1,name.length()-1);
				
				String type = null;
				if(i-1 < types.length)
					type = types[i-1];
//				Config.getNames().add(name);
				params.add(new Param(name, type, optional));
				if(s.endsWith("]"))
					optional--;
			}
		}
		
		private CommandLineArgument(String character, List<Param> params) {
			this.character = character;
			this.params = params;
		}
		
		@Override
		public String toString() {
			return name()+" [-"+character+"]: "+params;
		}

		public static Set<String> getNames() {
			if (names == null) {
				names  = new TreeSet<String>();
				for(CommandLineArgument c : CommandLineArgument.values()){
					for(Param p : c.params){
						names.add(p.name);
					}
				}
			}
			return names;
		}
		
	}
