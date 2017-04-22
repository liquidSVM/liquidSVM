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

import java.awt.BorderLayout;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import javax.swing.AbstractAction;
import javax.swing.DefaultComboBoxModel;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JSplitPane;
import javax.swing.JTextField;
import javax.swing.MutableComboBoxModel;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import de.uni_stuttgart.isa.liquidsvm.Config;
import de.uni_stuttgart.isa.liquidsvm.SVM;
import de.uni_stuttgart.isa.liquidsvm.LiquidData;

public class SVMFrame extends JFrame {
	

	public enum Scenario { LS(SVM.LS.class), MC(SVM.MC.class), QT(SVM.QT.class), EX(SVM.EX.class), NPL(SVM.NPL.class), ROC(SVM.ROC.class) ;
		public final Class type;
		private Scenario(Class type){
			this.type = type;
		}
		SVM createSVM(double[][] data, double[] labels, Config config){
			return new SVM(this.name(), data, labels, config);
		}
	};
	
	public enum ConfigOption{
		Display("0:1:4", "How much information should there be printed."),
		Threads("-1:0:"+SVM.MAX_CPU, "How many threads should be used"),
		Partition_choice("&*0:do not split the working sets"
				+ "&1:split random chunks using maximum <size> of each chunk&2:split the working sets in random chunks using <number> of chunks"
				+ "&3:split by Voronoi subsets using <radius>"
				+ "&4:split by Voronoi subsets of some <size>."
				+ "&5:split by Voronoi subsets <size> with overlaps."
				+ "&6:split by Voronoi subsets using <size> (faster).",
				"What Partitioning should be used."
				),
		Grid_choice("-2:0:2"," What Grid configuration should be used"),
		Folds("1:5:", "How many folds should be used in cross-validation"),
//		Gpus("0:0:", "How many Gpus should be used"),
		Kernel("&*0:Gaussian RBF&1:Poisson","What kernel to use"),
//		RandomSeed("-r <seed>","-1:-1:"+Integer.MAX_VALUE),
//		Clipp("-s <clipp> [<stop_eps>]","-1.0","0.0010"),
//		Weights("-w <weight1> <weight2> [<weights> <geometric> <class>]"),
//		Workingset("-W <type>","&*0:take the entire data set&1:multiclass 'all versus all'&2:multiclass 'one versus all'&3:bootstrap with <number> resamples of size <size>")
		;
		
		public String paramType;
		public String help;
		private ConfigOption(String paramType, String help){
			this.paramType = paramType;
			this.help = help;
		}
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private JPanel contentPane;
	private JTextField outputTextfield;
	private JPanel argumentsPanel;
	private JPanel configPanel;
	
	private SVM svm;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		System.setProperty("enable.load.without.library", "true");
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					SVMFrame frame = new SVMFrame();
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}
	
	public SVMFrame() {
		layoutComponents();
		layoutConfigOptions();
		layoutParameters();
		
		makeSVM();
	}
	
	public double[][] getTrainData(){
		return LiquidData.iris;
	}
	
	public double[] getTrainDataLabels(){
		return LiquidData.iris_labs;
	}
	
	public double[][] getTestData(){
		return LiquidData.iris;
	}
	
	public double[] getTestDataLabels(){
		return LiquidData.iris_labs;
	}
	
	private void makeSVM() {
		svm = ((Scenario) scenarioComboBox.getSelectedItem()).createSVM( getTrainData(), getTrainDataLabels(), configuration.train(false));
	}

	private Config configuration = new Config();
	
	private Map<CommandLineArgument, List<JComponent> > commandArgsComps = new TreeMap<>();
	private Map<CommandLineArgument, JCheckBox> commandArgUseCBs = new TreeMap<>();
	private Map<ConfigOption, JComponent> configComps = new TreeMap<>();
	private Map<ConfigOption, JCheckBox> configUseCBs = new TreeMap<>();
	private JComboBox<String> dataComboBox;
	private JComboBox<Scenario> scenarioComboBox;

	private void layoutParameters() {
		argumentsPanel.setLayout(new GridLayout(0, 1));
		for(CommandLineArgument conf : CommandLineArgument.values()) {
			List<JComponent> comps = new ArrayList<JComponent>();
			JPanel pan = new JPanel(new FlowLayout(FlowLayout.LEFT));
			final JCheckBox useCheckbox = new JCheckBox();
			pan.add(useCheckbox);
			commandArgUseCBs.put(conf, useCheckbox);
			pan.add(new JLabel(conf.name()+": -"+conf.character+" "));
			for(Param param : conf.params) {
				String name = param.name;
				String type = param.type;
				JComponent comp = null;
				if(type == null) {
					comp = new JTextField(5);
					((JTextField)comp).getDocument().addDocumentListener(new DocumentListener() {
						@Override
						public void removeUpdate(DocumentEvent e) {
							useCheckbox.setSelected(true);
						}
						@Override
						public void insertUpdate(DocumentEvent e) {
							useCheckbox.setSelected(true);
						}
						@Override
						public void changedUpdate(DocumentEvent e) {
							useCheckbox.setSelected(true);
						}
					});
				}else if(type.startsWith("&")) {
					String[] split = type.split("&");
					JComboBox<String> jc = new JComboBox<String>(split);
					((MutableComboBoxModel<String>)jc.getModel()).removeElementAt(0);
					if(type.contains("&*")) {
						for(String s : split)
							if(s.startsWith("*"))
								jc.setSelectedItem(s);
					}
					jc.addItemListener(new ItemListener() {
						@Override
						public void itemStateChanged(ItemEvent e) {
							useCheckbox.setSelected(true);
							updateOutput();
						}
					});
					comp = jc;
				}else if(type.contains(":")) {
					if(type.endsWith(":")) type += 1000;
					String[] arr = type.split(":");
					SpinnerModel sm;
					if(arr.length!=3) {
						sm = new SpinnerNumberModel(0,0,Integer.MAX_VALUE,1);
						System.err.println("Problem parsing: "+type+" => "+Arrays.toString(arr));
					}else {
					int[] numbers = new int[3];
					for (int i = 0; i < 3; i++) {
						if(arr[i].length()==0)
							numbers[i] = Integer.MAX_VALUE;
						else
							numbers[i] = Integer.parseInt(arr[i]);
					}
					sm = new SpinnerNumberModel(numbers[1],numbers[0],numbers[2], 1);
					}
					JSpinner js = new JSpinner(sm);
					comp = js;
					js.addChangeListener(new ChangeListener() {
						@Override
						public void stateChanged(ChangeEvent e) {
							useCheckbox.setSelected(true);
							updateOutput();
						}
					});
				}else {
					comp = new JTextField(type);
				}
				pan.add(new JLabel("<"+name+">: "));
				pan.add(comp);
				
				comps.add(comp);
				
				comp.addPropertyChangeListener(new PropertyChangeListener() {
					@Override
					public void propertyChange(PropertyChangeEvent evt) {
						updateOutput();
					}
				});
			}
			useCheckbox.addChangeListener(new ChangeListener() {
				@Override
				public void stateChanged(ChangeEvent e) {
					updateOutput();
				}
			});
			argumentsPanel.add(pan);
			commandArgsComps.put(conf, comps);
		}
	}
	private void layoutConfigOptions() {
		configPanel.setLayout(new GridLayout(0, 1));
		for(ConfigOption conf : ConfigOption.values()) {
			JPanel pan = new JPanel(new FlowLayout(FlowLayout.LEFT));
			final JCheckBox useCheckbox = new JCheckBox();
			pan.add(useCheckbox);
			configUseCBs.put(conf, useCheckbox);
			pan.add(new JLabel(conf.name()+" = "));
			String type = conf.paramType;
			JComponent comp = null;
			if(type == null) {
				comp = new JTextField(5);
				((JTextField)comp).getDocument().addDocumentListener(new DocumentListener() {
					@Override
					public void removeUpdate(DocumentEvent e) {
						useCheckbox.setSelected(true);
					}
					@Override
					public void insertUpdate(DocumentEvent e) {
						useCheckbox.setSelected(true);
					}
					@Override
					public void changedUpdate(DocumentEvent e) {
						useCheckbox.setSelected(true);
					}
				});
			}else if(type.startsWith("&")) {
				String[] split = type.split("&");
				JComboBox<String> jc = new JComboBox<String>(split);
				((MutableComboBoxModel<String>)jc.getModel()).removeElementAt(0);
				if(type.contains("&*")) {
					for(String s : split)
						if(s.startsWith("*"))
							jc.setSelectedItem(s);
				}
				jc.addItemListener(new ItemListener() {
					@Override
					public void itemStateChanged(ItemEvent e) {
						useCheckbox.setSelected(true);
						updateOutput();
					}
				});
				comp = jc;
			}else if(type.contains(":")) {
				if(type.endsWith(":")) type += 1000;
				String[] arr = type.split(":");
				SpinnerModel sm;
				if(arr.length!=3) {
					sm = new SpinnerNumberModel(0,0,Integer.MAX_VALUE,1);
					System.err.println("Problem parsing: "+type+" => "+Arrays.toString(arr));
				}else {
				int[] numbers = new int[3];
				for (int i = 0; i < 3; i++) {
					if(arr[i].length()==0)
						numbers[i] = Integer.MAX_VALUE;
					else
						numbers[i] = Integer.parseInt(arr[i]);
				}
				sm = new SpinnerNumberModel(numbers[1],numbers[0],numbers[2], 1);
				}
				JSpinner js = new JSpinner(sm);
				comp = js;
				js.addChangeListener(new ChangeListener() {
					@Override
					public void stateChanged(ChangeEvent e) {
						useCheckbox.setSelected(true);
						updateOutput();
					}
				});
			}else {
				comp = new JTextField(type);
			}
//			pan.add(new JLabel(conf.help));
			pan.add(comp);
			
			comp.addPropertyChangeListener(new PropertyChangeListener() {
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					updateOutput();
				}
			});
			useCheckbox.addChangeListener(new ChangeListener() {
				@Override
				public void stateChanged(ChangeEvent e) {
					updateOutput();
				}
			});
			configPanel.add(pan);
			configComps.put(conf, comp);
		}
	}
	
	protected String[] updateOutput() {
		List<String> args = new ArrayList<String>();
		configuration = new Config();
		for(ConfigOption conf : ConfigOption.values()) {
			if(!configUseCBs.get(conf).isSelected())
				continue;
			String type = conf.paramType;
			JComponent comp = configComps.get(conf);
			String value = null;
			if(type == null) {
				JTextField tf = (JTextField) comp;
				value = tf.getText();
			}else if(type.startsWith("&")) {
				String[] split = type.split("&");
				if(!(comp instanceof JComboBox<?>)) continue;
				@SuppressWarnings("unchecked")
				JComboBox<String> jc = (JComboBox<String>) comp;
				String sel = (String) jc.getSelectedItem();
				int start = 0;
				if(sel.startsWith("*"))
					start = 1;
				value = sel.substring(start, sel.indexOf(':'));
			}else if(type.contains(":")) {
				if(type.endsWith(":")) type += 1000;
				String[] arr = type.split(":");
				SpinnerModel sm;
				if(arr.length!=3) {
					sm = new SpinnerNumberModel(0,0,Integer.MAX_VALUE,1);
					System.err.println("Problem parsing: "+type+" => "+Arrays.toString(arr));
				}else {
				int[] numbers = new int[3];
				for (int i = 0; i < 3; i++) {
					if(arr[i].length()==0)
						numbers[i] = Integer.MAX_VALUE;
					else
						numbers[i] = Integer.parseInt(arr[i]);
				}
				sm = new SpinnerNumberModel(numbers[1],numbers[0],numbers[2], 1);
				}
				JSpinner js = (JSpinner) comp;
				value = js.getValue().toString();
			}else {
				JTextField tf = (JTextField) comp;
				value = tf.getText();
			}
			configuration.set(conf.name(), value);
		}
		parseCommandArgsFromConfig();
		for(CommandLineArgument conf : CommandLineArgument.values()) {
			if(!commandArgUseCBs.get(conf).isSelected())
				continue;
			List<String> localargs = new ArrayList<String>();
			localargs.add("-"+conf.character);
			boolean tainted = false;
			
			Iterator<JComponent> iterator = commandArgsComps.get(conf).iterator();
			for(Param param : conf.params) {
				String type = param.type;
				JComponent comp = iterator.next();
				if(type == null) {
					JTextField tf = (JTextField) comp;
					String text = tf.getText();
					if(text.length() > 0) {
						localargs.add(text);
						tainted = true;
					}
				}else if(type.startsWith("&")) {
					String[] split = type.split("&");
					if(!(comp instanceof JComboBox<?>)) continue;
					@SuppressWarnings("unchecked")
					JComboBox<String> jc = (JComboBox<String>) comp;
					String sel = (String) jc.getSelectedItem();
					int start = 0;
					if(sel.startsWith("*"))
						start = 1;
					else
						tainted = true;
					localargs.add(sel.substring(start, sel.indexOf(':')));
				}else if(type.contains(":")) {
					if(type.endsWith(":")) type += 1000;
					String[] arr = type.split(":");
					SpinnerModel sm;
					if(arr.length!=3) {
						sm = new SpinnerNumberModel(0,0,Integer.MAX_VALUE,1);
						System.err.println("Problem parsing: "+type+" => "+Arrays.toString(arr));
					}else {
						int[] numbers = new int[3];
						for (int i = 0; i < 3; i++) {
							if(arr[i].length()==0)
								numbers[i] = Integer.MAX_VALUE;
							else
								numbers[i] = Integer.parseInt(arr[i]);
						}
						sm = new SpinnerNumberModel(numbers[1],numbers[0],numbers[2], 1);
					}
					JSpinner js = (JSpinner) comp;
					String val = js.getValue().toString();
					localargs.add(val);
					if(!type.contains(":"+val+":"))
						tainted=true;
				}else {
					JTextField tf = (JTextField) comp;
					String text = tf.getText();
					localargs.add(text);
					if(!type.equals(text)) {
						tainted = true;
					}
				}
			}
//			if(tainted)
			args.addAll(localargs);
		}
		outputTextfield.setText(args.toString());
		return args.toArray(new String[args.size()]);
	}

	private void parseCommandArgsFromConfig() {
		svm.setConfigAll(configuration);
		for(CommandLineArgument conf : CommandLineArgument.values()) {
			if(commandArgUseCBs.get(conf).isSelected())
				continue;
			
		}
	}

	/**
	 * Create the frame.
	 */
	public void layoutComponents() {
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(10, 10, 1200, 900);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(new BorderLayout(0, 0));
		
		JPanel headPanel = new JPanel();
		
		JLabel lblData = new JLabel("Data:");
		
		dataComboBox = new JComboBox<String>();
		dataComboBox.setModel(new DefaultComboBoxModel<String>(new String[] {"Iris", "Trees", "CovType"}));
		
		JLabel lblSolver = new JLabel("Scenario:");
		
		scenarioComboBox = new JComboBox<Scenario>();
		scenarioComboBox.setModel(new DefaultComboBoxModel<Scenario>(Scenario.values()));
		GroupLayout gl_headPanel = new GroupLayout(headPanel);
		gl_headPanel.setHorizontalGroup(
			gl_headPanel.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_headPanel.createSequentialGroup()
					.addGap(5)
					.addComponent(lblData)
					.addGap(5)
					.addComponent(dataComboBox, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
					.addGap(5)
					.addComponent(lblSolver)
					.addGap(5)
					.addComponent(scenarioComboBox, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
		);
		gl_headPanel.setVerticalGroup(
			gl_headPanel.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_headPanel.createSequentialGroup()
					.addGap(10)
					.addComponent(lblData))
				.addGroup(gl_headPanel.createSequentialGroup()
					.addGap(5)
					.addComponent(dataComboBox, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
				.addGroup(gl_headPanel.createSequentialGroup()
					.addGap(10)
					.addComponent(lblSolver))
				.addGroup(gl_headPanel.createSequentialGroup()
					.addGap(5)
					.addComponent(scenarioComboBox, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
		);
		headPanel.setLayout(gl_headPanel);
		contentPane.add(headPanel, BorderLayout.NORTH);
		
		JPanel outputPanel = new JPanel();
		contentPane.add(outputPanel, BorderLayout.SOUTH);
		outputPanel.setLayout(new BorderLayout(0, 0));
		
		outputTextfield = new JTextField("Output");
		outputPanel.add(outputTextfield);
		outputTextfield.setColumns(10);
		outputTextfield.setEditable(false);
		
		outputPanel.add(BorderLayout.EAST,new JButton(new AbstractAction("SVM!") {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				SVM svm = ((Scenario) scenarioComboBox.getSelectedItem()).createSVM( LiquidData.iris, LiquidData.iris_labs, configuration.train(false));
				svm.train(updateOutput());
				svm.select();
				svm.test(LiquidData.iris, LiquidData.iris_labs);
			}
		}));
		
		configPanel= new JPanel();
		argumentsPanel = new JPanel();
		JScrollPane argumentsScroll = new JScrollPane(argumentsPanel, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, true, configPanel, argumentsScroll);
		split.setOneTouchExpandable(true);
		contentPane.add(split, BorderLayout.CENTER);
		
	}

	
}
