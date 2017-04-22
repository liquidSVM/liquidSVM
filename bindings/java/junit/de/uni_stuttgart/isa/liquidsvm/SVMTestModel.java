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

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import de.uni_stuttgart.isa.liquidsvm.SVM.MC;
import static org.hamcrest.CoreMatchers.*;
import org.hamcrest.CoreMatchers;
import org.hamcrest.Description;
import org.hamcrest.TypeSafeMatcher;

import java.io.*;

/**
 * @author Philipp Thomann
 *
 */
public class SVMTestModel {
	
	private static final double ls_err = 5;
	private static final double mc_err = 0.05;
	private double[][] mc_tr;
	private double[] mc_tr_labs;
	private double[][] mc_ts;
	private double[] mc_ts_labs;
	private int mc_levels;
	
	private double[][] reg_tr;
	private double[] reg_tr_labs;
	private double[][] reg_ts;
	private double[] reg_ts_labs;

	@Before
	public void setUp() throws Exception {
		mc_tr = LiquidData.iris;
		mc_tr_labs = LiquidData.iris_labs;
		mc_ts = LiquidData.iris;
		mc_ts_labs = LiquidData.iris_labs;
		mc_levels = 3;
		
		reg_tr = LiquidData.trees;
		reg_tr_labs = LiquidData.trees_labs;
		reg_ts = LiquidData.trees;
		reg_ts_labs = LiquidData.trees_labs;
	}
	
	@Test
	public void testSerialize() throws Exception {
		int n = mc_ts.length;
		int tasks = 1 + mc_levels * (mc_levels-1) / 2; // AvA
		
		SVM s = new SVM.MC(mc_tr,mc_tr_labs, theConfig());
		
		Class origClass = s.getClass();
		int origCookie = s.getCookie();
		
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ObjectOutputStream oos = new ObjectOutputStream(baos);
		oos.writeObject(s);
		
		s.clean();

		ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
		ObjectInputStream ois = new ObjectInputStream(bais);
		s = (SVM)ois.readObject();
		
		int newCookie = s.getCookie();
		
		assertFalse("Cookie needs to have changed", origCookie == newCookie);
		assertEquals(s.getClass(), origClass);

		ResultAndErrors res = s.test(mc_ts,mc_ts_labs);
		assertEquals("lengths are equal", n, res.result.length);
		assertEquals("number of error columns", tasks, res.errors.length);
		double handErr = 0;
		for(int i=0; i<n; i++)
			handErr += (mc_ts_labs[i] == res.result[i][0]) ? 0:1;
		handErr /= (double)n;
		assertLT(handErr,mc_err);
		assertLT(res.errors[0][0],mc_err);
		assertApprox(res.errors[0][0], handErr);
	}


	@Test
	public void testCover() throws Exception {
		int n = mc_ts.length;
		int tasks = mc_levels * (mc_levels-1) / 2; // AvA
		
		SVM s;
		s = new SVM.MC(mc_tr,mc_tr_labs, theConfig().set("VORONOI", "4 40"));

		int[] cover;
		for(int task=1; task<=tasks; task++){
			cover = s.getCover(task);
			assertTrue(cover.length>=3 && cover.length < 10);
			
			for(int i=0; i<cover.length; i++){
				assertTrue(cover[i]>=0 && cover[i]<n);
				// now check that samples belong to the correct label
				// task=1: not 3, task=2: not 2, task=1: not 1
				assertTrue(mc_tr_labs[cover[i]] != (4-task));
			}
		}
		
		cover = s.getCover(4);
		assertTrue(cover.length==0);
		
		// expected exception?
//		cover = s.getCover(-1);
	}

	@Test
	public void testSolution() throws Exception {
		int n = mc_ts.length;
		int tasks = mc_levels * (mc_levels-1) / 2; // AvA
		
		SVM s;
		s = new SVM.MC(mc_tr,mc_tr_labs, theConfig());
		
		int[] sv;
		double[] coeff;
		int cell = 1;
		int fold = 1;
		
		for(int task=1; task<=tasks; task++){
			sv = s.getSolutionSVs(task, cell, fold);
			coeff = s.getSolutionCoeffs(task, cell, fold);
			
			assertTrue(sv.length == coeff.length);
			assertTrue(sv.length <= n);
			
			for(int i=0; i<sv.length; i++){
				assertTrue(sv[i]>=0 && sv[i]<n);
				// now check that samples belong to the correct label
				// task=1: not 3, task=2: not 2, task=1: not 1
				assertTrue(mc_tr_labs[sv[i]] != (4-task));
			}
		}
		
		// expected exception?
//		cover = s.getCover(-1);
	}
	
	private static Config theConfig() {
		return new Config().threads(1).display(0);
	}
	
	public static void assertLT(double x, double than){
		assertTrue(x+" should be < "+than, x<than);
	}
	
	public static void assertApprox(double a, double b){
		assertTrue(Math.abs(a-b)<0.001);
	}
	
}
