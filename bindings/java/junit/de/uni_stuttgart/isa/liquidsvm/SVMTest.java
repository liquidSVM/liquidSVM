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

/**
 * @author Philipp Thomann
 *
 */
public class SVMTest {
	
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
	public void testMC_AvA_hinge() {
		int n = mc_ts.length;
		int tasks = 1 + mc_levels * (mc_levels-1) / 2; // AvA
		
		SVM s = new SVM.MC(mc_tr,mc_tr_labs, theConfig());
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
	public void testMC_OvA_ls() {
		int n = mc_ts.length;
		int tasks = 1 + mc_levels ; // OvA
		
		SVM s = new SVM.MC(mc_tr,mc_tr_labs, "OvA", theConfig());
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
	public void testMC_OvA_hinge() {
		int n = mc_ts.length;
		int tasks = 1 + mc_levels ; // OvA
		
		SVM s = new SVM.MC(mc_tr,mc_tr_labs, "OvA_hi", theConfig());
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
	public void testMC_AvA_ls() {
		int n = mc_ts.length;
		int tasks = 1 + mc_levels * (mc_levels-1) / 2; // AvA
		
		SVM s = new SVM.MC(mc_tr,mc_tr_labs, "AvA_ls", theConfig());
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
	public void testLS() {
		int n = reg_ts.length;
		int tasks = 1;
		
		SVM s = new SVM.LS(reg_tr,reg_tr_labs, theConfig());
		ResultAndErrors res = s.test(reg_ts,reg_ts_labs);
		assertEquals("lengths are equal", n, res.result.length);
		assertEquals("number of error columns", tasks, res.errors.length);
		double handErr = 0;
		for(int i=0; i<n; i++){
			System.out.print(reg_ts_labs[i] +":"+ res.result[i][0]+", ");
			handErr += Math.pow(reg_ts_labs[i] - res.result[i][0], 2);
		}
		handErr /= (double)n;
		assertLT(handErr,ls_err);
		assertLT(res.errors[0][0],ls_err);
		assertApprox(res.errors[0][0], handErr);
	}


	//@Ignore @Test
	public void testParallel() throws InterruptedException {
		for(int i=0; i<1000; i++){
		Thread a = new Thread(new Runnable(){ public void run(){
				int n = reg_ts.length;
				int tasks = 1;
				
		SVM s = new SVM.LS(reg_tr,reg_tr_labs, theConfig().set("THREADS","2 0"));
		ResultAndErrors res = s.test(reg_ts,reg_ts_labs);
		assertEquals("lengths are equal", n, res.result.length);
		assertEquals("number of error columns", tasks, res.errors.length);
		double handErr = 0;
		for(int i=0; i<n; i++){
			System.out.print(reg_ts_labs[i] +":"+ res.result[i][0]+", ");
			handErr += Math.pow(reg_ts_labs[i] - res.result[i][0], 2);
		}
		handErr /= (double)n;
		assertLT(handErr,ls_err);
		assertLT(res.errors[0][0],ls_err);
		assertApprox(res.errors[0][0], handErr);
			}});
		a.run();
		Thread b = new Thread(new Runnable(){ public void run(){
			int n = reg_ts.length;
			int tasks = 1;
			
			SVM s = new SVM.LS(reg_tr,reg_tr_labs, theConfig().set("THREADS","2 2"));
			ResultAndErrors res = s.test(reg_ts,reg_ts_labs);
			assertEquals("lengths are equal", n, res.result.length);
			assertEquals("number of error columns", tasks, res.errors.length);
			double handErr = 0;
			for(int i=0; i<n; i++){
				System.out.print(reg_ts_labs[i] +":"+ res.result[i][0]+", ");
				handErr += Math.pow(reg_ts_labs[i] - res.result[i][0], 2);
			}
			handErr /= (double)n;
			assertLT(handErr,ls_err);
			assertLT(res.errors[0][0],ls_err);
			assertApprox(res.errors[0][0], handErr);
		}});
		b.run();
		a.join();
		b.join();
		}
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
