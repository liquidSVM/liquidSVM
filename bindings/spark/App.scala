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

/* SimpleApp.scala */

package de.uni_stuttgart.isa.liquidsvm.spark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import de.uni_stuttgart.isa.liquidsvm.{SVM,ResultAndErrors,Config}
import org.apache.spark.storage.StorageLevel

import de.uni_stuttgart.isa.liquidsvm.spark._
import de.uni_stuttgart.isa.liquidsvm.spark.MyUtil._
import de.uni_stuttgart.isa.liquidsvm.spark.MyUtil2._

class Timeit {
  var last:Long = System.currentTimeMillis
  def time():Long = {
    var ret = -this.last
    var last = System.currentTimeMillis
    ret += last
    println("timing: "+ret/1000+"s")
    return ret / 1000
  }
}

object App {
  
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("liquidSVM - Spark App")
    val sc = new SparkContext(conf)
    
    println(s"===== Welcome to liquidSVM on Spark (${sc.version}) =====")
    println(s"You can use this App with the optional arguments: [ file [ trainSize [ CELL_SIZE [ VORONOI ] ] ] ]")
    
    println("driver java.library.path "+System.getProperty("java.library.path"))
    println("driver LD_LIBRARY_PATH   "+System.getenv("LD_LIBRARY_PATH"))
    
    sc.parallelize(Array(1)).map(x => {System.getProperty("java.library.path")}).collect.foreach(x=>println("worker java.library.path "+x))
    sc.parallelize(Array(1)).map(x => {System.getenv("LD_LIBRARY_PATH")}       ).collect.foreach(x=>println("worker LD_LIBRARY_PATH   "+x))
    
    val name = if(args.length > 0)
      args(0)
    else
      "/covtype-full"
    
    val trainSize = if(args.length > 1)
      args(1).toInt
    else
      10000
    
    val CELL_SIZE = if(args.length > 2)
      args(2).toInt
    else
      2000
      
    val VORONOI = if(args.length > 3)
      args(3).toInt
    else
      2000
    
    
    val res = doBenchmarksOur(name,trainSize=trainSize, CELL_SIZE=CELL_SIZE, VORONOI=VORONOI)
    
    println(s"${res._3}: time ${res._2} err ${res._1}")
    
    sc.stop()
  }
  
  def doBenchmarksOur(name: String,CELL_SIZE:Int=2000, VORONOI:Int=3000, SUBSET_SIZE: Int=100000,
	featureTransform:Double=>Double=changeNN, trainSize:Int=0, testSize:Int=0,
	threads:Int=6, num_hosts:Int=7, spacing:Int=6, avail_cores:Int=12) = {
  
  var config = new Config().scenario("MC").threads(1).set("VORONOI","6 "+VORONOI+" 1 100000")
  
  val data = loadData(name+".train.csv", size=trainSize, cache=StorageLevel.MEMORY_ONLY_2)
  val test = loadData(name+ ".test.csv", size=testSize,  cache=StorageLevel.MEMORY_ONLY_2)
  
  val time = new Timeit
  val d = new DistributedSVM("MC",data, SUBSET_SIZE=SUBSET_SIZE, CELL_SIZE=CELL_SIZE, config=config)
  var trainTestP = d.createTrainAndTest(test)
  
  //d.config.display(display).threads(1).set("VORONOI","6 4000 1 100000").set("WEIGHTS",weights.toString)//.set("WEIGHTS_BLABLA","on")

  var result = d.trainAndPredict(trainTestP=trainTestP,threads=threads,num_hosts=num_hosts,spacing=spacing,avail_cores=avail_cores).setName("res")
  
  val err = result.filter{case (x,y) => x != y(0)}.count / result.count.toDouble
  result.count
  var timeSplitTrainTest = time.time
  (err, timeSplitTrainTest, name, result, d, data, test, trainTestP)
}

}
