//! spark-shell --name liquidSVM --conf spark.app.id=liquidSVM-benchmarks --jars liquidSVM.jar --num-executors 7
//sc.addJar("bin/liquidSVM.jar") // FIXME do we need that?


// Copyright 2015-2017 Philipp Thomann
//
// This file is part of Simons' SVM.
//
// Simons' SVM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// Simons' SVM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with Simons' SVM. If not, see <http://www.gnu.org/licenses/>.

import de.uni_stuttgart.isa.liquidsvm.{SVM,ResultAndErrors,Config}
import de.uni_stuttgart.isa.liquidsvm.spark._

import de.uni_stuttgart.isa.liquidsvm.spark.MyUtil._
import de.uni_stuttgart.isa.liquidsvm.spark.MyUtil2._

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.storage.StorageLevel
/*
import org.apache.spark.{Partitioner,HashPartitioner,SparkEnv,SparkContext,TaskContext}
import org.apache.spark.AccumulatorParam
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.feature.StandardScaler

import scala.collection.mutable.{Map, LinkedHashMap, HashMap}
import scala.collection.parallel.mutable.ParArray
// import scala.collection.JavaConversions
import java.util.Calendar
*/
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.attribute.BinaryAttribute
//import org.apache.spark.sql.Row
//import org.apache.spark.sql.types.{BinaryType,StructType,StructField}

val OLD_SPARK = {
sc.version.split("\\.")(0)=="1"
}


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


def doBenchmarksOur(name: String,CELL_SIZE:Int=2000, VORONOI:Int=3000, SUBSET_SIZE: Int=100000,
	featureTransform:Double=>Double=changeNN, trainSize:Int=0, testSize:Int=0,
	threads:Int=6, num_hosts:Int=7, spacing:Int=6, avail_cores:Int=12) = {
  
  val data = loadData(name+".train.csv", size=trainSize, cache=StorageLevel.MEMORY_ONLY_2)
  val test = loadData(name+ ".test.csv", size=testSize,  cache=StorageLevel.MEMORY_ONLY_2)
  
  var config = new Config().scenario("MC").threads(6).set("VORONOI","6 "+VORONOI+" 1 100000")
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

def summarize(res: Array[scala.Tuple8[Double, Long, String,
      RDD[(Double, Array[Double])], DistributedSVM, RDD[LabeledPoint], RDD[LabeledPoint], RDD[(Array[LabeledPoint], Array[LabeledPoint], Int)]]]) = {
  val name = res.map(_._3.substring(1)).mkString(" & ")
  val trainSize = res.map(_._6.count).mkString(" & ")
  val testSize = res.map(_._7.count).mkString(" & ")
  val err = res.map{x=>(x._1*1000).toInt/1000d}.mkString(" & ")
  val time = res.map{x=>(x._2/60.0*10).toInt/10d}.mkString(" & ")
  println("name & "+ name)
  println("trainSize & "+ trainSize)
  println("testSize & "+ testSize)
  println("time & "+ time)
  println("err & "+ err)
}


val spark = new org.apache.spark.sql.SQLContext(sc)

def loadDataML(filename: String, size: Int=0, cache:StorageLevel = StorageLevel.MEMORY_ONLY_2) = {
  val formula = new RFormula().setFormula("(Y+1)/2 ~ .").setFeaturesCol("features").setLabelCol("label")
  var raw = if(OLD_SPARK){
    loadData(filename, size=size).toDF("label","features")
     .select((($"label"+1)/2).as("label",BinaryAttribute.defaultAttr.toMetadata), $"features")
//     loadData(filename, size=size).map{ x => Row((x.label+1)/2, x.features)}
//     val schema = StructType(Array(StructField("label",BinaryType, false),StructField("features",new org.apache.spark.mllib.linalg.VectorUDT(),false)))
//     spark.createDataFrame(a, schema)
  }else{
    var raw = spark.read.format("csv").option("inferSchema","true").load(filename).withColumnRenamed("_c0","Y")
    formula.fit(raw).transform(raw).select("label","features")
  }
  if(size>0){
    val fraction = size / raw.count.toDouble
    if(fraction < 1)
      raw = raw.sample(false, fraction)
  }
  raw.cache
}

def doBenchmarksOthersPipeline(name: String, numTrees: Int=100, maxDepth: Int = 5, trainSize:Int=0, testSize:Int=0) = {
  
  val time = new Timeit
  
  val data = loadDataML(name+".train.csv", size=trainSize)
  val test = loadDataML(name+".test.csv", size=testSize)
  
  val rf = new RandomForestClassifier()
    .setNumTrees(numTrees).setMaxDepth(maxDepth)
  val pipeline = new Pipeline()
    .setStages(Array(rf))

  // Fit the pipeline to training documents.
  val model = pipeline.fit(data)

  val result = model.transform(test)
  result.count
  
  val err = result.filter("label != prediction").count / result.count.toDouble
  (err,time.time,result,data,test)
  
}



def doBenchmarksOthersOld(name: String, numTrees: Int = 100, maxDepth: Int = 4, trainSize:Int=0, testSize:Int=0) = {
  
  val time = new Timeit
  
  val data = loadData(name+".train.csv", size=trainSize, cache=StorageLevel.MEMORY_ONLY_2)
    .map{ x => new LabeledPoint((x.label+1)/2, x.features) }
  val test = loadData(name+ ".test.csv", size=testSize,  cache=StorageLevel.MEMORY_ONLY_2)
    .map{ x => new LabeledPoint((x.label+1)/2, x.features) }
  
  val numClasses = 2
  val categoricalFeaturesInfo = scala.collection.immutable.Map[Int, Int]()
  //val numTrees = 1000 // Use more in practice.
  val featureSubsetStrategy = "auto" // Let the algorithm choose.
  val impurity = "gini"
  //val maxDepth = 4
  val maxBins = 32

  val model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  // Evaluate model on test instances and compute test error
  val result = test.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val err = result.filter(r => r._1 != r._2).count.toDouble / result.count.toDouble
 
  (err,time.time,result,data,test)
  
}

