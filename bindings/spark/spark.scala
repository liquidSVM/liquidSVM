//! spark-shell --name liquidSVM --conf spark.app.id=liquidSVM --jars bin/liquidSVM.jar --num-executors 7
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

package de.uni_stuttgart.isa.liquidsvm.spark


import de.uni_stuttgart.isa.liquidsvm.{SVM,ResultAndErrors,Config}
import org.apache.spark.{Partitioner,HashPartitioner,SparkEnv,SparkContext,TaskContext}
import org.apache.spark.AccumulatorParam
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StandardScaler

import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}

import org.apache.hadoop.fs._


import scala.collection.mutable.{Map, LinkedHashMap, HashMap}
import scala.collection.parallel.mutable.ParArray
// import scala.collection.JavaConversions
import java.util.Calendar
import java.io.{File,BufferedWriter,FileWriter}

// we need a class to partition by the cell id
class CellPartitioner(cells: Int) extends Partitioner {
  def numPartitions: Int = cells
  
  // the cell id is the key in the key/value-pair which gets partitioned
  def getPartition(key: Any): Int = {
    return key.asInstanceOf[Int]
  }
}

object MyUtil {
def lpFormat(x: LabeledPoint, separator: String=", ") = x.label.toString+separator+x.features.toArray.mkString(separator)

def vecAdd(a: Array[Double], b: Array[Double]) = a.zip(b).map{ case (x,y) => x+y}
def vecAdd(a: Array[Int], b: Array[Int]) = a.zip(b).map{ case (x,y) => x+y}
//def vecAdd(a: Product, b: Product) = a.productIterator.zip(b.productIterator).map{ case (x:Double,y:Double) => x+y}

}

import MyUtil._

class BinaryEvaluator(
  var truePos: Int=0,
  var trueNeg: Int=0,
  var pos: Int=0,
  var neg: Int=0,
  var num: Int=0
  ) extends Serializable {
  def this(a: Double, b: Double) = {
    this()
    val aa = if(a>0.5) 1 else 0
    val bb = if(b>0.5) 1 else 0
    truePos = aa*bb
    trueNeg = (1-aa)*(1-bb)
    pos = aa
    neg = 1-aa
    num = 1
  }
  def reduce(other: BinaryEvaluator) = {
    new BinaryEvaluator(truePos+other.truePos, trueNeg+other.trueNeg, pos+other.pos, neg+other.neg, num+other.num) 
  }
  override def toString() = {
    val tpr = truePos/pos.toDouble
    val tnr = trueNeg/neg.toDouble
    val acc = (truePos + trueNeg) / num.toDouble
    f"score ${tpr * tnr}%.3f (tp=$tpr%.3f tn=$tnr%.3f acc=$acc%.2f) after $num%d"
  }
}

// adapted from the spark Programming Guide:
object EvalAccumulatorParam extends AccumulatorParam[BinaryEvaluator] {
  def zero(initialValue: BinaryEvaluator) = {
    initialValue
  }
  def addInPlace(v1: BinaryEvaluator, v2: BinaryEvaluator) = {
    v1.reduce(v2)
  }
}


class DistributedSVM(scenario: String, training: RDD[LabeledPoint], SUBSET_SIZE: Int=50000, CELL_SIZE: Int=2000, var config: Config = new Config) extends Serializable {
  
  var data: RDD[LabeledPoint] = training
//   var scenario: String = scenario
  var centers: Array[Vector] = null
  var centersB: Broadcast[Array[Vector]] = null
  var dataP: RDD[(Int,LabeledPoint)] = null

  var distSVM: RDD[(Int,SVM)] = null
  
  def calculateCenters(doPartition: Boolean = true): Array[Vector] = {
    val theSample = (if(data.count <= SUBSET_SIZE){
      data.collect
    }else{
      data.takeSample(false, SUBSET_SIZE)
    }).map(_.features.toArray)
    
    val reductionFraction = theSample.size / data.count.toDouble
    println(s"Got ${theSample.size} centers")
    centers = SVM.calculateDataCover((CELL_SIZE * reductionFraction).toInt, theSample).map(Vectors.dense(_))
    centersB = SparkContext.getOrCreate.broadcast(centers)
    
    if(doPartition){
      dataP = partitionInCells(data)
      dataP.setName("dataP")
    }
//    dataP.persist(StorageLevel.MEMORY_ONLY_2)
    
    centers
  }
  
  // function to partition any rdd according to voronoi cells
  def partitionInCells(data: RDD[LabeledPoint]) = {
    // calculate the voronoi partition of data:
    
    val centers_dists = data.map(x=> centersB.value.map( Vectors.sqdist(x.features,_) ))
    
    val cell_id = centers_dists.map( s => s.indexOf(s.min) )
    
    // now produce key/value-pairs and partition by them
    
    cell_id.zip(data).partitionBy(new CellPartitioner(centersB.value.size))
  }
  
  def train(training: RDD[(Int,LabeledPoint)]): DistributedSVM = {
    run(training)
  }
  
  def run(input: RDD[(Int,LabeledPoint)] = null): DistributedSVM = {
    calculateCenters()
    var configB = SparkContext.getOrCreate.broadcast(config)
    println("Scenario: "+configB.value.get("scenario"))
    dataP.persist(StorageLevel.MEMORY_ONLY_2)
    distSVM = dataP.groupByKey.map{ case (key, cell) => {
      var s = new SVM(this.scenario, cell.map(_.features.toArray).toArray, cell.map(_.label).toArray, configB.value)
      (key, s)
    }}.cache
    this
  }
  
  def predict(x: RDD[Vector]): RDD[LabeledPoint] = {
    val testing = x.map(new LabeledPoint(0,_))
    test(testing).map(a => new LabeledPoint(a._2(0),a._1.features))
  }
  
  def test(testing: RDD[LabeledPoint]): RDD[(LabeledPoint,Array[Double])] = {
    // At the moment the decision functions/svm_managers are not movable,
    // hence we need to push the test data to the correct cells
    var testP = partitionInCells(testing)
    testP.persist(StorageLevel.MEMORY_ONLY_2)
    distSVM.cogroup(testP).flatMap{ case (key, svm_cell) => {
      val svm: SVM = svm_cell._1.toList(0)
      val cell = svm_cell._2
      val x = cell.map(_.features.toArray).toArray
      val y = cell.map(_.label).toArray
      val res = svm.test(x,y).result
      cell.zip(res)
    }}.cache
  }
    
  def createTrainAndTest(testing: RDD[LabeledPoint]): RDD[(Array[LabeledPoint],Array[LabeledPoint], Int)]  = {
      calculateCenters()
      var testP = partitionInCells(testing).setName("testP")
      var trainTestP = dataP.cogroup(testP).map{ case (k,v) => (v._1.toArray,v._2.toArray,k) }
      trainTestP.setName("TrainTestP")
// 	trainTestP.persist(StorageLevel.MEMORY_ONLY_2)
    return trainTestP
  }
  def trainAndPredict(testing: RDD[LabeledPoint]=null,
	trainTestP: RDD[(Array[LabeledPoint],Array[LabeledPoint], Int)] = null,
	threads:Int=6, num_hosts:Int=11, spacing:Int=6, avail_cores:Int=12): RDD[(Double, Array[Double])] = {
	var theTrainTestP = if(trainTestP==null){
	  createTrainAndTest(testing)
	}else{
	  trainTestP 
	}
	var configB = SparkContext.getOrCreate.broadcast(config)
	val accum = SparkContext.getOrCreate.accumulator(new BinaryEvaluator, "Eval")(EvalAccumulatorParam)
	trainTestP.flatMap( cell => {
	  //??? val log = LogManager.getRootLogger
	  val yTrain: Array[Double] = cell._1.map(_.label)
	  val y: Array[Double] = cell._2.map(_.label)
	  println(s"=============\n${Calendar.getInstance.getTime}: Training liquidSVM with size ${yTrain.size} and testing on ${ y.size } samples and key is ${ cell._3 }")
	  println(s"Got ${yTrain.filter(_>0.5).size} positive training and ${y.filter(_>0.5).size} positive test labels")
	  if(cell._1.size == 0){
	    // There is no training data??
	    println("Got no training data; returning 0")
	    y.zip(Array.fill[Array[Double]](cell._2.size)(Array(0)))
	  }else if(cell._2.size == 0){
	    // There is no testing data??
	    println("Got no testing data; returning empty")
	    y.zip(Array.fill[Array[Double]](0)(Array(0)))
	  }else if(yTrain.forall(_==yTrain(0))){
	    // label is constant on cell anyway, hence just predict that:
	    println("Got constant labels in cell; returning this label for all test samples")
	    y.zip(Array.fill[Array[Double]](cell._2.size)(Array(yTrain(0))))
	  }else{
//            print(cell._1.map(_.label).distinct)
	    val config = new Config(configB.value)
	    val se = SparkEnv.get
	    //println("========"+se.conf.get("spark.master").startsWith("local"))
	    if(!se.conf.get("spark.master").startsWith("local")){
	      val coreOffset = ( (se.executorId.toInt-1) / num_hosts) * spacing % avail_cores
	      config.set("THREADS",threads+" "+coreOffset)
	    }
	    
	    var s = if(config.has("WEIGHTS")){
	      val weights = config.get("WEIGHTS").split(" ")
	      println(s"Doing ${weights.size} weights")
	    
	      var s = new SVM(this.scenario, cell._1.map(_.features.toArray), yTrain, config.train(false))
	      s.train()
	      for(i <- 1 to weights.size){
		s.setConfig("WEIGHT_NUMBER",i)
		s.select()
	      }
	      s
	    }else new SVM(this.scenario, cell._1.map(_.features.toArray), yTrain, config)
	    
	    val trainIts = s.getTrainErrs.map(_(9))
	    val selectIts = s.getSelectErrs.map(_(9))
	    if(selectIts.exists(_>=80000)){
	      println(s"A long running solution was selected at key=${cell._3}")
	    }
	    if(trainIts.exists(_>=100000)){
//	      import java.io._
	      
	      var w: BufferedWriter = null
	      try{
		val file = File.createTempFile("cell-"+cell._3+"-", ".train.csv")
		println(s"Got a breaking example at key=${cell._3} and saving it at ${file.getAbsolutePath}")
		w = new BufferedWriter(new FileWriter(file))
		cell._1.foreach(x=>w.write(lpFormat(x)+"\n"))
	      }catch{
		case e: Exception => println(s"Excpetion during save of breaking cell: ${e} - anyway we just ignore and continue")
	      }finally{
		try{
		w.close
		}catch{case e: Throwable => println(e)}
	      }
	    }
	    
	    val x = cell._2.map(_.features).map(_.toArray)
// 	    val res: Array[Double] = s.predict(x)
	    val res: Array[Array[Double]] = s.test(x,null).result
            s.clean
	    val ret = y.zip(res)
	    val taskEval = ret.map(x=>new BinaryEvaluator(x._1,x._2(0))).reduce(_.reduce(_))
	    println(cell._3 + ": " + taskEval)
	    accum += taskEval
	    ret
          }
      }).cache
  }
  
  
//  def evalResult(testResult: RDD[ResultAndErrors]) = {
//    // don't forget to activate these e.g. using
//    
//    var res = testResult.map(_.result.map(_(0))).collect
//    
//    val testLabByCell = testP.map(_.label).glom().collect
//    val err = res.zip(testLabByCell).map( ab => ab._1.zip(ab._2).filter(cd=>cd._1==cd._2).size / ab._1.size.toDouble)
//    
//    val testCellSize = testP.glom.map(_.size).collect
//    val testError = 1 - (testCellSize, err).zipped.map(_*_).sum / testCellSize.sum
//    null
//  }
  
//   def setConfig(name: String, value: String): DistributedSVM = {
//     config(name) = value
//     this
//   }
//   
//   def getConfig(name: String): String = {
//     config(name)
//   }
  
//  def save(sc: SparkContext, path: String) = {
//  }
//  
//  def load(sc: SparkContext, path: String): DistributedSVM = {
//  }
  
}

object MyUtil2 {

// load the data
def loadData(filename: String, size: Int=0, separator: String=", ", cache: StorageLevel=StorageLevel.MEMORY_ONLY, featureTransform: Double=>Double = identity): RDD[LabeledPoint] = {
  var raw = SparkContext.getOrCreate.textFile(filename,SparkContext.getOrCreate.defaultParallelism*10).map(s => s.split(separator).map(_.toDouble)).map(
    x => new LabeledPoint(x(0),Vectors.dense(x.slice(1, x.size).map(featureTransform)))
  )
  if(size>0){
    val fraction = size / raw.count.toDouble
    if(fraction < 1)
      raw = raw.sample(false, fraction)
  }
  return raw.setName(filename).cache()
}


def saveTrainTestP(path: String, trainTestP: RDD[(Array[LabeledPoint],Array[LabeledPoint],Int)] , separator: String=", ") = {
  trainTestP.map(_._1).mapPartitionsWithIndex((i,cell) => cell.flatMap(a => a.map(y => i+", "+ lpFormat(y)))).saveAsTextFile(path+".train.splits")
  trainTestP.map(_._2).mapPartitionsWithIndex((i,cell) => cell.flatMap(a => a.map(y => i+", "+ lpFormat(y)))).saveAsTextFile(path+".test.splits")
}

def readTrainTestP(path: String, separator: String=", ", labelTrans: Double=>Double = identity, featureTransform:Double=>Double = identity,
	      storage: StorageLevel=StorageLevel.MEMORY_ONLY): RDD[(Array[LabeledPoint],Array[LabeledPoint], Int)] = {
  var number = FileSystem.get(SparkContext.getOrCreate.hadoopConfiguration).listStatus(new Path(path+".train.splits/"))
		.map(_.getPath.getName).filter(_.startsWith("part-")).size
  println(s"Reading ${number} cells")
  val trainSplits = SparkContext.getOrCreate.textFile(path+".train.splits").map(s => {
    val x = s.split(separator).map(_.toDouble);
    (x(0).toInt, new LabeledPoint(labelTrans(x(1)),Vectors.dense(x.slice(2, x.size))))
  }).setName(path+".train.splits")
  val testSplits = SparkContext.getOrCreate.textFile(path+".test.splits").map(s => {
    val x = s.split(separator).map(_.toDouble);
    (x(0).toInt, new LabeledPoint(labelTrans(x(1)),Vectors.dense(x.slice(2, x.size))))
  }).setName(path+"test.splits")
  trainSplits.cogroup(testSplits,new CellPartitioner(number)).map{ case (k,cell) => (cell._1.toArray,cell._2.toArray, k) }.setName(path+"-trainTestP").persist(storage)
}

def evalBinary(res: RDD[(Double,Array[Double])], index: Int=0) = {
  val eval: BinaryEvaluator = res.map{ case (a,b) => new BinaryEvaluator(a,b(index)) }.reduce(_.reduce(_))
  val pos = res.filter(_._1>=0.5).count
  val neg = res.filter(_._1<=0.5).count
  val truepos = res.filter(x=> x._1>0.5 && x._2(0)>=0.5 ).count / pos.toDouble
  val trueneg = res.filter(x=> x._1<0.5 && x._2(0)<=0.5 ).count / neg.toDouble
  ( eval, truepos * trueneg, truepos, trueneg )
}



def doExperimentSplit(file: String, size: Int=0, SUBSET_SIZE: Int=50000, CELL_SIZE: Int=2000, PREFIX: String = if(SparkContext.getOrCreate.isLocal) {"../../data/" }else{ "/"},
	separator: String=", ", config: Config = new Config()) = {

  val data = loadData(PREFIX + file + ".train.csv", size, separator=separator)
  val test = loadData(PREFIX + file + ".test.csv", size, separator=separator)

  var d = new DistributedSVM("MC", data, SUBSET_SIZE=SUBSET_SIZE, CELL_SIZE=CELL_SIZE, config=config)
  var trainTestP = d.createTrainAndTest(test)
  saveTrainTestP(file, trainTestP)
  trainTestP
}

def doExperimentTrainTest(weights: String="0.2 0.5 0.8",
      trainTestP: RDD[(Array[LabeledPoint],Array[LabeledPoint],Int)] = readTrainTestP("covtype-full", labelTrans=2*_-1,
      storage=StorageLevel.MEMORY_ONLY), display: Int=3,
      threads:Int=6, num_hosts:Int=11, spacing:Int=6, avail_cores:Int=12) = {
  val d = new DistributedSVM("MC", null)
  d.config.display(display).threads(1).set("VORONOI","6 4000 1 100000").set("WEIGHTS",weights.toString)//.set("WEIGHTS_BLABLA","on")

  var res = d.trainAndPredict(trainTestP=trainTestP,threads=threads,num_hosts=num_hosts,spacing=spacing,avail_cores=avail_cores).setName("res")
  var err = res.filter(x => x._1*x._2(0)<=0).count / res.count.toDouble

  (err, res, d)
}

def doExperimentBdcomp(file: String="BDCOMP-scaled", size: Int=0, SUBSET_SIZE: Int=50000, CELL_SIZE: Int=2000, PREFIX: String = if(SparkContext.getOrCreate.isLocal) {"../../data/" }else{ "/"},
	separator: String=",", config: Config = new Config) = {
  val data = loadData(PREFIX + file + ".train.csv", size, separator=separator)
  val test = loadData(PREFIX + file + ".test.csv", size, separator=separator)

  var d = new DistributedSVM("MC", data, SUBSET_SIZE=SUBSET_SIZE, CELL_SIZE=CELL_SIZE, config=config)
  d.config.display(3).threads(-1).set("VORONOI","6 4000 1 100000")

  data.count

  (d, data)
  //(d, d.createTrainAndTest(test))
  //var res = d.trainAndPredict(test)
  //var err = res.filter(x => x._1*x._2<=0).count / res.count.toDouble
  //
  //return (err, data, test, res, d)
}

def doExperimentBdcompTrainTest(weights: String = "0.975",
      trainTestP: RDD[(Array[LabeledPoint],Array[LabeledPoint],Int)] = readTrainTestP("bdcomp-scaled", labelTrans=2*_-1,
      storage=StorageLevel.MEMORY_ONLY),
      voronoi: String="6 20000 1 200000",
      threads:Int=6, num_hosts:Int=11, spacing:Int=6, avail_cores:Int=12) = {
  val d = new DistributedSVM("MC", null)
  d.config.display(3).threads(1).set("VORONOI",voronoi).set("WEIGHTS",weights.toString).gridChoice(2)
  //val trainTestP = readTrainTestP("bdcomp-scaled", labelTrans=2*_-1, storage=storage)
  val res = d.trainAndPredict(trainTestP=trainTestP,threads=threads,num_hosts=num_hosts,spacing=spacing,avail_cores=avail_cores).setName("res")
  //val eval = evalBinary(res)
  (res, trainTestP, d)
}

def scaleAndSave(path: String, data: RDD[LabeledPoint], test: RDD[LabeledPoint]) = {
  val scaler = new StandardScaler(withMean=true, withStd=true).fit(data.map(_.features))
  data.map(x => x.label.toString+", "+scaler.transform(x.features).toArray.mkString(", ")).saveAsTextFile(path+"-scaled.train.csv")
  test.map(x => x.label.toString+", "+scaler.transform(x.features).toArray.mkString(", ")).saveAsTextFile(path+"-scaled.test.csv")
}


def changeNN(x:Double) = if(x.isNaN) -1 else x

def splitData(path: String="/BDCOMP-all", output: String="BDCOMP-all-scaled", minSize: Int=5000,CELL_SIZE:Int=100000, SUBSET_SIZE: Int=100000,
	featureTransform:Double=>Double=changeNN, trainSize:Int=0, testSize:Int=0) = {

  // load the data, test has not to be cached!
  var dataOrig = loadData(path+".train.csv", size=trainSize, separator=",", cache=StorageLevel.MEMORY_ONLY, featureTransform=featureTransform)
  var testOrig = loadData(path+".test.csv", size=testSize, separator=",", cache=StorageLevel.NONE, featureTransform=featureTransform)
  
  // learn a scaler and apply it
  println(":::::::Scaling data...")
  
  val scaler = new StandardScaler(withMean=true, withStd=true).fit(dataOrig.map(_.features))
  var test = testOrig.map(x=>new LabeledPoint(x.label, scaler.transform(x.features)))
  // we don't cache the transformed data: this has to be computed 2+SUBSET-fraction times, but it would need more cache memory!
  var data = dataOrig.map(x=>new LabeledPoint(x.label, scaler.transform(x.features))).setName(path+".train.scaled").cache

  // now train the spatial decomposition
  println(":::::::Calculating first batch of centers...")
  var d = new DistributedSVM("MC",data,SUBSET_SIZE=SUBSET_SIZE,CELL_SIZE=CELL_SIZE)
  var centers = d.calculateCenters(false)
  println(":::::::Saving first batch of centers...")
  SparkContext.getOrCreate.parallelize(centers).zipWithIndex.map(x=> new LabeledPoint(x._2, x._1)).saveAsTextFile(output+".origcenters.csv")
  
  println(":::::::Calculating distances to centers...")
  var centersB = SparkContext.getOrCreate.broadcast(centers)
  val centers_dists = data.map(x=> centersB.value.map( Vectors.sqdist(x.features,_) )).setName(path+".train.dists")//.cache
  var cell_id = centers_dists.map( s => s.indexOf(s.min) )
  var cs = cell_id.countByValue

  // now which cells are too small and should be distributed among the others
  println(":::::::Calculating reduced distances...")
  // TODO: check whether some cells really stay alive...
  var delKeys = cs.filter(_._2<minSize).keys.toArray
  val a = cs.keys.toList
  // resplit we do for security d.partitionInCells anyway
  // cell_id = centers_dists.map(s=>{var s2=s; delKeys.foreach(s(_)=Int.MaxValue); a.indexOf(s2.indexOf(s2.min)) }).setName(path+".train.cell_id").cache
  // cs = cell_id.countByValue
  centers = a.map(centers(_)).toArray
  centersB = SparkContext.getOrCreate.broadcast(centers)

  // save the final centers together with their id's
  println(":::::::Saving centers...")
  SparkContext.getOrCreate.parallelize(centers).zipWithIndex.map(x=> x._2 + ", "+x._1.toArray.mkString(", ")).saveAsTextFile(output+".centers.csv")

  // partition data/test by centers:
  d.centersB = centersB
  println(":::::::Splitting partitions...")
  var dataP = d.partitionInCells(data)
  var testP = d.partitionInCells(test)

  println(":::::::Saving splits...")
  if(false){
//     var trainTestP = data.cogroup(test, new CellPartitioner(cs.size)).map{ case (k,v) => (v._1.toArray,v._2.toArray,k) }
//     var dataP = cell_id.zip(data)
//     var trainTestP = dataP.cogroup(testP, new CellPartitioner(cs.size)).map{ case (k,v) => (v._1.toArray,v._2.toArray,k) }
//     saveTrainTestP("BDCOMP-all-scaled", trainTestP)
  }else{
    dataP.mapPartitionsWithIndex((i,cell) => cell.map(a => i+", "+ lpFormat(a._2))).saveAsTextFile(output+".train.splits")
    testP.mapPartitionsWithIndex((i,cell) => cell.map(a => i+", "+ lpFormat(a._2))).saveAsTextFile(output+".test.splits")
  }
  
  (dataP, testP, centers)
}

def splitDataKmeans(data: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int=300, iterations:Int = 40, output: String="BDCOMP-all-scaled.kmeans", minSize: Int=5000,CELL_SIZE:Int=100000, SUBSET_SIZE: Int=100000,
	featureTransform:Double=>Double=changeNN, trainSize:Int=0, testSize:Int=0) = {
  val model = KMeans.train(data.map(_.features), 300, 20)
  //var model2 = new KMeans().setInitialModel(model).setK(300).run(data)
  model.save(SparkContext.getOrCreate, output+".kmeansmodel")
  
  val centers = model.clusterCenters
  SparkContext.getOrCreate.parallelize(centers).zipWithIndex.map(x=> x._2 + ", "+x._1.toArray.mkString(", ")).saveAsTextFile(output+".centers.csv")
  
  var d = new DistributedSVM("MC",data)
  d.centers = centers
  d.centersB = SparkContext.getOrCreate.broadcast(centers)
  var dataP = d.partitionInCells(data)
  var testP = d.partitionInCells(test)
  dataP.mapPartitionsWithIndex((i,cell) => cell.map(a => i+", "+ lpFormat(a._2))).saveAsTextFile(output+".train.splits")
  testP.mapPartitionsWithIndex((i,cell) => cell.map(a => i+", "+ lpFormat(a._2))).saveAsTextFile(output+".test.splits")
}

// var (dataP,testP,cs,centers) = splitData("/BDCOMP-all", "BDCOMP-all-scaled.2", 10000, 100000, 1000000 )

// var (err, data, test, res, d) = doExperiment("covtype-full", 100000)

// var (err, data, test, res, d) = doExperiment("covtype-full", 0, config=new Config().display(2))

/*

var d = new DistributedSVM("MC",data,SUBSET_SIZE=100000,CELL_SIZE=100000)

var centers = d.calculateCenters(false)
var centersB = SparkContext.getOrCreate.broadcast(centers)
val centers_dists = data.map(x=> centersB.value.par.map( Vectors.sqdist(x.features,_) )).cache
val cell_id = centers_dists.map( s => s.indexOf(s.min) )
var cs = cell_id.countByValue

var delKeys = cs.filter(_._2<5000).keys.toArray
val cell_id = centers_dists.map(s=>{var s2=s; delKeys.foreach(s(_)=10000); a.indexOf(s2.indexOf(s2.min)) }).cache
var cs = cell_id.countByValue

val a = cs.keys.toList
var centers2 = a.map(centers(_)).toArray
var centersOrig = centers
centers = centers2
centersB = SparkContext.getOrCreate.broadcast(centers)

*/


/*

Human splice site recognition: 50mio x 11,725,480 :-(
http://sonnenburgs.de/soeren/projects/coffin/splice_data.tar.xz

Possible largescale datasets:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/data
https://www.kaggle.com/c/wikichallenge
http://stat-computing.org/dataexpo/2009/the-data.html
Google BigQuery public-data:natality nyc-tlc (taxifahrten)

*/
}
