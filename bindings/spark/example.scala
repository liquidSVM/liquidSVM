import de.uni_stuttgart.isa.liquidsvm._
import de.uni_stuttgart.isa.liquidsvm.spark._

val data = MyUtil2.loadData("covtype.10000.train.csv")
val test = MyUtil2.loadData("covtype.10000.test.csv")

var config = new Config().scenario("MC").threads(1).set("VORONOI","6 2000")
val d = new DistributedSVM("MC",data, SUBSET_SIZE=50000, CELL_SIZE=5000, config=config)
var trainTestP = d.createTrainAndTest(test)
var result = d.trainAndPredict(trainTestP=trainTestP,threads=1,num_hosts=1,spacing=1)
val err = result.filter{case (x,y) => x != y(0)}.count / result.count.toDouble

// and now realise the training
println(err)
