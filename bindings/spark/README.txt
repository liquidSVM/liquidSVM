

LIQUIDSVM ON SPARK


We provide a simple version of liquidSVM that also can be executed on
any Spark Cluster. By this also much larger data sets can be attacked -
we used it for a data set with 30 million samples and 631 features on up
to 11 workers.

  NOTE This is a preview, stay tuned for a better interface and more
  documentation!

We tested it on Spark versions 1.6.1 and 2.1.0. It is only supported on
Linux. Generalisation to macOS should be straitforward. Windows should
not be impossible to achieve.


Quick start

Download Spark from http://spark.apache.org/downloads.html, e.g.
spark-2.1.0-bin-hadoop2.7.tgz, and unpack it. We assume that henceforth
$SPARK_HOME points to that directory. We also assume that $JAVA_HOME is
correctly set.

  SUGGESTION To avoid too much information, copy
  conf/log4j.properties.template to conf/log4j.properties and change in
  the latter the line

  log4j.rootCategory=INFO, console

  to

  log4j.rootCategory=WARN, console

Download
http://www.isa.uni-stuttgart.de/software/spark/liquidSVM-spark.zip,
unpack it and change into that directory. First do the following to
compile and to use the library:

    make lib
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

Then issue:

    $SPARK_HOME/bin/spark-submit --master local[2] \
      --class de.uni_stuttgart.isa.liquidsvm.spark.App \
      liquidSVM-spark.jar covtype.10000

This will start a local Spark environment with as many executors as
processors and train and test the covtype.10000 in that directory. You
can use any other liquidData. If Spark is configured with Hadoop you
also can give such urls.

While the job runs go to http://localhost:4040/ and monitor how the work
progresses.

You also can use the interactive Spark-shell. Currently, this works for
local only using at most the number of physical cores for executors, say
2:

    $SPARK_HOME/bin/spark-shell --master local[2] --jars liquidSVM-spark.jar

and then you can do the following:

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

or equivalently:

    :load example.scala

  NOTE At the moment --master local[n] crashes if n is bigger than the
  number of physical cores! --master local[*] gives usually the number
  of logical cores, which is therefore problematic. The above examples
  are all with --master local[2] because nowadays most computers have at
  least 2 physical cores.


Installation of native library (tested only for YARN)

The core routines of liquidSVM are written in C++ hence there has to be
our native JNI-library made available to all workers.

Copy on the fly

To sometimes use liquidSVM on Spark it is most easy you can let Spark
distribute it on the fly (if libliquidsvm.so is in the current
directory):

    $(SPARK_HOME)/bin/spark-submit \
      --conf spark.executor.extraLibraryPath=. --conf spark.driver.extraLibraryPath=. --files libliquidsvm.so
      ...

Local Install

If you will use liquidSVM more often maybe install the bindings locally.

We assume that the machines are homogeneous and every one has a
directory $LOCAL_LIB, e.g. /usr/local/lib/ or /export/user/lib/. It also
can be a shared NFS- or AFS-directory.

1)  put the libliquidsvm.so into all those $LOCAL_LIB directories

    for node in master slave1 slave2; do
      scp libliquidsvm.so $node:$LOCAL_LIB/
    done

or the one if it is shared:

    cp libliquidsvm.so $node:$LOCAL_LIB/

If your machines are of different types you also can

    for node in master slave1 slave2; do
      ssh $node cd $(SIMONSSVM_HOME)/bindings/java && make local-lib LOCAL=$LOCAL_LIB
    done

2)  add
    $LOCAL_LIB to the java.library.path for driver and workers. It seems that `$LD_LIBRARY_PATHis inherited, but it might be wise to put it into$SPARK_HOME/conf/spark-defaults.conf`.

On our machines I have $LOCAL_LIB=/export/user/thomann/lib and hence I
set:

    spark.driver.extraLibraryPath    /export/user/thomann/lib:/home/b/thomann/hd/hadoop/lib/native
    spark.executor.extraLibraryPath  /export/user/thomann/lib:/home/b/thomann/hd/hadoop/lib/native

Since I have $HADOOP_HOME=/home/b/thomann/hd/hadoop I there also include
the native libraries for HADOOP.

One also could add this on the command line:

    spark-shell \
    --conf spark.driver.extraLibraryPath /export/user/thomann/lib:/home/b/thomann/hd/hadoop/lib/native \
    --conf spark.executor.extraLibraryPath /export/user/thomann/lib:/home/b/thomann/hd/hadoop/lib/native\
    ...


Configuration

Configuring memory management can become the most difficult part when
working with liquidSVM for Spark. This is already for pure JVM
operations known to be challenging. However, in our case also there is
also the additional problem of C++ memory management. This is controlled
by the spark.yarn.executor.memoryOverhead configuration on YARN, which
we used.

We made the observation that it is beneficient to split every worker
node into several executors. Then one has to be carful to split the
available memory by the number of executors per node.

The executor memory needs to accomodate the data for all the cells in
that executor (controlled by spark.executor.memory). But it also needs
to have enough memory saved for the C++ structures (controlled by
spark.yarn.executor.memoryOverhead). If the latter cannot be made big
enough, consider using config.set("FORGET_TRAIN_SOLUTIONS","1") which
needs a little more time in the select phase to retrain the solutions.

Worked example

Here are some examples in $SPARK_HOME/conf/spark-defaults.conf on our
cluster. Every machine consists of two NUMA-nodes, each having 6
physical cores and 128GB memory.

For the driver and in general we use:

    spark.driver.memory              175g
    spark.driver.maxResultSize       25g
    spark.memory.fraction            0.875
    spark.network.timeout            120s

For 2 executors per node:

    spark.executor.memory            100g
    spark.yarn.executor.memoryOverhead  96000

For 4 executors per node

    spark.executor.memory            30g
    spark.yarn.executor.memoryOverhead  36000

For 12 executors per node

    spark.executor.memory            14g
    spark.yarn.executor.memoryOverhead  6000
