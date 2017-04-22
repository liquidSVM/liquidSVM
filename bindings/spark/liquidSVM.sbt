  name := "liquidSVM-spark"

  version := "1.0"

  scalaVersion := "2.10.5"

  libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.1" % "provided"
  libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.1" % "provided"

  autoScalaLibrary := false
