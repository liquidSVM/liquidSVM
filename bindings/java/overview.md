liquidSVM contains bindings to Ingo Steinwarts liquidSVM implementation.

# liquidSVM for Java

Welcome to the Java bindings for liquidSVM.

Summary:

  * Download <http://www.isa.uni-stuttgart.de/software/java/liquidSVM-java.zip>
  * Then to try it out issue on the command line on Linux
```
unzip liquidSVM-java.zip
cd liquidSVM-java
make lib
java -Djava.library.path=. -jar liquidSVM.jar
```
and on MacOS or Windows
```
unzip liquidSVM-java.zip
cd liquidSVM-java
java -Djava.library.path=. -jar liquidSVM.jar
```

Both liquidSVM and these bindings are provided under the AGPL 3.0 license.

## API Usage Example

The API can be investigated in the [javadoc](doc/index.html)
But to give you a heads up consider the File liquidSVM_java/Example.java:

```java
import de.uni_stuttgart.isa.liquidsvm.Config;
import de.uni_stuttgart.isa.liquidsvm.ResultAndErrors;
import de.uni_stuttgart.isa.liquidsvm.SVM;
import de.uni_stuttgart.isa.liquidsvm.SVM.LS;
import de.uni_stuttgart.isa.liquidsvm.LiquidData;

public class Example {

	public static void main(String[] args) throws java.io.IOException {
	
		String filePrefix = (args.length==0) ? "reg-1d" : args[0];
		
		// read comma separated training and testing data
		LiquidData data = new LiquidData(filePrefix);

		// Now train a least squares SVM on a 10by10 hyperparameter grid
		// and select the best parameters. The configuration displays
		// some progress information and specifies to only use two threads.
		SVM s = new LS(data.train, new Config().display(1).threads(2));

		// evaluate the selected SVM on the test features  
		double[] predictions = s.predict(data.testX);
		// or (since we have labels) do this and calculate the error
		ResultAndErrors result = s.test(data.test);
		
		System.out.println("Test error: " + result.errors[0][0]);
		for(int i=0; i<Math.min(result.result.length, 5); i++)
			System.out.println(predictions[i] + "==" + result.result[i][0]);

	}
}
```
The `reg-1d` data set is a artificial dataset provided by us. 

Compile and run this:
```
javac -classpath liquidSVM.jar Example.java
java -Djava.library.path=. -cp .:liquidSVM.jar Example reg-1d
```

## Using 


## Native Library Compilation

liquidSVM is implemented in C++ therefore a native library
needs to be compiled and included in the Java process.
Binaries for MacOS and Windows are included, however if it is
possible for you, we recommend you compile it for every machine
to get full performance. Two prerequisites have to be fulfilled:

1. the environment Variable `JAVA_HOME` has to be set
2. a Unix-type toolchain is available including make and a compiler like gcc or clang.

Then on the command line you can use different options:

`make native`
:	usually the fastest, but the resulting library is usually not portable to other machines. 
`make generic`
:	should be portable to most machines, yet slower (factor 2 to 4?) 
`make debug`
:	compiles with debugging activated (can be debugged e.g. with gdb) 
`make empty`
:	No special compilation options activated. 

To fulfill the prerequisites here follow some hints depending on your OS.

### Linux

If `echo $JAVA_HOME` gives nothing, in many cases it suffices to issue
```
export JAVA_HOME=/usr/lib/jvm/default-java
```
Which can be put e.g. into `~/.bashrc`.

### MacOS

The toolchain can be installed if Xcode is installed and then the optional
command line tools are installed from therein.

Usually `JAVA_HOME` is given under
```
export JAVA_HOME=/Library/Java/JavaVirtualMachines/*/Contents/Home
```

### Windows

To have `JAVA_HOME` correct use something like
```
set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_92
```
An easy possibility to install a Unix-type toolchain are the Rtools:

<https://cran.r-project.org/bin/windows/Rtools/Rtools33.exe>

They should be usable without installing R. We assume here:
```
path=%RTOOLS%\bin;%RTOOLS%\gcc-4.6.3\bin;%path% 
```
where `%RTOOLS%` is the location where they were installed (e.g. `C:\Rtools`).



