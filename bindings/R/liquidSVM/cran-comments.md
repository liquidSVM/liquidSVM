## Test environments
* rocker/r-devel, R 3.5.1, R-devel
* macOS 10.14.1, R 3.5.0, R-devel
* Windows 10, R 3.5.2, R-devel

## R CMD check results
There were no ERRORs or WARNINGs. 

There was 1 non-trivial NOTE:

* checking installed package size ... NOTE
  installed size is  9.3Mb
  sub-directories of 1Mb or more:
    libs   8.4Mb

  liquidSVM provides a comprehensive, self-contained library
  with heavy inlining for fast execution and hence produces
  a large shared library. We hope this is not a problem.
