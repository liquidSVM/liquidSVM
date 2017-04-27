## Test environments
* debian 8.6, R 3.3.2, R 3.4.0, R-devel
* macOS 10.12, R 3.4.0, R-devel
* Windows 8.1, R 3.4.0

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
