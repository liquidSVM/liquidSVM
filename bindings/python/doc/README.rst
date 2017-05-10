liquidSVM for Python
====================

Welcome to the Python bindings for liquidSVM.

Summary:

-  Install it using any of the following variants:

   ::

       pip install --user --upgrade liquidSVM
       easy_install --user --upgrade liquidSVM

-  If you want to compile liquidSVM for your machine download
   http://www.isa.uni-stuttgart.de/software/python/liquidSVM-python.tar.gz.
   For Windows there are binaries at
   `liquidSVM-python.win-amd64.zip <http://www.isa.uni-stuttgart.de/software/python/liquidSVM-python.win-amd64.zip>`__,
   for Mac at
   `liquidSVM-python.macosx.tar.gz <http://www.isa.uni-stuttgart.de/software/python/liquidSVM-python.macosx.tar.gz>`__

Then to try it out issue on the command line

::

    python -m liquidSVM covtype.1000 mc --display=1

    **NOTE**: it might be possible that there is a problem with the last
    line if there are files called ``liquidSVM*`` in the current
    directory, so change to some other or a newly created one.

Or use it in an interactive shell

.. code:: python

    from liquidSVM import *
    model = mcSVM(iris, iris_labs, display=1,threads=2)
    result, err = model.test(iris, iris_labs)
    result = model.predict(iris)

    reg = LiquidData('reg-1d')
    model = lsSVM(reg.test, display=1)
    result, err = model.test(reg.test)

More Information can be found in the
`demo <http://www.isa.uni-stuttgart.de/software/python/demo.html>`__
`[jupyter
notebook] <http://www.isa.uni-stuttgart.de/software/python/demo.ipynb>`__
and in

.. code:: python

    from liquidSVM import *
    help(SVM)
    help(doc.configuration)

Both liquidSVM and these bindings are provided under the AGPL 3.0
license.

Native Library Compilation
--------------------------

liquidSVM is implemented in C++ therefore a native library needs to be
compiled and included in the Python process. Binaries for Windows are
included, however if it is possible for you, we recommend you compile it
for every machine to get full performance.

To set compiler options use the the environment variable
``LIQUIDSVM_CONFIGURE_ARGS``. The first word in it can be any of the
following:

``native``
    usually the fastest, but the resulting library is usually not
    portable to other machines.
``generic``
    should be portable to most machines, yet slower (factor 2 to 4?)
``debug``
    compiles with debugging activated (can be debugged e.g. with gdb)
``empty``
    No special compilation options activated.

The remainder of the environment variable will be passed to the
compiler. Extract
http://www.isa.uni-stuttgart.de/software/python/liquidSVM-python.tar.gz
and change into the directory. On Linux and MacOS X command line use for
instance:

::

    LIQUIDSVM_CONFIGURE_ARGS="native -mavx2" python setup.py bdist
    LIQUIDSVM_CONFIGURE_ARGS=generic python setup.py bdist

*MacOS*:
    Install Xcode and then the optional command line tools are installed
    from therein.

*Windows*:
    If you have VisualStudio installed then you should have an
    environment variable like ``%VS90COMNTOOLS%`` (for VisualStudio
    2015). Still it seems that setup.py needs to have this information
    in ``%VS90COMNTOOLS%`` so copy that environment variable or use for
    example:

    ::

        set VS90COMNTOOLS=%VS140COMNTOOLS%

        **Note:** At the moment the Visual Studio for Python only gives
        Version 9.0 and this is too old for compilation.


