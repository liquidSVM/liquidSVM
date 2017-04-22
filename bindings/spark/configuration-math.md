### The configuration math

Unfortunately we do not know the memory consumption of all the involved data structures
and they have to be found by trial and error.
Hence let $rdd(n)\sim C_rdd n$ be the memory used to represent $n$ rows in Sparks memory.
Let $cpp(n,m)\sim C_cpp nm$ be the memory used by `libliquidsvm.so` for SVMs of $n$ samples which
it will split into cells of size $m$.

Let $Mem$ be the memory available on every machine, $JVM$ the memory for JVM objects, and $CPP$ the memory for C++ structures.
Further, let
$W$ be the number of worker machines, $E$ the number of executors per machine,
$N$ the entire training sample size and $K$ the number of coarse cells.
We assume that the most of the coarse cells have about $n=N/K$ samples.

Then we have the following natural constraints
\[
E * (JVM + CPP) <= Mem \\
rdd(N / (E*W) ) <= JVM (permanently) \\
cpp(N / (E*W) , m) <= CPP ()
\]
The middle inequality has to be fulfilled from beginning until the end of the whole training.
The last inequality needs to be fulfiiled only temporarily on the worker for any coarse cell that is trained,
and gets reset before the next coarse cell on that worker is trained.

This then gives
\[
E * (C_rdd + mC_cpp) N / (E*W) <= Mem
\iff C_rdd + m C_cpp <= Mem*W/N
\]


