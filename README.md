fast_2dmpc
==========

This is the code repository of a research project I am doing. It is an algorithm that can solve 2D MPC problem.
The algorithm follows from the description of a recent paper written by me (availabe soon).

Files
-----
+ blas.h  	    
    Linear algebra library

+ lapack.h	    
  Linear algebra library

+ mympc_step.c	
  The main algorithm code

+ mympc_test.m  
  A testing program

Install
-------
1. Fork the repository
2. Start MATLAB and change to the directory containing the source code file
3. At the MATLAB command prompt type

        >> mex -setup
4.Compile mympc_sim.c using

        >> mex mympc_sim.c -lblas -llapack
    
   If you want to link your own libraries, you can. For example,
   to link the libraries libacml.a, libacml_mv.a, and libgfortran.a 
   in the directory /opt/acml/lib, use the option '-L' to specify the 
   library search path and the option '-l' to specify the individual 
   libraries, as in
    
        >> mex mympc_sim.c -L/opt/acml/lib -lacml -lacml_mv -lgfortran
       
5. Test the installation by running the test code mympc_test

        >>mympc_test