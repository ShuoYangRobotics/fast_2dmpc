fast_2dmpc
==========

This is the code repository of a research project I am doing. It is an algorithm that can solve 2D MPC problem.
The algorithm follows from the description of a recent paper written by me (availabe soon).
Basically it solves the following optimization problem
![2D MPC](http://ihome.ust.hk/~syangag/AQP.png)

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
1. Download the repository
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

5. Install CVX package, please refer to http://cvxr.com/cvx/download/
6. Test the installation by running the test code mympc_test

        >>mympc_test
        
Function Description
----------------------
The function can be called with following command

    [z,t] = mympc_step(sys, param, DX0, E0, R0);

where inputs are

    System description (sys structure):

    sys.A        :   system matrix A
    sys.B        :   input matrix B
    sys.C        :   system matrix C
    sys.D        :   input matrix D
    sys.Q        :   state cost matrix Q
    sys.R        :   input cost matrix R
    sys.dxmax    :   state upper limits dx_{max}
    sys.dxmin    :   state lower limits dx_{min}
    sys.emax     :   input upper limits e_{max}
    sys.emin     :   input lower limits e_{min}
    sys.rmax     :   input upper limits r_{max}
    sys.rmin     :   input lower limits r_{min}
    sys.n        :   number of states
    sys.m        :   number of inputs
    sys.dx0      :   intial state
    sys.prev_e   :   errors of each time instance in the previous batch

    MPC parameters (params structure):

    param.T        :   MPC horizon T
    param.c_type   :   type of constraint on input r
    param.kappa    :   barrier parameter
    param.niters   :   number of newton iterations
    param.quiet    :   no output to display if true

    Other inputs
    DX0   :   warm start DX trajectory (n by T+1 matrix)
    E0    :   warm start E trajectory (n by T+1 matrix)
    R0    :   warm start R trajectory (n by T+2 matrix)
    
Outputs are

    z    :    Solved output sequence
    t    :    CPU time used for solving the problem
    
    
    