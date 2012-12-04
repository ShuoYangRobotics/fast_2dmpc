#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "blas.h"
#include "lapack.h"
#include "mex.h"

/* constant and global variable */
const int INT_ONE = 1;
const int INT_N_ONE = 1;
const int INT_TWO = 2;
const int INT_THREE = 3;
const int INT_SEVEN = 7;
const double DOU_HALF = 0.5;
const double DOU_ONE = 1;
const double DOU_TWO = 2;
const double DOU_ZERO = 0;
const double DOU_N_ONE = -1;
int quiet = 0; //use to control wheather to print iteration info or not

/* function declaration */
/*  Function to print matrix
 *  Inputs:
 * 		A : matrix
 *      m,n : matrix dimension
 */ 
void printmat(double *A, int m, int n);
/* Core function. Solve best z 
 * 
 *
 */ 
void fmpcsolve(double *A, double *B, double *C, double *D, 
		  double *At, double *Bt, double *Ct, double *Dt,
		  double *eyen, double *eye2n, double *eyem,
		  double *Q, double *R,
		  double *curr_DX0, double *prev_E0,
		  double *h1, double *h2, double *z0,
	      double *zmax, double *zmin,
		  int T, int n, int m,
		  int dim0, int dim1, int dim2,
		  int niters, double kappa, double c_type);

void rdrp1(double *z, double *nu,
	      double *Htwo, double *G, double *tmpG, double *phi, 
	      double *h1, double *h2, double *b, double *d1, double *d2,
		  double *PTd1, double *PTd2,
		  int dim0, int dim1, int dim2,
		  int T, int n, int m,
		  double kappa, double *rd, double *rp); 

void rdrp2(double *z, double *nu,
	      double *Htwo, double *G, double *tmpG, double *phi,
	      double *h1, double *h2, double *b, double *d1, double *d2,
		  double *PTd1, double *PTd2,
		  int dim0, int dim1, int dim2,
		  int T, int n, int m,
		  double kappa, double *rd, double *rp);
		
void dnudz1(double* Htwo, double *G, double *tmpG, double *phi, 
			double *PTd1, double *PTd2,
			int dim0, int dim1, int dim2, double kappa,
			double *rd, double *rp, double *dz, double *dnu);

void dnudz2(double* Htwo, double *G, double *tmpG, double *phi, 
			double *A, double *B, double *C, double *D,  double *CB, double *CA,
			double *At, double *Bt, double *Ct, double *Dt, 
			double *eyen, double *eye2n, double *eyem,
			double *CBt, double *CAt,
			double *PTd1, double *PTd2, double *be,
			int dim0, int dim1, int dim2, double kappa,
			double *PhiQ, double *PhiR, double *PhiS, double *Y, double *L,
			//six tmp storage
			double *tmps, double *tmps1, double *tmps2, 
			double *tmps3, double *tmps4, double *tmps5,
			//for computation storage
			double *tmps6, double *tmpmm, double *tmpnn,
			double *tmpns, double *tmpns1,
			//temp storage for compute L
			double *tmpll1,	double *tmpll2, double *tmpll3,
			int T, int n, int m,
			double *rd, double *rp, double *dz, double *dnu);

void resdresp(double *rd, double *rp, 
		      int dim1, int dim2,
			  double *resd, double *resp, double *res);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* problem setup */
	/*
	 * This is important! It determines the type of constraint
	 * c_type = 1: fluctuation constraint
	 * c_type = 0: box constraint
	 */
	int c_type; 
    int i, j, m, n, T, niters;
	int dim0, dim1, dim2; //dimensions of big matrices
    double kappa;
    double *dptr, *dptr1, *dptr2;
	double *A, *B, *C, *D;
	double *At, *Bt, *Ct, *Dt;
	double *Q, *R;
	double *rmax, *rmin, *dxmax, *dxmin, *emax, *emin, *x;
	double *h1, *h2;  
	double *eyen, *eye2n, *eyem;
	double *z, *zmax, *zmin, *zmaxp, *zminp; 
	double *DX0, *E0, *R0, *prev_E0, *curr_DX0;
	double *z_out;
    double *telapsed;
    clock_t t1, t2;
	
	printf("Entered the mex function");
	/* inputs */
	n = (int)mxGetScalar(mxGetField(prhs[0],0,"n"));
    m = (int)mxGetScalar(mxGetField(prhs[0],0,"m"));
    A = mxGetPr(mxGetField(prhs[0],0,"A"));
    B = mxGetPr(mxGetField(prhs[0],0,"B"));
    C = mxGetPr(mxGetField(prhs[0],0,"C"));
    D = mxGetPr(mxGetField(prhs[0],0,"D"));
    Q = mxGetPr(mxGetField(prhs[0],0,"Q"));
    R = mxGetPr(mxGetField(prhs[0],0,"R"));
	curr_DX0 = mxGetPr(mxGetField(prhs[0],0,"dx0"));
	//values of error in the previous batch T items * n
    prev_E0 = mxGetPr(mxGetField(prhs[0],0,"prev_e"));  
    
    rmax = mxGetPr(mxGetField(prhs[0],0,"rmax"));
    rmin = mxGetPr(mxGetField(prhs[0],0,"rmin"));
    dxmax = mxGetPr(mxGetField(prhs[0],0,"dxmax"));
    dxmin = mxGetPr(mxGetField(prhs[0],0,"dxmin"));
    emax = mxGetPr(mxGetField(prhs[0],0,"emax"));
    emin = mxGetPr(mxGetField(prhs[0],0,"emin"));

	c_type = (int)mxGetScalar(mxGetField(prhs[1],0,"c_type"));

    T = (int)mxGetScalar(mxGetField(prhs[1],0,"T"));
    kappa = (double)mxGetScalar(mxGetField(prhs[1],0,"kappa"));
    niters = (int)mxGetScalar(mxGetField(prhs[1],0,"niters"));
    quiet = (int)mxGetScalar(mxGetField(prhs[1],0,"quiet"));
		
	//they form z0	
    DX0 = mxGetPr(prhs[2]); //initial value for dx T+1 items * n
	E0 = mxGetPr(prhs[3]);  //initial value for e  T+1 items * n 
    R0 = mxGetPr(prhs[4]);  //initial value for r  T+2 items * m
	
	printf("Obtained inputs\n");
    /* outputs */
    plhs[0] = mxCreateDoubleMatrix((T+1)*(2*n+m)+m,1,mxREAL);

    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    z_out = mxGetPr(plhs[0]);   //dim1 items
    telapsed = mxGetPr(plhs[1]);

    dim0 = (T+1)*(2*n+m); 
	dim1 = dim0+m;			//dim1 is the dimension for most big matrices
	dim2 = (T+1)*(2*n);
	
    At = malloc(sizeof(double)*n*n);
    Ct = malloc(sizeof(double)*n*n);
    Bt = malloc(sizeof(double)*n*m);
    Dt = malloc(sizeof(double)*n*m);

    eyen = malloc(sizeof(double)*n*n);
	eye2n = malloc(sizeof(double)*n*n*4);
    eyem = malloc(sizeof(double)*m*m);

    z = malloc(sizeof(double)*dim1);
	zmax = malloc(sizeof(double)*dim1);
	zmaxp = malloc(sizeof(double)*dim1);
	zmin = malloc(sizeof(double)*dim1);
	zminp = malloc(sizeof(double)*dim1);
    h1 = malloc(sizeof(double)*dim0);
    h2 = malloc(sizeof(double)*dim0);
    
    /* create identity matrix with dimension n*n and m*m */
    dptr = eyen;
    for (i = 0; i < n*n; i++)
    {
        *dptr = 0;
        dptr++;
    }
    dptr = dptr-n*n;
    for (i = 0; i < n; i++)
    {
        *dptr = 1;
        dptr = dptr+n+1;
    }
	dptr = eye2n;
    for (i = 0; i < 4*n*n; i++)
    {
        *dptr = 0;
        dptr++;
    }
    dptr = dptr-4*n*n;
    for (i = 0; i < 2*n; i++)
    {
        *dptr = 1;
        dptr = dptr+2*n+1;
    }

    dptr = eyem;
    for (i = 0; i < m*m; i++)
    {
        *dptr = 0;
        dptr++;
    }
    dptr = dptr-m*m;
    for (i = 0; i < m; i++)
    {
        *(dptr+i*m+i) = 1;
    }
	
    dptr = z; //put DX0 ,E0 and R0 into z as initial value 
    for (i = 0; i < T+1; i++)
    {
        for (j = 0; j < m; j++)
        {
            *dptr = *(R0+j);
            dptr++;
        }
        for (j = 0; j < n; j++)
        {
            *dptr = *(DX0+j);
            dptr++; 
        }
		for (j = 0; j < n; j++)
        {
            *dptr = *(E0+j);
            dptr++; 
        }
    } 
 	for (j = 0; j < m; j++)
    {
        *dptr = *(R0+j);
        dptr++;
    }

	dptr1 = zmax;
	dptr2 = zmin;
	for (i = 0; i < T+1; i++)
    {
        for (j = 0; j < m; j++)
        {
            *dptr1 = *(rmax+j);
			*dptr2 = *(rmin+j);
            dptr1++;dptr2++;
        }
        for (j = 0; j < n; j++)
        {
            *dptr1 = *(dxmax+j);
	        *dptr2 = *(dxmin+j);
			dptr1++;dptr2++; 
        }
		for (j = 0; j < n; j++)
        {
            *dptr1 = *(emax+j);
	        *dptr2 = *(emin+j);
			dptr1++;dptr2++;
        }
    } 
 	for (j = 0; j < m; j++)
    {
        *dptr1 = *(rmax+j);
		*dptr2 = *(rmin+j);
        dptr1++;dptr2++;
    }
	
	/* project z 
     * z may not feasible, so project it into the box constraint
     * this z is the initial value of z in fmpc solve
	 */
    for (i = 0; i < dim1; i++) zminp[i] = zmin[i] + 0.01*(zmax[i]-zmin[i]);
    for (i = 0; i < dim1; i++) zmaxp[i] = zmax[i] - 0.01*(zmax[i]-zmin[i]);    
    for (i = 0; i < dim1; i++) z[i] = z[i] > zmaxp[i] ? zmaxp[i] : z[i];
    for (i = 0; i < dim1; i++) z[i] = z[i] < zminp[i] ? zminp[i] : z[i];
	
	/* formulate h1, h2 */
	dptr1 = h1;
	dptr2 = h2;
	for (i = 0; i < T+1; i++)
    {
        for (j = 0; j < m; j++)
        {
            *dptr1 = *(rmax+j);
			*dptr2 = *(rmin+j);
            dptr1++;dptr2++;
        }
        for (j = 0; j < n; j++)
        {
            *dptr1 = *(dxmax+j);
	        *dptr2 = *(dxmin+j);
			dptr1++;dptr2++; 
        }
		for (j = 0; j < n; j++)
        {
            *dptr1 = *(emax+j);
	        *dptr2 = *(emin+j);
			dptr1++;dptr2++;
        }
    }
	
	    /* At, Bt, Ct, Dt */
	//get transpose of A, B, C, and D
	F77_CALL(dgemm)("t","n",&n,&n,&n,&DOU_ONE,A,&n,eyen,&n,&DOU_ZERO,At,&n);
	F77_CALL(dgemm)("n","t",&m,&n,&m,&DOU_ONE,eyem,&m,B,&n,&DOU_ZERO,Bt,&m);
	F77_CALL(dgemm)("t","n",&n,&n,&n,&DOU_ONE,C,&n,eyen,&n,&DOU_ZERO,Ct,&n);
	F77_CALL(dgemm)("n","t",&m,&n,&m,&DOU_ONE,eyem,&m,D,&n,&DOU_ZERO,Dt,&m);

	printf("Pre-process the inputs\n");
    t1 = clock();
    fmpcsolve(A, B, C, D, At, Bt, Ct, Dt,
			  eyen, eye2n, eyem,
			  Q, R,
			  curr_DX0, prev_E0,
			  h1, h2, z,
			  zmax, zmin,
			  T, n, m,
			  dim0, dim1, dim2,
			niters, kappa, c_type);
    t2 = clock();
    *telapsed = (double)(t2-t1)/(CLOCKS_PER_SEC);
	
	printf("Algorithm completed\n");
	dptr = z_out; dptr1 = z;
	/* put solved z into output trajectory matrix */
    for (i = 0; i < dim1; i++)
    {

            *dptr = *dptr1;
            dptr++;dptr1++;
    }  
 
	free(eyen); free(eyem); free(eye2n);
	free(At); free(Bt); free(Ct); free(Dt);
    free(z); free(zmax); free(zmin); free(zmaxp); free(zminp);
	free(h1); free(h2);
    return;
}

void printmat(double *A, int m, int n)
{
    double *dptr;
    int j, i;
    dptr = A;
    for (j = 0; j < m; j++)
    {
        for (i = 0; i < n; i++)
        {
            printf("%5.4f\t", *(dptr+m*i+j));
        }
        printf("\n");
    }
    printf("\n");
    return;
}

void fmpcsolve(double *A, double *B, double *C, double *D, 
		  double *At, double *Bt, double *Ct, double *Dt,
		  double *eyen, double *eye2n, double *eyem,
		  double *Q, double *R,
		  double *curr_DX0, double *prev_E0,
		  double *h1, double *h2, double *z0,
	      double *zmax, double *zmin,
		  int T, int n, int m,
		  int dim0, int dim1, int dim2,
		  int niters, double kappa, double c_type)
{
	int i, j, k, iter, cont;
	long l;
	int maxiter = niters;
	double min_kappa = 1;
	double ns, cost;
	double alpha = 0.01;
    double beta = 0.95;
    double tol = 0.1;
	double *s;
	double resd, resp, res;
	double *b;
	double *rd, *rp, *PTd1, *PTd2, *be;
	double *z, *dz, *newz;
	double *nu, *dnu, *newnu;
	double *tmpn1,*tmpn2, *CB, *CA, *CBt, *CAt;
	double *tmpm;
	double *dptr, *dptr1, *dptr2, *dptr3,*curr_col;
	double *Htwo, *G, *tmpG, *phi;
	double *d1, *d2;
	double *PhiQ, *PhiR, *PhiS;
	double *tmps, *tmps1, *tmps2, *tmps3, *tmps4, *tmps5, *tmps6;
	double *tmpns, *tmpns1, *tmpmm, *tmpnn;
	double *tmpll1, *tmpll2, *tmpll3;
	double *Y, *L;

	d1 = malloc(sizeof(double)*dim0);
	d2 = malloc(sizeof(double)*dim0);
	
	rd = malloc(sizeof(double)*dim1);
	PTd1 = malloc(sizeof(double)*dim1);
	PTd2 = malloc(sizeof(double)*dim1);
	rp = malloc(sizeof(double)*dim2);
	be = malloc(sizeof(double)*dim2);
	
	b = malloc(sizeof(double)*dim2);
	for (i = 0; i< dim2; i++) b[i] = 0;

	CB = malloc(sizeof(double)*n*m);
	CA = malloc(sizeof(double)*n*n);
	CBt = malloc(sizeof(double)*n*m);
	CAt = malloc(sizeof(double)*n*n);
		
	tmpn1 = malloc(sizeof(double)*n);
	tmpn2 = malloc(sizeof(double)*n);
	tmpm = malloc(sizeof(double)*m);
	
	z = malloc(sizeof(double)*dim1);
	s = malloc(sizeof(double)*dim1);
	dz = malloc(sizeof(double)*dim1);
	newz = malloc(sizeof(double)*dim1);
	
	nu = malloc(sizeof(double)*dim2);
	dnu = malloc(sizeof(double)*dim2);
	newnu = malloc(sizeof(double)*dim2);
	
	phi = malloc(sizeof(double)*dim1*dim1);	
	Htwo = malloc(sizeof(double)*dim1*dim1);
	G = malloc(sizeof(double)*dim2*dim1);
	tmpG = malloc(sizeof(double)*dim2*dim1);
	for (i = 0; i< dim2*dim1; i++) G[i] = 0;
	for (i = 0; i< dim1*dim1; i++) Htwo[i] = 0;
	
	PhiQ = malloc(sizeof(double)*n*n*(T+1));
    PhiR = malloc(sizeof(double)*m*m*(T+2));
    PhiS = malloc(sizeof(double)*n*n*(T+1));

	Y = malloc(sizeof(double)*12*n*n*(T+1));
	L = malloc(sizeof(double)*dim2*dim2);

	tmps = malloc(sizeof(double)*n*n);
	tmps1 = malloc(sizeof(double)*n*n);
	tmps2 = malloc(sizeof(double)*n*n);
	tmps3 = malloc(sizeof(double)*n*n);
	tmps4 = malloc(sizeof(double)*n*n);
	tmps5 = malloc(sizeof(double)*n*n);
	tmps6 = malloc(sizeof(double)*n*n);
	tmpns = malloc(sizeof(double)*n*m);
	tmpns1 = malloc(sizeof(double)*n*m);
	tmpll1 = malloc(sizeof(double)*4*n*n);
	tmpll2 = malloc(sizeof(double)*4*n*n);
	tmpll3 = malloc(sizeof(double)*4*n*n);
	
	tmpmm = malloc(sizeof(double)*m*m);
	tmpnn = malloc(sizeof(double)*n*n);
	//construct b
	//tmpn1  = A*curr_DX0
	//tmpn2 = -CA*curr_DX0
	F77_CALL(dgemv)("n",&n,&n,&DOU_ONE,A,&n,curr_DX0,&INT_ONE,&DOU_ZERO,tmpn1,&INT_ONE);
	F77_CALL(dgemv)("n",&n,&n,&DOU_N_ONE,C,&n,tmpn1,&INT_ONE,&DOU_ZERO,tmpn2,&INT_ONE);

	dptr = b;
	dptr1 = b+n;
	for (j = 0; j < n; j++)
	{
		*dptr = *(tmpn1+j);
		*dptr1 = *(tmpn2+j);
		dptr++; dptr1++;
	}
	dptr = b;
	for (i = 0; i < T+1; i++)
	{
		dptr += n;
		for (j = 0; j < n; j++)
		{
			*dptr += *(prev_E0+i*n+j);
			dptr++;
		}
	}
	
	//compute CB, CA
	F77_CALL(dgemm)("n","n",&n,&m,&n,&DOU_ONE,C,&n,B,&n,&DOU_ZERO,CB,&n);
	F77_CALL(dgemm)("n","n",&n,&n,&n,&DOU_ONE,C,&n,A,&n,&DOU_ZERO,CA,&n);
	//get transpose of CB, CA
	F77_CALL(dgemm)("t","n",&n,&n,&n,&DOU_ONE,CA,&n,eyen,&n,&DOU_ZERO,CAt,&n);
	F77_CALL(dgemm)("n","t",&m,&n,&m,&DOU_ONE,eyem,&m,CB,&n,&DOU_ZERO,CBt,&m);
	
	//z = z0
	dptr = z;
	for (i=0; i < dim1; i++) *dptr++ = *(z0+i);
	//nu = nu0  !!!!!!!!!! choose 5, is that OK? !!!!!!!!!!!!!!!!
	dptr = nu;
	for (i=0; i < dim2; i++) nu[i] = 10;
	
	/*
	 * construct G
	 */
	dptr = curr_col=G; 
	//block row1
	dptr1 = B;dptr2 = CB;
	for(i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = -*(dptr1);
			dptr++;dptr1++;	
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *(dptr2);
			dptr++;dptr2++;	
		}
		dptr = G+(i+1)*dim2;
	}
	curr_col = G+m*dim2;
	//block row2
	dptr = curr_col;
	dptr1 = A;dptr2 = CA;
	for(i = 0; i < n; i++)
	{
		dptr+=i;*(dptr) = 1;
		dptr+=n-i;
		dptr+=n;
		for (j = 0; j < n; j++)
		{
			*dptr = -*(dptr1);
			dptr++;dptr1++;	
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *(dptr2);
			dptr++;dptr2++;	
		}
		dptr = curr_col+(i+1)*dim2;
	}
	curr_col += n*dim2;
	//block row 3
	dptr = curr_col;
	for(i = 0; i < n; i++)
	{
		dptr+=n+i;*(dptr) = 1;
		dptr = curr_col+(i+1)*dim2;
	}
	curr_col += n*dim2;
	//block rows 
	for (i = 0; i < (T-1); i++)
	{
		dptr = curr_col;
		dptr1 = B; dptr2 = CB; dptr3 = D;
		for (j = 0; j < m; j++)
		{
			dptr+=i*2*n+n;
			for (k = 0; k < n; k++)
			{
				*dptr = *(dptr3);
				dptr++; dptr3++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = -*(dptr1);
				dptr++; dptr1++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *(dptr2);
				dptr++; dptr2++;
			}
			dptr = curr_col+(j+1)*dim2;
		}
		curr_col +=m*dim2;
		dptr = curr_col;
		dptr1 = A; dptr2 = CA; 
		for (j = 0; j < n; j++)
		{
			dptr+=i*2*n+n*2;
			dptr+=j;*(dptr) = 1; dptr+=n-j;
			for (k = 0; k < n; k++)
			{
				*dptr = -*(dptr1);
				dptr++; dptr1++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *(dptr2);
				dptr++; dptr2++;
			}
			dptr = curr_col+(j+1)*dim2;
		}
		curr_col +=n*dim2;
		dptr = curr_col;
		for (j = 0; j < n; j++)
		{
			dptr+=i*2*n+n*2+n;
			dptr+=j;*(dptr) = 1;
			dptr = curr_col+(j+1)*dim2;
		}
		curr_col +=n*dim2;
	}
	//last block row
	dptr = curr_col;
	dptr1 = B; dptr2 = CB; dptr3 = D;
	for (j = 0; j < m; j++)
	{
		dptr+=(T-1)*2*n+n;
		for (k = 0; k < n; k++)
		{
			*dptr = *(dptr3);
			dptr++; dptr3++;
		}
		for (k = 0; k < n; k++)
		{
			*dptr = -*(dptr1);
			dptr++; dptr1++;
		}
		for (k = 0; k < n; k++)
		{
			*dptr = *(dptr2);
			dptr++; dptr2++;
		}
		dptr = curr_col+(j+1)*dim2;
	}
	curr_col +=m*dim2;
	dptr = curr_col;
	for (j = 0; j < n; j++)
	{
		dptr+=(T-1)*2*n+n*2;
		dptr+=j;*(dptr) = 1;
		dptr = curr_col+(j+1)*dim2;
	}
	curr_col +=n*dim2;
	dptr = curr_col;
	for (j = 0; j < n; j++)
	{
		dptr+=T*2*n+n;
		dptr+=j;*(dptr) = 1;
		dptr = curr_col+(j+1)*dim2;
	}
	curr_col +=n*dim2;
	dptr = curr_col;
	dptr1 = D;
	for (j = 0; j < m; j++)
	{
		dptr+=T*2*n+n;
		for (k = 0; k < n; k++)
		{
			*dptr = *(dptr1);
			dptr++; dptr1++;
		}
		dptr = curr_col+(j+1)*dim2;
	}
	
	/*
	 * construct Htwo (2H)
	 */
	curr_col = Htwo;
	for (i = 0; i < T+1; i++)
	{
		dptr = curr_col; dptr1 = R;
		for (j = 0; j < m; j++)
		{
			dptr+= i*(2*n+m);
			for (k = 0; k < m; k++)
			{
				*dptr = *(dptr1)*2;
				dptr++; dptr1++;
			}
			dptr = curr_col+(j+1)*dim1;
		}
		curr_col +=(m+n)*dim1;
		dptr = curr_col; dptr1 = Q;
		for (j = 0; j < n; j++)
		{
			dptr+= i*(2*n+m)+(m+n);
			for (k = 0; k < n; k++)
			{
				*dptr = *(dptr1)*2;
				dptr++; dptr1++;
			}
			dptr = curr_col+(j+1)*dim1;
		}
		curr_col +=n*dim1;
	}
	dptr = curr_col; dptr1 = R;
	for (j = 0; j < m; j++)
	{
		dptr+= (T+1)*(2*n+m);
		for (k = 0; k < m; k++)
		{
			*dptr = *(dptr1)*2;
			dptr++; dptr1++;
		}
		dptr = curr_col+(j+1)*dim1;
	}
	
	printf("Construction of problem matrices done.\n");
	if (quiet == 0)   
        printf("\nobjective  iteration \t step \t\t rd \t\t\t rp\n");
	while (kappa>min_kappa)
	{
		for (iter = 0; iter < maxiter; iter++)
		{
			//get rd rp dz dnu
			if (c_type == 0)
			{
				rdrp1(z, nu,
				     Htwo, G, tmpG, phi,
				     h1, h2, b, d1, d2,
				     PTd1, PTd2,
					 dim0, dim1, dim2,
					 T, n, m,
					 kappa, rd, rp);
				dnudz1(Htwo, G, tmpG, phi, 
					  PTd1, PTd2,
					  dim0, dim1, dim2, kappa,
					  rd, rp, dz, dnu);
			}
			else
			{
				rdrp2(z, nu,
				     Htwo, G, tmpG, phi,
				     h1, h2, b, d1, d2,
				     PTd1, PTd2,
					 dim0, dim1, dim2,
					 T, n, m,
					 kappa, rd, rp);
				dnudz2(Htwo, G, tmpG, phi, 
						A, B, C, D, CB, CA,
						At, Bt, Ct, Dt, 
						eyen, eye2n, eyem,
						CBt, CAt,
						PTd1, PTd2, be,
						dim0, dim1, dim2, kappa,
						PhiQ, PhiR, PhiS, Y, L,
						tmps, tmps1, tmps2, 
						tmps3, tmps4, tmps5,
						tmps6, tmpmm, tmpnn,
						tmpns, tmpns1,
						tmpll1,	tmpll2, tmpll3,
						T, n, m,
						rd, rp, dz, dnu);
			}	

			resdresp(rd, rp, 
					 dim1, dim2,
					 &resd, &resp, &res);
			if (res < tol) break;
			for (i = 0; i < dim1; i++) s[i] = 1; 
	        /* feasibility search */
	        while (1)
	        {
				cont = 0;
	            dptr = z; dptr1 = dz; dptr2 = zmax; dptr3 = zmin;
	            for (i = 0; i < dim1; i++)
	            {
	                if (*dptr+*(s+i)*(*dptr1) >= *dptr2) 
					{
						cont = 1;
						*(s+i) *= beta;
					}
	                if (*dptr+*(s+i)*(*dptr1) <= *dptr3) 
					{
						cont = 1;
						*(s+i) *= beta;
					}
	                dptr++; dptr1++; dptr2++; dptr3++;
	            }
	            if (cont == 0)
	                break;
	        }
		
			//printmat(s,1,dim1);
			//printmat(z,1,dim1);
			//printmat(dnu,1,dim2);
			dptr = z; dptr2 = dz;
	        for (i = 0; i < dim1; i++)
	        {
	            *dptr += s[i]*(*dptr2);
	            dptr++; dptr2++;
	        }
			dptr = nu; dptr2 = dnu;
	        for (i = 0; i < dim2; i++)
	        {
	            *dptr += s[i]*(*dptr2);
	            dptr++; dptr2++;
	        }
			
			F77_CALL(dgemv)("n",&dim1,&dim1,&DOU_HALF,Htwo,&dim1,z,&INT_ONE,&DOU_ZERO,newz,&INT_ONE);
			cost = F77_CALL(ddot)(&dim1, z,&INT_ONE, newz,&INT_ONE);
			printf("%5.4f\t",cost);
			if (quiet == 0)
	        {
				ns = F77_CALL(dnrm2)(&dim2,s,&INT_ONE);
	            printf("    %d \t\t %5.4f \t %0.5e \t\t %0.5e\n"
					,iter,ns,resd,resp);
	        }
		}
		kappa = kappa/5;
	}
	dptr = z0; dptr2 = z;
    for (i = 0; i < dim1; i++)
    {
        *dptr = *dptr2;
        dptr++; dptr2++;
    }
	free(d1); free(d2);
	free(rd); free(PTd1); free(PTd2); free(rp); free(be);
	free(b); free(tmpn1); free(tmpn2); free(tmpm);
	free(z); free(dz); free(newz); free(s);
	free(nu); free(dnu); free(newnu);
	free(phi); free(Htwo); free(G); free(tmpG);
	free(CA); free(CB); free(CAt); free(CBt);
	free(PhiQ); free(PhiR); free(PhiS); free(Y); free(L); 
	free(tmpns); free(tmpns1);
	free(tmps); free(tmps1); free(tmps2); free(tmps3); free(tmps4); 
	free(tmps5); free(tmps6);
	free(tmpll1); free(tmpll2); free(tmpll3); free(tmpnn); free(tmpmm); 
	return;
}

void rdrp1(double *z, double *nu,
	      double *Htwo, double *G, double *tmpG, double *phi,
	      double *h1, double *h2, double *b, double *d1, double *d2,
		  double *PTd1, double *PTd2,
		  int dim0, int dim1, int dim2,
		  int T, int n, int m,
		  double kappa, double *rd, double *rp)
{
	int i,j,curr_blk;
	
	//d1 d2 dim0
	//z dim1 
	//h1 h2 dim0
	//PTd1 PTd2 dim1
	//calculate d1 d2
	for (i = 0; i < T+1; i++)
	{
		curr_blk = i*(2*n+m);
		for (j = 0; j < m; j++)
			d1[curr_blk+j] = 
				1/( h1[curr_blk+j] - (z[curr_blk+2*n+m+j]-z[curr_blk+j]) );
		for (j = 0; j < n; j++)
			d1[curr_blk+m+j] = 1/(h1[curr_blk+m+j] - z[curr_blk+m+j]);
		for (j = 0; j < n; j++)
			d1[curr_blk+m+n+j] = 1/(h1[curr_blk+m+n+j] - z[curr_blk+m+n+j]);	
			
		for (j = 0; j < m; j++)
			d2[curr_blk+j] = 1/(h2[curr_blk+j] -
			(z[curr_blk+j]-z[curr_blk+2*n+m+j])) ;
		for (j = 0; j < n; j++)
			d2[curr_blk+m+j] = 1/(h2[curr_blk+m+j] + z[curr_blk+m+j]);
		for (j = 0; j < n; j++)
			d2[curr_blk+m+n+j] = 1/(h2[curr_blk+m+n+j] + z[curr_blk+m+n+j]);		 
	}
	//printmat(d1,1,dim0);
	//printmat(d2,1,dim0);
	/*
	 *  Compute rd and rp
	 */
	//rd = 2Hz
	F77_CALL(dgemv)("n",&dim1,&dim1,&DOU_ONE,Htwo,&dim1,z,&INT_ONE,&DOU_ZERO,rd,&INT_ONE);
	//rd = rd + G^Tnu
	F77_CALL(dgemv)("t",&dim2,&dim1,&DOU_ONE,G,&dim2,nu,&INT_ONE,&DOU_ONE,rd,&INT_ONE);
	
	//rd = rd+kappa*(P1^Td1+P2^Td2)
	for (j = 0; j < 2*n+m; j++)
	{
		rd[j] +=kappa*(d1[j]-d2[j]);
		PTd1[j] = d1[j];
		PTd2[j] = -d2[j];
	}
	for (i = 1; i < T+1; i++)
	{
		curr_blk = i*(2*n+m);
		for (j = 0; j < m; j++)
		{
			rd[curr_blk+j] +=kappa*((d1[curr_blk-2*n-m+j]-d1[curr_blk+j])-
									(d2[curr_blk-2*n-m+j]-d2[curr_blk+j]));
			PTd1[curr_blk+j] = d1[curr_blk-2*n-m+j]-d1[curr_blk+j];
			PTd2[curr_blk+j] = -(d2[curr_blk-2*n-m+j]-d2[curr_blk+j]);									
		}
		for (j = 0; j < 2*n; j++)
		{
			rd[curr_blk+m+j] +=kappa*(d1[curr_blk+m+j]-d2[curr_blk+m+j]);
			PTd1[curr_blk+m+j] = d1[curr_blk+m+j];
			PTd2[curr_blk+m+j] = -d2[curr_blk+m+j];
		}
	}
	curr_blk = (T+1)*(2*n+m);
	for (j = 0; j < m; j++)
	{
		rd[curr_blk+j] +=kappa*((d1[curr_blk-2*n-m+j])-
								(d2[curr_blk-2*n-m+j]));
		PTd1[curr_blk+j] = d1[curr_blk-2*n-m+j];
		PTd2[curr_blk+j] = -(d2[curr_blk-2*n-m+j]);									
	}
	//printmat(PTd1,1,dim1);
	//printmat(PTd2,1,dim0);
	//printmat(rd,1,dim1);
	
	for (i = 0; i< dim2; i++) rp[i] = -b[i];
	//printmat(rp,1,dim2);
	//printmat(z,1,dim1);
	//printmat(G,dim2,dim1);
	//rp = Gz - b 
	F77_CALL(dgemv)("n",&dim2,&dim1,&DOU_ONE,G,&dim2,z,&INT_ONE,&DOU_ONE,rp,&INT_ONE);
	//printmat(rp,1,dim2);
	return;		
}

void dnudz1(double* Htwo, double *G, double *tmpG, double *phi, 
			double *PTd1, double *PTd2,
			int dim0, int dim1, int dim2, double kappa,
			double *rd, double *rp, double *dz, double *dnu)
{
	int i, lwork=-1;
	double *tmp;
	double tmpopt;
	/*
	 *   compute phi
	 */
	//phi = 2H
	for (i = 0; i< dim1*dim1; i++) phi[i] = Htwo[i];
	//printmat(phi,dim1,dim1);
	//first compute P1^Td1^2P1, it is (PTd1)(PTd1)^T
	//phi = phi + kappa*(PTd1)(PTd1)^T
	F77_CALL(dgemm)("n","t",&dim1,&dim1,&INT_ONE,&kappa,PTd1,&dim1,PTd1,&dim1,&DOU_ONE,phi,&dim1);
	//phi = phi + kappa*(PTd2)(PTd2)^T
	F77_CALL(dgemm)("n","t",&dim1,&dim1,&INT_ONE,&kappa,PTd2,&dim1,PTd2,&dim1,&DOU_ONE,phi,&dim1);
	//printmat(phi,dim1,dim1);
	
	/*
	 *   compute dnu dz
	 */
	// G dz = -rp
	//printmat(rp,1,dim2);
	//printmat(G,dim2,dim1);
	tmp = malloc(sizeof(double)*INT_ONE);
	for (i = 0; i< dim1*dim2; i++) tmpG[i] = G[i];
	for (i = 0; i< dim2; i++) dz[i] = -rp[i];
	
	//printmat(z,1,dim1);
	//printmat(dz,1,dim1);
	//printmat(dz,1,dim1);
	F77_CALL(dgels)("n",&dim2,&dim1,&INT_ONE,tmpG,&dim2,dz,&dim1,&tmpopt,&lwork,&i);
	lwork = (int)tmpopt;
	//printmat(dz,1,dim1);
	tmp = (double*)malloc( lwork*sizeof(double) );
	F77_CALL(dgels)("n",&dim2,&dim1,&INT_ONE,tmpG,&dim2,dz,&dim1,tmp,&lwork,&i);
	//printmat(dz,1,dim1);
	free(tmp);
	// PTd1(as a tmp storage) = -rd - phi*dz
	for (i = 0; i< dim1; i++) PTd1[i] = -rd[i];
	F77_CALL(dgemv)("n",&dim1,&dim1,&DOU_N_ONE,phi,&dim1,dz,&INT_ONE,&DOU_ONE,PTd1,&INT_ONE);
	
	//G^T dnu = PTd1
	lwork = -1;
	//printmat(PTd1,1,dim1);
	F77_CALL(dgels)("t",&dim2,&dim1,&INT_ONE,tmpG,&dim2,PTd1,&dim1,&tmpopt,&lwork,&i);
	lwork = (int)tmpopt;
	//printmat(PTd1,1,dim1);
	tmp = (double*)malloc( lwork*sizeof(double) );
	F77_CALL(dgels)("t",&dim2,&dim1,&INT_ONE,tmpG,&dim2,PTd1,&dim1,tmp,&lwork,&i);
	//printmat(PTd1,1,dim1);
	for (i = 0; i< dim2; i++) dnu[i] = PTd1[i];
	free(tmp);
	//printmat(z,1,dim1);
	//printmat(dz,1,dim1);
	//printmat(dnu,1,dim2);
	return;
}

void rdrp2(double *z, double *nu,
	      double *Htwo, double *G, double *tmpG, double *phi,
	      double *h1, double *h2, double *b, double *d1, double *d2,
		  double *PTd1, double *PTd2,
		  int dim0, int dim1, int dim2,
		  int T, int n, int m,
		  double kappa, double *rd, double *rp)
{
	int i,j,curr_blk;
	
	//d1 d2 dim0
	//z dim1 
	//h1 h2 dim0
	//PTd1 PTd2 dim1
	//calculate d1 d2
	for (i = 0; i < dim0; i++)
	{
		d1[i] = 1/(h1[i]-z[i]+1);
		d2[i] = 1/(h2[i]+z[i]+1);
	}
	/*
	 *  Compute rd and rp
	 */
	//rd = 2Hz
	F77_CALL(dgemv)("n",&dim1,&dim1,&DOU_ONE,Htwo,&dim1,z,&INT_ONE,&DOU_ZERO,rd,&INT_ONE);
	//rd = rd + G^Tnu
	F77_CALL(dgemv)("t",&dim2,&dim1,&DOU_ONE,G,&dim2,nu,&INT_ONE,&DOU_ONE,rd,&INT_ONE);
	//rd = rd+kappa*(P1^Td1+P2^Td2)
	for (i = 0; i < dim0; i++)
	{	
		rd[i] += kappa*(d1[i]-d2[i]);
		PTd1[i] = d1[i];
		PTd2[i] = -d2[i];
	}
	for (i = 0; i<m; i++)
	{
		PTd1[(T+1)*(2*n+m)+i] =PTd2[(T+1)*(2*n+m)+i] = 0;
	}
	//printmat(PTd1, dim1, 1);
	//rp = Gz - b 
	for (i = 0; i< dim2; i++) rp[i] = -b[i];
	F77_CALL(dgemv)("n",&dim2,&dim1,&DOU_ONE,G,&dim2,z,&INT_ONE,&DOU_ONE,rp,&INT_ONE);
	return;		
}

//this function solves problem K = P*inv(X)*Q by Cholesky factorize X
void KeqPXQ(double *K, double *P,double *X,double *Q, 
            double *tmps1, double *tmpns1, 
            int dimx, int dimpl, int dimqr, double sign, 
			double *tmpx)
{
	//printf("enter KeqPXQ ");
	long i;
	int info;
	double *mytmp;
	//may be moved out to save time
	if (dimx == dimqr) mytmp = tmps1;
	else mytmp = tmpns1;
	//solve XK_1 = Q, K_1 = mytmp
	for (i = 0; i < dimx*dimx; i++) tmpx[i] = X[i];
	for (i = 0; i < dimx*dimqr; i++) mytmp[i] = Q[i];
	F77_CALL(dposv)("L",&dimx,&dimqr,tmpx,&dimx,mytmp,&dimx,&info);
	// K = PK_1
	F77_CALL(dgemm)("n","n",&dimpl,&dimqr,&dimx,&sign,P,&dimpl,mytmp,&dimx,&DOU_ZERO,K,&dimpl);
	//printf("left KeqPXQ\n");
	return;
}

void dnudz2(double* Htwo, double *G, double *tmpG, double *phi, 
			double *A, double *B, double *C, double *D,  double *CB, double *CA,
			double *At, double *Bt, double *Ct, double *Dt, 
			double *eyen, double *eye2n, double *eyem,
			double *CBt, double *CAt,
			double *PTd1, double *PTd2, double *be,
			int dim0, int dim1, int dim2, double kappa,
			double *PhiQ, double *PhiR, double *PhiS, double *Y, double *L,
			//six tmp storage
			double *tmps, double *tmps1, double *tmps2, 
			double *tmps3, double *tmps4, double *tmps5,
			//for computation storage
			double *tmps6, double *tmpmm, double *tmpnn,
			double *tmpns, double *tmpns1,
			//temp storage for compute L
			double *tmpll1,	double *tmpll2, double *tmpll3,
			int T, int n, int m,
			double *rd, double *rp, double *dz, double *dnu)
{
	long i, j, k;
	int p, q, lwork=-1, curr_blk, ntwo = 2*n, info;
	double *tmp;
	double tmpopt;
	double *dptr, *dptr1, *dptr2, *dptr3, *dptr4, *dptr5, *dptr6;
	double *curr_PhiR, *curr_PhiS, *curr_PhiQ;
	//printf("Entered function dnudz2. Calculate dnu, dz\n");
	/*
	 *   compute phi
	 */
	//phi = 2H
	for (i = 0; i< dim1*dim1; i++) phi[i] = Htwo[i];
	//phi = phi + kappa*(PTd1)(PTd1)^T + kappa*(PTd2)(PTd2)^T
	for (i = 0; i< dim0; i++) 
		phi[i+i*dim1] = kappa*(PTd1[i]*PTd1[i]+PTd1[i]*PTd1[i]);
	
	/*
	 *   compute dnu dz
	 */
	//first get PhiQ, PhiR, PhiS
	dptr = PhiR; dptr1 = phi;
	for (i = 0; i < T+1; i++)
	{
		curr_blk = i*(dim1)*(2*n+m);
		dptr1 = phi + curr_blk;
		for (j = 0; j < m; j++)
		{
			dptr1 = phi + curr_blk + j*dim1;
			dptr1 += i*(2*n+m); 
			for (k = 0; k < m; k++)
			{
				*(dptr) = *(dptr1);
				dptr++; dptr1++;
			}	
		}
	}
	curr_blk = (T+1)*(dim1)*(2*n+m);
	dptr1 = phi + curr_blk;
	for (j = 0; j < m; j++)
	{
		dptr1 = phi + curr_blk + j*dim1;
		dptr1 += (T+1)*(2*n+m); 
		for (k = 0; k < m; k++)
		{
			*(dptr) = *(dptr1);
			dptr++; dptr1++;
		}	
	}
	//printf("finish PhR ");
	// printf("=====\n");
	// printmat(PhiR,m*m*(T+2),1);
	// printmat(PhiR,m*m,1);
	// printf("=====\n");
	dptr1 = phi; dptr2 = PhiS; 
	for (i = 0; i < T+1; i++)
	{
		curr_blk = i*(dim1)*(2*n+m);
		curr_blk +=m*dim1;
		for (j = 0; j < n; j++)
		{
			dptr1 = phi + curr_blk + j*dim1;
			dptr1 += i*(2*n+m)+m; 
			for (k = 0; k < n; k++)
			{
				*(dptr2) = *(dptr1);
				dptr2++; dptr1++;
			}	
		}
	}
	//printf("finish PhS ");
	dptr1 = phi; dptr3 = PhiQ; 
	for (i = 0; i < T+1; i++)
	{
		curr_blk = i*(dim1)*(2*n+m);
		curr_blk +=(n+m)*dim1;
		for (j = 0; j < n; j++)
		{
			dptr1 = phi + curr_blk + j*dim1;
			dptr1 += i*(2*n+m)+m+n; 
			for (k = 0; k < n; k++)
			{
				*(dptr3) = *(dptr1);
				dptr3++; dptr1++;
			}	
		}
	}
	//printf("finish PhQ \n");
	//printmat(phi,dim1,dim1);
	//second formulate Y
	//printf("ready to initialize Y \n");
	//for (i = 0; i < dim2*dim2; i++) Y[i] = 0;
	//printf("finish initialize Y \n");
	dptr = Y; 
	//Y_00
	//tmps = Y_00_11
	KeqPXQ(tmps, B,PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	// printf("=====\n");
	// printmat(PhiR,m*m*(T+2),1);
	// printmat(PhiR,m*m,1);
	// printf("=====\n");
	// printf("==\n");
	// printmat(tmps,n,n);
	// printf("==\n");
	for (i = 0; i < n; i++) tmps[i*n+i] += PhiS[i*n+i];
	//tmps1 = Y_00_21
	KeqPXQ(tmps1, CB,PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
	//tmps2 = Y_00_12
	KeqPXQ(tmps2, B,PhiR,CBt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
	//tmps3 = Y_00_22
	KeqPXQ(tmps3, CB,PhiR,CBt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	KeqPXQ(tmps4, D,PhiR+m*m,Dt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	for (i = 0; i < n*n; i++) tmps3[i] = tmps3[i] + tmps4[i] + PhiS[i];
	//("Calculated Y_00\n");
	//put Y_00 into Y
	dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr1;
			dptr++;dptr1++;
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr2;
			dptr++;dptr2++;
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr3;
			dptr++;dptr3++;
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr4;
			dptr++;dptr4++;
		}
	}

	//Y_01
	//tmps = Y_01_11
	KeqPXQ(tmps, eyen,PhiS,At, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
	//tmps1 = Y_01_21
	KeqPXQ(tmps1, D,PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
	//tmps2 = Y_01_12
	KeqPXQ(tmps2, eyen,PhiS+n*n,CAt, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
	//tmps3 = Y_01_22
	KeqPXQ(tmps3, D,PhiR+m*m,CBt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	//put Y_01 into Y
	dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr1;
			dptr++;dptr1++;
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr2;
			dptr++;dptr2++;
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr3;
			dptr++;dptr3++;
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr4;
			dptr++;dptr4++;
		}
	}

	//Y_10
	//tmps = Y_10_11
	KeqPXQ(tmps, A,PhiS,eyen, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
	//tmps1 = Y_10_21
	KeqPXQ(tmps1, CAt,PhiS+n*n,eyen, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
	//tmps2 = Y_10_12
	KeqPXQ(tmps2, B,PhiR,Dt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
	//tmps3 = Y_10_22
	KeqPXQ(tmps3, CB,PhiR+m*m,Dt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	//put Y_10 into Y
	dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr1;
			dptr++;dptr1++;
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr2;
			dptr++;dptr2++;
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr3;
			dptr++;dptr3++;
		}
		for (j = 0; j < n; j++)
		{
			*dptr = *dptr4;
			dptr++;dptr4++;
		}
	}
	for (i = 1; i<T; i++)
	{
		curr_PhiR = PhiR+i*m*m;
		curr_PhiS = PhiS+i*n*n;
		curr_PhiQ = PhiQ+i*n*n;
		//Y_ii
		//tmps = Y_ii_11
		KeqPXQ(tmps, A,curr_PhiS-n*n,At, tmps6, tmpns1, n, n, n, DOU_ONE, tmpnn);
		KeqPXQ(tmps1, B,curr_PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
		for (j = 0; j < n*n; j++) tmps[j] = tmps[j] + tmps1[j] + *(curr_PhiS+n*n+j);

		//tmps1 = Y_ii_21
		KeqPXQ(tmps1, CA,curr_PhiS-n*n,At, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
		KeqPXQ(tmps2, CB,curr_PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
		for (j = 0; j < n*n; j++) tmps1[j] = tmps1[j] + tmps2[j];
		//tmps2 = Y_ii_12
		KeqPXQ(tmps2, A,curr_PhiS-n*n,CAt, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
		KeqPXQ(tmps3, B,curr_PhiR,CBt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
		for (j = 0; j < n*n; j++) tmps2[j] = tmps2[j] + tmps3[j];
		//tmps3 = Y_ii_22
		KeqPXQ(tmps3, CA,curr_PhiS-n*n,CAt, tmps6, tmpns1, n, n, n, DOU_ONE, tmpnn);
		KeqPXQ(tmps4, CB,curr_PhiR,CBt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
		KeqPXQ(tmps5, D,curr_PhiR+m*m,Dt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
		for (j = 0; j < n*n; j++) 
			tmps3[j] += tmps4[j] + tmps5[j] + curr_PhiQ[j];
		//put Y_ii into Y
		dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr1;
				dptr++;dptr1++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr2;
				dptr++;dptr2++;
			}
		}
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr3;
				dptr++;dptr3++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr4;
				dptr++;dptr4++;
			}
		}
		//Y_ii+1
		//tmps = Y_ii+1_11
		KeqPXQ(tmps, eyen,curr_PhiS,At, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
		//tmps1 = Y_ii+1_21
		KeqPXQ(tmps1, D,curr_PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
		//tmps2 = Y_ii+1_12
		KeqPXQ(tmps2, eyen,curr_PhiS+n*n,CAt, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
		//tmps3 = Y_ii+1_22
		KeqPXQ(tmps3, D,curr_PhiR+m*m,CBt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
		//put Y_ii+1 into Y
		dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr1;
				dptr++;dptr1++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr2;
				dptr++;dptr2++;
			}
		}
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr3;
				dptr++;dptr3++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr4;
				dptr++;dptr4++;
			}
		}
		//Y_i+1i
		//tmps = Y_i+1i_11
		KeqPXQ(tmps, A,curr_PhiS,eyen, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
		//tmps1 = Y_i+1i_21
		KeqPXQ(tmps1, CAt,curr_PhiS+n*n,eyen, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
		//tmps2 = Y_i+1i_12
		KeqPXQ(tmps2, B,curr_PhiR,Dt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
		//tmps3 = Y_i+1i_22
		KeqPXQ(tmps3, CB,curr_PhiR+m*m,Dt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
		//put Y_i+1i into Y
		dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr1;
				dptr++;dptr1++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr2;
				dptr++;dptr2++;
			}
		}
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr3;
				dptr++;dptr3++;
			}
			for (k = 0; k < n; k++)
			{
				*dptr = *dptr4;
				dptr++;dptr4++;
			}
		}

	}
	//printf("finish Y except for Y_TT\n");
	curr_PhiR = PhiR+T*m*m;
	curr_PhiS = PhiS+T*n*n;
	curr_PhiQ = PhiQ+T*n*n;
	//Y_TT
	//tmps = Y_TT_11
	KeqPXQ(tmps, A,curr_PhiS-n*n,At, tmps6, tmpns1, n, n, n, DOU_ONE, tmpnn);
	KeqPXQ(tmps1, B,curr_PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	for (j = 0; j < n*n; j++) tmps[j] = tmps[j] + tmps1[j];
	//tmps1 = Y_TT_21
	KeqPXQ(tmps1, CA,curr_PhiS-n*n,At, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
	KeqPXQ(tmps2, CB,curr_PhiR,Bt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
	for (j = 0; j < n*n; j++) tmps1[j] = tmps1[j] + tmps2[j];
	//tmps2 = Y_TT_12
	KeqPXQ(tmps2, A,curr_PhiS-n*n,CAt, tmps6, tmpns1, n, n, n, DOU_N_ONE, tmpnn);
	KeqPXQ(tmps3, B,curr_PhiR,CBt, tmps6, tmpns1, m, n, n, DOU_N_ONE, tmpmm);
	for (j = 0; j < n*n; j++) tmps2[j] = tmps2[j] + tmps3[j];
	//tmps3 = Y_TT_22
	KeqPXQ(tmps3, CA,curr_PhiS-n*n,CAt, tmps6, tmpns1, n, n, n, DOU_ONE, tmpnn);
	KeqPXQ(tmps4, CB,curr_PhiR,CBt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	KeqPXQ(tmps5, D,curr_PhiR+m*m,Dt, tmps6, tmpns1, m, n, n, DOU_ONE, tmpmm);
	for (j = 0; j < n*n; j++) 
		tmps3[j] += tmps4[j] + tmps5[j] + curr_PhiQ[j];
	//put Y_TT into Y
	dptr1 = tmps; dptr2 = tmps1; dptr3 = tmps2; dptr4 = tmps3;
	for (j = 0; j < n; j++)
	{
		for (k = 0; k < n; k++)
		{
			*dptr = *dptr1;
			dptr++;dptr1++;
		}
		for (k = 0; k < n; k++)
		{
			*dptr = *dptr2;
			dptr++;dptr2++;
		}
	}
	for (j = 0; j < n; j++)
	{
		for (k = 0; k < n; k++)
		{
			*dptr = *dptr3;
			dptr++;dptr3++;
		}
		for (k = 0; k < n; k++)
		{
			*dptr = *dptr4;
			dptr++;dptr4++;
		}
	}
	//printf("finish Y\n");
	//formulate L
	for (i = 0; i < dim2*dim2; i++) L[i] = 0;
	dptr = L; 
	// tmpll1 = L_00
	for (i = 0; i < 4*n*n; i++) tmpll1[i] = *(Y+i);
	// printf("=====\n");
	// printmat(tmpll1,2*n,2*n);
	F77_CALL(dpotrf)("L",&ntwo,tmpll1,&ntwo,&info);
	// printf("\n info is %d \n",info);
	// printmat(tmpll1,2*n,2*n);
	// printf("=====\n");
	// tmpll2 = L_10
	for (i = 0; i < 4*n*n; i++) tmpll3[i] = *(Y+4*n*n+i);
	F77_CALL(dtrtrs)("L","N","N",&ntwo,&ntwo,tmpll1,&ntwo,tmpll3,&ntwo,&info);
	F77_CALL(dgemm)("t","n",&ntwo,&ntwo,&ntwo,
					&DOU_ONE,tmpll3,&ntwo,eye2n,&ntwo,&DOU_ZERO,tmpll2,&ntwo);
	dptr1 = tmpll1; dptr2 = tmpll2;
	// printf("===\n");
	// printmat(tmpll1,2*n,2*n);
	// printmat(tmpll2,2*n,2*n);
	// printf("===\n");
	for (i = 0; i < 2*n; i++)
	{
		dptr = L + i*dim2;
		for (j = 0; j < 2*n; j++)
		{
			if (j>=i)
				*dptr = *dptr1;
			else
				*dptr = 0;
			dptr++; dptr1++;
		}
		for (j = 0; j < 2*n; j++)
		{
			*dptr = *dptr2;
			dptr++; dptr2++;
		}
	}
	//printmat(L,dim2,dim2);
	
	for (i = 1; i<T;i++)
	{
		//printf("%d",i);
		//L_ii
		for (j = 0; j < 4*n*n; j++) tmpll1[j] = *(Y+i*3*4*n*n+j);
		F77_CALL(dgemm)("n","t",&ntwo,&ntwo,&ntwo,
						&DOU_N_ONE,tmpll2,&ntwo,tmpll2,&ntwo,&DOU_ONE,tmpll1,&ntwo);
		F77_CALL(dpotrf)("L",&ntwo,tmpll1,&ntwo,&info);
		//L_i+1i
		for (j = 0; j < 4*n*n; j++) tmpll3[j] = *(Y+i*3*4*n*n+4*n*n+j);
		F77_CALL(dtrtrs)("L","N","N",&ntwo,&ntwo,tmpll1,&ntwo,tmpll3,&ntwo,&info);
		F77_CALL(dgemm)("t","n",&ntwo,&ntwo,&ntwo,
						&DOU_ONE,tmpll3,&ntwo,eye2n,&ntwo,&DOU_ZERO,tmpll2,&ntwo);
		dptr1 = tmpll1; dptr2 = tmpll2;
		for (j = 0; j < 2*n; j++)
		{
			dptr = L + i*2*n*dim2 + j*dim2 + i*2*n;
			for (k = 0; k < 2*n; k++)
			{
				if (k>=j)
					*dptr = *dptr1;
				else
					*dptr = 0;
				dptr++; dptr1++;
			}
			for (k = 0; k < 2*n; k++)
			{
				*dptr = *dptr2;
				dptr++; dptr2++;
			}
		}
	}
	for (j = 0; j < 4*n*n; j++) tmpll1[j] = *(Y+T*3*4*n*n+j);
	F77_CALL(dgemm)("n","t",&ntwo,&ntwo,&ntwo,
					&DOU_N_ONE,tmpll2,&ntwo,tmpll2,&ntwo,&DOU_ONE,tmpll1,&ntwo);
	F77_CALL(dpotrf)("L",&ntwo,tmpll1,&ntwo,&info);
	dptr1 = tmpll1;
	for (j = 0; j < 2*n; j++)
	{
		dptr = L + T*2*n*dim2 + j*dim2 + T*2*n;
		for (k = 0; k < 2*n; k++)
		{
			if (k>=j)
				*dptr = *dptr1;
			else
				*dptr = 0;
			dptr++; dptr1++;
		}
	}
	//printf("finish L\n");
			
	//PTd1 = invPhird
	for (i = 0; i < dim1; i++) PTd1[i] = rd[i];
	for (i = 0; i < T+1; i++)
	{
		curr_blk = i*(2*n+m);
		F77_CALL(dposv)("L",&m,&INT_ONE,PhiR+i*m*m,&m,PTd1+curr_blk,&m,&info);
		curr_blk = i*(2*n+m)+m;
		for (j = 0; j < n; j++)
		{
			*(PTd1+curr_blk+j) *= 1 / *(PhiS+i*n*n+j*n+j);
		}
		curr_blk = i*(2*n+m)+m+n;
		F77_CALL(dposv)("L",&n,&INT_ONE,PhiQ+i*n*n,&m,PTd1+curr_blk,&n,&info);
	}
	curr_blk = (T+1)*(2*n+m);
	F77_CALL(dposv)("L",&m,&INT_ONE,PhiR+(T+1)*m*m,&m,PTd1+curr_blk,&m,&info);
	
	//printmat(PTd1,1,dim1);
	//beta = -rp+ GinvPhird
	for (i = 0; i< dim2; i++) be[i] = -rp[i];
	F77_CALL(dgemv)("n",&dim2,&dim1,&DOU_ONE,G,&dim2,PTd1,&INT_ONE,&DOU_ONE,be,&INT_ONE);
	
	//solve dnu
	//solve dnu 1 . Lx = be
	F77_CALL(dtrsv)("L","N","N",&dim2,L,&dim2,be,&INT_ONE);
	//solve dnu 2 . Lt nu = x
	F77_CALL(dtrsv)("L","T","N",&dim2,L,&dim2,be,&INT_ONE);
	for (i = 0; i < dim2; i++) dnu[i] = be[i];
	
	//solve dz 
	//PTd1 = -rd - G^Tdnu
	for (i = 0; i < dim1; i++) PTd1[i] = -rd[i];
	F77_CALL(dgemv)("t",&dim2,&dim1,&DOU_N_ONE,G,&dim2,dnu,&INT_ONE,&DOU_ONE,PTd1,&INT_ONE);
	F77_CALL(dposv)("L",&dim1,&dim1,phi,&dim1,PTd1,&INT_ONE,&info);
	for (i = 0; i < dim1; i++) dz[i] = PTd1[i];
	
	// printf("==\n");
	// 	printmat(dz,dim1,1);
	// 	printmat(dnu,dim2,1);
	// 	printf("==\n");
	//printf("Left dnudz.\n");
	return;
}

void resdresp(double *rd, double *rp, 
		      int dim1, int dim2,
		      double *resd, double *resp, double *res)
{
    *resp = F77_CALL(dnrm2)(&dim2,rp,&INT_ONE);
    *resd = F77_CALL(dnrm2)(&dim1,rd,&INT_ONE);
    *res = sqrt((*resp)*(*resp)+(*resd)*(*resd));
    return;
}