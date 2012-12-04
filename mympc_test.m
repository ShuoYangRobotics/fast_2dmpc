clear
format long
%variable setting
n=4;
m=2;
T=30;
dim0 = (T+1)*(2*n+m); 
dim1 = dim0+m;		
dim2 = (T+1)*(2*n);

niters = 5;
kappa = 1000;
c_type = 1;
quiet = 1;

%model matrices
A = randn(n,n);
A = A./(max(abs(eig(A))));
B = rand(n,m);
C = randn(n,n);
C = C./(max(abs(eig(C))));
D = rand(n,m);

% objective matrices
Q = randn(n,n); Q = Q*Q';      
R = randn(m,m); R = R*R';

% state and control limits
dXmax = 0.3; Emax = 0.02; Rmax = 0.75;
dXmin = -0.02; Emin = -0.001; Rmin = -0.15;

dxmin = dXmin*ones(n,1);
dxmax = dXmax*ones(n,1);
emin = Emin*ones(n,1);
emax = Emax*ones(n,1);
rmin = Rmin*ones(m,1);
rmax = Rmax*ones(m,1);

DX0 = 0.2*ones(n,1); 
E0 = 0.01*ones(n,1); 
R0 = 0.38*ones(m,1); 

prev_E0 =0.05*ones(n*(T+1),1);
curr_DX0 = 0.13*ones(n,1);

sys.n = n; sys.m = m;
sys.A = A; sys.B = B; sys.C = C; sys.D = D;
sys.Q = Q; sys.R = R;
sys.dx0 = curr_DX0;
sys.prev_e = prev_E0;
sys.rmax = rmax; sys.dxmax = dxmax; sys.emax = emax;
sys.rmin = rmin; sys.dxmin = dxmin; sys.emin = emin;

param.c_type = c_type;
param.T = T;
param.kappa = kappa;
param.niters = niters;
param.quiet = quiet;

tic
[myz,t] = mympc_step(sys, param, DX0, E0, R0);
t1 = toc

tic
H = zeros(dim1,dim1);
for i=1:T+1
    %1==n (2*n+m)+1==(2*n+m)+n
    H((i-1)*(2*n+m)+1:(i-1)*(2*n+m)+m,...
        (i-1)*(2*n+m)+1:(i-1)*(2*n+m)+m) = R;
    H((i-1)*(2*n+m)+1+n+m:(i-1)*(2*n+m)+n+n+m,...
        (i-1)*(2*n+m)+1+n+m:(i-1)*(2*n+m)+n+n+m) = Q;
end
H((T+1)*(2*n+m)+1:(T+1)*(2*n+m)+m,...
    (T+1)*(2*n+m)+1:(T+1)*(2*n+m)+m) = R;

P1 = [eye(dim0) zeros(dim0,m)];
P2 = -P1;

G = zeros(dim2,dim1);
G(1:n,1:m) = -B;
G(1:n,1+m:n+m) = eye(n);
G(1+n:n+n,1:m) = C*B;
G(1+n:n+n,1+m+n:n+m+n) = eye(n);
G(1+n:n+n,1+m+n+n:m+m+n+n) = D;
mvv = 2*n;
mvh = 2*n+m;
for i=2:T+1
    G(1+mvv*(i-1):n+mvv*(i-1),...
        1+mvh*(i-1):m+mvh*(i-1)) = -B;
    G(1+mvv*(i-1):n+mvv*(i-1),...
        1+mvh*(i-1)-2*n:n+mvh*(i-1)-2*n) = -A;
    G(1+mvv*(i-1):n+mvv*(i-1),...
        1+m+mvh*(i-1):n+m+mvh*(i-1)) = eye(n);
    G(1+n+mvv*(i-1):n+n+mvv*(i-1),...
        1+mvh*(i-1):m+mvh*(i-1)) = C*B;
    G(1+n+mvv*(i-1):n+n+mvv*(i-1),...
        1+mvh*(i-1)-2*n:n+mvh*(i-1)-2*n) = C*A;
    G(1+n+mvv*(i-1):n+n+mvv*(i-1),...
        1+m+n+mvh*(i-1):n+m+n+mvh*(i-1)) = eye(n);
    G(1+n+mvv*(i-1):n+n+mvv*(i-1),...
        1+m+n+n+mvh*(i-1):m+m+n+n+mvh*(i-1)) = D;
end

h1 = zeros(dim0,1);
mvv = (2*n+m);
for i=1:T+1
    h1(1+(i-1)*mvv:m+(i-1)*mvv) = rmax;
    h1(1+m+(i-1)*mvv:n+m+(i-1)*mvv) = dxmax;
    h1(1+m+n+(i-1)*mvv:n+m+n+(i-1)*mvv) = emax;
end
h2 = zeros(dim0,1);
for i=1:T+1
    h2(1+(i-1)*mvv:m+(i-1)*mvv) = -rmin;
    h2(1+m+(i-1)*mvv:n+m+(i-1)*mvv) = -dxmin;
    h2(1+m+n+(i-1)*mvv:n+m+n+(i-1)*mvv) = -emin;
end
b = zeros(dim2,1);
b(1:n) = A*curr_DX0;
b(1+n:n+n) = -C*A*curr_DX0;
for i=1:T+1
    b(1+n+(i-1)*2*n:n+n+(i-1)*2*n) = b(1+n+(i-1)*2*n:n+n+(i-1)*2*n) ...
        + prev_E0(1+(i-1)*n:n+(i-1)*n);
end


cvx_begin
    variable z(dim1) 
    minimize(z'*H*z)
    subject to
        P1*z<=h1
        -P1*z<=h2
        Gz = b
cvx_end
t2 = toc
% 
[t2 t1]








