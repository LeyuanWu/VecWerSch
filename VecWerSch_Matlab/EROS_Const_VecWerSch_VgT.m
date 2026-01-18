clear;
fprintf('\n ###################################################### ');
fprintf('\n %s',datetime);
%% Model parameters, computation grid, density
%%%%%%%% EROS model
load EROS
nVpF=size(eros11272_22540,1);
nFace=22540;nVert=nVpF-nFace;
Vert=eros11272_22540(1:nVert,:);
Faces=eros11272_22540(nVert+1:end,:)+1; % Note the index start from 0 for EROS data
%%%%%%%% The external cuboid of the model
X1=min(Vert(:,1));X2=max(Vert(:,1));
Y1=min(Vert(:,2));Y2=max(Vert(:,2));
Z1=min(Vert(:,3));Z2=max(Vert(:,3));
%%%%%%%% Computation grid
% < ---------------------------- Parameter ---------------------------- > %
xmin=-20;dx=0.1;xmax=20;
ymin=-10;dy=0.1;ymax=10;
z0=-6.3;
% < ------------------------------ End  ------------------------------- > %
xgv=xmin:dx:xmax;ygv=ymin:dy:ymax;
nx=length(xgv);ny=length(ygv);
[X2d,Y2d]=meshgrid(xgv,ygv);
%%%%%%%% Density
% < ---------------------------- Parameter ---------------------------- > %
rho0=2670;
% < ------------------------------ End  ------------------------------- > %
fprintf('\n ------------------------------------------------------ ');
fprintf('\n Model parameters...');
fprintf('\n Number of vertices: %d.',nVert);
fprintf('\n Number of faces: %d.',nFace);
fprintf('\n Density: %.2f kg/m^3.',rho0);
fprintf('\n The external cuboid of the model: X:[%.2f,%.2f] km; Y:[%.2f,%.2f] km; Z:[%.2f,%.2f] km',...
    X1,X2,Y1,Y2,Z1,Z2);
fprintf('\n Computation plane altitude: %.2f km',z0);
fprintf('\n dx: %.2f km; dy: %.2f km; nx: %d; ny: %d; total computation points: %d',dx,dy,nx,ny,nx*ny);
%% Start computaion
%%%%%%%% Werner_1997_CMDA
fprintf('\n ------------------------------------------------------ ');
fprintf('\n Analytical solution of Werner_1997_CMDA...');
tic;
[dV_Werner,gx_Werner,gy_Werner,gz_Werner,...
    Txx_Werner,Tyy_Werner,Tzz_Werner,Txy_Werner,Txz_Werner,Tyz_Werner]...
    =phi_xyz_Werner_PolyFace(xgv,ygv,z0,Vert,Faces,rho0);
time_dVgT_Werner=toc;
%%%% Print the total number of computation points and time cost
fprintf('\n Number of computation points: %d',nx*ny);
fprintf('\n Time cost: %.2f sec',time_dVgT_Werner);
fprintf('\n Number of facets computed per second: %d faces/sec',nx*ny*nFace/time_dVgT_Werner);




