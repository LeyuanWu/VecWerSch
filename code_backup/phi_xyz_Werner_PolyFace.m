function [dV,gx,gy,gz,Txx,Tyy,Tzz,Txy,Txz,Tyz]...
    =phi_xyz_Werner_PolyFace(xgv,ygv,zp,Vert,PolyFaces,rho0)
% ----------------------------------------------------------------------- %
% vectorization calculation of gravity field of homogeneous density polyhedron
% "with arbitrary polygon as surface"
% Can also be used to calculate the gravitational effects of 
% "partial faces", where each edge may belong to both faces or only one face
% M-file to calculate the complete gravity caused by a polyhedron with
% constant density using the algorithm in 1997 Werner CMDA
% #########################################################################
% Author: Leyuan Wu <leyuanwu@zjut.edu.cn>
%   College of Science, Zhejiang University of Technology, Hangzhou, China
% #########################################################################
% ---- Reference ---- %
% Werner, R.A., Scheeres, D.J.
% Exterior gravitation of a polyhedron derived and compared with harmonic
% and mascon gravitation representations of asteroid 4769 Castalia.
% Celestial Mech Dyn Astr 65, 313-344 (1996). https://doi.org/10.1007/BF00053511
% ----------------------------------------------------------------------- %
% ---- input ---- %
% xgv,ygv,zp: Computation points
% Note: 
% If zp is a scalar and xgv and ygv are vectors, the computing plane
% If size(zp)=[ny,nx], and xgv,ygv are vectors, calculation surface
% If the three vectors have the same size, discrete calculation points;
% Vert,PolyFaces: Vertex list, face list (cell array)
% Note!! For a non-convex polygon surface, the outer normal vector formed 
% by the first three points is required to be the correct outer-normal of the surface
% rho0: density
% ---- output ---- %
% dV: gravity potential
% gx,gy,gz: gravity vector
% Txx,Tyy,Tzz,Txy,Txz,Tyz: gravity gradient tensor
% ---- Units ----%
% length in km, density in kg/m^3,
% gravity potential in m^2/s^2, vector in mGal, tensor in 1e-9/s^2 (Eotvos)
% -------------------------------------------------------------------------
%%%%%%%% If PolyFaces is a three-column array (when the input is all triangles), convert it to a cellular array
if(~iscell(PolyFaces))
    PolyFaces=mat2cell(PolyFaces,ones(size(PolyFaces,1),1));
end
%%%%%%%% Determine whether it is a calculation plane or surface
if(isscalar(zp) && ~isscalar(xgv)) % Plane
    flag=1;
    nx=length(xgv);ny=length(ygv);
    [X2d,Y2d]=meshgrid(xgv,ygv);
    Z2d=zp*ones(size(X2d));
    XPs=X2d(:);YPs=Y2d(:);ZPs=Z2d(:);
elseif(size(zp,1)==length(ygv) && size(zp,2)==length(xgv) && isvector(ygv) && isvector(xgv)) % Surface on regular grid
    flag=1;
    nx=length(xgv);ny=length(ygv);
    [X2d,Y2d]=meshgrid(xgv,ygv);
    XPs=X2d(:);YPs=Y2d(:);ZPs=zp(:);
else                                    % irregular set of points
    flag=2;
    sizeComp=size(zp);
    XPs=xgv(:);YPs=ygv(:);ZPs=zp(:);
end
%%%%%%%% Constants
% G=6.673e-11;
G=6.67430e-11;
CdV=G*rho0*1e6;
Cg=G*rho0*1e5*1e3;
Cten=G*rho0*1e9;
%%%%%%%% Generate an edge list, four columns of data
% **** Start vertex; End vertex; Face number 1; Face number 2 (nan if none exists) **** %
Edges=PolyFace2Edge(PolyFaces);
%%%%%%%% Some significant quantity
nP=length(XPs);
nPolyFace=size(PolyFaces,1);
nEdge=size(Edges,1);
%%%%%%%% Surface gravity effect initialization
dV_F=zeros(nP,1);
gx_F=zeros(nP,1);gy_F=zeros(nP,1);gz_F=zeros(nP,1);
Txx_F=zeros(nP,1);Tyy_F=zeros(nP,1);Tzz_F=zeros(nP,1);
Txy_F=zeros(nP,1);Txz_F=zeros(nP,1);Tyz_F=zeros(nP,1);
%%%% Loop the effects on each facet
fprintf('\n ------------------------- ');
fprintf('\n Calculating the effects of each facet...');
hnfs=zeros(nPolyFace,3); % The face normal vector is stored for later use
for iPolyFace=1:1:nPolyFace
    face=PolyFaces{iPolyFace};
    rfs=Vert(face,:);
    rf1=rfs(1,:);rf2=rfs(2,:);rf3=rfs(3,:);
    rf12=rf2-rf1;rf13=rf3-rf1;
    nf=[rf12(2)*rf13(3)-rf12(3)*rf13(2),rf12(3)*rf13(1)-rf12(1)*rf13(3),rf12(1)*rf13(2)-rf12(2)*rf13(1)];
    hnf=nf/sqrt(nf(1)^2+nf(2)^2+nf(3)^2);
    DyadF=hnf'*hnf; 
    hnfs(iPolyFace,:)=hnf;
    %%%% Computation points
    rPs=[XPs,YPs,ZPs];
    % ****************** Vectorized part ****************** %
    for iface=1:1:length(face)-2
        if(iface==1) % The first group requires no judgment
            flagRev=1;
        else
            rf1=rfs(1,:);rf2=rfs(iface+1,:);rf3=rfs(iface+2,:);
            rf12=rf2-rf1;rf13=rf3-rf1;
            nf_cur=[rf12(2)*rf13(3)-rf12(3)*rf13(2),rf12(3)*rf13(1)-rf12(1)*rf13(3),rf12(1)*rf13(2)-rf12(2)*rf13(1)];
            if(nf_cur*hnf'>=0)
                flagRev=1;
            else
                flagRev=-1;
                temp=rf2;
                rf2=rf3;
                rf3=temp;
            end
        end
        %%%% Solid Angle
        rfP1s=rf1-rPs; 
        rfP2s=rf2-rPs; 
        rfP3s=rf3-rPs; 
        norm_rfP1s=sqrt(sum(rfP1s.^2,2));
        norm_rfP2s=sqrt(sum(rfP2s.^2,2));
        norm_rfP3s=sqrt(sum(rfP3s.^2,2));
        crosProd=horzcat(rfP2s(:,2).*rfP3s(:,3)-rfP2s(:,3).*rfP3s(:,2),...
            rfP2s(:,3).*rfP3s(:,1)-rfP2s(:,1).*rfP3s(:,3),...
            rfP2s(:,1).*rfP3s(:,2)-rfP2s(:,2).*rfP3s(:,1));
        triProd=sum(rfP1s.*crosProd,2);
        denume=norm_rfP1s.*norm_rfP2s.*norm_rfP3s...
            +norm_rfP1s.*sum(rfP2s.*rfP3s,2)+norm_rfP2s.*sum(rfP1s.*rfP3s,2)+norm_rfP3s.*sum(rfP1s.*rfP2s,2);
        Wf=flagRev*2*atan2(triProd,denume);
        %%%% Gravity contribution
        dV_F=dV_F+sum((rfP1s*DyadF).*rfP1s,2).*Wf;
        gxyz=(rfP1s*DyadF).*repmat(Wf,1,3);
        gx_F=gx_F+gxyz(:,1);gy_F=gy_F+gxyz(:,2);gz_F=gz_F+gxyz(:,3);
        Txx_F=Txx_F+DyadF(1,1)*Wf;Tyy_F=Tyy_F+DyadF(2,2)*Wf;Tzz_F=Tzz_F+DyadF(3,3)*Wf;
        Txy_F=Txy_F+DyadF(1,2)*Wf;Txz_F=Txz_F+DyadF(1,3)*Wf;Tyz_F=Tyz_F+DyadF(2,3)*Wf;
    end
    % ****************** Vectorized part ****************** %
    if(mod(iPolyFace,floor(nPolyFace/10))==0)
        fprintf('\n %d/%d calculated...%s',iPolyFace,nPolyFace,datetime);
    end
end
%%%%%%%% Edge effect gravity initialization
dV_E=zeros(nP,1);
gx_E=zeros(nP,1);gy_E=zeros(nP,1);gz_E=zeros(nP,1);
Txx_E=zeros(nP,1);Tyy_E=zeros(nP,1);Tzz_E=zeros(nP,1);
Txy_E=zeros(nP,1);Txz_E=zeros(nP,1);Tyz_E=zeros(nP,1);
%%%% Loop the effect of each edge
fprintf('\n ------------------------- ');
fprintf('\n Calculating the effects of each edge...');
for iEdge=1:1:nEdge
    %%%% Edge length
    re1=Vert(Edges(iEdge,1),:);
    re2=Vert(Edges(iEdge,2),:);
    re12=re2-re1;
    norm_re12=sqrt(re12(1)^2+re12(2)^2+re12(3)^2);
    %%%% The normal vector and the dyadic vector for each side
    edge1=Edges(iEdge,3)*re12;
    iFace1=Edges(iEdge,4);
    hnf1=hnfs(iFace1,:);
    w1=[edge1(2)*hnf1(3)-edge1(3)*hnf1(2),edge1(3)*hnf1(1)-edge1(1)*hnf1(3),edge1(1)*hnf1(2)-edge1(2)*hnf1(1)];
    hw1=w1/sqrt(w1(1)^2+w1(2)^2+w1(3)^2);% The unit outer-normal of the edge 1
    if(~isnan(Edges(iEdge,5))) % An edge belonging to two faces
        edge2=-edge1;
        iFace2=Edges(iEdge,5);
        hnf2=hnfs(iFace2,:);
        w2=[edge2(2)*hnf2(3)-edge2(3)*hnf2(2),edge2(3)*hnf2(1)-edge2(1)*hnf2(3),edge2(1)*hnf2(2)-edge2(2)*hnf2(1)];
        hw2=w2/sqrt(w2(1)^2+w2(2)^2+w2(3)^2);% The unit outer-normal of the edge 2
        DyadE=hnf1'*hw1+hnf2'*hw2; 
    else
        DyadE=hnf1'*hw1; 
    end
    % ****************** Vectorized part ****************** %
    rPs=[XPs,YPs,ZPs];
    %%%% The line integral on each side: Le
    reP1s=re1-rPs;
    reP2s=re2-rPs;
    norm_reP1s=sqrt(sum(reP1s.^2,2));
    norm_reP2s=sqrt(sum(reP2s.^2,2));
    Le=log((norm_reP1s+norm_reP2s+norm_re12)./(norm_reP1s+norm_reP2s-norm_re12));
    %%%% Gravity contribution
    dV_E=dV_E+sum((reP1s*DyadE).*reP1s,2).*Le;
    gxyz=reP1s*DyadE.*repmat(Le,1,3);
    gx_E=gx_E+gxyz(:,1);gy_E=gy_E+gxyz(:,2);gz_E=gz_E+gxyz(:,3);
    Txx_E=Txx_E+DyadE(1,1)*Le;Tyy_E=Tyy_E+DyadE(2,2)*Le;Tzz_E=Tzz_E+DyadE(3,3)*Le;
    Txy_E=Txy_E+DyadE(1,2)*Le;Txz_E=Txz_E+DyadE(1,3)*Le;Tyz_E=Tyz_E+DyadE(2,3)*Le;
    % ****************** Vectorized part ****************** %
    if(mod(iEdge,floor(nEdge/10))==0)
        fprintf('\n %d/%d calculated...%s',iEdge,nEdge,datetime);
    end
end
%%%%%%%% Overall gravity effect
dV=1/2*CdV*dV_E-1/2*CdV*dV_F;
gx=-Cg*gx_E+Cg*gx_F;gy=-Cg*gy_E+Cg*gy_F;gz=-Cg*gz_E+Cg*gz_F;
Txx=Cten*Txx_E-Cten*Txx_F;Tyy=Cten*Tyy_E-Cten*Tyy_F;Tzz=Cten*Tzz_E-Cten*Tzz_F;
Txy=Cten*Txy_E-Cten*Txy_F;Txz=Cten*Txz_E-Cten*Txz_F;Tyz=Cten*Tyz_E-Cten*Tyz_F;
%%%%%%%% Transform to plane anomaly accordingly
if(flag==1)
    dV=reshape(dV,ny,nx);% m^2/s^2
    gx=reshape(gx,ny,nx);% mGal
    gy=reshape(gy,ny,nx);% mGal
    gz=reshape(gz,ny,nx);% mGal
    Txx=reshape(Txx,ny,nx);% Eotvos
    Tyy=reshape(Tyy,ny,nx);% Eotvos
    Tzz=reshape(Tzz,ny,nx);% Eotvos
    Txy=reshape(Txy,ny,nx);% Eotvos
    Txz=reshape(Txz,ny,nx);% Eotvos
    Tyz=reshape(Tyz,ny,nx);% Eotvos
end
if(flag==2)
    dV=reshape(dV,sizeComp);% m^2/s^2
    gx=reshape(gx,sizeComp);% mGal
    gy=reshape(gy,sizeComp);% mGal
    gz=reshape(gz,sizeComp);% mGal
    Txx=reshape(Txx,sizeComp);% Eotvos
    Tyy=reshape(Tyy,sizeComp);% Eotvos
    Tzz=reshape(Tzz,sizeComp);% Eotvos
    Txy=reshape(Txy,sizeComp);% Eotvos
    Txz=reshape(Txz,sizeComp);% Eotvos
    Tyz=reshape(Tyz,sizeComp);% Eotvos
end
















