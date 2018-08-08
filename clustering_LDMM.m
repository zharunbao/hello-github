clear;
run vlfeat-0.9.20/toolbox/vl_setup;
fi=imread('barbarahead.png');
% fi=rgb2gray(fi);
f=double(fi);
   

 
[m,n] = size(f);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
% sample the data randomly
load sample_matrix2;
% id1 = randperm(m*n);
% num_sam=floor(m*n*0.1); % the sample rate
% 
% 
% id = id1(1:num_sam);
% id0=id1(num_sam+1:end);
% id_matrix=zeros(m,n);
% id_matrix(id)=1;
% 
% 
% 
% id_1=(id_matrix>1/2);




% generate the initial guess
% fw=zeros(m*n,1);
% fw(id)=f(id);
% fw(id0) = mean(fw(id))+std(fw(id))*randn(size(id0)); % fill in the missing pixels with random number
% 
% fw=reshape(fw,m,n);
load fw;

% parameters of patches
px=10; % size of patch in x direction
py=10; % size of patch in y direction

% location of the patches
x1=[1:1:m];
x2=[1:1:n];
[X,Y]=meshgrid(x1,x2);

X1=reshape(X,[],1);
X2=reshape(Y,[],1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

uw = image2patch(fw,X1,X2,px,py);% uw是65536*100的矩阵，矩阵的每一行都是一个patch拉成的向量。
load cluster.mat;
% idx = kmeans(uw,3);
% i1 =find(idx == 1);
% i2 =find(idx == 2);
% i3 =find(idx == 3);
patches1 = uw(i1,:);
patches2 = uw(i2,:);
patches3 = uw(i3,:);
[r,k]=size(uw);
[r1,k1]=size(patches1);
[r2,k2]=size(patches2);
[r3,k3]=size(patches3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%参数
outerloop=100;
lambda1=10e-1; 
lambda2=10e-1;
lambda3=10e-1;

out = cell(outerloop,1);
v=zeros(r,k);
u_image=zeros(m,n);
A=zeros(outerloop,0);
for ii=1:outerloop 
    
    W_p1 = weight_ann(uw(i1,:)');
    W_p2 = weight_ann(uw(i2,:)');
    W_p3 = weight_ann(uw(i3,:)');% Compute the weight matrix
    

%   Assemble the coefficient matrix
    DW_p1 =sparse([1:r1],[1:r1],sum(W_p1,2),r1,r1);
    DW_p2 =sparse([1:r2],[1:r2],sum(W_p2,2),r2,r2);
    DW_p3 =sparse([1:r3],[1:r3],sum(W_p3,2),r3,r3);
    
    LW_p1=DW_p1-W_p1;
    LW_p2=DW_p2-W_p2;
    LW_p3=DW_p3-W_p3;
    
    coe_matrix_p1=LW_p1+lambda1*W_p1;
    coe_matrix_p2=LW_p2+lambda2*W_p2;
    coe_matrix_p3=LW_p3+lambda3*W_p3;
    
    for kk=1:1 % Bregman iteration, usually the number of iterations is set to be 1


        b_p1=lambda1*W_p1*(uw(i1,:)-v(i1,:));
        b_p2=lambda2*W_p2*(uw(i2,:)-v(i2,:));
        b_p3=lambda3*W_p3*(uw(i3,:)-v(i3,:));
        [uw(i1,:),flag,relres]=gmresm(coe_matrix_p1,b_p1,[],1e-2,50,[],[],uw(i1,:)); 
        [uw(i2,:),flag,relres]=gmresm(coe_matrix_p2,b_p2,[],1e-2,50,[],[],uw(i2,:));
        [uw(i3,:),flag,relres]=gmresm(coe_matrix_p3,b_p3,[],1e-2,50,[],[],uw(i3,:));% solving the linear system using GMRES


        uw_old=uw;
        u_image=patch2image(uw+v,X1,X2,px,py,m,n); % recover the 2D data from patches
      
        
        u_image(id_1)=fi(id_1); % assign the value at sample points to be the given value

            
        uw=image2patch(u_image,X1,X2,px,py); % generate patches
       
        v=v+uw_old-uw; % update the Lagrange multiplier
        PSNR=psnr(u_image,f);

        fprintf('step=%d, PSNR=%f\n', ii, PSNR); % display the PSNR at current step

    end
    out{ii} = u_image; % save the data
    A(ii)=PSNR;


end
    