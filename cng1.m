% % ************************************************************************************
% % *** Pore pressure prediction using ANFIS, ARD-BNN, GPRN                          ***
% % *** and SVM model                                                                ***
% % *** To run the script Netlab and LSSVM softwares/toolboxes are required to be 
% % *** downloaded in the same folder. 
% % *** Tested using Netlab software developed by I. Nabney, [Nabney, I., 2002. NETLAB: 
% % *** algorithms for pattern recognition. Springer Science & Business
% % *** Media]; Netlab can be downloaded at https://www2.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/downloads
% % *** Tested using LSSVM software/toolbox which cab be downladed at http://www.esat.kuleuven.be/sista/lssvmlab/
% % *** Tested using Matlab R2017a under Windows 7                                   ***
% % ************************************************************************************
% % Addtional data;                                                                  ***
% % Four Excel data/dat files:                                                           ***
% % Two files include well log and pore pressure data.                               ***         
% % Pore pressure are calculated using Porosity method and Eaton's method            ***                                       *** 
% % Other two files include depth data                                               ***                                                       
% **************************************************************************************

close all;clc;close all;
clc;
% %********** ARD-BNN MODELLING******************************************%%
load por_pp_1344.dat;%input data
load dep_por_1344.dat;
datain1=por_pp_1344;
dep=dep_por_1344;
dept=dep(2:3151);
x1p=datain1(:,1);            %training travel time data.
x2p=datain1(:,2);            %training density data.
x3p=datain1(:,3);            %training hydrostatic pressure data.
x4p=datain1(:,4);            %training porosity data.
x5p=datain1(:,5);            %training pore pressure data
x=[x1p(1:3150),x2p(1:3150),x3p(1:3150),x4p(1:3150),x5p(1:3150)];
t=[x5p(2:3151)];% target data
%[xn,minx,maxx]=premnmx(x);
% [tn,mint,maxt] = premnmx(t);
x=x';
t=t';

[xn,minx,maxx,tn,mint,maxt] = premnmx(x,t);
[xtrans,transMat] = prepca(xn,0.000000000000000000000000002);
[R,Q] = size(xtrans);
iitst = 2:4:Q;
iival = 4:4:Q;
iitr = [1:4:Q 3:4:Q];
valX = xtrans(:,iival); val.T = tn(:,iival);
testX = xtrans(:,iitst); test.T = tn(:,iitst);
xtr = xtrans(:,iitr); ttr = tn(:,iitr);
xr=xtr';
zr=ttr';
% % [Initialization of network after Nabney, I., 2002. NETLAB algorithm]
% % [Hyper-parameters; alpha is lamda and beta is mu in the manuscript for ARD-BNN];  
noise =0.01;
nin = 5;			% Number of inputs.
nhidden = 7;		% Number of hidden units.
nout = 1;			% Number of outputs.
aw1 = 0.001*ones(1, nin);% First-layer ARD-BNN hyperparameters.
ab1 = 0.001;			% Hyperparameter for hidden unit biases.
aw2 = 0.001;			% Hyperparameter for second-layer weights.
ab2 = 0.001;			% Hyperparameter for output unit biases.
beta = 20.0;			% Coefficient of data error.

% mlp and mlpprior function after Nabney, I., 2002. NETLAB
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'linear', prior, beta);

% Set up vector of options for the optimiser after Nabney, I., 2002. NETLAB.
nouter = 2;			    % Number of outer loops
ninner = 1;		        % Number of inner loops
options = zeros(1,18);	% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence betast occur
options(3) = 1.0e-7;
options(14) = 500;		% Number of training cycles in inner loop. 

% Train using scaled conjugate gradients(scg), re-estimating alpha and beta after Nabney, I., 2002. NETLAB
for k = 1:nouter
  net = netopt(net, options, xr, zr, 'scg');%netopt after Nabney, I., 2002. NETLAB
  [net, gamma] = evidence(net, xr, zr, ninner);%evidence after Nabney, I., 2002. NETLAB
  fprintf(1, '\n\nRe-estimation cycle %d:\n', k);
  disp('The first three alphas are the hyperparameters for the corresponding');
  disp('input to hidden unit weights.  The remainder are the hyperparameters');
  disp('for the hidden unit biases, second layer weights and output unit');
  disp('biases, respectively.');
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma);
   
end
fprintf(1, 'true beta: %f\n', 1/(noise*noise));
an = mlpfwd(net,xr);%mlpfwd after Nabney, I., 2002. NETLAB
[y] = postmnmx(an',mint,maxt);
tr=t(:,iitr);
dev=tr-y';
 dev=dev';
 dbp=dev(:,1);
 msedbp=mse(dbp)%mse of trainind set
maedbp=mae(dbp);
retrain=1.0-sum((tr-y').^2)./sum(tr.^2);
dtrain=1.0-sum((tr-y').^2)./sum((y'-mean(tr)+tr-mean(tr)).^2);
Rtrain=corrcoef(tr,y);
ate = mlpfwd(net,testX');
[ate] = postmnmx(ate',mint,maxt);
tat=t(:,iitst);
devt=tat-ate;
devt=devt';
tdbp=devt(:,1);
msetdbp=mse(tdbp);% mse of test set
maetdbp=mae(tdbp);
retest=1.0-sum((tat-ate).^2)./sum(tat.^2);
dtest=1.0-sum((tat-ate).^2)./sum((tat-mean(tat)+tat-mean(tat)).^2);
Rtest=corrcoef(tat,ate);
av = mlpfwd(net,valX');
[av] = postmnmx(av',mint,maxt);
tav=t(:,iival);
devv=tav-av;
devv=devv';
vdbp=devv(:,1);
msevdbp=mse(vdbp);%mse of validation set
maevdbp=mae(vdbp);
reval=1.0-sum((tav-av).^2)./sum(tav.^2);
dval=1.0-sum((tav-av).^2)./sum((av-mean(av)+tav-mean(tav)).^2);
Rval=corrcoef(tav,av);

% fprintf(1, '    alpha1: %8.5f\n', net.alpha(1));
% fprintf(1, '    alpha2: %8.5f\n', net.alpha(2));
% disp('This is confirmed by looking at the corresponding weight values:')
% disp(' ');
% fprintf(1, '    %8.5f    %8.5f\n', net.w1');

lag=[1 2 3  ];
 x1c=inv(net.alpha(1))*100;
 x2c=inv(net.alpha(2))*100;
xall=[x1c x2c  ];
xall=xall./max(xall)*100;

figure
subplot(3,1,1)
plot(tr,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(y,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','ARD-BNN-Pred.');
title('Training Interval:Original Vs. ARD-BNN-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight

subplot(3,1,2)
plot(tav,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(av,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','ARD-BNN-Pred.');
title('Validation Interval:Original Vs. ARD-BNN-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
subplot(3,1,3)
plot(tat,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(ate,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','ARD-BNN-Pred.');
title('Test Interval:Oroginal Vs. ARD-BNN-Predicted');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
%plotregression(tr,y);
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('Target','Fontsize',18);
ylabel('Predicted','Fontsize',18);
%plotregression(tav,av,'Regression:Validation Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('Target','Fontsize',18);
ylabel('Predicted','Fontsize',18);
%plotregression(tat,ate,'Regression:Test Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('Target','Fontsize',18);
ylabel('Predicted','Fontsize',18);



% % *******ANFIS MODELING************************************************%%
[R,Q] = size(xtrans);
iitst = 2:4:Q;
iival = 4:4:Q;
iitr = [1:4:Q 3:4:Q];
valX = xtrans(:,iival); val.T = tn(:,iival);
testX = xtrans(:,iitst); test.T = tn(:,iitst);
xtr = xtrans(:,iitr); ttr = tn(:,iitr);
x=[x1p(1:3150),x2p(1:3150),x3p(1:3150),x4p(1:3150),x5p(1:3150)];
t=[x5p(2:3151)];
[xn,minx,maxx]=premnmx(x);
[tn,mint,maxt] = premnmx(t);
Data=[x,t];
numMFs=2;
mfType='gbellmf';% membership function
trnData1=Data(1:4:Q, :);
trnData2=Data(3:4:Q, :);
valData=Data(4:4:Q, :);
testData=Data(2:4:Q, :);
trnData=[trnData1' trnData2'];
trnData=trnData';
trnData=[trnData1' trnData2'];
trnData=trnData';
fismat = genfis1(trnData,numMFs,mfType);
figure
subplot(2,2,1)
plotmf(fismat, 'input', 1);
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Input','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
subplot(2,2,2)
plotmf(fismat, 'input', 2)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Input','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
subplot(2,2,3)
plotmf(fismat, 'input', 1)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Input','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
subplot(2,2,4)
plotmf(fismat, 'input', 1)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Input','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
trnOpt(1)=50;
trnOpt(2)=0;
trnOpt(3)=.001;
trnOpt(4)=10;
trnOpt(5)=1.1;
dispOpt=1.0;
method=1;
[fismat1,trnError,ss,fismat2,valError] = ...
anfis(trnData,fismat,trnOpt,dispOpt,valData,method);
figure
subplot(2,2,1)
plotmf(fismat2, 'input', 1)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Longitude','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
subplot(2,2,2)
plotmf(fismat2, 'input', 2)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Latitude','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
subplot(2,2,3)
plotmf(fismat2, 'input', 1)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Variable','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
subplot(2,2,4)
plotmf(fismat2, 'input', 1)
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',20);axis tight;
xlabel('Lag distance','Fontsize',18);
ylabel('Degree of Membership','Fontsize',18);
plot(trnError,'r','LineWidth',3);
xlabel('Epoch','Fontsize',24);
ylabel('MSE (Mean Squared Error)','Fontsize',24);
title('Error Curves','Fontsize',24);
legend('Training Error');
grid on;
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',24);axis tight;
plot(valError,'r','LineWidth',3);
xlabel('Epoch','Fontsize',24);
ylabel('MSE (Mean Squared Error)','Fontsize',24);
title('Error Curves','Fontsize',24);
grid on;
legend('Testing Error');
set(gca,'LineWidth',3,'FontName','Helvetica','Fontsize',24);axis tight;

 yt = evalfis([trnData(:,1:5); valData(:,1:5)],fismat2);

 y1 = evalfis([trnData(:,1:5)],fismat1);

y2 = evalfis([valData(:,1:5)],fismat2);
y3 = evalfis([testData(:,1:5)],fismat2);
xnew=x;
anfis_output = evalfis([xnew(:,1:5)],fismat2);
trnT1=t(1:4:Q, :);
trnT2=t(3:4:Q, :);
valT=t(4:4:Q, :);
testT=t(2:4:Q,:);
trnT=[trnT1' trnT2'];
trnT=trnT';
%%%%%***************************plotting************************%%%%%%%%%
subplot(3,1,1)
plot(trnT,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(y1,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','ANFIS-Pred.');
ylabel('Depth[Km]','Fontsize',18);
title('Training Interval:Original Vs. ANFIS-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight

subplot(3,1,2)
plot(valT,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(y2,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','ANFIS-Pred.');
ylabel('Depth[Km]','Fontsize',18);
title('Validation Interval:Original Vs. ANFIS-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight

subplot(3,1,3)
plot(testT,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(y3,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','ANFIS-Pred.');
xlabel('No. of Data','Fontsize',18);
ylabel('Depth[Km]','Fontsize',18);
title('Test Interval:Oroginal Vs. ANFIS-Predicted');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
plotregression(trnT,y1,'Regression:Training Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('Target Depth[Km]','Fontsize',18);
ylabel('Predicted Depth[Km]','Fontsize',18);
legend('Training Interval')
dbp=trnT-y1;
msedbp=mse(dbp)%mse of trainind set
maedbp=mae(dbp)
retrain=1.0-sum((trnT-y1).^2)./sum(trnT.^2)
dtrain=1.0-sum((trnT-y1).^2)./sum((y1-mean(trnT)+trnT-mean(trnT)).^2)
Rtrain=corrcoef(trnT,y1)
plotregression(valT,y2,'Regression:Validation Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('Target Depth[Km]','Fontsize',18);
ylabel('Predicted Depth[Km]','Fontsize',18);
vdbp=valT-y2;
msevdbp=mse(vdbp)%mse of validation set
maevdbp=mae(vdbp)
reval=1.0-sum((valT-y2).^2)./sum(valT.^2)
dval=1.0-sum((valT-y2).^2)./sum((y2-mean(y2)+valT-mean(valT)).^2)
Rval=corrcoef(valT,y2)
plotregression(testT,y3,'Regression:Test Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('Target Depth[Km]','Fontsize',18);
ylabel('Predicted Depth[Km]','Fontsize',18);
tdbp=testT-y3;
msetdbp=mse(tdbp)%mse of test set
maetdbp=mae(tdbp)
retest=1.0-sum((testT-y3).^2)./sum(testT.^2)
dtest=1.0-sum((testT-y3).^2)./sum((y3-mean(y3)+testT-mean(testT)).^2)
Rtest=corrcoef(testT,y3)

%%********GAUSSIAN PROCESS REGRESSION[GPRN] MODELLING *******************%%

x=[x2p(1:3150),x1p(1:3150),x3p(1:3150),x4p(1:3150),x5p(1:3150)];
t=[x5p(2:3151)];
x=x';
t=t';
[xn,minx,maxx,tn,mint,maxt] = premnmx(x,t);
[xtrans,transMat] = prepca(xn,0.000000000000000000000000000000002);
[R,Q] = size(xtrans);
iitst = 2:4:Q;
iival = 4:4:Q;
iitr = [1:4:Q 3:4:Q];
valX = xtrans(:,iival); val.T = tn(:,iival);
testX = xtrans(:,iitst); test.T = tn(:,iitst);
xtr = xtrans(:,iitr); ttr = tn(:,iitr);
xr=xtr';
zr=ttr';
xn=xn';
tn=tn';
net = gp(5, 'sqexp');
prior.pr_mean = 0;
prior.pr_var =3;
net = gpinit(net, xr, zr, prior);
% Now train to find the hyperparameters.
options = foptions;
options(1) = 1;
optionzs(14) = 14;
[net, options] = netopt(net, options, xr, zr, 'scg');
fprintf(1, '\nfinal hyperparameters:\n')
format_string = strcat('  bias:\t\t\t%10.6f\n  noise:\t\t%10.6f\n', ...
  '  inverse lengthscale:\t%10.6f\n  vertical scale:\t%10.6f\n');
fprintf(1, format_string, ...
    exp(net.bias), exp(net.noise), exp(net.inweights(1)), exp(net.fpar(1)));
an = gpfwd(net,xr);
[y] = postmnmx(an',mint,maxt);
tr=t(:,iitr);
dev=tr-y;
dev=dev';
dbp=dev(:,1);
msedbp=mse(dbp)
maedbp=mae(dbp)
retrain=1.0-sum((tr-y).^2)./sum(tr.^2)
dtrain=1.0-sum((tr-y).^2)./sum((y-mean(tr)+tr-mean(tr)).^2)
Rtrain=corrcoef(tr,y)
av = gpfwd(net,valX');
[av] = postmnmx(av',mint,maxt);
tav=t(:,iival);
devv=tav-av;
devv=devv';
vdbp=devv(:,1);
msevdbp=mse(vdbp)%MSE of validation set
maevdbp=mae(vdbp)%MAE of validation set
reval=1.0-sum((tav-av).^2)./sum(tav.^2)% RE of validation set
dval=1.0-sum((tav-av).^2)./sum((av-mean(av)+tav-mean(tav)).^2)%'d' value of of validation set
Rval=corrcoef(tav,av)
ate = gpfwd(net,testX');
[ate] = postmnmx(ate',mint,maxt);
tat=t(:,iitst);
devt=tat-ate;
devt=devt';
tdbp=devt(:,1);
msetdbp=mse(tdbp)%MSE of test set
maetdbp=mae(tdbp)%MAE of test set
retest=1.0-sum((tat-ate).^2)./sum(tat.^2)%RE of test set
dtest=1.0-sum((tat-ate).^2)./sum((tat-mean(tat)+tat-mean(tat)).^2)%'d' value of of validation set
Rtest=corrcoef(tat,ate)
subplot(3,1,1)
plot(tr,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(y,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','GP-Pred.');
ylabel('Pore pressure','Fontsize',18);
title('Training Interval:Original Vs. GP-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight

subplot(3,1,2)
plot(tav,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(av,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','GP-Pred.');
ylabel('pore pressure','Fontsize',18);
title('Validation Interval:Original Vs. GP-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
subplot(3,1,3)
plot(tat,'o-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(ate,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','GP-Pred.');
xlabel('No. of Data','Fontsize',18);
ylabel('pore pressure','Fontsize',18);
title('Test Interval:Original Vs. GP-Predicted');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
plotregression(tr,y,'Regression:Training Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('observed pore pressure','Fontsize',18);
ylabel('Predicted pore pressure','Fontsize',18);
plotregression(tav,av,'Regression:Validation Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('observed pore pressure','Fontsize',18);
ylabel('Predicted pore pressure','Fontsize',18);
plotregression(tat,ate,'Regression:Test Interval');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);
xlabel('observed pore pressure]','Fontsize',18);
ylabel('Predicted pore pressure','Fontsize',18);


%%***********SVM  MODELLING**********************************************%%
x=[x1p(1:3150),x2p(1:3150),x4p(1:3150),x3p(1:3150)];
t=[x5p(2:3151)];
x=x';
t=t';
[xn,minx,maxx,tn,mint,maxt] = premnmx(x,t);
[xtrans,transMat] = prepca(xn,0.000000000000000000000000000000002);
[R,Q] = size(xtrans);
iitst = 2:4:Q;
iival = 4:4:Q;
iitr = [1:4:Q 3:4:Q];
valX = xtrans(:,iival); val.T = tn(:,iival);
testX = xtrans(:,iitst); test.T = tn(:,iitst);
xtr = xtrans(:,iitr); ttr = tn(:,iitr);
xr=xtr';
zr=ttr';
% % LS-SVMlab toolbox is at http://www.esat.kuleuven.be/sista/lssvmlab/ 
% ----------------Parameter specifications--------------------------------------
C =100;
lambda = .0000001;
epsilon = .01;
kerneloption = 01;
 %kernel='poly';
 kernel='Gaussian';
 %kernel='htrbf';
  %kernel='wavelet';
%kernel='polyhomog';
 %kernel='frame';		
verbose=1;
[xsup,ysup,w,w0] = svmreg(xr,zr,C,epsilon,kernel,kerneloption,lambda,verbose);
% --------------------------------------------------------
ypredtr = svmval(xr,xsup,w,w0,kernel,kerneloption);
[ypred] = postmnmx(ypredtr',mint,maxt);
% ..........................plotting and analysis......................
tr=t(:,iitr);
plotregression(tr,ypred)
xlabel('Obs.','FontName','Verdana','Fontsize',20);ylabel('SVR','FontName','Verdana','Fontsize',20);
set(gca,'LineWidth',3,'FontName','Verdana','Fontsize',20);
xtest=valX';
ypredtr = svmval(xtest,xsup,w,w0,kernel,kerneloption);
[ysvrval] = postmnmx(ypredtr',mint,maxt);
tav=t(:,iival);
plotregression(tav,ysvrval)
xlabel('Obs.','FontName','Verdana','Fontsize',20);ylabel('SVR','FontName','Verdana','Fontsize',20);
set(gca,'LineWidth',3,'FontName','Verdana','Fontsize',20);
xtest=testX';
ypredtr = svmval(xtest,xsup,w,w0,kernel,kerneloption);
[ysvrtst] = postmnmx(ypredtr',mint,maxt);
tat=t(:,iitst);
plotregression(tat,ysvrtst)
xlabel('Obs.','FontName','Verdana','Fontsize',20);ylabel('SVR','FontName','Verdana','Fontsize',20);
set(gca,'LineWidth',3,'FontName','Verdana','Fontsize',20);
meantr=mean(tr)
stdtr=std(tr)
dev=tr-ypred;
dev=dev';
dbp=dev(:,1);
msedbp=mse(dbp)
maedbp=mae(dbp)
retrain=1.0-sum((tr-ypred).^2)./sum(tr.^2)
dtrain=1.0-sum((tr-ypred).^2)./sum((ypred-mean(tr)+tr-mean(tr)).^2)
Rtrain=corrcoef(tr,ypred)
subplot(3,1,1)
plot(tr,'>-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(ypred,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','SVM-Pred.');
ylabel('Depth[Km]','Fontsize',18);
title('Training Interval:Original Vs. SVM-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
aval = svmval(valX',xsup,w,w0,kernel,kerneloption);
[av] = postmnmx(aval',mint,maxt);
tav=t(:,iival);
meanval=mean(tav)
stdval=std(tav)
devv=tav-av;
devv=devv';
vdbp=devv(:,1);
msevdbp=mse(vdbp)
maevdbp=mae(vdbp)
reval=1.0-sum((tav-av).^2)./sum(tav.^2)
dval=1.0-sum((tav-av).^2)./sum((av-mean(av)+tav-mean(tav)).^2)
Rval=corrcoef(tav,av)
subplot(3,1,2)
plot(tav,'>-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(av,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','SVM-Pred.');
ylabel('Depth[Km]','Fontsize',18);
title('Validation Interval:Original Vs. SVM-Predicted')
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
atest = svmval(testX',xsup,w,w0,kernel,kerneloption);
[ate] = postmnmx(atest',mint,maxt);
tat=t(:,iitst);
devt=tat-ate;
devt=devt';
tdbp=devt(:,1);
msetdbp=mse(tdbp)%mse of test set
maetdbp=mae(tdbp)
retest=1.0-sum((tat-ate).^2)./sum(tat.^2)
dtest=1.0-sum((tat-ate).^2)./sum((tat-mean(tat)+tat-mean(tat)).^2)
Rtest=corrcoef(tat,ate)
subplot(3,1,3)
plot(tat,'>-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6); hold on;plot(ate,'r-','LineWidth',3,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',6);
legend('Data','SVM-Pred.');
xlabel('No. of Data','Fontsize',18);
ylabel('Depth[Km]','Fontsize',18);
title('Test Interval:Original Vs. SVM-Predicted');
set(gca,'LineWidth',3,'FontName','Times New Roman','Fontsize',18);axis tight
