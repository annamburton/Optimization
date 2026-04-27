function curve_fitting_model1_last
global tforward
% clear all
close all
clc

load AFluDat05-09.txt


tdata = AFluDat05_09(:,1);
qdata = AFluDat05_09(:,2);

tforward = (1:1:1825)';
format long

k =1.0e+03 *[0.000000005104570   0.000000052590476   0.000000000000019   1.976827190426774...
   0.002092395869505  0.000000351177667];

 lb = [0.0           0.0              0.0                   0.0      0.0       0.0];
 
 
[k,~] = lsqcurvefit(@model1,k,tdata,qdata,lb,[],...
                             optimset('Disp','iter','TolX',10^(-15),'TolFun',10^(-15)));
 
 figure1 = figure(2);
 
 axes1 = axes('Parent',figure1,...
    'YTickLabel',{'0.001','0.002','0.003','0.004','0.005'},...
    'YTick',[0.001 0.002 0.003 0.004 0.005],...
    'XTickLabel',{'2005','2006','2007','2008','2009','2010'},...
    'XTick',[0 365 730 1095 1460 1825],...
    'XGrid','on',...
    'AmbientLightColor',[0.941176470588235 0.941176470588235 0.941176470588235]);

box(axes1,'on');
hold(axes1,'all');

[~,Y] = ode15s(@(t,y)(model_1(t,y,k)),tforward,[k(4)  k(5)  65000  .00047  .00047]);
 plot(tdata, qdata, 'Marker','.','Color',[1 0 0],...
                'MarkerSize',10,'LineStyle','none')
 plot(tforward, Y(:,5), '-b','LineWidth',1.2)

 xlabel({'Time(days since 01/01/2005)'},'LineWidth',2,'FontSize',11,'FontName','Arial');
 ylabel({'Cumulative number of human cases (\times 10^5)'},'LineWidth',2,...
    'FontSize',11,...
    'FontName','Computer Modern');                      
         
end

function dy = model_1(t,y,k)

Lb = 1020/365;
L  = 1000/365;
mb = 1/(2*365);
nb = 0.1;
mu = 1/(65*365);
nu = 0.15; 
om = 127;

dy = zeros(5,1);   

dy(1) = Lb -(k(1)*sin(2*pi*(t+om)/365)+k(2))*y(1)*y(2)-mb*y(1);    
dy(2) = (k(1)*sin(2*pi*(t+om)/365)+k(2))*y(1)*y(2)-(nb+mb)*y(2);
dy(3) = L - k(3)*y(3)*y(2)-mu*y(3);
dy(4) = k(3)*y(3)*y(2)-(mu+nu)*y(4);
dy(5) = k(3)*y(3)*y(2);

end
    
function q = model1(k,tdata)
global tforward

[~,Y] = ode23s(@(t,y)(model_1(t,y,k)),tforward,[k(4)  k(5)  65000  k(6)  .00047]);

q = Y(tdata(:),5);

end
