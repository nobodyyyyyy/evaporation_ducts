% function nps_mode=duct_height(t,ts,RH,u,P)
 


t = 20;
RH = 79;
ts = 23.78681;
u = 4.1;
P = 1011.4;
zu = 12.5;       % 风速测量高度12.5
zt = 12.5;       % 温度测量高度
zq = 12.5;       % 湿度测量高度
zp = 12.5;       % 气压测量高度

% --------------
% 参数设定
% --------------    
     
     Beta = 1.25;   % 阵风系数
     von = 0.4;    % Karman常数
     tdk = 273.16; % 开尔文温度转换量
     grav = 9.82;  % 重力加速度
     zi = 600;     % 设定近地层厚度
     us = 0;
    
     es = R_S(t,P)*RH/100;
     Q = es*0.622/(P-0.378*es);  % 相对湿度转化为比湿
     Qs = qsee(ts,P);            % 海表面饱和比湿
     %disp(Qs) 
     %*************  air constants ************
    
     Rgas = 287.04;    % 干空气气体常数
     cpa = 1004.67;    % 干空气比热
     %cpv=cpa*(1+0.84*Q);
     %rhoa = P*100/(Rgas*(t+tdk)*(1+0.61*Q));   % 湿空气密度
     visa = 1.326e-5*(1+6.542e-3*t+8.301e-6*t*t-4.84e-9*t*t*t);    % 运动粘滞系数
     du = u-us;
     dt = ts-t-0.00976*zt;
     dq = Qs-Q; 
     ta = t+tdk;
     %tv = ta*(1+0.61*Q)*(1000/P)^(Rgas/cpa);  % 虚温
     %disp('tv');disp(tv);
     tv = (ta+0.00976*zt)*(1+0.61*Q);
     %disp('tv2');disp(tv2);
      
     %***************   迭代程序开始 *******
     
     %***************  首次猜测 ************
   
    ug = 0.5;
    ut = sqrt(du*du+ug*ug);
	u10 = ut*log(10/1e-4)/log(zu/1e-4);
    usr = 0.035*u10; 
	zo10 = 0.011*usr*usr/grav+0.11*visa/usr;
	Cd10 = (von/log(10/zo10))^2;
	Ch10 = 0.00115;
	Ct10 = Ch10/sqrt(Cd10);
	zot10 = 10/exp(von/Ct10);
    %-----减少迭代次数--------
	Cd = (von/log(zu/zo10))^2;
	Ct = von/log(zt/zot10);
	CC = von*Ct/Cd;
	Ribcu = -zu/zi/0.004/Beta^3;
	Ribu = -grav*zu/ta*(dt+0.61*ta*dq)/ut^2;
    disp(Ribu)
	nits = 3;    % 设置迭代次数
	if Ribu<0;
		zetu=CC*Ribu/(1+Ribu/Ribcu);
	else;
		zetu=CC*Ribu*(1+27/9*Ribu/CC);
		end;		
	L10=zu/zetu;
	if zetu>50;
		nits=1;
	end;
     usr = ut*von/(log(zu/zo10)-psiu_nps(zu/L10));
     tsr = -dt*von/(log(zt/zot10)-psit_nps(zt/L10));
     qsr = -dq*von/(log(zq/zot10)-psit_nps(zq/L10));

   % 选择charn系数	
   charn=0.011;
   if ut>10
      charn=0.011+(ut-10)/(18-10)*(0.018-0.011);
   end;
   if ut>18
      charn=0.018;
   end;
   
     %disp(usr)
     
     %***************  bulk loop ************
   for i=1:nits;
     
     zet=von*grav*zu*(tsr*(1+0.6078*Q)+0.6078*ta*qsr)/tv/(usr*usr);
     %严格按照NPS的文献所述的稳定度函数形式，区别位温和虚位温。
     %disp(zet);
     zo=charn*usr*usr/grav+0.11*visa/usr;
     rr=zo*usr/visa;
     L=zu/zet;
     zoq=min(1.15e-4,5.5e-5/rr^.6); zot=zoq;    % Fairall COARE3.O
     %zoq=zo*exp(2-2.28*rr^0.25);zot=zoq;         % Brutsaert 1975
     %zot=5.4*rr^(4/3)/(1.75*rr+1)^2;zoq=zot;    % 见 P.A.Federichson的文献
     %zoq=zo*exp(-2.67*rr^0.25+2.57);zot=zoq;    % Brutsaert 1982
     %zoq=zo*exp(3.4-3.5*rr^0.25);zot=zoq;
     
     usr=ut*von/(log(zu/zo)-psiu_nps(zu/L));
     tsr=-dt*von/(log(zt/zot)-psit_nps(zt/L));
     qsr=-dq*von/(log(zq/zoq)-psit_nps(zq/L));
     Bf=-grav/tv*usr*(tsr+0.61*ta*qsr);
     if Bf>0
     ug = Beta*(Bf*zi)^0.333;
     else
     ug = 0.2;
     end;
     ut = sqrt(du*du+ug*ug);
       
   end;%bulk iter loop
     %disp(zot)
     %disp(zoq)
  
     %usr = ut*von/(log(zu/zo)-psiu_nps(zu/L));
     %tsr = -dt*von/(log(zt/zot)-psit_nps(zt/L));
     %qsr = -dq*von/(log(zq/zoq)-psit_nps(zq/L));
     %zet = von*grav*zu*(tsr*(1+0.6078*Q)+0.6078*ta*qsr)/tv/(usr*usr);
     %L = zu/zet;
     %disp(zet)
%-------------------------------------------
%     计算波导高度
%-------------------------------------------
    
ee = 0.62197;%the ratic of the gas constant for dry air to that of water vapor


%--------------------------------------------
%     计算温度、比湿廓线
%--------------------------------------------
h0=zo;
disp (h0)
h=[0:1:50];
T_kuoxian(1)=ts+tsr/von*(log(h0/zot)-psit_nps(h0/L))-0.00976*h0;
Q_kuoxian(1)=Qs+qsr/von*(log(h0/zoq)-psit_nps(h0/L));
for i=2:length(h)
    T_kuoxian(i)=ts+tsr/von*(log(h(i)/zot)-psit_nps(h(i)/L))-0.00976*h(i);
    Q_kuoxian(i)=Qs+qsr/von*(log(h(i)/zoq)-psit_nps(h(i)/L));
end
disp ( [h' T_kuoxian' Q_kuoxian'])



%--------------------------------------------
%     计算修气压廓线
%--------------------------------------------

%
P_kuoxian(1)=P*exp(2*grav*(zp)/Rgas/(ta*(1+0.61*Q)+(ts+tdk)*(1+0.61*Qs)));
%disp(P_kuoxian(1));
%首先利用测量高度上的气压计算海平面气压
for i=2:length(h)
 P_kuoxian(i)=P_kuoxian(i-1)*exp(2*grav*(-1)/Rgas/((T_kuoxian(i-1)+tdk)*(1+0.61*Q_kuoxian(i-1))+(T_kuoxian(i)+tdk)*(1+0.61*Q_kuoxian(i))));%0.1gaiwei1
end
disp ( [P_kuoxian']) 


%--------------------------------------------
%     计算修正折射率廓线
%--------------------------------------------
for i=1:length(h)
    E(i)=Q_kuoxian(i)*P_kuoxian(i)/(ee+(1-ee)*Q_kuoxian(i));
    rh(i)=100*E(i)/R_S(T_kuoxian(i),P_kuoxian(i));% 相对湿度廓线
    M(i)=77.6*P_kuoxian(i)/(T_kuoxian(i)+tdk)-(5.6/(T_kuoxian(i)+tdk)-3.75e5/(T_kuoxian(i)+tdk)^2)*E(i)+0.1568*h(i);
end
   [mmin,ii]=min(M')
   hh=h(ii)             %修正折射率最小值对应的高度hh，即为蒸发波导高度



     
    