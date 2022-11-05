function y=cor26g(x)
%version with shortened iteration; modified Rt and Rq

%x=[5.5 0 28.7 27.2 24.2 18.5 141 419 0 600 1010 15 15 15 0 1 1 5 1 ];%sample data stream
u=x(1);%wind speed (m/s)  at height zu (m)
us=x(2);%surface current speed in the wind direction (m/s)
ts=x(3);%bulk water temperature (C) if jcool=1, interface water T if jcool=0  
t=x(4);%bulk air temperature (C), height zt
Qs=x(5);%bulk water spec hum (g/kg) if jcool=1, ...
Q=x(6);%bulk air spec hum (g/kg), height zq
zi=x(7);%PBL depth (m)
P=x(8);%Atmos surface pressure (mb)
zu=x(9);%wind speed measurement height (m)
zt=x(10);%air T measurement height (m)
zq=x(11);%air q measurement height (m)


     Rgas = 287.1;    % 干空气气体常数
     cpa = 1004.67;   % 干空气比热
     Le=(2.501-.00237*ts)*1e6;
     tdk = 273.16;
     grav = 9.82; 
     Beta = 1.2;
     von = 0.4;
     %cpv=cpa*(1+0.84*Q);
     rhoa = P*100/(Rgas*(t+tdk)*(1+0.61*Q));   % 湿空气密度
     visa = 1.326e-5*(1+6.542e-3*t+8.301e-6*t*t-4.84e-9*t*t*t);    % 运动粘滞系数
    
      
     %***************   迭代程序开始 *******
     
     %***************  首次猜测 ************
    du = u-us;
    dt = ts-t-0.0098*zt;
    dq = Qs-Q; 
    ta = t+tdk;
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
     usr = ut*von/(log(zu/zo10)-psiu_26(zu/L10));
     tsr = -dt*von/(log(zt/zot10)-psit_26(zt/L10));
     qsr = -dq*von/(log(zq/zot10)-psit_26(zq/L10));

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
     
     zet=von*grav*zu*(tsr*(1+0.61*Q)+0.61*ta*qsr)/(ta*(1+0.61*Q))/(usr*usr);
      %disp(usr)
      %disp(zet);
     zo=charn*usr*usr/grav+0.11*visa/usr;
     rr=zo*usr/visa;
     L=zu/zet;
     zoq=min(1.15e-4,5.5e-5/rr^.63);
     %zoq=5.5e-5/rr^0.63;
     zot=zoq;
     usr=ut*von/(log(zu/zo)-psiu_26(zu/L));
     tsr=-dt*von/(log(zt/zot)-psit_26(zt/L));
     qsr=-dq*von/(log(zq/zoq)-psit_26(zq/L));
     Bf=-grav/ta*usr*(tsr+.61*ta*qsr);
     if Bf>0
     ug=Beta*(Bf*zi)^0.333;
     else
     ug=0.2;
     end;
    
       
   end;%bulk iter loop
     %disp(zot)
     ut=sqrt(du*du+ug*ug);
     usr = ut*von/(log(zu/zo)-psiu_26(zu/L));
     tsr = -dt*von/(log(zt/zot)-psit_26(zt/L));
     qsr = -dq*von/(log(zq/zoq)-psit_26(zq/L));
     L = 1/(von*grav/ta*(tsr+0.61*ta*qsr)/(usr*usr));
     zet = von*grav*zu/ta*(tsr+.61*ta*qsr)/(usr*usr);
     tau=rhoa*usr*usr*du/ut;                %stress
     hsb=-rhoa*cpa*usr*tsr;
     hlb=-rhoa*Le*usr*qsr;
     
%-------------
%  Babin计算波导高度
%-------------
    
    ee = 0.62197;%the ratic of the gas constant for dry air to that of water vapor
    e  = Q*P/(ee+(1-ee)*Q);
    Np = 77.6/ta+4810*77.6*Q/(ta*ta*(ee+(1-ee)*Q));%折射率随气压变化的梯度
    Nt = -77.6*Q/(ta*ta)-2*4810*77.6*P*Q/(ta*ta*ta*(ee+(1-ee)*Q));%折射率随温度变化的梯度
    Nh = 4810*77.6*P*ee/(ta*ta*(ee+(1-ee)*Q)^2);%折射率随湿度变化的梯度
    C1 = -0.01*rhoa*grav*Np;
    C2 = Nt;
    C3 = Nh;
    A  = C1-C2*grav*(P-(1-ee)*e)/(cpa*P);
    B  = C2*(P/1000)^(Rgas/cpa);
    C  = C3;

    if zet>=0
       %fprintf('\n稳定条件下的蒸发波导高度计算\n');
       H=-(B*tsr+C*qsr)/von/((A+0.157)+5/L*(B*tsr+C*qsr))
   else
       %fprintf('\n不稳定条件下的蒸发波导高度计算\n');
       H1=-(B*tsr+C*qsr)*fait(zet)/von/(A+0.157);
   
       while Conv == 1 
       zet1=von*grav*H1*(tsr+0.61*ta*qsr)/ta/(usr^2);
       H=-(B*tsr+C*qsr)*fait(zet1)/von/(A+0.157);
        if abs(H-H1)<=0.0001 
           Conv = 0;
        else
           H1=H;
        end
  end
   

   end

     %************  10-m neutral coeff realtive to ut ********
     Cdn_10=von*von/log(10/zo)/log(10/zo);
     Chn_10=von*von/log(10/zo)/log(10/zot);
     Cen_10=von*von/log(10/zo)/log(10/zoq);
   
    
   y=[ zo zot zoq L zet usr tsr qsr Cdn_10 Chn_10 Cen_10  H ];
   %   1   2   3  4  5   6   7   8     9     10     11    12 
