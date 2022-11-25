%function babin_mode=duct_height(t,ts,RH,u,P)
 
%------------
%算法名称：      计算波导高度
%算法描述：      根据babin模式描述，计算波导高度
%调用格式：      [H] = duct_height(Ta,Ts,Rh,U,P)
%编写时间：      2006年 5月22日
%改写时间：    
%输入参数：      t      气温   摄氏度
%                ts     海表水温   摄氏度
%                q      相对湿度
%                u      风速    米/秒
%                P      大气压    百帕
%输出参数：      H      波导高度    米
%------------
%调用子函数
%                R_S.m       相对湿度转化为比湿
%                qsee.m      求海面饱和比湿
%                psiu_25.m    
%                psit_25.m
%                fait.m      计算fai函数
% clear;
% clc;

% --------------
% 入口参数设定
% --------------

t = 20;
RH = 79;
ts = 23.78681;
u = 4.1;
P = 1011.4;
zu = 2.5;       % 风速测量高度12.5
zt = 2.5;       % 温度测量高度
zq = 2.5;       % 湿度测量高度

% --------------
% 参数设定
% --------------    
     
     Beta = 1.2;   % 阵风系数
     von = 0.4;    % Karman常数
     tdk = 273.16; % 开尔文温度转换量
     grav = 9.8;   % 重力加速度
     zi = 600;     % 设定近地层厚度
     us = 0;
  
    %****************************************************
     es = R_S(t,P)*RH/100;
     Q = es.*0.622/(P-0.378*es);  % 相对湿度转化为比湿
     Qs = qsee(ts,P);            % 海表面饱和比湿
      
     %*************  air constants ************
    
     Rgas = 287.1;    % 干空气气体常数
     cpa = 1004.67;   % 干空气比热
     %cpv=cpa*(1+0.84*Q);
     %rhoa = P*100/(Rgas*(t+tdk)*(1+0.61*Q));   % 湿空气密度
     rhoa = 1.29; %dry空气密度
     visa = 1.326e-5*(1+6.542e-3*t+8.301e-6*t*t-4.84e-9*t*t*t);    % 运动粘滞系数
     %disp(rhoa)
     
      
     %***************   迭代程序开始 *******

     %***************  首次猜测 ************
     du = u-us;
     dt = ts-t-0.0098*zt;
     dq = Qs-Q; 
     ta = t+tdk;
     
     Cdn = 0.001;
     Ctn = 0.0012;
     zo  = 10*exp(-1*von/sqrt(Cdn));  % 粗糙长度
     zot = 10*exp(-1*von/sqrt(Ctn));  % 温湿粗糙长度
     Cuz = von/log(zu/zo);            % 风速输送系数
     Cqz = von/log(zt/zot);           % 湿度输送系数
     Ctz = von/log(zq/zot);           % 温度输送系数
     
     ug=0.5;
     usr= Cuz*sqrt(du*du+ug*ug);      % 风速特征尺度 
     qsr = -1*dq*Cqz;                 % 湿度特征尺度 
     tsr = -1*dt*Ctz;                 % 温度特征尺度 
    
     %***************   迭代程序开始 ******* 
    for i=1:20    
     zet=von*grav*zu/ta*(tsr+0.61*ta*qsr)/(usr*usr);  
     L=zu/zet;
      %disp(usr)
      %disp(zet)
     zo=0.011*usr*usr/grav+0.11*visa/usr;
     rr=zo*usr/visa;
     if rr<=.11,
     rt=.177;
     rq=.292;
     elseif rr<=.825,
     rt=1.376*rr^.929;
     rq=1.808*rr^.826;
     elseif rr<=3.0,
     rt=1.026*rr^(-.599);
     rq=1.393*rr^(-.528);
     elseif rr<=10.0,
     rt=1.625*rr^(-1.018);
     rq=1.956*rr^(-.870);
     elseif rr<=30.0,
     rt=4.661*rr^(-1.475);
     rq=4.994*rr^(-1.297);
     elseif rr<=100.0,
     rt=34.904*rr^(-2.067);
     rq=30.709*rr^(-1.845);
     elseif rr<=300.0,
     rt=1667.19*rr^(-2.907);
     rq=1448.68*rr^(-2.682);
     elseif rr<=1000.0,
     rt=5.88e5*rr^(-3.935);
     rq=2.98e5*rr^(-3.616);
     end;
     
  
     Cdn = (von/(log(zu/zo)))^2;
     Cuz = von/log(zu/zo);            % 风速输送系数
     Cqz = von/log(zt/zot);           % 湿度输送系数
     Ctz = von/log(zq/zot); 
     zot=rt*visa/usr;
     zoq=rq*visa/usr;
     usr=u*von/(log(zu/zo)-psiu_25(zu/L));
     tsr=-dt*von/(log(zt/zot)-psit_25(zt/L));
     qsr=-dq*von/(log(zq/zoq)-psit_25(zq/L));
     Bf=-grav/ta*usr*(tsr+0.61*ta*qsr);
     if Bf>0
     ug=Beta*(Bf*zi)^.333;
     else
     ug=0;
     end;
     u=sqrt(du*du+ug*ug);
 end
 
   %disp (zot)
   %disp (zoq)

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
tz = tsr/(von*(zu+zo))*fait((zu+zo)/L);
qz = qsr/(von*(zu+zo))*fait((zu+zo)/L);
Mz = A+B*qz+C*qz;
%syms z
%M=int(Mz,z,zo,50);

Conv = 1;

if zet>=0
   fprintf('\n稳定条件下的蒸发波导高度计算\n');
    H=-(B*tsr+C*qsr)/(von*(A+0.157)+5/L*(B*tsr+C*qsr));
else
   fprintf('\n不稳定条件下的蒸发波导高度计算\n');
    H1=-(B*tsr+C*qsr)*fait(zet)/von/(A+0.157);
   %%for i=1:20
  while Conv == 1 
    zet1=von*grav*H1*(tsr+0.61*ta*qsr)/ta/(usr^2);
    H=-(B*tsr+C*qsr)*fait(zet1)/von/(A+0.157);
    if abs(H-H1)<=0.0001 
           Conv = 0;
    else
           %g(i)= H; 
           H1=H;
    end
  end
end
    
   
%
disp(H)

a=load('17.txt');
x=a(1:6,4);
y1=a(1:6,1);
y2=a(1:6,2);
y3=a(1:6,3);
dyx=gradient(y3)./gradient(x);
figure(1)
subplot(1,3,1)
plot(t,h,'-r',y1,x,'.k')
title('01072719');
ylabel(gca,'垂直高度（米）');
xlabel(gca,'温度（k）');
subplot(1,3,2)
plot(e,h,'-r',y2,x,'.k');
title('01072719');
ylabel(gca,'垂直高度（米）');
xlabel(gca,'水汽压（hPa）');
subplot(1,3,3)
plot(Nh,h,'-r',y3,x,'.k')
title('01072719');
ylabel(gca,'垂直高度（米）');
xlabel(gca,'修正折射指数（M）');
print(gcf,'-dtiff','蒸发波导01072719-2','-r300');