%function Mz=M_Wei_N(T,To,p,u,RH)
clc;clear;
k=0.4;      %��������
s=0.62197;  %�ɿ���Ħ����
g=9.72;     %�������ٶ�
loa=1.293;  %0��ʱ�Ŀ����ܶ�

Rd=287.05;  %�ɿ��������峣��
Cpa=1004.67;%�ɿ�����ѹ����
R=6.371e+6; %����İ뾶
v=1.461e-5; %�˶�ճ��ϵ��
global hpop_val zu Number
%input atmospheric parameter 9-6-6: 40 2006-9-6-8:30 2006 -9-5-8:10
zo=0.000015;%�ֲڶ�
zu=6;        %̽��߶�

% u=7.98;         %����
% t=3.8;  %����
% to=2; %ˮ�� 
% p=1022.07;          %��ѹ
% RH=78.8/100;    %ʪ��
t =10.4;
RH =57;
to =15.8;
u = 3.66;
p = 1015.36;
% t =23;
% RH =100;
% to =28;
% u = 7;
% p = 1009;
T=t+273.16;  %����
To=to+273.16; %ˮ��
Es=6.1078*exp(17.27*(T-273.16)/(T-35.86));%���¶��µı���ˮ��ѹ
Esea=6.1078*exp(17.27*(To-273.16)/(To-35.86));%���¶��µı���ˮ��ѹ
ep=RH*Es;  %���¶ȵ�ˮ��ѹ

qs=0.622*Esea/(p-0.378*Esea); %���ͱ�ʪ
q=0.622*ep/(p-0.378*ep);      %��ʪ
theta=T*(1000./p).^0.286;     %λ��
theta0=To*(1000./p).^0.286;     %����λ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if Number==1
%    A=xlsread('E:\refractivit\�Ϻ�����ˮ������'.9,'C3:E41');
%    T1=A(hpop_val,1)+273.16; %����
%    p1=A(hpop_val,2)         %��ѹ
%    RH1=A(hpop_val,3)/100;   %ʪ��
%    R=6.371e+6;              %����İ뾶
%    Es1=6.1078*exp(17.27*(T1-273.16)/(T1-35.86));%���¶��µı���ˮ��ѹ
%    ep1=RH1*Es1;  %���¶ȵ�ˮ��ѹ
%    M61=77.6*p1./T1+(3.73e+5)*ep1./(T1.^2)+z./R*(1e+6); %̽������
% else
%    R=6.371e+6;              %����İ뾶 
%    Es=6.1078*exp(17.27*(T-273.16)./(T-35.86));%���¶��µı���ˮ��ѹ
%    ep=RH.*Es;  %���¶ȵ�ˮ��ѹ
   M61=77.6*p./T+(3.73e+5)*ep./(T.^2)+zu./R*(1e+6); %̽������
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%����Lֵ�ж�������ȶ���%%%%%%%%%%%%%%%%%%%%%%
%unstablem unstableh����������Grachev et al(2000),��NPL
%unstablemD unstablehD����������NWAģ�ͺ�NRLģ�͵��㷨
L1=100;   %����L�ĳ�ֵ
for i=1:100
    ux=(u*k)./(log(zu./zo)-unstablemD(L1))/0.74;
    Tx=k*(T-To)./(log(zu./zo)-unstablehD(L1))/0.74;
    thetax=k.*(theta-theta0)./(log(zu./zo)-unstablehD(L1))/0.74;
    qx=k*(q-qs)./(log(zu./zo)-unstablehD(L1))/0.74;
    ex=k*(ep-Esea)./(log(zu./zo)-unstablehD(L1))/0.74;
    L3=ux.^2*T./(k*g*(Tx+0.61*T.*qx));
    L1=L3;
end
disp('Ī��-�²����򳤶�');
L1
%%%%%%%%%%%%%%%%%%%%%%%%%%�����߶�%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
if L1<0
    syms zd1 ;
    t=diff(unstablehDex(L1,zd1),zd1);
    zd1=100;   %�趨��ֵ����ѭ��
    for i=1:100
    zd2=(25.26*Tx-89.9*ex).*(1-zd1.*eval(t));
    zd1=zd2;
    end
    disp('���ȶ�ʱ�����������߶�');
    zd1
    %o1=1-zd1.*eval(t)
    %z1=linspace(0,300,1000);
    if zd1>40
       zd1=0;
    end

M0=M61-(-0.125*zd1/(1-zd1.*eval(t)).*(log((zo+zu)/zo)-unstablehDex(L1,zo+zu)+unstablehDex(L1,zo))+0.125*zu);
zu=0.001:0.2:200;
Mz=M0-0.125*zd1/(1-zd1.*eval(t)).*(log((zo+zu)/zo)-unstablehDex(L1,zo+zu)+unstablehDex(L1,zo))+0.125;
else
    %%%%%%%%%%%%%%%%%%%%%%%%%�ȶ�״̬��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L1=1258;  %����L�ĳ�ֵ
for i=1:20
    ux=(u*k)./(log(zu./zo)-stablemD(L1))/0.74;
    Tx=k*(T-To)./(log(zu./zo)-stablemD(L1))/0.74;
    thetax=k*(theta-theta0)./(log(zu./zo)-stablemD(L1))/0.74;
    qx=k*(q-qs)./(log(zu./zo)-stablemD(L1))/0.74;
    ex=k*(ep-Esea)./(log(zu./zo)-stablemD(L1))/0.74;
    L3=ux.^2*T/(k*g*(Tx+0.61*T.*qx));
    L1=L3;
end 
L1
syms zd1 ;
t=diff(stablehexN(L1,zd1),zd1);
zd1=100;
for i=1:100
    zd2=(25.26*Tx-89.9*ex).*(1-zd1*eval(t));
    zd1=zd2;
end
disp('�ȶ�ʱ�����������߶�');
zd1
%0=1-zd1.*eval(t)
if zd1>40
    zd1=0;
end

M0=M61-(-0.125*zd1/(1-zd1.*eval(t)).*(log((zo+zu)/zo)-stablehexN(L1,zo+zu)+stablehexN(L1,zo))+0.125*zu);
zu=0.001:0.2:200;
%z1=lujing
Mz=M0-0.125*zd1/(1-zd1.*eval(t)).*(log((zo+zu)/zo)-stablehexN(L1,zo+zu)+stablehexN(L1,zo))+0.125*zu;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
zd1
%figure(1)
%hold on
%plot(Mz,z,'-g','LinVidth',2)
%hold off
