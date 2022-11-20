function phih2=stablehexN(L,z)
%under stable conditions,phih2
%z=10
global zu
ZL2=zu./L;
% phih2=1-(1+(2*ZL2/3)).^1.5-(2/3)*(ZL2-(5/0.35)).*exp(-0.35*ZL2)-(2/3)*(5/0.35);%NPS模型
phih2=-7*ZL2;%来自NRL模型
%phih2=-5*ZL2;%来自BYC模型