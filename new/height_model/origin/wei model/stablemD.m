function phim2=stablemD(L)
%under stable conditions,phim2
%z=10;
global zu
ZL2=zu/L;
% phim2=-ZL2-(2/3)*(ZL2-(5/0.35)).*exp(-0.35*ZL2)-(2/3)*(5/0.35);
phim2=-7*ZL2;%来自NRL模型