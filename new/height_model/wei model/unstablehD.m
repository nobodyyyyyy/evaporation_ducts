function phih1=unstablehD(L)
%under unstable conditions,phih1 采用了NWA和NRL模型一样
%z=10;
global zu
ZL1=zu./L;
y=(1-16*ZL1).^0.25;
phih1=2*log((1+y.^2)./2);