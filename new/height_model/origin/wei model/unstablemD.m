function phim1=unstablemD(L)
%under unstable conditions,phim1 采用了NWA模型一样
%z=10;
global zu
ZL1=zu./L;
x=(1-16*ZL1).^0.25;
phim1=2*log((1+x)./2).*log((1+x.^2)/2)-2*atan(x)+pi/2;