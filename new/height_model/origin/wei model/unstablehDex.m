function phih1=unstablehDex(L,z)
%under unstable conditions,phih1 ������NWA\NRlģ��һ��
%z=10;
global zu
ZL1=zu./L;
y=(1-16*ZL1).^0.25;
phih1=2*log((1+y.^2)./2);