
function psi=fait(zet)
	x=(1-16*zet).^0.5;	%psik=2*log((1+x)/2);
	y=(1-12.87*zet).^0.3333;
	%psic=1.5*log((1+x+x.*x)/3)-sqrt(3)*atan((1+2*x)/sqrt(3))+4*atan(1)/sqrt(3);
	f=zet.*zet./(1+zet.*zet);    psi=(1-f)/x+f/y;  

