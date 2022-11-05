
function psi=psit(zet)
	
    x=(1-15*zet).^.5;
	psik=2*log((1+x)/2);
	x=(1-34*zet).^.3333;%toga2.6原为34.15
	psic=1.5*log((1+x+x.*x)/3)-sqrt(3)*atan((1+2*x)/sqrt(3))+4*atan(1)/sqrt(3);
	f=zet.*zet./(1+zet.*zet);
    %f=zet.*zet/40./(1+zet.*zet/40);
    psi=(1-f).*psik+f.*psic;  
   
   ii=find(zet>0);
if ~isempty(ii);
	%%psi=-4.7*zet;
    c=min(50,.35*zet);
    psi(ii)=-((1+2/3*zet(ii)).^1.5+.6667*(zet(ii)-14.28)./exp(c(ii))+8.525);
    %psi=1-(1+2/3*zet).^1.5-2/3*(zet-14.28)./exp(0.35*zet)-2/3*5/0.35;    % 见 babin lkb 的文献
    %psi=-(1+2/3*zet).^1.5-2/3*(zet-14.28)./exp(0.35*zet)-2/3*5/0.35; 
end;