
   function psi=psiu(zet)
%  zet=0.5
	x=(1-15*zet).^.25;
	psik=2*log((1+x)/2)+log((1+x.*x)/2)-2*atan(x)+2*atan(1);
	x=(1-10*zet).^.3333;%2.6原为10.15
	psic=1.5*log((1+x+x.*x)/3)-sqrt(3)*atan((1+2*x)/sqrt(3))+4*atan(1)/sqrt(3);
	f=zet.*zet./(1+zet.*zet);
    %f=(zet.*zet)/40/(1+zet.*zet/40);
	psi=(1-f).*psik+f.*psic;                                               
  
    ii=find(zet>0);
   if ~isempty(ii);
	%%psi(ii)=-4.7*zet;
  	c(ii)=min(50,.35*zet(ii));
    %psi=-zet-2/3*(zet-5/0.35)/exp(0.35*zet)-2/3*5/0.35;  % 见 babin lkb 的文献
    %psi=-1-zet-2/3*(zet-5/0.35)/exp(0.35*zet)-2/3*5/0.35;  % 见 fairall 2.6
    psi(ii)=-((1+2/3*zet(ii)).^1.5+2/3*(zet(ii)-14.28)./exp(c(ii))+8.525);
end;