function qs=qsee(ts,P)
x=ts;
p=P;
es=6.112.*exp(17.502.*x./(x+240.97))*0.98*(1.0007+3.46e-6*p);
qs=es*0.622/(p-0.378*es);