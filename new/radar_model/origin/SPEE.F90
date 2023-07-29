subroutine SPEE(aa,frq,height,xxx 输出)

	use pemod
    !pemod �����߷���
	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)
	real(kind=4)::aa 蒸发波导高度 ,dd  ,frq 雷达频率 , height 天线高度
	real::xxx(1024,200)


	! �״�ϵͳ�����ʹ�������������
	! ����Gerstoft et al. [2003, Radio Sci.] �����е�����
	rmax = 100.
	rmax = rmax * 1.d3 水平距离 100km
	nrout = 200 水平距离 100km 200份
	dr = rmax/dble(nrout) 间隔距离
   ! write(*,*)aa,dd,ee

	! �״�ϵͳ������Ĭ�ϼ�����ʽΪˮƽ����
	freq = frq	! ����Ƶ�ʣ���λMHz
	antht = height		! ���߸߶ȣ���λm
	ipat = 2		! ��������
	bw = 16			! ������ȣ���λdeg
	elv = 1.		! �������ǣ���λdeg

	wl = c0 / freq
	fko = 2.d0 * pi / wl  
	!����
	con = 1.e-6 * fko

   !��ʼ��Fourier�任����
	call getfftsz

	delp = pi / zmax	! determined by Nyquist theorem
	fnorm = 2. / n		! Fourier�任�еĹ�һ��ϵ��	
	cnst = delp / fko	! ����ȷ�����ɿռ���λ����(phase factors)
	nm1 = n - 1

	! Initialize variables and set-up filter array.
	no4 = n / 4
	n34 = 3. *  no4
	cn75 = 4. * pi / n

	if( allocated( filt ) ) deallocate( filt, stat=ierror )
	allocate( filt(0:no4), stat=ierror )
	if( ierror .ne. 0 ) stop 
	filt = 0.d0  电磁波不断往高度进行计算，200m就不算了，截止之后可能不好数值，用这个截断

	filt = .5 + .5 * dcos( (/(i, i=0, no4)/) * cn75 ) !�˲�����

	call allarray

	do i = 0, n
		ht(i) = dble(i) * delz   存高度
	end do

     call MM(aa)  !���������ȷ�������� 公式2里的 m2-1

	! Initialize starter field
	call xyinit(ROUT)	! rout��¼�������
	call fft(u)			! transform to z-space
	pobs(:,0) = u    公式2 U(x0,p)

	call phase1
	call phase2

    free=20 * log10(2*fko)  自由空间传播损耗

	do i = 1, nrout
		call pestep( rout )
		pobs(:,i) = u

		do j = 1, 1024
		    xxx(j,i)=10.*log10(rout/1000)+free-20.*log10(abs(pobs(j,i))) !�������
		end do
	end do
!    print *,  xxx(1023,nrout)

end subroutine

!#################################################################################################
subroutine allarray 

	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)

	ierror = 0

	if( allocated( ht ) ) deallocate( ht, stat=ierror )
	allocate( ht(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	ht = 0.d0  n 的矩阵

	if( allocated( ref ) ) deallocate( ref, stat=ierror )
	allocate( ref(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	ref = 0.d0

	if( allocated( mloss ) ) deallocate( mloss, stat=ierror )
	allocate( mloss(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	mloss = 0.d0


	if( allocated( u ) ) deallocate( u, stat=ierror )
	allocate( u(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	u = cmplx( 0., 0., 8 )

	if( allocated( frsp ) ) deallocate( frsp, stat=ierror )
	allocate( frsp(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	frsp = cmplx( 0., 0., 8 )

	if( allocated( envpr ) ) deallocate( envpr, stat=ierror )
	allocate( envpr(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	envpr = cmplx( 0., 0., 8 )

	if( allocated( pobs ) ) deallocate( pobs, stat=ierror )
	allocate( pobs(0:n,0:nrout), stat=ierror )
	if( ierror .ne. 0 ) return 
	pobs = cmplx( 0., 0., 8 )

 !   if( allocated( xxx ) ) deallocate( xxx, stat=ierror )
!	allocate( xxx(0:n,0:nrout), stat=ierror )
!	if( ierror .ne. 0 ) return 
!	xxx = ( 0., 0. )

	if( allocated( ulst ) ) deallocate( ulst, stat=ierror )
	allocate( ulst(0:n), stat=ierror )
	if( ierror .ne. 0 ) return 
	ulst = cmplx( 0., 0., 8)

	return

end subroutine
!#############################################################################################
! Purpose: Determines the antenna pattern factor for angle passed to routine

subroutine antpat( sang, patfac )

	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)

	common /pattern/ pelev, afac, umax, sbw

!	In the following pattern definitions, "ua" refers to the angle for which 
!	the antenna pattern is sought, and "u0" refers to the elevation angle.
!	ipat = 0 gives Omnidirectional antenna pattern factor : f(ua) = 1 

	patfac = 1. !antenna pattern factor

	if( ipat .gt. 1 ) then			!ipat=type of antenna pattern.
		ua = dasin( sang )			
		udif = ua - elv				!alpha_pat=alpha-miu_or
	end if

!	ipat = 1 gives Gaussian antenna pattern based on
!	f(p-p0) = exp(-w**2 * ( p-p0 )**2 ) / 4, where p = sin(ua) and p0 = sin(u0)

	if( ipat .eq. 1 ) then
		pr = sang - pelev 
		patfac = dexp( -pr * pr * afac ) !afac=the antenna factor

!	ipat = 2 gives sin(x)/x pattern based on 
!	f(ua-u0) = sin(x) / x where x = afac * sin(ua-u0) for |ua-u0| <= umax
!	f(ua-u0) = .03 for |ua-u0| > umax
!	ipat = 4 gives height-finder pattern which is a special case of sin(x)/x
        
	elseif (( ipat .eq. 2 ) .or. ( ipat .eq. 4 )) then
		if( ipat .eq. 4 ) then
			dirang = dabs( sang )  !dirang:sine of direct ray angle
			if( dirang .gt. elv ) udif = ua - dirang
		end if

		if( dabs(udif) .le. 1.e-6 ) then
			patfac = 1. 
		elseif( dabs( udif ) .gt. umax ) then
			patfac = .03 
		else
			arg = afac * dsin( udif )                
			patfac = dmin1( 1., dmax1( .03, dsin( arg ) / arg ) )
		end if

!	ipat = 3 gives csc-sq pattern based on
!	f(ua) = 1 for ua-u0 <= bw
!	f(ua) = sin(bw) / sin(ua-u0) for ua-u0 > bw
!	f(ua) = maximum of .03 or [1+(ua-u0)/bw] for ua-u0 < 0

	elseif( ipat .eq. 3 ) then
		if( udif .gt. bw ) then
			patfac = sbw / dsin( udif )
		elseif( udif .lt. 0 ) then
			patfac = dmin1( 1., dmax1( .03, (1. + udif / bw) ) )
		end if               
	end if
    
	return

end subroutine
!#############################################################################################
subroutine fft( udum )

	use pemod

	implicit integer(kind=4) (i-n)

	include 'fftsiz.inc'

	complex( kind=8 ) udum(0:*)

	dimension x(0:maxpts), y(0:maxpts)

	do i = 0, n
		x(i) = real( udum(i) )  实部
		y(i) = imag( udum(i) )  虚部
	end do

	call sinfft( ln, x )
	call sinfft( ln, y )

	do i = 0, n
		udum(i) = cmplx( x(i), y(i), 8 )
	end do

	return

end
!################################################################################################
subroutine getfftsz

	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)

	! Ϊ�˼��㷽�㣬����ֱ�Ӷ�����ر�����ֵ����ʹ֮����Nyquist����

	delz = 1. 1m还是1km
	ln = 10
	n = 2**ln
	zmax = delz * dble(n)          最大高度最高能算到多少
	return

end
!#################################################################################################
subroutine pestep( rout )

	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)

	save r

	if( rout .le. 1.e-3 ) r = 0
	rout = rout + dr
	
	rlast = r
	ulst = u

	r = r + dr
 !   write(*,*)rmax,r
	!  TRANSFORM TO FOURIER SPACE  
	call fft( u )

	!  Multiply by free-space propagator.        
	u = u * frsp

	!  TRANSFORM BACK TO Z-SPACE  
	call fft( u )

	! Multiply by environment term.
	u = u * envpr

!	do i = 0, n
!		write(*,*) u(i)
!	end do

	return
end subroutine
!#############################################################
! Purpose: Initialize free-space propagator array FRSP() using narrow-angle propagator

! Local Variables:
!	AK = Term used in ANG for each bin, i.e., I*DELP
!	AKSQ = Square of AK
!	ANG = Exponent term: ANG = -i * dr * (p**2)/(2*k0), where k is the free-space wavenumber,
!						 p is the transform variable (p=k*sin(theta)),
!	ATTN = Attenuation factor for filtering


! Reference: AD-A248112, Frank J. Ryan, 1991

subroutine phase1

	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)
	
	double precision cak

	do i = 0, n
		ak = dble(i) * delp
		aksq = ak * ak
		ang = aksq * dr / ( 2.d0 * fko )
		ca = dcos( ang )
		sa = -dsin( ang )
		frsp(i) = fnorm * cmplx( ca, sa, 8) !��Fourier��任������ϵ��fnorm
	end do

	! Filter the upper 1/4 of the propagator arrays.
	frsp(n34:n) = filt(0:no4) * frsp(n34:n)  对应

!	do i = 0, n
!		write(30,*) dble(i) * delp, frsp(i)
!	end do

	return

end
!############################################################################
! Purpose: Calculates the environmental phase term for a given profile, 
!		   then stores in the array envpr().

subroutine phase2
 
	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)
	real(kind=8):: m

	do i = 0, n
		m = 1 + ref(i)*1.d-6
	!	write(*,*)m		
		m = m*m        
		ang = fko * ( m - 1.d0 ) * dr / 2.d0 
		ca = dcos( ang )
		sa = dsin( ang )
		envpr(i) = cmplx( ca, sa, 8 )

	end do

	! Filter upper 1/4 of the arrays.
	envpr(n34:n) = filt(0:nf4) * envpr(n34:n)

!	do i = 0, n
!		write(30,*) i, envpr(i)
!	end do

	return

end
!!!!!!!#################################################################
subroutine xyinit( ROUT )

	use pemod

	implicit integer(kind=4) (i-n)
	implicit real(kind=8) (a-h, o-z)    

	common /pattern/ pelev, afac, umax, sbw 
	
	complex(kind=8):: refcoef, rterm, dterm  

! Reflection coefficient is defaulted to -1 for horizontal polarization.
	 
	refcoef = dcmplx(-1., 0.)	! complex reflection coefficient       
	sgain = dsqrt(WL) / zmax	!the normalization factor ��һ������
	dtheta = delp / fko			!the angle difference between mesh points in p-space      
	antko = fko * antht			!the height-gain value at the source�������ߴ��ĸ߶�����ֵ

!	write(*,*) wl, zmax, sgain, delp, fko, dtheta, antht, antko, refcoef

!	Calculate constants used to determine antenna pattern factor 
!		IPAT = 0 -> omni
!		IPAT = 1 -> gaussian
!		IPAT = 2 -> sinc x
!		IPAT = 3 -> csc**2 x
!		IPAT = 4 -> generic height-finder    
      
	bw = bw * radc			!convert degree to arc
	elv = elv * radc
	bw2 = .5 * bw


	if( ipat .eq. 1 ) then
		afac = .34657359 / (dsin( bw2 ))**2 ! constant used in determining 
	                                        ! antenna pattern factors
		pelev = dsin( elv )                 ! sine of elevation angle
	elseif( ipat .eq. 3 ) then
		sbw = dsin( bw )                    !sine of the beamwidth
	elseif( ipat .ne. 0 ) then
		afac = 1.39157 / dsin( bw2 )
		a = pi / afac 
		umax = datan( a / dsqrt(1. - a**2)  ) 
	end if

	do I=1,N
		pk = dble(i) * dtheta	!the direct-path ray elevation angle 
		zpk = pk * antko
         
!		Get antenna pattern factors for the direct and reflected rays.
		call antpat( pk, FACD ) 
		call antpat( -pk, FACR )

		rterm = dcmplx( dcos( zpk ), dsin( zpk ) )
		dterm = dconjg( rterm ) !����

		u(i) = sgain * ( facd * dterm + refcoef * facr * rterm )

	end do

	! Filter upper 1/4 of the field.
	do i = n34, n
		attn = filt( i-n34 )
		u(i) = attn * u(i)		!�����ĳ�ʼ����p�ռ�
	end do

!	do i = 0, n
!		write(30,*) dble(i)*dtheta, u(i)
!	end do

	rout = 0

	return

end