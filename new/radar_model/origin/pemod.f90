module pemod

	real( kind=8 ) pi
	parameter ( pi = 3.1415926535897932d0 )    !Self-explanatory

!	common / pevar / freq, antht, bw, elv, wl, fko, con, delz, zmax, &
!				 delp, fnorm, cnst, rmax, dr, ipat, ln, n, nm1, no4, n34, n75
						
	real( kind=8 ) freq, antht, bw, elv, wl, fko, con, delz, zmax, delp, &
				   fnorm, cnst, rmax, dr,free,a
	integer( kind=4 ) ipat, ln, n, nm1, no4, n34, n75, nrout


	real(kind=8), allocatable :: filt(:), ht(:), ref(:), mloss(:)
	public :: filt, ht, ref, mloss

	complex(kind=8), allocatable :: u(:), frsp(:), envpr(:), pobs(:,:), ulst(:)
	public :: u, frsp, envpr, pobs, ulst

! CONSTANT
	real( kind=8 ) c0, radc, gamma
	complex( kind=8 ) qi

	data c0 / 299.79245d0 /				! speed of light x 1e.-6 m/s
	data radc / 1.74533d-2 /			! degree to radian conversion factor
	data qi / (0.d0, 1.d0) /			! Imaginary i
	data gamma / 0.01d0 /

end module