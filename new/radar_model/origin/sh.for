
      !��Ӧֵ����
      program sh 

      implicit real*8 (a-h,o-z)
      save

      real(kind=4)::a,b,c,d,frq,height
	real::TTT(1024,200),TTT1(1024,200),noise
      complex xxx(1024,200)

	open(10,file='test2.txt')
	    read(10,*)a
	    read(10,*)frq
	    read(10,*)height
	close(10) 
c      write(*,*)a,b,c,d
      funcval=1.0d0

	   call SPEE(a,frq,height,TTT)
c	   TTT(:,:)=xxx(:,:)

	open(10,file='clode2.txt')
      do i=1,1024
          do j =1,200
            write(10,*) TTT(i,j)
          end do
      end do
	close(10)



      
      end