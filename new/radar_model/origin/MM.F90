   
   
   subroutine MM(c1)
        use pemod
		implicit none

		!m:������ָ��, zt:�ݻ����, c1:��ϲ�б��, zb:�ݻ��׺��, q:�����������
		real:: m,zt,c1,zb,height,MMM,M1
		integer i
 !       write(*,*)m,zt,c1,zb,q

	   
	   !�Ĳ���ȷ������������

       do i=0,n
	      height=ht(i)
		  if (height==0) then
		     MMM=330		  
		  else
		     MMM=330+0.12513*(ht(i)-c1*log(ht(i)./1.5*10000));!米为单位
		  end if
		     ref(i)=MMM
	   end do      

   end subroutine