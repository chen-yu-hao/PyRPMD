! Same full-Cayley BCB propagation, but the complete NVE loop is native Fortran.
! q(3,3,P) has exactly the same bytes as C q[P,3,3] / PyRPMD arrays.
module pyrpmd_fortran_loop
  use iso_c_binding, only: c_int32_t,c_int64_t,c_double
  use, intrinsic :: ieee_arithmetic
  implicit none
  private
  public :: pyrpmd_loop_fortran
contains
  subroutine tri_solve(n,alpha,d,l,x)
    integer, intent(in) :: n
    real(c_double),intent(in) :: alpha,d(n),l(n)
    real(c_double),intent(inout) :: x(n)
    integer :: k
    do k=2,n
      x(k)=x(k)-l(k)*x(k-1)
    enddo
    x(n)=x(n)*d(n)
    do k=n-1,1,-1
      x(k)=(x(k)+alpha*x(k+1))*d(k)
    enddo
  end subroutine

  integer(c_int32_t) function pyrpmd_loop_fortran(p,mass,beta,dt,steps,q,momentum,mode) &
                                 bind(C,name="pyrpmd_loop_fortran") result(rc)
    integer(c_int32_t),value,intent(in)::p,mode
    integer(c_int64_t),value,intent(in)::steps
    real(c_double),value,intent(in)::beta,dt
    real(c_double),intent(in)::mass(3)
    real(c_double),intent(inout)::q(3,3,p),momentum(3,3,p)
    interface
      subroutine bkmp2_batch(n,pos,en,f,st) bind(C,name="pyrpmd_bkmp2_batch")
        import c_int32_t,c_double
        integer(c_int32_t),value,intent(in)::n
        real(c_double),intent(in)::pos(3,3,n)
        real(c_double),intent(out)::en(n),f(3,3,n)
        integer(c_int32_t),intent(out)::st(n)
      end subroutine
    end interface
    real(c_double),allocatable:: d(:),l(:),z1(:),z2(:),x(:),f(:,:,:),en(:)
    integer(c_int32_t),allocatable::status(:)
    real(c_double)::alpha,omega2,w,diag,det,si(4),a,b,rr,ss,old,mu
    integer::k,j,c1,jp,jm
    integer(c_int64_t)::it
    rc=0
    if (p<1.or.p>100000.or.steps<0.or.beta<=0.or..not.ieee_is_finite(beta).or. &
        .not.ieee_is_finite(dt).or.any(mass<=0).or.any(.not.ieee_is_finite(mass)).or. &
        (mode/=0.and.mode/=1)) then
      rc=-1;return
    endif
    omega2=(real(p,c_double)/beta)**2
    alpha=.25_c_double*dt*dt*omega2
    if(.not.ieee_is_finite(omega2).or..not.ieee_is_finite(alpha))then
      rc=-2;return
    endif
    allocate(d(p),l(p),z1(p),z2(p),x(p),f(3,3,p),en(p),status(p))
    d=0;l=0;z1=0;z2=0;si=0
    if(p>=3)then
      diag=1+2*alpha
      d(1)=1/diag
      do k=2,p
        l(k)=-alpha*d(k-1)
        d(k)=1/(diag+alpha*l(k))
      enddo
      z1(1)=-alpha;z2(p)=-alpha
      call tri_solve(p,alpha,d,l,z1)
      call tri_solve(p,alpha,d,l,z2)
      a=1+z1(p);b=z2(p);rr=z1(1);ss=1+z2(1)
      det=a*ss-b*rr
      if(det<=0.or..not.ieee_is_finite(det))then
        rc=-2;return
      endif
      si(1)=ss/det;si(2)=-b/det;si(3)=-rr/det;si(4)=a/det
    endif
    do it=1,steps
      if(mode==0)then
        call bkmp2_batch(p,q,en,f,status)
        if(any(status/=0))then
          rc=int(it,c_int32_t);return
        endif
      else
        f=-.005_c_double*q
      endif
      momentum=momentum+.5_c_double*dt*f
      ! Each Cartesian component is one periodic ring. This is the full ring,
      ! never P isolated beads. Keep the same loop order as the C/C++ versions.
      do j=1,3
        mu=mass(j)
        do c1=1,3
          if(p==1)then
            q(c1,j,1)=q(c1,j,1)+dt*momentum(c1,j,1)/mu
            cycle
          endif
          do k=1,p
            x(k)=q(c1,j,k)+.5_c_double*dt*momentum(c1,j,k)/mu
          enddo
          if(p==2)then
            a=x(1);b=x(2);diag=1+4*alpha
            x(1)=((1+2*alpha)*a+2*alpha*b)/diag
            x(2)=(2*alpha*a+(1+2*alpha)*b)/diag
          else
            call tri_solve(p,alpha,d,l,x)
            rr=si(1)*x(p)+si(2)*x(1)
            ss=si(3)*x(p)+si(4)*x(1)
            do k=1,p
              x(k)=x(k)-z1(k)*rr-z2(k)*ss
            enddo
          endif
          do k=1,p
            jm=modulo(k-2,p)+1
            jp=modulo(k,p)+1
            q(c1,j,k)=2*x(k)-q(c1,j,k)
            momentum(c1,j,k)=momentum(c1,j,k)-dt*mu*omega2*(2*x(k)-x(jm)-x(jp))
          enddo
        enddo
      enddo
      if(mode==0)then
        call bkmp2_batch(p,q,en,f,status)
        if(any(status/=0))then
          rc=int(it,c_int32_t);return
        endif
      else
        f=-.005_c_double*q
      endif
      momentum=momentum+.5_c_double*dt*f
    enddo
  end function
end module
