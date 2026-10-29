! PyRPMD MIT-licensed binding; the fetched BKMP2 source retains its own terms.
! C-contiguous q[B,3,3] has the same memory layout as q(3,3,B) here.
subroutine pyrpmd_bkmp2_radial(r, energy, gradient, status) bind(C)
  use iso_c_binding, only: c_double, c_int32_t
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(c_double), intent(in) :: r(3)
  real(c_double), intent(out) :: energy, gradient(3)
  integer(c_int32_t), intent(out) :: status
  energy = ieee_value(0.0_c_double, ieee_quiet_nan)
  gradient = energy
  status = 1
  ! Guard the upstream CHGEOM STOP paths before calling legacy Fortran.
  if (any(.not. ieee_is_finite(r))) return
  if (minval(r) < 0.2_c_double) return
  if (2.0_c_double * maxval(r) > sum(r) + 1.e-6_c_double) return
  call pote(r, energy, gradient)
  status = 2
  if (.not. ieee_is_finite(energy)) return
  if (any(.not. ieee_is_finite(gradient))) return
  status = 0
end subroutine

subroutine pyrpmd_bkmp2_batch(n, q, energy, force, status) bind(C)
  use iso_c_binding, only: c_double, c_int32_t
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer(c_int32_t), value, intent(in) :: n
  real(c_double), intent(in) :: q(3,3,n)
  real(c_double), intent(out) :: energy(n), force(3,3,n)
  integer(c_int32_t), intent(out) :: status(n)
  integer :: frame, pair, a, b
  integer, parameter :: left(3) = [1,2,1], right(3) = [2,3,3]
  real(c_double) :: delta(3,3), r(3), gradient(3), contribution(3)
  interface
    subroutine pyrpmd_bkmp2_radial(r, e, g, info) bind(C)
      use iso_c_binding, only: c_double, c_int32_t
      real(c_double), intent(in) :: r(3)
      real(c_double), intent(out) :: e, g(3)
      integer(c_int32_t), intent(out) :: info
    end subroutine
  end interface
  do frame = 1,n
    do pair = 1,3
      delta(:,pair) = q(:,right(pair),frame) - q(:,left(pair),frame)
      r(pair) = sqrt(sum(delta(:,pair)**2))
    enddo
    call pyrpmd_bkmp2_radial(r, energy(frame), gradient, status(frame))
    force(:,:,frame) = ieee_value(0.0_c_double, ieee_quiet_nan)
    if (status(frame) /= 0) cycle
    force(:,:,frame) = 0.0_c_double
    do pair = 1,3
      a = left(pair); b = right(pair)
      contribution = gradient(pair) * delta(:,pair) / r(pair)
      force(:,a,frame) = force(:,a,frame) + contribution
      force(:,b,frame) = force(:,b,frame) - contribution
    enddo
    if (any(.not. ieee_is_finite(force(:,:,frame)))) status(frame) = 2
  enddo
end subroutine
