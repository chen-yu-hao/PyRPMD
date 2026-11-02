! PyRPMD binding. Fetched CBE/PES-2008 files retain their upstream terms.
! C-contiguous q[B,6,3] is the memory layout q(3,6,B) here.
! COMMON blocks and destructive PREPOT unit conversion require one initializer
! per loaded library and serialized calls. The Python owner supplies the lock.
subroutine pyrpmd_cbe2008_batch(n, q, energy, force, status) bind(C)
  use iso_c_binding, only: c_double, c_int32_t
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer(c_int32_t), value, intent(in) :: n
  real(c_double), intent(in) :: q(3,6,n)
  real(c_double), intent(out) :: energy(n), force(3,6,n)
  integer(c_int32_t), intent(out) :: status(n)
  logical, save :: initialized = .false.
  logical :: invalid
  integer :: frame, a, b, natoms, nbeads, info
  real(c_double) :: frame_q(3,6,1), value(1), gradient(3,6,1), distance2
  if (.not. initialized) then
    call initialize_potential()
    initialized = .true.
  endif
  do frame = 1,n
    energy(frame) = ieee_value(0.0_c_double, ieee_quiet_nan)
    force(:,:,frame) = energy(frame)
    status(frame) = 1
    if (any(.not. ieee_is_finite(q(:,:,frame)))) cycle
    invalid = .false.
    do a = 1,5
      do b = a+1,6
        distance2 = sum((q(:,a,frame)-q(:,b,frame))**2)
        if (distance2 < 0.04_c_double) invalid = .true.
      enddo
    enddo
    ! A numerical singularity guard only; no claim of a physical PES domain.
    if (invalid) cycle
    frame_q(:,:,1) = q(:,:,frame)
    natoms = 6
    nbeads = 1
    call get_potential(frame_q, natoms, nbeads, value, gradient, info)
    status(frame) = 2
    if (info /= 0) cycle
    if (.not. ieee_is_finite(value(1))) cycle
    if (any(.not. ieee_is_finite(gradient))) cycle
    energy(frame) = value(1)
    force(:,:,frame) = -gradient(:,:,1)
    status(frame) = 0
  enddo
end subroutine
