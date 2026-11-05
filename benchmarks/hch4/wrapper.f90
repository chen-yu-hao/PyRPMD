! PyRPMD binding. Fetched CBE/PES-2008 files retain their upstream terms.
! C-contiguous q[B,6,3] is the memory layout q(3,6,B) here.
! COMMON blocks and destructive PREPOT unit conversion require one initializer
! per loaded library and serialized calls. The Python owner supplies the lock.
!
! The upstream GET_POTENTIAL already accepts the whole ring q(3,6,P). Admitted
! frames are passed to it in one call, writing V and dV/dq directly into the
! caller's arrays; only a batch containing rejected frames is compacted.
subroutine pyrpmd_cbe2008_batch(n, q, energy, force, status) bind(C)
  use iso_c_binding, only: c_double, c_int32_t
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer(c_int32_t), value, intent(in) :: n
  real(c_double), intent(in) :: q(3,6,n)
  real(c_double), intent(out) :: energy(n), force(3,6,n)
  integer(c_int32_t), intent(out) :: status(n)
  logical, save :: initialized = .false.
  integer :: frame, k, a, b, natoms, nbeads, info, admitted
  integer, allocatable :: index(:)
  real(c_double), allocatable :: packed_q(:,:,:), packed_v(:), packed_g(:,:,:)
  real(c_double) :: nan
  if (n <= 0) return
  if (.not. initialized) then
    call initialize_potential()
    initialized = .true.
  endif
  nan = ieee_value(0.0_c_double, ieee_quiet_nan)
  ! One fused admission pass: finite coordinates and every pair >= 0.2 bohr.
  ! A numerical singularity guard only; no claim of a physical PES domain.
  admitted = 0
  do frame = 1,n
    status(frame) = 1
    if (any(.not. ieee_is_finite(q(:,:,frame)))) cycle
    pairs: do a = 1,5
      do b = a+1,6
        if (sum((q(:,a,frame)-q(:,b,frame))**2) < 0.04_c_double) exit pairs
      enddo
    enddo pairs
    if (a <= 5) cycle
    status(frame) = 0
    admitted = admitted + 1
  enddo
  natoms = 6
  if (admitted == n) then
    ! Common case: the full ring in one upstream call, no copies.
    nbeads = n
    call get_potential(q, natoms, nbeads, energy, force, info)
    if (info /= 0) status = 2
  else
    energy = nan
    force = nan
    if (admitted == 0) return
    allocate(index(admitted), packed_q(3,6,admitted), packed_v(admitted), packed_g(3,6,admitted))
    k = 0
    do frame = 1,n
      if (status(frame) /= 0) cycle
      k = k + 1
      index(k) = frame
      packed_q(:,:,k) = q(:,:,frame)
    enddo
    nbeads = admitted
    call get_potential(packed_q, natoms, nbeads, packed_v, packed_g, info)
    do k = 1,admitted
      energy(index(k)) = packed_v(k)
      force(:,:,index(k)) = packed_g(:,:,k)
      if (info /= 0) status(index(k)) = 2
    enddo
  endif
  ! Upstream returns the positive gradient; report force = -dV/dq in place.
  do frame = 1,n
    if (status(frame) == 1) cycle
    if (status(frame) == 0) then
      if (ieee_is_finite(energy(frame)) .and. all(ieee_is_finite(force(:,:,frame)))) then
        force(:,:,frame) = -force(:,:,frame)
        cycle
      endif
      status(frame) = 2
    endif
    energy(frame) = nan
    force(:,:,frame) = nan
  enddo
end subroutine
