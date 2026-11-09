! PyRPMD binding. Fetched CBE/PES-2008 files retain their upstream terms.
! C-contiguous q[B,6,3] is the memory layout q(3,6,B) here.
! COMMON blocks and destructive PREPOT unit conversion require one initializer
! per loaded library and serialized calls. The Python owner supplies the lock.
!
! The upstream GET_POTENTIAL already accepts the whole ring q(3,6,P). Admitted
! frames are passed to it in one call, writing V and dV/dq directly into the
! caller's arrays; only a batch containing rejected frames is compacted.
!
! This unit deliberately does not USE IEEE_ARITHMETIC: gfortran then saves and
! restores the complete floating-point environment on every call. Finiteness
! is tested as abs(x) <= huge(x), which is false exactly for NaN and +-Inf.
subroutine pyrpmd_cbe2008_batch(n, q, energy, force, status) bind(C)
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  integer(c_int32_t), value, intent(in) :: n
  real(c_double), intent(in) :: q(3,6,n)
  real(c_double), intent(out) :: energy(n), force(3,6,n)
  integer(c_int32_t), intent(out) :: status(n)
  logical, save :: initialized = .false.
  real(c_double), parameter :: largest = huge(1.0_c_double)
  ! Pair guard: |r_a-r_b| >= 0.2 bohr. The fused test admits only squared
  ! distances clearly above 0.04, so a frame within rounding of the guard
  ! takes the exact original comparison below.
  real(c_double), parameter :: guard = 0.04_c_double, clear = 0.04_c_double*(1.0_c_double+1.0e-12_c_double)
  integer :: frame, k, a, b, natoms, nbeads, info, admitted
  integer, allocatable :: index(:)
  real(c_double), allocatable :: packed_q(:,:,:), packed_v(:), packed_g(:,:,:)
  real(c_double) :: nan, dx, dy, dz, d2
  logical :: ok
  if (n <= 0) return
  if (.not. initialized) then
    call initialize_potential()
    initialized = .true.
  endif
  nan = transfer(9221120237041090560_c_int64_t, 1.0_c_double)
  ! One fused admission pass. A nonfinite coordinate makes the squared
  ! distance of every pair containing its atom NaN or +Inf, so the single
  ! test clear <= d2 <= huge covers finiteness and the 0.2 bohr guard.
  ! A numerical singularity guard only; no claim of a physical PES domain.
  admitted = 0
  do frame = 1,n
    ok = .true.
    do a = 1,5
      do b = a+1,6
        dx = q(1,a,frame)-q(1,b,frame)
        dy = q(2,a,frame)-q(2,b,frame)
        dz = q(3,a,frame)-q(3,b,frame)
        d2 = dx*dx+dy*dy+dz*dz
        ok = ok .and. d2 >= clear .and. d2 <= largest
      enddo
    enddo
    if (.not. ok) ok = admissible(q(:,:,frame))
    if (ok) then
      status(frame) = 0
      admitted = admitted + 1
    else
      status(frame) = 1
    endif
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
  ! Upstream returns the positive gradient; report force = -dV/dq in place,
  ! testing finiteness in the same pass.
  do frame = 1,n
    if (status(frame) == 1) cycle
    if (status(frame) == 0) then
      ok = abs(energy(frame)) <= largest
      do b = 1,6
        do a = 1,3
          force(a,b,frame) = -force(a,b,frame)
          ok = ok .and. abs(force(a,b,frame)) <= largest
        enddo
      enddo
      if (ok) cycle
      status(frame) = 2
    endif
    energy(frame) = nan
    force(:,:,frame) = nan
  enddo
contains
  ! The original per-frame admission rule, used only when the fused test
  ! fails: every coordinate finite and every pair at least 0.2 bohr apart.
  logical function admissible(x)
    real(c_double), intent(in) :: x(3,6)
    integer :: i, j
    admissible = .false.
    if (any(.not. (abs(x) <= largest))) return
    do i = 1,5
      do j = i+1,6
        if (sum((x(:,i)-x(:,j))**2) < guard) return
      enddo
    enddo
    admissible = .true.
  end function admissible
end subroutine
