! PyRPMD MIT binding. Fetched BrHCl_LEPS1.f retains its upstream terms.
! The C++ caller serializes all accesses to the process-global COMMON blocks.
! q(3,3) views a C array q[atom][xyz]; order is Br,H,Cl, in bohr.
subroutine pyrpmd_brhcl_bare_frame(q, energy, force, radial_gradient, status) bind(C)
  use iso_c_binding, only: c_double, c_int32_t
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(c_double), intent(in) :: q(3,3)
  real(c_double), intent(out) :: energy, force(3,3), radial_gradient(3)
  integer(c_int32_t), intent(out) :: status
  real(c_double) :: cart(25,3), anuzero
  integer :: nulbl(25), nflag(20), nasurf(6,6), nder
  common /usricm/ cart, anuzero, nulbl, nflag, nasurf, nder
  real(c_double) :: pengygs, pengyes(5), pengyij(15)
  real(c_double) :: dgscart(25,3), descart(25,3,5), dijcart(25,3,15)
  common /usrocm/ pengygs, pengyes, pengyij, dgscart, descart, dijcart
  real(c_double) :: r(75), engygs, degsdr(75)
  common /pt1cm/ r, engygs, degsdr
  logical, save :: initialized = .false.
  integer :: a, b
  real(c_double) :: distance
  energy = ieee_value(0.0_c_double, ieee_quiet_nan)
  force = energy
  radial_gradient = energy
  status = 2_c_int32_t
  if (any(.not. ieee_is_finite(q))) return
  do a = 1, 2
    do b = a+1, 3
      distance = sqrt(sum((q(:,a)-q(:,b))**2))
      if (.not. ieee_is_finite(distance) .or. distance < 0.2_c_double) return
    end do
  end do
  if (.not. initialized) then
    ! Upstream PREPOT converts its constants in place: call exactly once.
    call prepot
    initialized = .true.
  end if
  cart = 0.0_c_double
  cart(1:3,:) = transpose(q)
  nder = 1
  call pot
  energy = pengygs
  force = -transpose(dgscart(1:3,:))
  radial_gradient = degsdr(1:3)
  status = 3_c_int32_t
  if (.not. ieee_is_finite(energy) .or. any(.not. ieee_is_finite(force)) &
      .or. any(.not. ieee_is_finite(radial_gradient))) then
    energy = ieee_value(0.0_c_double, ieee_quiet_nan)
    force = energy
    radial_gradient = energy
    return
  end if
  status = 0_c_int32_t
end subroutine
