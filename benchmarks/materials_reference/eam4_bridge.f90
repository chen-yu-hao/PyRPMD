! A static C interface to the unmodified Cambridge/Suleimanov EAM4 routines.
! This file does not implement the potential or a mixed-bead RPMD model.
! Coordinates: angstrom. Energy: eV. Returned derivatives: eV/angstrom.
! Atom order: one H, mobile Ni, fixed Ni. Fixed-Ni derivatives are zero
! by the upstream contract; its fixed--fixed pair-energy constant is omitted.
subroutine pyrpmd_eam4_static(nmobile, nfixed, periodic_xy, energy_only, &
        q, lengths, energy, gradient, status) bind(C)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
    implicit none
    integer(c_int), value, intent(in) :: nmobile, nfixed, periodic_xy, energy_only
    real(c_double), intent(in) :: q(3,1+nmobile+nfixed), lengths(3)
    real(c_double), intent(out) :: energy, gradient(3,1+nmobile+nfixed)
    integer(c_int), intent(out) :: status
    real(c_double) :: qm(3,nmobile,1), qr(3,nfixed,1), qh(3,1,1)
    real(c_double) :: gm(3,nmobile,1), gr(3,nfixed,1), gh(3,1,1)
    character(len=6) :: boundary

    status = 1
    energy = ieee_value(0.0_c_double, ieee_quiet_nan)
    gradient = energy
    if (nmobile < 0 .or. nfixed < 0 .or. nmobile+nfixed < 1) return
    if (periodic_xy < 0 .or. periodic_xy > 1) return
    if (energy_only < 0 .or. energy_only > 1) return
    if (.not. all(ieee_is_finite(q))) return
    if (.not. all(ieee_is_finite(lengths))) return
    if (any(lengths <= 0.0_c_double)) return
    ! The upstream minimum-image cutoff is 10 angstrom, not a general Ewald sum.
    if (periodic_xy == 1 .and. any(lengths(1:2) <= 20.0_c_double)) return
    qh(:,1,1) = q(:,1)
    qm(:,:,1) = q(:,2:1+nmobile)
    qr(:,:,1) = q(:,2+nmobile:1+nmobile+nfixed)
    boundary = 'afixed'
    if (periodic_xy == 1) boundary = 'bfixed'
    if (energy_only == 1) then
        call eamnih_energy(nmobile, nfixed, 1, lengths(1), lengths(2), lengths(3), &
            boundary, .true., qm, qr, qh, energy)
        gradient = ieee_value(0.0_c_double, ieee_quiet_nan)
    else
        call eamnih_forces_umb(nmobile, nfixed, 1, lengths(1), lengths(2), lengths(3), &
            boundary, .true., qm, qr, qh, energy, gm, gr, gh)
        gradient(:,1) = gh(:,1,1)
        gradient(:,2:1+nmobile) = gm(:,:,1)
        gradient(:,2+nmobile:1+nmobile+nfixed) = gr(:,:,1)
    end if
    status = 2
    if (.not. ieee_is_finite(energy)) return
    if (energy_only == 0 .and. .not. all(ieee_is_finite(gradient))) return
    status = 0
end subroutine pyrpmd_eam4_static
