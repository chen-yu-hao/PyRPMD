! PyRPMD MIT. One isolated Caracal state per persistent worker process.
! Caracal is linked as a separately built, source-pinned optional dependency.
program pyrpmd_caracal_worker
use iso_c_binding
use general
use evb_mod
use pbc_mod
use ieee_arithmetic
implicit none
interface
  function c_read(fd, buffer, nbytes) bind(C, name="read") result(rc)
    import c_int, c_ptr, c_size_t, c_int64_t
    integer(c_int), value :: fd
    type(c_ptr), value :: buffer
    integer(c_size_t), value :: nbytes
    integer(c_int64_t) :: rc
  end function
  function c_write(fd, buffer, nbytes) bind(C, name="write") result(rc)
    import c_int, c_ptr, c_size_t, c_int64_t
    integer(c_int), value :: fd
    type(c_ptr), value :: buffer
    integer(c_size_t), value :: nbytes
    integer(c_int64_t) :: rc
  end function
end interface
integer(c_int64_t), target :: count, ready
integer(c_int64_t) :: rc, b
integer :: chdir_status, atom_index, component, displacement
character(len=40) :: force_mode, step_string
real(c_double) :: step_size, samples(4)
real(c_double), allocatable :: displaced(:,:), discarded_gradient(:,:)
character(len=4096) :: directory
real(c_double), allocatable, target :: coordinates(:,:,:), energies(:), forces(:,:,:)
call get_command_argument(3, force_mode)
call get_command_argument(4, step_string)
read(step_string,*,iostat=chdir_status) step_size
if (chdir_status /= 0 .or. step_size <= 0) stop 78
if (force_mode /= "analytic" .and. force_mode /= "fd") stop 79
call get_command_argument(2, directory)
if (len_trim(directory) < 1) stop 71
call chdir(trim(directory), chdir_status)
if (chdir_status /= 0) stop 72
use_mpi = .false.
use_rpmd = .true.
use_explore = .true.
use_calc_rate = .false.
use_stick_coeff = .false.
calc_freq_int = .false.
call prog_initial(0)
open(unit=15, file="caracal.log", status="replace")
call getkey(0)
call getxyz(0)
nbeads = 1
call read_pes(0)
allocate(displaced(3,natoms),discarded_gradient(3,natoms))
ready = int(natoms, c_int64_t)
rc = c_write(3, c_loc(ready), 8_c_size_t)
if (rc /= 8) stop 73
! The complete batch is transferred in one transaction. No Python callback.
do
  call transfer_read(c_loc(count), 8_c_size_t)
  if (count == 0) exit
  if (count < 0 .or. count > 1000000 .or. natoms < 1) stop 74
  if (count * int(natoms,c_int64_t) > 10000000) stop 75
  allocate(coordinates(3,natoms,count), energies(count), forces(3,natoms,count))
  call transfer_read(c_loc(coordinates), 24_c_size_t*natoms*count)
  do b = 1, count
    spline_all = .true. ! stateless full-path search; never reuse another bead's local neighborhood
    call gradient(coordinates(:,:,b), energies(b), forces(:,:,b), 1, 0)
    if (force_mode == "fd") then
      ! Explicit fourth-order energy differentiation for TREQ; not the original
      ! approximate analytical gradient. Cost: 1+12*N energy/gradient calls.
      do atom_index = 1, natoms
        do component = 1, 3
          do displacement = 1, 4
            displaced = coordinates(:,:,b)
            select case(displacement)
              case(1)
                displaced(component,atom_index)=displaced(component,atom_index)-2*step_size
              case(2)
                displaced(component,atom_index)=displaced(component,atom_index)-step_size
              case(3)
                displaced(component,atom_index)=displaced(component,atom_index)+step_size
              case(4)
                displaced(component,atom_index)=displaced(component,atom_index)+2*step_size
            end select
            spline_all = .true.
            call gradient(displaced,samples(displacement),discarded_gradient,1,0)
          end do
          forces(component,atom_index,b)=(samples(1)-8*samples(2)+8*samples(3)-samples(4))/(12*step_size)
        end do
      end do
    end if
  end do
  forces = -forces
  call transfer_write(c_loc(energies), 8_c_size_t*count)
  call transfer_write(c_loc(forces), 24_c_size_t*natoms*count)
  deallocate(coordinates, energies, forces)
end do
contains
  subroutine transfer_read(ptr, size)
    type(c_ptr), value :: ptr
    integer(c_size_t), value :: size
    integer(c_int8_t), pointer :: buf(:)
    integer(c_int64_t) :: used, got
    call c_f_pointer(ptr, buf, [size])
    used = 0
    do while (used < size)
      got = c_read(3, c_loc(buf(used+1)), size-used)
      if (got <= 0) stop 76
      used = used+got
    end do
  end subroutine
  subroutine transfer_write(ptr, size)
    type(c_ptr), value :: ptr
    integer(c_size_t), value :: size
    integer(c_int8_t), pointer :: buf(:)
    integer(c_int64_t) :: used, sent
    call c_f_pointer(ptr, buf, [size])
    used = 0
    do while (used < size)
      sent = c_write(3, c_loc(buf(used+1)), size-used)
      if (sent <= 0) stop 77
      used = used+sent
    end do
  end subroutine
end program
