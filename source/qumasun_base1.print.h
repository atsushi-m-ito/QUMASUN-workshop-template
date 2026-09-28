#pragma once
#include <mpi.h>
#include "qumasun_base1.h"
#include "grid_cutoff_energy.h"


inline
void QUMASUN_BASE1::PrintCondition() {
    using namespace QUMASUN;
    const bool is_root = IsRoot(m_mpi_comm);
    if (!is_root)return;

    printf("\nSimulation Condition====================\n");
    printf("Box width: %f, %f, %f [Bohr]\n", m_box_x, m_box_y, m_box_z);
    printf("         : %f, %f, %f [Ang]\n", m_box_x * Bohr_as_Ang, m_box_y * Bohr_as_Ang, m_box_z * Bohr_as_Ang);
    printf("Grid size: %d, %d, %d\n", m_global_grid.SizeX(), m_global_grid.SizeY(), m_global_grid.SizeZ());
    printf("Grid width(dx,dy,dz): %f, %f, %f [Bohr]\n", m_dx, m_dy, m_dz);
    printf("                    : %f, %f, %f [Ang]\n", m_dx * Bohr_as_Ang, m_dy * Bohr_as_Ang, m_dz * Bohr_as_Ang);
    printf("High-resolution (HR) ratio for core charge: %d, %d, %d\n", m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);

    const double Ecut_x = CutoffEnergyFromGridWidth(m_dx);
    const double Ecut_y = CutoffEnergyFromGridWidth(m_dy);
    const double Ecut_z = CutoffEnergyFromGridWidth(m_dz);
    printf("Ecut = 0.5*(2pi/2dx)^2\n"
        "  (Cutoff energy corresponding to plane wave basis)\n");
    printf("Ecut(x): %f [Hartree] = %f [Ry] = %f [eV]\n", Ecut_x, Ecut_x * 2.0, Ecut_x * Hartree_as_eV);
    printf("Ecut(y): %f [Hartree] = %f [Ry] = %f [eV]\n", Ecut_y, Ecut_y * 2.0, Ecut_y * Hartree_as_eV);
    printf("Ecut(z): %f [Hartree] = %f [Ry] = %f [eV]\n", Ecut_z, Ecut_z * 2.0, Ecut_z * Hartree_as_eV);
    printf("\n");
    printf("k-point sample: %d, %d, %d\n", m_kpoint_sampling[0], m_kpoint_sampling[1], m_kpoint_sampling[2]);
    printf("Spin polarization: %s\n", (is_spin_on ? "on" : "off"));
    printf("number of orbitals(num_solutions) = %d\n", num_solution);
    printf("====================Simulation Condition\n\n");
    fflush(stdout);
}

