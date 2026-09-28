#pragma once
#include <cstdio>
#include "vec3.h"
#include "nucleus.h"
#include "atomic_number.h"
#include "physical_param.h"
#include "qumasun_input.h"

inline 
void FprintRVM(const char* filepath, int num_nuclei, Nucleus* nuclei, const vec3d* p, const double* m, const double* box_axis)
{
    FILE* fp = fopen(filepath, "w");

    fprintf(fp, "#Simulation Box===========================\n\n");

    fprintf(fp, "Material.SuperCell     1 1 1\n");
    fprintf(fp, "Material.UnitLatticeVector.Unit    Bohr\n");
    fprintf(fp, "Material.UnitLatticeVector.Begin\n");
    fprintf(fp, "  %.15f  %.15f  %.15f\n", box_axis[0], box_axis[1], box_axis[2]);
    fprintf(fp, "  %.15f  %.15f  %.15f\n", box_axis[3], box_axis[4], box_axis[5]);
    fprintf(fp, "  %.15f  %.15f  %.15f\n", box_axis[6], box_axis[7], box_axis[8]);
    fprintf(fp, "Material.UnitLatticeVector.End\n");
    fprintf(fp, "\n\n");

    fprintf(fp, "#Last Positons of Nuclei===========================\n\n");

    fprintf(fp, "Material.UnitCell.Unit    Bohr\n");
    fprintf(fp, "Material.UnitCell.Begin\n");
    for (int i = 0; i < num_nuclei; ++i) {
        fprintf(fp, "  %s  %.15f  %.15f  %.15f\n", msz::GetAtomicSymbol(nuclei[i].Z), nuclei[i].Rx, nuclei[i].Ry, nuclei[i].Rz);
    }
    fprintf(fp, "Material.UnitCell.End\n");
    fprintf(fp, "\n\n");

    fprintf(fp, "#Last Mass and Velocity of Nuclei===========================\n\n");

    fprintf(fp, "Dynamics.Mass.Unit        u  # u, Da, me, or a.u.\n");
    fprintf(fp, "Dynamics.Velocity.Unit    a.u.  # m / s, or a.u.\n");
    fprintf(fp, "\n");
    fprintf(fp, "#note: velocity of 1[eV] and 1[u] atom corresponds to v = 13891.38738921143[m / s] = 0.0063497933278141[a.u.]\n");
    fprintf(fp, "#note : if mass is m[u] and energy is E[eV], v' = v * sqrt(E/m)\n");
    fprintf(fp, "\n");
    fprintf(fp, "Dynamics.Velocity.Begin\n");
    for (int i = 0; i < num_nuclei; ++i) {
        fprintf(fp, "  %.15f  %.15f  %.15f  %.15f\n", m[i] / mass_u_as_au, (p[i].x / m[i]), (p[i].y / m[i]), (p[i].z / m[i]));
    }
    fprintf(fp, "Dynamics.Velocity.End\n");

    fprintf(fp, "\n");

    fclose(fp);

}
