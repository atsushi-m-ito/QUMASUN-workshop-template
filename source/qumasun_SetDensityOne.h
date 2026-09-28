#pragma once
#include <cstdlib>
#include "soacomplex.h"

namespace QUMASUN {
namespace Basic {

    inline
    void SetDensityOne(GridRange& l_grid, double* __restrict l_rho, const SoAComplex* l_psi, const double* __restrict occupancy, int num_solution) {

        const int64_t size_3d = l_grid.Size3D();

#ifdef GY_WITH_CUDA_OR_HIP 
        const double* __restrict psi_l_re = l_psi[0].re;
        //bundle幅をnum_solution*2にすることで虚数部分も足しこめる.SoA構造が前提//
        gy::For(size_3d, GY_LAMBDA(int64_t i){
            double rho_i = 0.0;
            for (int s = 0; s < num_solution * 2; ++s) {
                int n = s / 2;
                const double factor = occupancy[n];
                const int64_t ii = i + size_3d * s;
                rho_i += (psi_l_re[ii] * psi_l_re[ii]) * factor;
            }
            l_rho[i] += rho_i;
        });
#else
        const double* __restrict psi_l_re = l_psi[0].re;

        for (int s = 0; s < num_solution * 2; ++s) {
            int n = s / 2;
            const double factor = occupancy[n];

            //bundle幅をnum_solution*2にすることで虚数部分も足しこめる.SoA構造が前提//
            gy::For(size_3d, GY_LAMBDA(int64_t i){
                const int64_t ii = i + size_3d * s;
                l_rho[i] += (psi_l_re[ii] * psi_l_re[ii]) * factor;
            });
        }
#endif

    }

}
}
