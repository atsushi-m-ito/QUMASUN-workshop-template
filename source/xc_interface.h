#pragma once
#include "GridRange.h"
#include "Vxc.h"

class XC_Interface {
public:
    
    void SetPotential_LDA(GridRangeMPI& l_grid, const double* rho, const double* pcc_rho, const double* rho_diff, double* Vx_up, double* Vc_up, double* Vx_down, double* Vc_down) {
        const size_t local_size = l_grid.Size3D();
        for (size_t i = 0; i < local_size; ++i) {
            double rho_tot = rho[i] + pcc_rho[i];
            auto ret = Calc_XC_LDA(rho_tot);
            Vx_up[i] = ret.V_x;
            Vc_up[i] = ret.V_c;
        }
    }

    void SetPotential_LSDA(GridRangeMPI& l_grid, const double* rho, const double* pcc_rho, const double* rho_diff, double* Vx_up, double* Vc_up, double* Vx_down, double* Vc_down) {
        const size_t local_size = l_grid.Size3D();
        for (size_t i = 0; i < local_size; ++i) {            
#ifdef XC_IMPLE_VER2
            auto ret = Calc_XC_LSDA_v2(rho[i] + pcc_rho[i], rho_diff[i]);
#else
            auto ret = Calc_XC_LSDA(rho[i] + pcc_rho[i], rho_diff[i] / (rho[i] + pcc_rho[i]));
#endif
            Vx_up[i] = ret.V_x_up;
            Vc_up[i] = ret.V_c_up;
            Vx_down[i] = ret.V_x_down;
            Vc_down[i] = ret.V_c_down;
        }
    }

};
