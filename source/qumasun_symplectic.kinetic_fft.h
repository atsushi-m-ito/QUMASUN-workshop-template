#pragma once

#include "qumasun_symplectic.h"
#include "fftw_executor.h"

namespace QUMASUN {
    /*
    * 時間発展を行う。
    * 戻り値として時間発展後の運動エネルギーを返す
    * [重要]時間発展では位相回転するだけであり、
    * 時間発展の前後で運動エネルギーは変化しない
    */
    inline
    double EvolveKineticInKspace(OneComplex* psi_k, double dt, int size_x, int size_y, int size_z, double dx, double dy, double dz,
            int kx_end = 0) {

        //kx_end is used for FFTW_R2C mode//
        if (kx_end == 0) {
            kx_end = size_x;
        }
        const int kz_end = size_z;

        //NOTE: 
        //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
        //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

        auto SQ = [](double x) { return x * x; };

        double Ekin = 0.0;

        for (int kz = 0; kz < kz_end; ++kz) {
            const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz ));
            for (int ky = 0; ky < size_y; ++ky) {
                const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

                for (int kx = 0; kx < kx_end; ++kx) {
                    const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
                    const size_t i = kx + kx_end * (ky + (size_y * kz));

                    const double k2_2 = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
                    const double phase = -dt*k2_2;
                    const double cos_p = cos(phase);
                    const double sin_p = sin(phase);
                    const double psi_re = psi_k[i].r;
                    const double psi_im = psi_k[i].i;
                    psi_k[i].r = cos_p * psi_re - sin_p * psi_im;
                    psi_k[i].i = cos_p * psi_im + sin_p * psi_re;

                    Ekin += k2_2 * (psi_k[i].r * psi_k[i].r + psi_k[i].i * psi_k[i].i);
                    //Ekin += k2_2 * (psi_re * psi_re + psi_im * psi_im);
                }

            }
        }

        return Ekin * dx * dy * dz;
    }
}

inline 
double QUMASUN_SYMPLECTIC::mEvolveK_FFT(double dt) {
    const bool is_root_ddm = IsRoot(m_ddm_comm);
    const int64_t global_size = m_global_grid.Size3D();
    const double invV = 1.0 / global_size;

    double Ekin = 0.0;


    DDMGatherScatter exchanger;

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        auto& l_psi = ml_wave_set[sk].l_psi_set;

        const auto range = exchanger.Estimate(m_global_grid, ml_grid, num_solution);
        exchanger.GatherExchangeD2Z(range, m_global_grid, m_work, ml_grid, l_psi[0].re, num_solution, m_work + 2 * m_size_3d * range.num_bundle);
        watch.Record(83);


#ifdef USE_FFT_MANY 

        //error check//
        if (m_fft_many_3d.GetNumBundle() != range.num_bundle) {            
            printf("num_bundle != num_gather_scatter, %s, %d\n", __FILE__, __LINE__); fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        m_fft_many_3d.ForwardDirect((OneComplex*)&m_work[0]);
        watch.Record(87);

        for (int s = 0; s < range.num_bundle; ++s) {

            auto* psi_whole_complx = (OneComplex*)&m_work[s * 2 * global_size];

            double res_K = QUMASUN::EvolveKineticInKspace(psi_whole_complx, dt, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
            
            const double occupancy = ml_wave_set[sk].occupancy[range.offset + s];
            Ekin += occupancy * res_K;
        }
        watch.Record(84);

        m_fft_many_3d.BackwardDirect((OneComplex*)&m_work[0]);
        watch.Record(88);

        //このスケーリングを逆FFTの前に持ってくると、数値誤差が乗って収束しなくなる. FFTのバグかも.//
        const int64_t i_end = global_size * 2 * range.num_bundle;
        for (int64_t i = 0; i < i_end; ++i) {
            m_work[i] *= invV;
        }
        watch.Record(84);

#else

        for (int s = 0; s < range.num_bundle; ++s) {

            auto* psi_whole_complx = (OneComplex*)&m_work[s * 2 * global_size];

            m_fftw->ForwardDirect(psi_whole_complx);
            double res_K = QUMASUN::EvolveKineticInKspace(psi_whole_complx, dt, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
            m_fftw->BackwardDirect(psi_whole_complx);

            for (int64_t i = 0; i < global_size; ++i) {
                psi_whole_complx[i].r *= invV;
                psi_whole_complx[i].i *= invV;
            }
            const double occupancy = ml_wave_set[sk].occupancy[range.offset + s];
            Ekin += occupancy * res_K;
        }
        watch.Record(84);
#endif
        

        exchanger.ScatterExchangeZ2D(range, ml_grid, l_psi[0].re, m_global_grid, m_work, num_solution, m_work + 2 * m_size_3d * range.num_bundle);
        watch.Record(85);

    }

    Ekin *= invV;

    //exchanger.MergeTimer(95, watch);

    double tot_EK = 0.0;
    MPI_Reduce(&Ekin, &tot_EK, 1, MPI_DOUBLE, MPI_SUM, 0, m_mpi_comm);
    return tot_EK;


}
