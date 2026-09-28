//simple mixing

#pragma once
#include "mpi.h"
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include "mixing_kerker.h"

class MixingBase {
    
public:    
    virtual ~MixingBase() {};
    virtual void Initialize(const double* l_rho_init, const double* l_rho_diff) = 0;
    virtual double Predict(double* l_rho_in_out, double* l_rho_diff_in_out, double* params) = 0;
};


class MixingSimple : public MixingBase {
private:
    const size_t m_local_size;
    double* ml_rho_prev = nullptr;
    MPI_Comm m_ddm_comm;
public:
    MixingSimple(size_t grid_size, const MPI_Comm& ddm_comm) : 
        m_local_size(grid_size),
        m_ddm_comm(ddm_comm)
    {
        //m_rho_prev = new double[grid_size];
    }

    ~MixingSimple() {
        delete[] ml_rho_prev;
    }

    void Initialize(const double* l_rho_init, const double* l_rho_diff) {
        if (ml_rho_prev == nullptr) {
            ml_rho_prev = new double[m_local_size];
        }

        memcpy(ml_rho_prev, l_rho_init, sizeof(double) * m_local_size);

    }

    /*
    * rho_in_outにはpsiの二乗として得られた電子密度を入力する
    * すると、入力値と過去の履歴から予測される最適値を書き戻す
    * 戻り値として、前回の予測値との差を返す
    */
    double Predict(double* l_rho_in_out, double* l_rho_diff_in_out, double* params) {
        if (ml_rho_prev == nullptr) {//事前にInitializeを呼んでいないエラー
            return 0.0;
        }

        const double mixing_ratio = params[0];

        double diff_abs_rho = 0.0;

        for (size_t i = 0; i < m_local_size; ++i) {
            const double diff = l_rho_in_out[i] - ml_rho_prev[i];
            l_rho_in_out[i] = diff * mixing_ratio + ml_rho_prev[i];
            diff_abs_rho += fabs(diff);

            ml_rho_prev[i] = l_rho_in_out[i];
        }
        
        double tot_diff_abs_rho = 0.0;
        MPI_Reduce(&diff_abs_rho, &tot_diff_abs_rho, 1, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);



        return tot_diff_abs_rho;
    }


};


class MixingKerker : public MixingBase {
private:
    const size_t m_local_size;
    double* ml_rho_prev = nullptr;
    int is_spin; 
    MPI_Comm m_ddm_comm;
    TransKerker kerker;
public:
    MixingKerker(size_t grid_size, int is_spin_, const MPI_Comm& ddm_comm,
        GridRange& global_grid, GridRangeMPI& l_grid, double dx, double dy, double dz) :
        m_local_size(grid_size),
        is_spin(is_spin_),
        m_ddm_comm(ddm_comm),
        kerker(global_grid, l_grid, dx, dy, dz)
    {
        //m_rho_prev = new double[grid_size];
    }

    ~MixingKerker() {
        delete[] ml_rho_prev;
    }

    void Initialize(const double* l_rho_init, const double* l_rho_diff) {
        if (ml_rho_prev == nullptr) {
            ml_rho_prev = new double[m_local_size * (is_spin ? 2 : 1)];
        }

        memcpy(ml_rho_prev, l_rho_init, sizeof(double) * m_local_size);
        if (is_spin) {
            memcpy(ml_rho_prev + m_local_size, l_rho_diff, sizeof(double) * m_local_size);
        }
    }

    /*
    * rho_in_outにはpsiの二乗として得られた電子密度を入力する
    * すると、入力値と過去の履歴から予測される最適値を書き戻す
    * 戻り値として、前回の予測値との差を返す
    */
    double Predict(double* l_rho_in_out, double* l_rho_diff_in_out, double* params) {
        if (ml_rho_prev == nullptr) {//事前にInitializeを呼んでいないエラー
            return 0.0;
        }

        double diff_abs_rho[2]{ 0.0, 0.0 };
        auto OnePredict = [&params](double* l_rho_in_out, double* l_rho_prev, int local_size, double& diff_abs_rho, TransKerker& kerker) {
            for (size_t i = 0; i < local_size; ++i) {
                l_rho_in_out[i] -= l_rho_prev[i];
                diff_abs_rho += fabs(l_rho_in_out[i]);
            }

            kerker.Convert(l_rho_in_out, params[0], params[1]);

            for (size_t i = 0; i < local_size; ++i) {
                l_rho_prev[i] += l_rho_in_out[i];
                l_rho_in_out[i] = l_rho_prev[i];
            }
            };

        OnePredict(l_rho_in_out, ml_rho_prev, m_local_size, diff_abs_rho[0], kerker);

        if (is_spin) {
            OnePredict(l_rho_diff_in_out, ml_rho_prev + m_local_size, m_local_size, diff_abs_rho[1], kerker);
            if (IsRoot(m_ddm_comm)) {
                printf( "diff_abs_rho1 = %.12f\n", diff_abs_rho[1]);
            }
        }

        double tot_diff_abs_rho[2];
        MPI_Reduce(&diff_abs_rho[0], &tot_diff_abs_rho[0], 2, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);


        return tot_diff_abs_rho[0];
    }


};

