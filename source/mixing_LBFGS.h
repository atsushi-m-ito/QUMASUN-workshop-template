//simple mixing

#pragma once
#include "mpi.h"
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <vector>
#include "mixing_simple.h"

class MixingLBFGS : public MixingBase {
private:
    const size_t m_local_size;
    double* ml_rho_prev = nullptr;
    MPI_Comm m_ddm_comm;
    int m_current_step = 0;
    const int m_steps_history;
    const int m_steps_simple;
    double* ml_g_prev = nullptr;  //g_{k+1}が収められている//
    std::vector<double*> s_list;
    std::vector<double*> y_list;
    std::vector<double> c_list;  //c[m] is inner prod of s[m] and y[m]//

    TransKerker* kerker = nullptr;
    double dx;
    double dy;
    double dz;
    
public:
    MixingLBFGS(size_t grid_size, const MPI_Comm& ddm_comm, int steps_history,
        double dx_, double dy_, double dz_,
        TransKerker* kerker_) :
        m_local_size(grid_size),
        m_ddm_comm(ddm_comm),
        m_steps_history(steps_history),
        m_steps_simple(steps_history),
        s_list(steps_history),
        y_list(steps_history),
        c_list(steps_history),
        kerker(kerker_),
        dx(dx_), dy(dy_), dz(dz_)
    {
        s_list[0] = nullptr;
        //m_rho_prev = new double[grid_size];
    }

    ~MixingLBFGS() {
        delete[] ml_rho_prev;
        delete[] s_list[0];
        delete[] ml_g_prev;
        delete kerker;
    }

    void Initialize(const double* l_rho_init, const double* l_rho_diff) {
        if (ml_rho_prev == nullptr) {
            ml_rho_prev = new double[m_local_size];
            ml_g_prev = new double[m_local_size];

            double* p = new double[m_local_size * m_steps_history * 2];
            for (int k = 0; k < m_steps_history; ++k) {
                s_list[k] = p;
                p += m_local_size;
                y_list[k] = p;
                p += m_local_size;
            }


        }

        memcpy(ml_rho_prev, l_rho_init, sizeof(double) * m_local_size);
        memset(ml_g_prev, 0, sizeof(double) * m_local_size);
        memset(s_list[0], 0, sizeof(double) * m_local_size* m_steps_history * 2);

        m_current_step = 0;


        double total_diff0 = Integrate(l_rho_init) * dx * dy * dz;
        double total_diff1 = Integrate(ml_rho_prev) * dx * dy * dz;

        if (IsRoot(m_ddm_comm)) {
            printf("[0] TEST:LBFGS:0, %.15f, %.15f\n", total_diff0, total_diff1);
        }
    }


    inline 
    int Index(int m) {
        return (m + m_steps_history) % m_steps_history;
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
        
        const int k = Index(m_current_step); 
        const int km1 = Index(k-1); //k - 1
        auto* y_km1 = y_list[km1];  //y_{ k - 1 }
        const auto* s_km1 = s_list[km1];  //y_{ k - 1 }

#define LBFGSKerkerMode 2
#if (LBFGSKerkerMode == 1) 
        double diff_abs_rho = 0.0;
        for (size_t i = 0; i < m_local_size; ++i) {
            l_rho_in_out[i] = (l_rho_in_out[i] - ml_rho_prev[i]);
            
            diff_abs_rho += fabs(l_rho_in_out[i]);
        }

        kerker->Convert(l_rho_in_out, params[0], params[1]);

        for (size_t i = 0; i < m_local_size; ++i) {
            double g_k = -l_rho_in_out[i];
            y_km1[i] = g_k - ml_g_prev[i];
            ml_g_prev[i] = g_k;
        }

#else
        
        double diff_abs_rho = 0.0;
        for (size_t i = 0; i < m_local_size; ++i) {
            double g_k = -(l_rho_in_out[i] - ml_rho_prev[i]);
            y_km1[i] = g_k - ml_g_prev[i];
            ml_g_prev[i] = g_k;

            diff_abs_rho += fabs(g_k);
        }
        //sum_diff_rho *= 

        
        double total_diff[3]{ 0.0,0.0,0.0 };
        total_diff[0] = Integrate(ml_g_prev) * dx * dy * dz;
        total_diff[1] = Integrate(l_rho_in_out) * dx * dy * dz;
        total_diff[2] = Integrate(ml_rho_prev) * dx * dy * dz;
        
        if (IsRoot(m_ddm_comm)) {
            printf("[0] TEST:LBFGS:0, %.15f, %.15f, %.15f\n", total_diff[0], total_diff[1], total_diff[2]);
        }
#endif

        double gamma_k = 0.0;
        
        {//calculate c_{k-1}
            double c[2]{ 0.0, 0.0 };
            for (size_t i = 0; i < m_local_size; ++i) {
                c[0] += y_km1[i] * s_km1[i];
                c[1]+= y_km1[i] * y_km1[i];
            }
            double c_tot[2];
            MPI_Allreduce(&c, &c_tot, 2, MPI_DOUBLE, MPI_SUM, m_ddm_comm);
            c_list[km1] = 1.0 / c_tot[0];
            //gamma_k = c_tot[0] / c_tot[1];
            //gamma_k = mixing_ratio;
            gamma_k = 1.0;
        }
        if (m_current_step <= m_steps_simple) {
            gamma_k = mixing_ratio;
        }
        if (IsRoot(m_ddm_comm)) {
            printf("gamma_k = %.15f\n", gamma_k);
        }

        
        double* q = new double[m_local_size];
        //double* z = new double[m_local_size];
        memcpy(q, ml_g_prev, sizeof(double) * m_local_size);

        std::vector<double> a_list(m_steps_history);

        const int m_end = std::max(m_current_step - m_steps_history, 0);
        for (int m1 = m_current_step - 1; m1 >= m_end; --m1) {
            const int m = Index(m1);
            const double& c_m = c_list[m];
            const auto* s_m = s_list[m];
            const auto* y_m = y_list[m];

            double a_m = 0.0;
            for (size_t i = 0; i < m_local_size; ++i) {
                a_m += s_m[i] * q[i];
            }
            a_m *= c_m;
            double a_tot;
            MPI_Allreduce(&a_m, &a_tot, 1, MPI_DOUBLE, MPI_SUM, m_ddm_comm);
            a_m = a_tot;
            a_list[m] = a_m;


            if (m_current_step > m_steps_simple) {
                //q_{m} -> q_{m-1}
                for (size_t i = 0; i < m_local_size; ++i) {
                    q[i] -= a_m * y_m[i];
                }
            }
        }

        double* z = q;
        for (size_t i = 0; i < m_local_size; ++i) {
            z[i] = gamma_k * q[i];
        }
        
        if (m_current_step > m_steps_simple) {
            for (int ms = m_end; ms < m_current_step; ++ms) {
                const int m = Index(ms);
                const double& c_m = c_list[m];
                const auto* s_m = s_list[m];
                const auto* y_m = y_list[m];
                const double& a_m = a_list[m];

                double b_m = 0.0;
                for (size_t i = 0; i < m_local_size; ++i) {
                    b_m += y_m[i] * z[i];
                }
                b_m *= c_m;
                double b_tot;
                MPI_Allreduce(&b_m, &b_tot, 1, MPI_DOUBLE, MPI_SUM, m_ddm_comm);
                b_m = b_tot;


                for (size_t i = 0; i < m_local_size; ++i) {
                    z[i] += (a_m - b_m) * s_m[i];
                }
            }
            /*
            for (size_t i = 0; i < m_local_size; ++i) {
                z[i] *= mixing_ratio;
            }
            */
        }


#if (LBFGSKerkerMode == 2)
        if (kerker) {
            kerker->Convert(z, params[0], params[1]);
        }
#endif


        auto* s_k = s_list[k];  //y_{ k - 1 }
        for (size_t i = 0; i < m_local_size; ++i) {
            l_rho_in_out[i] = ml_rho_prev[i] - z[i];
            //const double s = l_rho_in_out[i] - ml_rho_prev[i];
            //s_k[i] = s;
            s_k[i] = -z[i];
            ml_rho_prev[i] = l_rho_in_out[i];
        }


        double tot_diff_abs_rho = 0.0;
        MPI_Reduce(&diff_abs_rho, &tot_diff_abs_rho, 1, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
        
        m_current_step++;
        delete[] q;

        return tot_diff_abs_rho;
    }

    double Integrate(const double* l_rho) {

        double sum_diff_rho[3]{ 0.0,0.0,0.0 };
        double diff_abs_rho = 0.0;
        for (size_t i = 0; i < m_local_size; ++i) {
            
            sum_diff_rho[0] += l_rho[i];
        }
        

        double total_diff[3]{ 0.0,0.0,0.0 };
        MPI_Reduce(sum_diff_rho, total_diff, 1, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
        return total_diff[0];
    }
};
