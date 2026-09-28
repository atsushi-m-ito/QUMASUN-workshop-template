//simple mixing

#pragma once
#include "mpi.h"
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <vector>
#include "mixing_simple.h"
#include "inverse_m.h"

class MixingDIIS : public MixingBase {
private:
    const size_t m_local_size;
    bool is_spin;
    //double* ml_rho_prev = nullptr;
    MPI_Comm m_ddm_comm;
    int m_current_step = 0;
    const int m_steps_history;
    const int m_steps_simple;
    double* opt_R = nullptr;
    std::vector<double*> rho_list;
    std::vector<double*> R_list;
    double* matRR = nullptr;

public:
    MixingDIIS(size_t grid_size, int is_spin_, const MPI_Comm& ddm_comm, int steps_history ) :
        m_local_size(grid_size),
        is_spin(is_spin_ > 0),
        m_ddm_comm(ddm_comm),
        m_steps_history(steps_history),
        m_steps_simple(steps_history),
        rho_list(steps_history),
        R_list(steps_history)
        
    {
        rho_list[0] = nullptr;
    }

    ~MixingDIIS() {
        delete[] rho_list[0];
        
        delete[] matRR;
        delete[] opt_R;
    }

    void Initialize(const double* l_rho_init, const double* l_rho_diff) {
        const size_t local_spin_size = m_local_size * (is_spin ? 2 : 1);

        if (rho_list[0] == nullptr) {           

            double* p = new double[local_spin_size * m_steps_history * 2];
            for (int k = 0; k < m_steps_history; ++k) {
                rho_list[k] = p;
                p += local_spin_size;
            }
            for (int k = 0; k < m_steps_history; ++k) {
                R_list[k] = p;
                p += local_spin_size;
            }

            matRR = new double[(m_steps_history + 1) * (m_steps_history + 1)];
            opt_R = new double[local_spin_size];
        }

        memset(rho_list[0], 0, sizeof(double) * local_spin_size * m_steps_history * 2);
        memcpy(rho_list[0], l_rho_init, sizeof(double) * m_local_size);
        if (is_spin) {
            memcpy(rho_list[0] + m_local_size, l_rho_diff, sizeof(double) * m_local_size);
        }

        memset(matRR, 0, sizeof(double) * (m_steps_history + 1) * (m_steps_history + 1));

        m_current_step = 0;


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
        if (rho_list[0] == nullptr) {//事前にInitializeを呼んでいないエラー
            return 0.0;
        }

        const size_t local_spin_size = m_local_size * (is_spin ? 2 : 1);
        const double mixing_ratio = params[0];
        
        const int k = Index(m_current_step); 
        

        const auto* rho_k = rho_list[k];
        auto* R_k = R_list[k];

        double diff_abs_rho[2]{ 0.0,0.0 };
        for (size_t i = 0; i < m_local_size; ++i) {
            const double g  = (l_rho_in_out[i] - rho_k[i]);
            R_k[i] = g;
            diff_abs_rho[0] += fabs(g);
        }
        if (is_spin) {
            for (size_t i = 0; i < m_local_size; ++i) {
                const double g = (l_rho_diff_in_out[i] - rho_k[i + m_local_size]);
                R_k[i + m_local_size] = g;
                //diff_abs_rho[0] += fabs(g);
            }
        }

        double* vecRR = new double[m_steps_history+1];
        const int m_end = std::max(m_current_step - m_steps_history + 1, 0);
        for (int m1 = m_current_step; m1 >= m_end; --m1) {
            const int m = Index(m1);
            const auto* R_m = R_list[m];
            double c = 0.0;
            for (size_t i = 0; i < local_spin_size; ++i) {
                c += R_k[i] * R_m[i];                
            }
            vecRR[m] = c;
        }
        MPI_Reduce(&vecRR[0], &matRR[(m_steps_history + 1) * k], m_steps_history, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
        for (int m = 0; m < m_steps_history; ++m) {
            matRR[k + (m_steps_history + 1) * m] = matRR[m + (m_steps_history + 1) * k];
            matRR[m_steps_history + (m_steps_history + 1) * m] = 1.0;
            matRR[m + (m_steps_history + 1) * m_steps_history] = 1.0;
            vecRR[m] = 0.0;
        }
        matRR[m_steps_history + (m_steps_history + 1) * m_steps_history] = 0.0;
        vecRR[m_steps_history] = 1.0;

        if (m_current_step + 1 >= m_steps_history) {
            //DIIS//

            //estimate coefficient from matrix//
            if (IsRoot(m_ddm_comm)) {
                double* invRR = new double[(m_steps_history + 1) * (m_steps_history + 1) * 2];
                double* cpyRR = invRR + (m_steps_history + 1) * (m_steps_history + 1);
                memcpy(cpyRR, &matRR[0], sizeof(double) * (m_steps_history + 1) * (m_steps_history + 1));
                InverseM(cpyRR, (m_steps_history + 1), invRR);//入力も破壊されるので複製//
                for (int m = 0; m < m_steps_history + 1; ++m) {
                    vecRR[m] = invRR[m_steps_history + (m_steps_history + 1) * m];
                }

#ifdef DEBUG_PRINT_DIIS
                std::string matrix_out_path("DIIS_matrix");
                matrix_out_path += std::to_string(m_current_step) + ".txt";
                OutputMatrix(matRR, (m_steps_history + 1), (m_steps_history + 1), matrix_out_path.c_str());
#endif

                delete[] invRR;
            }

            MPI_Bcast(&vecRR[0], m_steps_history + 1, MPI_DOUBLE, 0, m_ddm_comm);

            const int next = Index(m_current_step + 1);
            auto* rho_n = rho_list[next];
            auto* R_n = R_list[next];            
            for (size_t i = 0; i < local_spin_size; ++i) {
                R_n[i] *= vecRR[next];
                rho_n[i] *= vecRR[next];
            }
            
            for (int m = 0; m < m_steps_history; ++m) {
                if (m != next) {
                    const auto* rho_m = rho_list[m];
                    const auto* R_m = R_list[m];
                    for (size_t i = 0; i < local_spin_size; ++i) {
                        R_n[i] += R_m[i] * vecRR[m];
                        rho_n[i] += rho_m[i] * vecRR[m];
                    }
                }
            }

            for (size_t i = 0; i < local_spin_size; ++i) {
                double g = R_n[i];
                rho_n[i] += g * mixing_ratio;
                diff_abs_rho[1] += fabs(g);
            }

            for (size_t i = 0; i < m_local_size; ++i) {
                l_rho_in_out[i] = rho_n[i];
            }
            if (is_spin) {
                for (size_t i = 0; i < m_local_size; ++i) {
                    l_rho_diff_in_out[i] = rho_n[i + m_local_size];
                }
            }


            delete[] vecRR;
        } else {
            //SimpleMixing//

            const int next = Index(m_current_step + 1);
            auto* rho_n = rho_list[next];            
            for (size_t i = 0; i < m_local_size; ++i) {
                double g = R_k[i];
                rho_n[i] = rho_k[i] + g * mixing_ratio;
                l_rho_in_out[i] = rho_n[i];
                diff_abs_rho[1] += fabs(g);
            }
            if (is_spin) {
                for (size_t i = 0; i < m_local_size; ++i) {
                    double g = R_k[i + m_local_size];
                    rho_n[i + m_local_size] = rho_k[i + m_local_size] + g * mixing_ratio;
                    l_rho_diff_in_out[i] = rho_n[i + m_local_size];
                    //diff_abs_rho[1] += fabs(g);
                }
            }
        }

        double tot_diff_abs_rho[2]{ 0.0, 0.0 };
        MPI_Reduce(&diff_abs_rho[0], &tot_diff_abs_rho[0], 2, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
        if (IsRoot(m_ddm_comm)) {
            printf("DIIS_optR = %.15f, %.15f\n", tot_diff_abs_rho[0], tot_diff_abs_rho[1]);
        }

        m_current_step++;

        return tot_diff_abs_rho[0];
        //return tot_diff_abs_rho[1];
    }

};
