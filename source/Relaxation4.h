#pragma once
#include "mpi.h"
#include <float.h>
#include "vec3.h"
#include "nucleus.h"

namespace glips {

    /*
    * Relaxation4
    * to relax atomic positions with 2 layers loop.
    * The 1st layer is the loop of relaxation by executing SCF.
    * The 2nd layer is the loop of relaxation by keeping electromic state solved by SCF in th 1st layer.
    * That is, atomic positions moves in static electronic density, 
    * and then only Eext, Enn and Exc(by pcc) are changed, while Ekin and Ehart are fixed.
    * 
    * だめだ。むしろ遅くなってしまう。
    */
    class Relaxation4Q
    {
    private:
        double m_lower_force_limit = 1.0e-2;
        double m_upper_move_limit = 0.1;

        MPI_Comm m_mpi_comm;
        Relaxation4Q() = delete;
    public:

        Relaxation4Q(MPI_Comm mpi_comm) :
            m_mpi_comm(mpi_comm)
        {

        };

        void Folding(double& rx, double& ry, double& rz, const double* box_axis) {
            double ix = rx / box_axis[0];
            ix -= std::floor(ix);
            rx = ix * box_axis[0];

            double iy = ry / box_axis[4];
            iy -= std::floor(iy);
            ry = iy * box_axis[4];

            double iz = rz / box_axis[8];
            iz -= std::floor(iz);
            rz = iz * box_axis[8];
        }

        /*
        指定したステップだけ構造緩和を行う
        */
        template<class QUMASUN_T>
        void Evolve(QUMASUN_T& qumasun, const int num_steps, double init_dt, int num_nuclei, Nucleus* nuclei, double* box_axis, const int fix_count, double* total_energy)
        {
            const bool is_root = IsRoot(m_mpi_comm);


            //dt *= m_dt_scale;            
            //const double dt_half = init_dt / 2.0;
            double prev_U = 0.0;//DBL_MAX
            double max_f = 0.0;
            double dU = 0.0;
            double dt = init_dt;
            const double dt_upper_limit = init_dt * 10.0;
            const double dt_lower_limit = init_dt * 0.1;


            double prev_abs_f = 1.0;
            auto prev_f = std::make_unique<vec3d[]>(num_nuclei);
            auto prev2_f = std::make_unique<vec3d[]>(num_nuclei);
            auto f = std::make_unique<vec3d[]>(num_nuclei);

            {
                for (int i = 0; i < num_nuclei; ++i) {
                    prev_f[i].Clear();
                }

                /*
                note: DDM使用時はForceの計算によって粒子のソートや入れ替えが起こるため、
                m_prev_fもソートする必要がある。困った.
                解決:運動量pを前回の力を格納するバッファにすることで解決
                */
            }

            std::string cube_path;
            const int zero_length = std::to_string(num_steps).size();

            for (int istep = 0; istep < num_steps; istep++) {

                if (is_root) {
                    printf("====================================\n");
                    printf("Relaxation Step = %d\n", istep);
                    fflush(stdout);
                }

                qumasun.Execute(istep);


                std::string num_str = std::string(zero_length - std::to_string(istep).size(), '0') + std::to_string(istep);
                cube_path = "relax_" + num_str + ".cube";
                qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, cube_path.c_str());

                cube_path = "test_eigenvalue_" + num_str + ".txt";
                qumasun.OutputEigenValue(cube_path.c_str());


                if (is_root) {
                    qumasun.GetForce(&f[0]);				//力の計算.
#if 0
                    if (fix_count > 0) {
                        atom_container->ClearFByState();
                    }
#endif

                    double U = qumasun.GetTotalEnergy();
                    //potential->GetU(&U);
                    dU = U - prev_U;

                    prev_U = U;
                    //const double* m = atom_container->M(); mass is 1.0, tentatively.

                    double inner_f_prev_f = 0.0;
                    double abs_f = 0.0;
                    max_f = 0.0;
                    for (int i = 0; i < num_nuclei; i++) {
                        //if (m[i] != 0.0) 
                        {
                            double f2 = f[i] * f[i];
                            if (max_f < f2) {
                                max_f = f2;
                            }
                            abs_f += f2;
                            inner_f_prev_f += f[i] * prev_f[i];
                        }
                    }


                    max_f = sqrt(max_f);
                    abs_f = sqrt(abs_f);


                    /*dtの増減/////////////////////
                      前回のforceと向きが同じならdtを増やし、逆なら減らす。直行なら変化なし。
                      これを内積から決める
                      */
                    const double x = inner_f_prev_f / (abs_f * prev_abs_f);
                    const double f1 = 0.5;
                    const double fm1 = -0.5;
                    const double a2 = (f1 + fm1) / 2.0;
                    const double a1 = (f1 - fm1) / 2.0;
                    const double ratio = 1.0 + a1 * x + a2 * x * x;
                    dt *= ratio;
                    prev_abs_f = abs_f;

                    dt = ((max_f * dt > m_upper_move_limit) ? m_upper_move_limit / max_f : dt);
                    dt = std::min(dt, dt_upper_limit);
                    dt = std::max(dt, dt_lower_limit);


                    uint8_t check_convergence = 0;

                    printf("\nRelaxation info===================================\n");
                    printf("istep = %d, max_f = %g, dU = %f, dt = %f\n", istep, max_f, dU, dt);
                    if (max_f < m_lower_force_limit) {
                        printf("force-convergence: max_f = %g < lower_force_limit = %g\n", max_f, m_lower_force_limit);
                        if (istep == 0) {
                            dU = 0.0;
                        }
                        check_convergence = 1;
                    }


                    MPI_Bcast(&check_convergence, 1, MPI_UINT8_T, 0, m_mpi_comm);
                    if (check_convergence) {
                        break;
                    }


#if 0
                    if (use_state == UseState::FixAtom) {
                        for (int i = fix_count; i < num_nuclei; i++) {
                            if (atom_container->State(i) ^ IAtomContainer::STATE_FLAG_FIX) {
                                r[i] += dt * f[i];
                                prev_f[i] = f[i];
                            }
                        }
                    } else
#endif
                    {
                        for (int i = 0; i < num_nuclei; i++) {
                            //if (m[i] != 0.0) 
                            {
                                nuclei[i].Rx += dt * f[i].x;
                                nuclei[i].Ry += dt * f[i].y;
                                nuclei[i].Rz += dt * f[i].z;


                                //should be folding 
                                Folding(nuclei[i].Rx, nuclei[i].Ry, nuclei[i].Rz, box_axis);

                                prev_f[i] = f[i];
                            }
                        }
                    }
                } else { //slave process//
                    uint8_t check_convergence = 0;
                    MPI_Bcast(&check_convergence, 1, MPI_UINT8_T, 0, m_mpi_comm);
                    if (check_convergence) {
                        break;
                    }
                }

#if 0
                //next step//
                MPI_Bcast(nuclei, sizeof(Nucleus)* num_nuclei, MPI_BYTE, 0, m_mpi_comm);
                qumasun.MoveNuclei(num_nuclei, &nuclei[0]);
#endif

                //The 2nd layer loop, in which the electronic state is kept//
                {
                    const int NUM_2ND_STEPS = 20;
                    double prev_U2 = prev_U; //U is energy if 1st layer
                    double prev2_abs_f = prev_abs_f;
                    double dt2 = dt * 0.1;
                    for (int i = 0; i < num_nuclei; i++) {
                        prev2_f[i] = prev_f[i];
                    }


                    for (int i2nd = 0; i2nd < NUM_2ND_STEPS; ++i2nd) {

                        double U2 = qumasun.RecalculateEnergy();
                        double dU2 = U2 - prev_U2;
                        qumasun.GetForce(&f[0]);				//力の計算.

                        if (is_root) {

                            double inner_f_prev_f = 0.0;
                            double abs_f = 0.0;
                            double max_f = 0.0;
                            for (int i = 0; i < num_nuclei; i++) {
                                //if (m[i] != 0.0) 
                                {
                                    double f2 = f[i] * f[i];
                                    if (max_f < f2) {
                                        max_f = f2;
                                    }
                                    abs_f += f2;
                                    inner_f_prev_f += f[i] * prev2_f[i];
                                }
                            }


                            max_f = sqrt(max_f);
                            abs_f = sqrt(abs_f);

                            /*dtの増減/////////////////////
                              前回のforceと向きが同じならdtを増やし、逆なら減らす。直行なら変化なし。
                              これを内積から決める
                              */
                            const double x = inner_f_prev_f / (abs_f * prev2_abs_f);
                            const double f1 = 0.5;
                            const double fm1 = -0.5;
                            const double a2 = (f1 + fm1) / 2.0;
                            const double a1 = (f1 - fm1) / 2.0;
                            const double ratio = 1.0 + a1 * x + a2 * x * x;
                            //dt2 *= ratio;
                            prev2_abs_f = abs_f;

                            dt2 = ((max_f * dt2 > m_upper_move_limit) ? m_upper_move_limit / max_f : dt2);
                            dt2 = std::min(dt2, dt_upper_limit);
                            //dt2 = std::max(dt2, dt_lower_limit);


                            uint8_t check_convergence = 0;

                            printf("\nRelaxation info===================================\n");
                            printf("2nd_step = %d-%d, max_2f = %g, dU = %f, dt = %f\n", istep, i2nd, max_f, dU2, dt2);
                            if (max_f < m_lower_force_limit) {
                                printf("2nd_force-convergence: max_2f = %g < lower_force_limit = %g\n", max_f, m_lower_force_limit);

                                check_convergence = 1;
                            }


                            MPI_Bcast(&check_convergence, 1, MPI_UINT8_T, 0, m_mpi_comm);
                            if (check_convergence) {
                                break;
                            }

                            for (int i = 0; i < num_nuclei; i++) {
                                //if (m[i] != 0.0) 
                                {
                                    nuclei[i].Rx += dt2 * f[i].x;
                                    nuclei[i].Ry += dt2 * f[i].y;
                                    nuclei[i].Rz += dt2 * f[i].z;


                                    //should be folding 
                                    Folding(nuclei[i].Rx, nuclei[i].Ry, nuclei[i].Rz, box_axis);

                                    prev2_f[i] = f[i];
                                }
                            }

                        } else { //slave process//
                            uint8_t check_convergence = 0;
                            MPI_Bcast(&check_convergence, 1, MPI_UINT8_T, 0, m_mpi_comm);
                            if (check_convergence) {
                                break;
                            }
                        }
                    
                        //next step for 2nd loop//
                        MPI_Bcast(nuclei, sizeof(Nucleus) * num_nuclei, MPI_BYTE, 0, m_mpi_comm);
                        qumasun.MoveNuclei(num_nuclei, &nuclei[0]);

                    }//end of 2nd loop//

                }
                

                
            }//end of loop "istep"

            if (total_energy) {

                double U = 0.0;
                //potential->GetU(&U);
                total_energy[0] = U;
                total_energy[1] = (num_steps <= 1) ? 0.0 : dU;//dt//
                total_energy[2] = max_f;
            }


        }


        struct Parameters {
            double lower_force_limit;
        };

        /*
        Integratorの再初期化
        */
        void Reset(const void* si_vars) {
            const Parameters& params = *(const Parameters*)si_vars;
            m_lower_force_limit = params.lower_force_limit;
        }
        void OutputLog(const char* filepath) {};



    };

}//namespace//
