#pragma once
#include <mpi.h>
#include "mpi_helper.h"
#include "qumasun_input.h"
#include "Relaxation2.h"
#include "folding.h"
#include "md3_writer2.h"
#include "qumasun_td.h"
#include "print_rvm.h"
#include "StopWatch.h"



inline
int TimeEvoQUMASUN(QUMASUN_TD& qumasun, QUMASUN::Input& input, MPI_Comm& mpi_comm, QUMASUN::DynamicsMode mode, const int* ddm_num) {
	using namespace QUMASUN;
	
    const bool is_root = IsRoot(mpi_comm);
    int proc_id = GetProcessID(mpi_comm);

    StopWatch<10, true> watch;

    auto RhoFileName = [](int istep, int max_step) {
        int64_t length = std::to_string(max_step).length();
        std::string filename("rho_t");
        length -= std::to_string(istep).length();
        if (length > 0) {
            filename += std::string(length, '0');
        }
        filename += std::to_string(istep) + ".cube";
        return filename;
        };
    auto DiffRhoFileName = [](int istep, int max_step) {
        int64_t length = std::to_string(max_step).length();
        std::string filename("diff_rho_t");
        length -= std::to_string(istep).length();
        if (length > 0) {
            filename += std::string(length, '0');
        }
        filename += std::to_string(istep) + ".cube";
        return filename;
        };


	switch (mode) {
		case DynamicsMode::TDDFT:
		{

            const double dt = input.dynamics_timestep_dt;
            const int total_steps = input.dynamics_step;
            const int incremental_step = input.dynamics_output_step;
            qumasun.Initialize();
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, RhoFileName(0, total_steps).c_str());
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DiffDensity, DiffRhoFileName(0, total_steps).c_str());

            for (int istep = 0; istep < total_steps; istep += incremental_step) {
                qumasun.Evolve(istep, incremental_step, dt);
                qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, RhoFileName(istep+ incremental_step, total_steps).c_str());
                qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DiffDensity, DiffRhoFileName(istep + incremental_step, total_steps).c_str());
            }
			

            break;
		}		
        case DynamicsMode::EhrenfestMDv1:
        {

            const double* box_axis =input.box_axis;

            const double dt = input.dynamics_timestep_dt;
            const int start_steps = input.dynamics_start_step;
            const int total_steps = input.dynamics_step + start_steps;
            const int incremental_step = input.dynamics_output_step;

            
            msz::MD3_Writer2<double> md3;
            md3.Open("nucl_position.md3");
            auto WriteMD3_one = [&md3, &box_axis](int num_nuclei, Nucleus* nuclei) {
                std::vector<int> atoms_Z(num_nuclei);
                std::vector<vec3d> r(num_nuclei);
                for (int i = 0; i < num_nuclei; ++i) {
                    atoms_Z[i] = nuclei[i].Z;
                    r[i].x = nuclei[i].Rx;
                    r[i].y = nuclei[i].Ry;
                    r[i].z = nuclei[i].Rz;
                }

                md3.BeginFrame(num_nuclei);
                md3.WriteZ(&atoms_Z[0]);
                md3.WriteR(&r[0]);
                md3.WriteBox(box_axis);
                md3.EndFrame();
                };
            WriteMD3_one(input.num_nuclei, &input.nuclei[0]);

            //show velocity//
            if (is_root) {
                printf("Mass and Velocity [a.u.] ===========================\n");
                for (int i = 0; i < input.num_nuclei; ++i) {
                    printf("%d: %f, %f, %f, %f\n", i, input.mass[i], input.velocity[i*3], input.velocity[i * 3+1], input.velocity[i * 3+2]);
                }
                printf("====================================================\n");
                fflush(stdout);
            }
            std::vector<vec3d> P(input.num_nuclei );
            std::vector<vec3d> F(input.num_nuclei );
            for (int i = 0; i < input.num_nuclei; ++i) {
                P[i].x = input.velocity[i * 3] * input.mass[i];
                P[i].y = input.velocity[i * 3+1] * input.mass[i];
                P[i].z = input.velocity[i * 3+2] * input.mass[i];
            }

            auto EvolveR = [&box_axis](int num_nuclei, Nucleus* nuclei, const vec3d* p, const double* m, double dt) {
                for (int i = 0; i < num_nuclei; ++i) {
                    nuclei[i].Rx += p[i].x / m[i] * dt;
                    nuclei[i].Ry += p[i].y / m[i] * dt;
                    nuclei[i].Rz += p[i].z / m[i] * dt;

                    Folding(nuclei[i].Rx, nuclei[i].Ry, nuclei[i].Rz, box_axis);
                }
                };
            auto EvolveP = [](int num_nuclei, vec3d* p, const vec3d* f, double dt) {
                for (int i = 0; i < num_nuclei; ++i) {
                    p[i].x += f[i].x * dt;
                    p[i].y += f[i].y * dt;
                    p[i].z += f[i].z * dt;
                }
                };
            
            auto KineticEnergy = [](int num_nuclei, const vec3d* p, const double* m) {
                double K = 0.0;
                for (int i = 0; i < num_nuclei; ++i) {
                    K += ((p[i].x * p[i].x) + (p[i].y * p[i].y) + (p[i].z * p[i].z)) / (2.0 * m[i]);
                }
                return K;
                };

            watch.Restart();
            qumasun.Initialize();
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, RhoFileName(0, total_steps).c_str());
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DiffDensity, DiffRhoFileName(0, total_steps).c_str());

            MPI_Bcast(&input.nuclei[0], sizeof(Nucleus)* input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
            
            watch.Record(0);
            const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
            watch.Record(2);

            double MD_E_0 = 0.0;
            double MD_delta_E_max = 0.0;
            double MD_delta_E_min = 0.0;

            for (int istep = start_steps; istep < total_steps; ++istep) {
#if 1
// Algorithm 3: like a symplectic//                
                //note: qumasun.MoveNuclei() が二度発生するのでコアポテンシャルと擬ポテンシャルの準備のコストが2倍となる//

                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                
                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus)* input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                watch.Record(3);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
                watch.Record(0);
                qumasun.Evolve(istep, 1, dt);
                watch.Record(1);
                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus)* input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                watch.Record(3);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
                watch.Record(0);
                const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
                watch.Record(2);
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                watch.Record(3);

#elif 1
// Algrithm 2: energy decreases with time//
                //note: Evolve(psiの発展)とGetForce(原子核に働く力)の間にEvolveR(原子核の位置の発展)を入れることと//
                //そうすることで、原子核と共に動く電子の軌道関数が、原子核に引っ張られずに済む//
                //例として等速運動する単原子の場合、原子核は電子軌道のセンターにいる//
                //ここで、先に原子核を動かしてから次に電子波動関数を動かすとすると、
                //後者の波動関数の発展の時には、原子核が先に進んでしまっていて引っ張られる//
                //先に電子の波動関数を動かすとすると、その時には原子核と電子波動関数でセンターがそろっていて引っ張られない//
                //よって、先に電子の波動関数を動かす。今の時点でセンターは一旦ずれる。
                //続いて原子核を動かす際は、電子の波動関数には依存せず、全ステップでセットされているはずの運動量に従って動くので、
                //センターがずれていても(電子のセンターが先行していても)問題ない。
                //原子核も動かし終わって、再びセンターがそろったところで、原子核にかかる力を計算し、運動量を更新する。
                //ここでもセンターは揃っているので、原子核が電子に引っ張られることはない//

                //問題発生: 電子が原子核に引っ張られて十分に進めない。//

                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                
                qumasun.Evolve(istep, 1, dt);

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt);

                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus)* input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.

                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);


#else
// Algrithm 1: energy increases with time//
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                
                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus)* input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                qumasun.Evolve(istep, 1, dt);
                //MPI_Barrier(mpi_comm);
                //printf("[%d]emd:4\n", proc_id); fflush(stdout);
                //MPI_Barrier(mpi_comm);

                const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
                //MPI_Barrier(mpi_comm);
                //printf("[%d]emd:5\n", proc_id); fflush(stdout);
                //MPI_Barrier(mpi_comm);

                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
#endif

                const double E_kin_nucl = KineticEnergy(input.num_nuclei, &P[0], &input.mass[0]);
                watch.Record(3);
                if (is_root) {
                    if (istep == start_steps) {
                        MD_E_0 = E_kin_nucl + E_ele;
                    } else {
                        if (MD_delta_E_max < E_kin_nucl + E_ele - MD_E_0) MD_delta_E_max = E_kin_nucl + E_ele - MD_E_0;
                        if (MD_delta_E_min > E_kin_nucl + E_ele - MD_E_0) MD_delta_E_min = E_kin_nucl + E_ele - MD_E_0;
                    }
                    printf("MD_E_kin_nucl: %d, %f\n", istep + 1, E_kin_nucl);
                    printf("MD_E_electron: %d, %f\n", istep + 1, E_ele);
                    printf("MD_E_total   : %d, %f\n", istep + 1, E_kin_nucl + E_ele);
                    printf("MD_delta_E_max     : %d, %f\n", istep + 1, MD_delta_E_max);
                    printf("MD_delta_E_min     : %d, %f\n", istep + 1, MD_delta_E_min);
                    fflush(stdout);
                }

                

                if (incremental_step > 0) {
                    if ((istep + 1) % incremental_step == 0) {
                        qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, RhoFileName(istep + 1, total_steps).c_str());
                        qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DiffDensity, DiffRhoFileName(istep + 1, total_steps).c_str());

                        WriteMD3_one(input.num_nuclei, &input.nuclei[0]);

                        qumasun.PrintTime();
                        watch.Record(5);
                    }
                }
                

                //MPI_Barrier(mpi_comm);
                //printf("[%d]emd:8\n", proc_id); fflush(stdout);
                //MPI_Barrier(mpi_comm);
            }
            //MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);


            FprintRVM("final_positions.txt", input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], box_axis);
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "final_rho.cube");
            

            break;
        }
        default:
        {
            return -1;
        }
        
	}

    
    if (input.output_state_vector != "none") {
        qumasun.OutputEigenVector(0, input.output_state_vector.c_str(), "state_file_list.txt");
    }

    if (input.output_orbital_density != "none") {
        qumasun.OutputEigenVector(1, input.output_orbital_density.c_str(), "");
    }

    qumasun.PrintTime();

    if (is_root) {
        printf("\nTotal calculation time: %f[s]\n", watch.Total());
        watch.Print("MoveNuclei   ", 0);
        watch.Print("EvolvePsi    ", 1);
        watch.Print("GetForce     ", 2);
        watch.Print("EvolveRandP  ", 3);
        watch.Print("Output       ", 5);
        printf("\n");
    }

    return 0;

}
