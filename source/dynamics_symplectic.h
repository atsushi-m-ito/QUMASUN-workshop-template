#pragma once
#include <mpi.h>
#include <filesystem>
#include "mpi_helper.h"
#include "qumasun_input.h"
#include "Relaxation2.h"
#include "folding.h"
#include "md3_writer2.h"
#include "qumasun_symplectic.h"
#include "print_rvm.h"


//#define CHECK_KINETIC

inline constexpr static bool IS_BARRIER_WATCH = true;

template<bool IS_WATCH>
void Watch_Barrier(MPI_Comm comm) {
    if constexpr (IS_WATCH) {
        MPI_Barrier(comm);
    }
}

inline double Distance(const Nucleus& n1, const Nucleus& n2) {
    return sqrt((n1.Rx - n2.Rx) * (n1.Rx - n2.Rx) 
        + (n1.Ry - n2.Ry) * (n1.Ry - n2.Ry) 
        + (n1.Rz - n2.Rz) * (n1.Rz - n2.Rz));
}


int TimeEvoSymplectic(QUMASUN_SYMPLECTIC& qumasun, QUMASUN::Input& input, MPI_Comm& mpi_comm, QUMASUN::DynamicsMode mode, const int* ddm_num) {
	using namespace QUMASUN;
	
    const bool is_root = IsRoot(mpi_comm);
    int proc_id = GetProcessID(mpi_comm);

    StopWatch<10, true> watch;
    watch.Restart();

    auto FormattedNumber = [](int istep, int max_step) {
        int64_t length = std::to_string(max_step).length();
        std::string filename;
        length -= std::to_string(istep).length();
        if (length > 0) {
            filename += std::string(length, '0');
        }
        filename += std::to_string(istep);
        return filename;
        };

    auto RhoFileName = [&FormattedNumber](int istep, int max_step) {
        
        std::string filename("rho_t");        
        return filename + FormattedNumber(istep, max_step) + ".cube";
        };

    auto DiffRhoFileName = [&FormattedNumber](int istep, int max_step) {
        
        std::string filename("diff_rho_t");
        return filename + FormattedNumber(istep, max_step) + ".cube";
        };


    const double* box_axis = input.box_axis;
    msz::MD3_Writer2<double> md3;
    auto WriteMD3_one = [&md3, &box_axis](int num_nuclei, Nucleus* nuclei, vec3d* p) {
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
        md3.WriteP(&p[0]);
        md3.WriteBox(box_axis);
        md3.EndFrame();
        };


	switch (mode) {        
        case DynamicsMode::SymplecticEhrenfestMD:  //SymplecticIntegrator, 波動関数と原子核座標・運動量を共に正準変数とみなす//
        {


            const double dt = input.dynamics_timestep_dt;
            const int start_steps = input.dynamics_start_step;
            const int total_steps = input.dynamics_step + start_steps;
            const int output_step = input.dynamics_output_step;


            auto P = std::make_unique<vec3d[]>(input.num_nuclei);
            auto F = std::make_unique<vec3d[]>(input.num_nuclei);
            for (int i = 0; i < input.num_nuclei; ++i) {
                P[i].x = input.velocity[i * 3] * input.mass[i];
                P[i].y = input.velocity[i * 3 + 1] * input.mass[i];
                P[i].z = input.velocity[i * 3 + 2] * input.mass[i];
            }

            if (is_root) {
                md3.Open("nucl_position.md3");
                WriteMD3_one(input.num_nuclei, &input.nuclei[0], P.get());
            
                printf("Mass and Velocity [a.u.] ===========================\n");
                for (int i = 0; i < input.num_nuclei; ++i) {
                    printf("%d: %f, %f, %f, %f\n", i, input.mass[i], input.velocity[i * 3], input.velocity[i * 3 + 1], input.velocity[i * 3 + 2]);
                }
                printf("====================================================\n");
                fflush(stdout);
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

            auto IsOutput = [&output_step](int istep) {
                if (output_step == 0)return false;
                return ((istep + 1) % output_step == 0);
                };

            auto PrintStep = [&mpi_comm](const char* text) {

                MPI_Barrier(mpi_comm);
                if (IsRoot(mpi_comm)) {
                    printf("%s\n", text); fflush(stdout);
                }
                MPI_Barrier(mpi_comm);
                };

            MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            qumasun.Initialize(input.num_nuclei, &input.nuclei[0]);
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, RhoFileName(0, total_steps).c_str());
            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DiffDensity, DiffRhoFileName(0, total_steps).c_str());
            if (!input.dynamics_output_orbital_density.empty() ) {
                qumasun.OutputEigenVectorList(1, "p_rho", input.dynamics_output_orbital_density, FormattedNumber(0, total_steps).c_str());
            }
            qumasun.OutputEigenValue("initial_occupancy.txt");
            {
                std::string tail;
                tail += "_t" + FormattedNumber(0, total_steps) + ".txt";
                qumasun.OutputOverlapMatrix("overlap_matrix", tail.c_str());
            }


            int save_point = 0;
            int64_t save_point_id = 0;
            if (input.save_point.size() > save_point_id) {
                save_point = input.save_point[save_point_id];
            }

            if (qumasun.CheckError()) {
                return -1;
            }

            MPI_Barrier(mpi_comm);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
            //const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
            //MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            watch.Record(0);
            
            double MD_E_0 = 0.0;
            double MD_delta_E_max = 0.0;
            double MD_delta_E_min = 0.0;

            {
                const double E_ele = qumasun.GetEnergy(false);
                //PrintStep("Evo GetEnergy");

                const double E_kin_nucl = KineticEnergy(input.num_nuclei, &P[0], &input.mass[0]);
                watch.Record(5);
                if (is_root) {
                    MD_E_0 = E_kin_nucl + E_ele;

                    printf("MD_E_kin_nucl: %d, %f\n", start_steps, E_kin_nucl);
                    printf("MD_E_electron: %d, %f\n", start_steps, E_ele);
                    printf("MD_E_total   : %d, %f\n", start_steps, E_kin_nucl + E_ele);
                    if (input.num_nuclei == 2) {
                        const double distance = Distance(input.nuclei[0], input.nuclei[1]);
                        printf("MD_2_distance: %d, %f\n", start_steps, distance);
                    }
                    fflush(stdout);
                }
            }

            for (int istep = start_steps; istep < total_steps; ++istep) {

                // Algorithm : just symplectic//     

#if 0
                //Algorithm v2: combine nonlocal potential and other potentials//
                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);


                qumasun.EvolveKinetic(dt / 2.0);

                qumasun.EvolvePotentialAll(input.num_nuclei, (double*)&F[0], dt);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        //printf("Fpot = %.10f, %.10f, %.10f\n", F[n].x, F[n].y, F[n].z);
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                EvolveP(input.num_nuclei, &P[0], &F[0], dt);



                qumasun.EvolveKinetic(dt / 2.0);

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                const double E_ele = qumasun.GetEnergy();
#elif 1         
                //Argorithm 2: e_kin -> e_pot -> e_nl(foward) -> n_R -> e_nl(back) -> e_pot -> e_kin

                double Ekin = qumasun.EvolveKinetic(dt / 2.0);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                watch.Record(1);
#ifdef CHECK_KINETIC
                if (is_root) {
                    printf("Ekin(1) = %.10f\n", Ekin);
                }
#endif

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                watch.Record(2);
#ifdef CHECK_KINETIC
                Ekin = qumasun.GetKineticEnergy();
                if (is_root) {
                    printf("Ekin(2) = %.10f\n", Ekin);
                }
#endif
#ifndef IGNORE_NONLOCAL
                //v18: forward and backward, 衝突がある場合は明らかに精度が良い
                qumasun.EvolvePotentialNonlocal_v2(input.num_nuclei, (double*)&F[0], dt / 2.0, 1);
                EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                watch.Record(3);
#ifdef CHECK_KINETIC
                Ekin = qumasun.GetKineticEnergy();
                if (is_root) {
                    printf("Ekin(3) = %.10f\n", Ekin);
                }
#endif
#endif

                //Algorithm v1: split nonlocal potential//
                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt );
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus)* input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm); 
                
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm); 
                watch.Record(4);

                
#ifndef IGNORE_NONLOCAL
                //v18: forward and backward, 衝突がある場合は明らかに精度が良い                
                qumasun.EvolvePotentialNonlocal_v2(input.num_nuclei, (double*)&F[0], dt / 2.0, -1);
                EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                watch.Record(3);
#ifdef CHECK_KINETIC
                Ekin = qumasun.GetKineticEnergy();
                if (is_root) {
                    printf("Ekin(4) = %.10f\n", Ekin);
                }
#endif

#endif

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm); 
                watch.Record(2);
#ifdef CHECK_KINETIC
                Ekin = qumasun.GetKineticEnergy();
                if (is_root) {
                    printf("Ekin(5) = %.10f\n", Ekin);
                }
#endif


                Ekin = qumasun.EvolveKinetic(dt / 2.0);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                watch.Record(1);
#ifdef CHECK_KINETIC
                if (is_root) {
                    printf("Ekin(6) = %.10f\n", Ekin);
                }
#endif


                const double E_ele = qumasun.GetEnergy(true);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                watch.Record(5);

#elif 1
                //Algorithm v1: split nonlocal potential//
                //Argorithm v1: n_R -> e_kin -> e_pot -> e_nl(foward) -> e_nl(back) -> e_pot -> e_kin -> n_R

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
                watch.Record(4);

                qumasun.EvolveKinetic(dt / 2.0);
                watch.Record(1);

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                watch.Record(2);

#ifndef IGNORE_NONLOCAL
#if 1       //v18: forward and backward, 衝突がある場合は明らかに精度が良い
                qumasun.EvolvePotentialNonlocal_v2(input.num_nuclei, (double*)&F[0], dt / 2.0, 1);
                EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
                watch.Record(3);
                qumasun.EvolvePotentialNonlocal_v2(input.num_nuclei, (double*)&F[0], dt / 2.0, -1);
                EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
                watch.Record(3);
#elif 1     //v17: 衝突のない場合の安定版, 衝突がある場合は明らかに精度が悪い
                qumasun.EvolvePotentialNonlocal(input.num_nuclei, (double*)&F[0], dt);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        //F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
#elif 1
                qumasun.TestEvolveNL4(dt);
#else
                qumasun.TestEvolveNL8(dt);
#endif
#endif

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        //printf("Fpot = %.10f, %.10f, %.10f\n", F[n].x, F[n].y, F[n].z);
                        //F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                watch.Record(2);
                qumasun.EvolveKinetic(dt / 2.0);
                watch.Record(1);

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
                watch.Record(4);
                
                const double E_ele = qumasun.GetEnergy(true);
                watch.Record(5);

#endif
                const double E_kin_nucl = KineticEnergy(input.num_nuclei, &P[0], &input.mass[0]);
                Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm); 
                watch.Record(5);
                if (is_root) {
                    
                    if (MD_delta_E_max < E_kin_nucl + E_ele - MD_E_0) MD_delta_E_max = E_kin_nucl + E_ele - MD_E_0;
                    if (MD_delta_E_min > E_kin_nucl + E_ele - MD_E_0) MD_delta_E_min = E_kin_nucl + E_ele - MD_E_0;
                    
                    printf("MD_E_kin_nucl: %d, %f\n", istep + 1, E_kin_nucl);
                    printf("MD_E_electron: %d, %f\n", istep + 1, E_ele);
                    printf("MD_E_total   : %d, %f\n", istep + 1, E_kin_nucl + E_ele);
                    printf("MD_delta_E_max     : %d, %f\n", istep + 1, MD_delta_E_max);
                    printf("MD_delta_E_min     : %d, %f\n", istep + 1, MD_delta_E_min);
                    if (input.num_nuclei == 2) {
                        const double distance = Distance(input.nuclei[0], input.nuclei[1]);
                        printf("MD_2_distance: %d, %f\n", istep + 1, distance);
                    }
                    fflush(stdout);
                }



                if (IsOutput(istep)) {

                    qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, RhoFileName(istep + 1, total_steps).c_str());

                    if (!input.dynamics_output_orbital_density.empty()) {
                        qumasun.OutputEigenVectorList(1, "p_rho", input.dynamics_output_orbital_density, FormattedNumber(istep + 1, total_steps).c_str());
                    }


                    if (is_root) {
                        WriteMD3_one(input.num_nuclei, &input.nuclei[0], P.get());
                    }
                        
                    qumasun.PrintTime();

                    std::string tail;
                    tail += "_t" + FormattedNumber(istep + 1, total_steps) + ".txt";
                    qumasun.OutputOverlapMatrix("overlap_matrix", tail.c_str());

                    Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                    watch.Record(6);
                }

                //状態セーブ//
                if (save_point == istep+1) {
                    //save//
                    //directory 生成//
                    std::string dirname("savepoint_t");
                    dirname += FormattedNumber(istep + 1, total_steps);
                    auto res = std::filesystem::create_directory(dirname);
                    dirname += "/";
                    //state save//
                    qumasun.OutputEigenVector(0, "state", "state_file_list.txt", dirname.c_str());
                    qumasun.OutputEigenValue((dirname + "test_eigenvalue.txt").c_str());

                    //next save point
                    ++save_point_id;
                    if (input.save_point.size() > save_point_id) {
                        save_point = input.save_point[save_point_id];
                    }
                    Watch_Barrier<IS_BARRIER_WATCH>(mpi_comm);
                    watch.Record(6);
                }
            } //end of loop: istep//

            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "final_rho.cube");
            if (is_root) {
                FprintRVM("final_positions.txt", input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], box_axis);
            }



            break;
        }
        case DynamicsMode::SemiSymplecticTest1: //symplectic integratorとの比較用//
        {    
            //波動関数だけみるとSuzuki-Trotter展開されているが,
            //原子核位置の発展とは独立したアルゴリズムになっている//
        

            const double* box_axis = input.box_axis;

            const double dt = input.dynamics_timestep_dt;
            const int start_steps = input.dynamics_start_step;
            const int total_steps = input.dynamics_step + start_steps;
            const int incremental_step = input.dynamics_output_step;


            auto P = std::make_unique<vec3d[]>(input.num_nuclei);
            auto F = std::make_unique<vec3d[]>(input.num_nuclei);
            for (int i = 0; i < input.num_nuclei; ++i) {
                P[i].x = input.velocity[i * 3] * input.mass[i];
                P[i].y = input.velocity[i * 3 + 1] * input.mass[i];
                P[i].z = input.velocity[i * 3 + 2] * input.mass[i];
            }

            //show velocity//
            if (is_root) {
                md3.Open("nucl_position.md3");
                WriteMD3_one(input.num_nuclei, &input.nuclei[0], &P[0]);

                printf("Mass and Velocity [a.u.] ===========================\n");
                for (int i = 0; i < input.num_nuclei; ++i) {
                    printf("%d: %f, %f, %f, %f\n", i, input.mass[i], input.velocity[i * 3], input.velocity[i * 3 + 1], input.velocity[i * 3 + 2]);
                }
                printf("====================================================\n");
                fflush(stdout);
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

            auto PrintStep = [&mpi_comm](const char* text) {

                MPI_Barrier(mpi_comm);
                if (IsRoot(mpi_comm)) {
                    printf("%s\n", text); fflush(stdout);
                }
                MPI_Barrier(mpi_comm);
                };

            MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            qumasun.Initialize(input.num_nuclei, &input.nuclei[0]);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
            //const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
            //MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

            double MD_E_0 = 0.0;
            double MD_delta_E_max = 0.0;
            double MD_delta_E_min = 0.0;

            for (int istep = start_steps; istep < total_steps; ++istep) {

                // Algorithm : symplectic for psi and symplectic for nuclei, independently//



                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                qumasun.GetForce(&F[0]);				//力の計算.
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                for (int n = 0; n < input.num_nuclei; ++n) {
                    //printf("Fpot = %.10f, %.10f, %.10f\n", F[n].x, F[n].y, F[n].z);
                }


                qumasun.EvolveKinetic(dt / 2.0);

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);


#ifndef IGNORE_NONLOCAL
#if 1
                qumasun.EvolvePotentialNonlocal(input.num_nuclei, (double*)&F[0], dt);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
#elif 1
                qumasun.TestEvolveNL4(dt);
#else
                qumasun.TestEvolveNL8(dt);
#endif
#endif

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);

                qumasun.EvolveKinetic(dt / 2.0);


                qumasun.GetForce(&F[0]);				//力の計算.
                EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);
                for (int n = 0; n < input.num_nuclei; ++n) {
                    //printf("Fpot = %.10f, %.10f, %.10f\n", F[n].x, F[n].y, F[n].z);
                }

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                const double E_ele = qumasun.GetEnergy(true);


                const double E_kin_nucl = KineticEnergy(input.num_nuclei, &P[0], &input.mass[0]);
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

                        if (is_root) {
                            WriteMD3_one(input.num_nuclei, &input.nuclei[0], &P[0]);
                        }
                    }
                }

        }


            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "final_rho.cube");
            if (is_root) {
                FprintRVM("final_positions.txt", input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], box_axis);
            }



            break;
        }

        case DynamicsMode::SemiSymplecticTest2: //symplectic integratorとの比較用//
        {
            //波動関数だけみるとSuzuki-Trotter展開されているが,
            //原子核位置の発展とは独立したアルゴリズムになっている//
            //Test1より雑に, forceの計算は一度に行う//


            const double* box_axis = input.box_axis;

            const double dt = input.dynamics_timestep_dt;
            const int start_steps = input.dynamics_start_step;
            const int total_steps = input.dynamics_step + start_steps;
            const int incremental_step = input.dynamics_output_step;


            auto P = std::make_unique<vec3d[]>(input.num_nuclei);
            auto F = std::make_unique<vec3d[]>(input.num_nuclei);
            for (int i = 0; i < input.num_nuclei; ++i) {
                P[i].x = input.velocity[i * 3] * input.mass[i];
                P[i].y = input.velocity[i * 3 + 1] * input.mass[i];
                P[i].z = input.velocity[i * 3 + 2] * input.mass[i];
            }

            //show velocity//
            if (is_root) {
                md3.Open("nucl_position.md3");
                WriteMD3_one(input.num_nuclei, &input.nuclei[0], &P[0]);

                printf("Mass and Velocity [a.u.] ===========================\n");
                for (int i = 0; i < input.num_nuclei; ++i) {
                    printf("%d: %f, %f, %f, %f\n", i, input.mass[i], input.velocity[i * 3], input.velocity[i * 3 + 1], input.velocity[i * 3 + 2]);
                }
                printf("====================================================\n");
                fflush(stdout);
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

            auto PrintStep = [&mpi_comm](const char* text) {

                MPI_Barrier(mpi_comm);
                if (IsRoot(mpi_comm)) {
                    printf("%s\n", text); fflush(stdout);
                }
                MPI_Barrier(mpi_comm);
                };

            MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            qumasun.Initialize(input.num_nuclei, &input.nuclei[0]);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
            //const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
            MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

            double MD_E_0 = 0.0;
            double MD_delta_E_max = 0.0;
            double MD_delta_E_min = 0.0;

            for (int istep = start_steps; istep < total_steps; ++istep) {

                // Algorithm : symplectic for psi and symplectic for nuclei, independently//



                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
                

                qumasun.EvolveKinetic(dt / 2.0);

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);


#ifndef IGNORE_NONLOCAL
#if 1
                qumasun.EvolvePotentialNonlocal(input.num_nuclei, (double*)&F[0], dt);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
#elif 1
                qumasun.TestEvolveNL4(dt);
#else
                qumasun.TestEvolveNL8(dt);
#endif
#endif

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);

                qumasun.EvolveKinetic(dt / 2.0);


                qumasun.GetForce(&F[0]);				//力の計算.
                EvolveP(input.num_nuclei, &P[0], &F[0], dt ); //一度で行うのでTest1の倍の時間刻み//
                for (int n = 0; n < input.num_nuclei; ++n) {
                    //printf("Fpot = %.10f, %.10f, %.10f\n", F[n].x, F[n].y, F[n].z);
                }

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                const double E_ele = qumasun.GetEnergy(true);


                const double E_kin_nucl = KineticEnergy(input.num_nuclei, &P[0], &input.mass[0]);
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

                        if (is_root) {
                            WriteMD3_one(input.num_nuclei, &input.nuclei[0], P.get());
                        }
                    }
                }

            }


            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "final_rho.cube");
            if (is_root) {
                FprintRVM("final_positions.txt", input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], box_axis);
            }



            break;
        }

        case DynamicsMode::SemiSymplecticTest3: //symplectic integratorとの比較用//
        {
            //波動関数だけみるとSuzuki-Trotter展開されているが,
            //原子核位置の発展とは独立したアルゴリズムになっている//
            //Test1より雑に, forceの計算は一度に行う//


            const double* box_axis = input.box_axis;

            const double dt = input.dynamics_timestep_dt;
            const int start_steps = input.dynamics_start_step;
            const int total_steps = input.dynamics_step + start_steps;
            const int incremental_step = input.dynamics_output_step;



            auto P = std::make_unique<vec3d[]>(input.num_nuclei);
            auto F = std::make_unique<vec3d[]>(input.num_nuclei);
            for (int i = 0; i < input.num_nuclei; ++i) {
                P[i].x = input.velocity[i * 3] * input.mass[i];
                P[i].y = input.velocity[i * 3 + 1] * input.mass[i];
                P[i].z = input.velocity[i * 3 + 2] * input.mass[i];
            }

            //show velocity//
            if (is_root) {
                md3.Open("nucl_position.md3");
                WriteMD3_one(input.num_nuclei, &input.nuclei[0], P.get());

                printf("Mass and Velocity [a.u.] ===========================\n");
                for (int i = 0; i < input.num_nuclei; ++i) {
                    printf("%d: %f, %f, %f, %f\n", i, input.mass[i], input.velocity[i * 3], input.velocity[i * 3 + 1], input.velocity[i * 3 + 2]);
                }
                printf("====================================================\n");
                fflush(stdout);
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

            auto PrintStep = [&mpi_comm](const char* text) {

                MPI_Barrier(mpi_comm);
                if (IsRoot(mpi_comm)) {
                    printf("%s\n", text); fflush(stdout);
                }
                MPI_Barrier(mpi_comm);
                };

            MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            qumasun.Initialize(input.num_nuclei, &input.nuclei[0]);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);
            //const double E_ele = qumasun.GetForce(&F[0]);				//力の計算.
            //MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
            //qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

            double MD_E_0 = 0.0;
            double MD_delta_E_max = 0.0;
            double MD_delta_E_min = 0.0;

            for (int istep = start_steps; istep < total_steps; ++istep) {

                // Algorithm : symplectic for psi and symplectic for nuclei, independently//



                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);


                qumasun.EvolveKinetic(dt / 2.0);

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);


#ifndef IGNORE_NONLOCAL
#if 1
                qumasun.EvolvePotentialNonlocal_v2(input.num_nuclei, (double*)&F[0], dt / 2.0, 1);
                qumasun.EvolvePotentialNonlocal_v2(input.num_nuclei, (double*)&F[0], dt / 2.0, -1);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], 1.0);//note: dt is included in return value of Nonlocal term// 
#elif 1
                qumasun.TestEvolveNL4(dt);
#else
                qumasun.TestEvolveNL8(dt);
#endif
#endif

                qumasun.EvolvePotential1(input.num_nuclei, (double*)&F[0], dt / 2.0);
                if (IsRoot(mpi_comm)) {
                    for (int n = 0; n < input.num_nuclei; ++n) {
                        F[n].x = F[n].y = F[n].z = 0.0;
                    }
                }
                //EvolveP(input.num_nuclei, &P[0], &F[0], dt / 2.0);

                qumasun.EvolveKinetic(dt / 2.0);


                qumasun.GetForce(&F[0]);				//力の計算.
                EvolveP(input.num_nuclei, &P[0], &F[0], dt); //一度で行うのでTest1の倍の時間刻み//
                for (int n = 0; n < input.num_nuclei; ++n) {
                    //printf("Fpot = %.10f, %.10f, %.10f\n", F[n].x, F[n].y, F[n].z);
                }

                EvolveR(input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], dt / 2.0);
                MPI_Bcast(&input.nuclei[0], sizeof(Nucleus) * input.num_nuclei, MPI_BYTE, 0, mpi_comm);
                qumasun.MoveNuclei(input.num_nuclei, &input.nuclei[0]);

                const double E_ele = qumasun.GetEnergy(true);


                const double E_kin_nucl = KineticEnergy(input.num_nuclei, &P[0], &input.mass[0]);
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
                        //show velocity//
                        if (is_root) {
                            WriteMD3_one(input.num_nuclei, &input.nuclei[0], P.get());
                        }
                    }
                }

            }


            qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "final_rho.cube");
            if (is_root) {
                FprintRVM("final_positions.txt", input.num_nuclei, &input.nuclei[0], &P[0], &input.mass[0], box_axis);
            }



            break;
        }

        default:
        {
            return -1;
        }
        
	}

    
    if (input.output_state_vector != "none") {
        qumasun.OutputEigenVector(0,input.output_state_vector.c_str(), "state_file_list.txt");
    }

    if (input.output_orbital_density != "none") {
        qumasun.OutputEigenVector(1, input.output_orbital_density.c_str(), "");
    }

    
    qumasun.PrintTime();


    if (is_root) {
        printf("\nTotal time      : %f[s]\n", watch.Total());
        printf("\nCalculation time: %f[s]\n", watch.Total({1,2,3,4,5}));
        watch.Print("--EvolvePsi_Kin ", 1);
        watch.Print("--EvolvePsi_Pot1", 2);
        watch.Print("--EvolvePsi_NL  ", 3);
        watch.Print("--MoveNuclei    ", 4);
        watch.Print("--Energy        ", 5);
        printf("\nOthers          : %f[s]\n", watch.Total({ 0,6 }));
        watch.Print("--Initialize    ", 0);
        watch.Print("--Output        ", 6);
        printf("\n");
    }

    return 0;

}
