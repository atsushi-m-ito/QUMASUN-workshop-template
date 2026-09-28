#pragma once
#include <mpi.h>
#include "mpi_helper.h"
#include "qumasun_input.h"
#include "Relaxation1.h"
#include "Relaxation2.h"
#include "Relaxation3.h"
#include "Relaxation4.h"
#include "Scaling1.h"
#include "folding.h"
#include "atomic_number.h"
#include "qumasun_kpoint.h"
#include "atomic_orbital_info.h"

void PrintPositions(int num_nuclei, Nucleus* nuclei)
{    
    printf("Positons of Nuclei===========================\n");
    for (int i = 0; i < num_nuclei; ++i) {
        printf("  %s\t%.15f\t%.15f\t%.15f\n", msz::GetAtomicSymbol(nuclei[i].Z), nuclei[i].Rx, nuclei[i].Ry, nuclei[i].Rz);
    }
    printf("\n");
}


int ExecuteQUMASUN(QUMASUN::Input& input, MPI_Comm& mpi_comm, QUMASUN::DynamicsMode mode, const int* ddm_num) {
	using namespace QUMASUN;
	
    const bool is_root = IsRoot(mpi_comm);

    if (input.initial_state.empty()) {
        QUMASUN::Input input_sub = input;
        std::set<int> kinds_Z;
        for (const auto& nucl : input.nuclei) {
            kinds_Z.insert(nucl.Z);
        }
        //ここでuniqueになっているはず//
        for (const auto& sub_Z : kinds_Z) {
            input_sub.num_nuclei = 1;
            input_sub.nuclei.clear();
            input_sub.nuclei.push_back(Nucleus{ sub_Z, 0.0,0.0,0.0 });
            const int ve = input.ve_for_Z[sub_Z];

            auto ve_anglar = GetNumValenceAnglar(sub_Z, ve);
            //ve_anglar.size();
        }

        

    }

    //小さいグリッドでSCFのプレ計算を行う//
    double* pre_small_state = nullptr;
    double* eigen_values_small = nullptr;
    double* occupancy_small = nullptr;
    QUMASUN_BASE1::StateMemInfo info_small;

    if (input.scf_pre_step > 0) {
        auto input_small = input;
        input_small.scf_step = input.scf_pre_step;
        input_small.eigen_step_per_scf = input.scf_pre_eigen_step;
        input_small.eigen_step_initial = input.scf_pre_eigen_step_initial;
        input_small.grid_size[0] = input_small.scf_pre_grid_size[0];
        input_small.grid_size[1] = input_small.scf_pre_grid_size[1];
        input_small.grid_size[2] = input_small.scf_pre_grid_size[2];

        if (IsRoot(mpi_comm)) {
            printf("Begin Pre-calculation to generate initial state===========\n");
        }
        QUMASUN_KPOINT qumasun_small(input_small, mpi_comm, ddm_num);
        qumasun_small.Initialize();
        qumasun_small.Execute();
        
        info_small = qumasun_small.GetStateMemInfo();
        pre_small_state = gy::AlignedAlloc<double>(info_small.required_byte_size / 8);
        eigen_values_small = gy::AlignedAlloc<double>(info_small.num_solution * info_small .num_having_spin_kpoint * 2);
        occupancy_small = eigen_values_small + info_small.num_solution * info_small.num_having_spin_kpoint;
        
        qumasun_small.SaveStateToMemory(pre_small_state, eigen_values_small, occupancy_small);
        
        if (IsRoot(mpi_comm)) {
            printf("End of Pre-calculation====================================\n");
        }

        input.initial_density = "none";        
    }

    //mainのqumasun初期化
    QUMASUN_KPOINT qumasun(input, mpi_comm, ddm_num);

    //プレ計算をした場合は初期状態のロード
    if (pre_small_state) {
        qumasun.LoadStateFromMemory(info_small, pre_small_state, eigen_values_small, occupancy_small);

        gy::AlignedFree(pre_small_state);
        gy::AlignedFree(eigen_values_small);
    }


	switch (mode) {
		case DynamicsMode::None:
		{            
    
            qumasun.Initialize();
			qumasun.Execute();
			qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "test.cube");
            //qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DensityHR, "test_HR.cube");
            if (input.spin_polarization != 0) {
                qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::DiffDensity, "test_diff.cube");
            }
			qumasun.OutputEigenValue("test_eigenvalue.txt");
            break;
		}
		case DynamicsMode::Test:
		{
            if (is_root) {
                printf("====================================\n"); 
                printf("Dynamics Step = 0\n");
                fflush(stdout);
            }
            qumasun.Initialize();
			qumasun.Execute();

			const int num_nuclei = input.num_nuclei;
			std::vector<Nucleus> nuclei = input.nuclei; //copy

            const int STEPS = input.dynamics_step;
            double delta_x = input.box_axis[0] / (double)(input.grid_size[0] * (STEPS));

			for (int istep = 1; istep <= STEPS; ++istep) {
                if (is_root) {
                    printf("====================================\n");
                    printf("Dynamics Step = %d\n", istep);
                    fflush(stdout);
                }

				for (int i = 0; i < num_nuclei; ++i) {
					nuclei[i].Rx += delta_x;
                    Folding(nuclei[i].Rx, nuclei[i].Ry, nuclei[i].Rz, input.box_axis);
				}
				qumasun.MoveNuclei(num_nuclei, &nuclei[0]);
				qumasun.Execute(istep);
			}
			qumasun.OutputDensity(QUMASUN::OUTPUT_TARGET::Density, "test.cube");
            qumasun.OutputEigenValue("test_eigenvalue.txt");

            break;
		}

        case DynamicsMode::Relaxation1:
        {

            const int num_nuclei = input.num_nuclei;
            std::vector<Nucleus> nuclei = input.nuclei; //copy
            glips::Relaxation1Q relax(mpi_comm);
            glips::Relaxation1Q::Parameters params;
            params.lower_force_limit = input.dynamics_force_threshold;
            relax.Reset(&params);
            const int steps = input.dynamics_step;
            double init_dt = input.dynamics_timestep_dt;
            qumasun.Initialize();
            relax.Evolve(qumasun, steps, init_dt, num_nuclei, &nuclei[0], input.box_axis, 0, nullptr);
            qumasun.OutputEigenValue("test_eigenvalue.txt");
            if (is_root) {
                PrintPositions(num_nuclei, &nuclei[0]);
            }
            break;
        }

        case DynamicsMode::Relaxation2:
        {

            const int num_nuclei = input.num_nuclei;
            std::vector<Nucleus> nuclei = input.nuclei; //copy
            glips::Relaxation2Q relax(mpi_comm);
            glips::Relaxation2Q::Parameters params;
            params.lower_force_limit = input.dynamics_force_threshold;
            relax.Reset(&params);
            const int steps = input.dynamics_step;
            double init_dt = input.dynamics_timestep_dt;
            qumasun.Initialize();
            relax.Evolve(qumasun, steps, init_dt, num_nuclei, &nuclei[0], input.box_axis, 0, nullptr);
            qumasun.OutputEigenValue("test_eigenvalue.txt");
            if (is_root) {
                PrintPositions(num_nuclei, &nuclei[0]);
            }
            break;
        }

        case DynamicsMode::Relaxation3:
        {

            const int num_nuclei = input.num_nuclei;
            std::vector<Nucleus> nuclei = input.nuclei; //copy
            glips::Relaxation3Q relax(mpi_comm);
            glips::Relaxation3Q::Parameters params;
            params.lower_force_limit = input.dynamics_force_threshold;
            relax.Reset(&params);
            const int steps = input.dynamics_step;
            double init_dt = input.dynamics_timestep_dt;
            qumasun.Initialize();
            relax.Evolve(qumasun, steps, init_dt, num_nuclei, &nuclei[0], input.box_axis, 0, nullptr);
            qumasun.OutputEigenValue("test_eigenvalue.txt");
            if (is_root) {
                PrintPositions(num_nuclei, &nuclei[0]);
            }
            break;
        }

        case DynamicsMode::Relaxation4:
        {

            const int num_nuclei = input.num_nuclei;
            std::vector<Nucleus> nuclei = input.nuclei; //copy
            glips::Relaxation4Q relax(mpi_comm);
            glips::Relaxation4Q::Parameters params;
            params.lower_force_limit = input.dynamics_force_threshold;
            relax.Reset(&params);
            const int steps = input.dynamics_step;
            double init_dt = input.dynamics_timestep_dt;
            qumasun.Initialize();
            relax.Evolve(qumasun, steps, init_dt, num_nuclei, &nuclei[0], input.box_axis, 0, nullptr);
            qumasun.OutputEigenValue("test_eigenvalue.txt");
            if (is_root) {
                PrintPositions(num_nuclei, &nuclei[0]);
            }
            break;
        }

        case DynamicsMode::Scaling1:
        {

            const int num_nuclei = input.num_nuclei;
            std::vector<Nucleus> nuclei = input.nuclei; //copy
            DynamicScaling1 solver(mpi_comm);
            const int steps = input.dynamics_step;

            auto& params = input.optional_parameters["Scaling1"];
            size_t pos=0, len;
            const double begin_scale = std::stod(params, &len); pos += len;
            const double end_scale = std::stod(params.substr(pos), &len); pos += len;
            const int count = std::stol(params.substr(pos), &len, 10);
            
            qumasun.Initialize();
            solver.Evolve(qumasun, begin_scale, end_scale, count, num_nuclei, &nuclei[0], input.box_axis);
            solver.OutputLog("test_scaling.txt");
            qumasun.OutputEigenValue("test_eigenvalue.txt");
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


    return 0;

}
