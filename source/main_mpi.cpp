/*******************************

QUantum MAterial Simulation UNraveler (QUMASUN)

QUMASUN is numerical simulation code for Density Functional Theory(DFT) and Time-dependent DFT based on the real space grid.

********************************/
#include <mpi.h>
#include "mpi_helper.h"

#include <cstdlib>
#include <cstdio>
#define TIME_PP_MPI


#include "qumasun_td.h"
#include "qumasun_symplectic.h"
#include "atomic_number.h"
#include "field_interpolation.h"
#include "GridRange.h"
#include "GridFor.h"
#include "GetArg.h"
#include "GridScatterGather.h"
#include "qumasun_make_input.h"
#include "dynamics.h"
#include "dynamics_td.h"
#include "dynamics_symplectic.h"


//simple DFT

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	const int proc_id = GetProcessID(mpi_comm);
	const int num_procs = GetNumProcess(mpi_comm);
	if (argc < 2) {
		if (proc_id == 0) {
			printf("ERROR: no input file.\n");
		}
		MPI_Finalize();
		return -1;
	}


#if defined(GY_WITH_CUDA) || defined(GY_WITH_HIP)
    if (num_procs>1) {
        int dev_count = 0;
        gyGetDeviceCount(&dev_count);
        printf("GPU-DeviceCount = %d\n", dev_count);

        if (dev_count > 1) {
            gySetDevice(proc_id % dev_count);
        }

    }
#endif

	//for MPI/////////////////////////////////////
	int ddm_num[4] = { 1,1,1, 1 }; //4th element is for state decomposition//
	GetArgumentNumList<int>(argc, argv, "-ddm", 3, ddm_num);
    ddm_num[3] = GetArgumentNum<int>(argc, argv, "-sd", 1);
	

	
	//begin calculation///////////////////////////////
    QUMASUN::Input input;
    int res = QUMASUN::MakeInput(argv[1], input, mpi_comm);
    if ( res != 0) {
        if (proc_id == 0) {
            if (res == -1) {
                printf("ERROR: cannot find the input file: %s\n", argv[1]);
            } else if (res == -2) {
                printf("ERROR: invalid semantics in input file: %s\n", argv[1]);
            }
        }
        MPI_Finalize();
        return -1;
    }

    if ((input.dynamics_mode == QUMASUN::DynamicsMode::SymplecticEhrenfestMD)
        || (input.dynamics_mode == QUMASUN::DynamicsMode::SemiSymplecticTest1)
        || (input.dynamics_mode == QUMASUN::DynamicsMode::SemiSymplecticTest2)
        || (input.dynamics_mode == QUMASUN::DynamicsMode::SemiSymplecticTest3)){

        if (IsRoot(mpi_comm)) {
            printf("Begin calculation TDDFT on QUMASUN\n");
        }

        QUMASUN_SYMPLECTIC qumasun(input, mpi_comm, ddm_num);
        TimeEvoSymplectic(qumasun, input, mpi_comm, input.dynamics_mode, ddm_num);

        
    } else if ((input.dynamics_mode == QUMASUN::DynamicsMode::TDDFT) || (input.dynamics_mode == QUMASUN::DynamicsMode::EhrenfestMDv1)) {

        if (IsRoot(mpi_comm)) {
            printf("Begin calculation TDDFT on QUMASUN\n");
        }

        QUMASUN_TD qumasun(input, mpi_comm, ddm_num);
        TimeEvoQUMASUN(qumasun, input, mpi_comm, input.dynamics_mode, ddm_num);

    } else { //DFT(SCF) calculation /////////////////////////

        if (input.kpoint_sample[0] * input.kpoint_sample[1] * input.kpoint_sample[2] > 1) {
            if (IsRoot(mpi_comm)) {
                printf("Begin calculation on QUMASUN with kpoint\n");
            }
        } else {
            if (IsRoot(mpi_comm)) {
                printf("Begin calculation on QUMASUN on gamma point\n"); fflush(stdout);
            }
        }

        ExecuteQUMASUN(input, mpi_comm, input.dynamics_mode, ddm_num);
    }

    MPI_Barrier(mpi_comm);
	MPI_Finalize();

}

