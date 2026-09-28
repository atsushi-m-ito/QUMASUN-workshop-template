#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qumasun_td.h"
#include "mpi_helper.h"
#include "qumasun_note.h"

inline
void QUMASUN_TD::mInitializeElectrons() {
    //printf("[%d] test0\n", proc_id); fflush(stdout);
    mInitializeState();
    watch.Record(0);
    //printf("[%d] test1\n", proc_id); fflush(stdout);

    //mSetOccupancy();
    mSetOccupancyInitial();
    watch.Record(2);
    //printf("[%d] test2\n", proc_id); fflush(stdout);
    //MPI_Barrier(m_mpi_comm);
    //printf("[%d] test2-2\n", proc_id); fflush(stdout);

    //calculate electron density in real space///////////////
    mInitializeDensity();
    watch.Record(7);
    //MPI_Barrier(m_mpi_comm);
    //printf("[%d] test3-2\n", proc_id); fflush(stdout);

    //calculate potential in real space//////////////////////
    mSetPotentialVhart();
    watch.Record(4);

}

inline
void QUMASUN_TD::Initialize() {
    //initialize////////////////////
    PrintCondition();
    if (!mCheckConditions()) {
        return;
    }

    const bool is_root_global = IsRoot(m_mpi_comm);
    watch.Restart();

    if (!is_state_initialized) {
        mInitializeElectrons();
        is_valid_electron_density = true;
    }

    if (!is_valid_core_density) {
        mPrepareCore();
        watch.Record(1);
        is_valid_core_density = true;
    }



    if (!is_state_initialized) {
        if (is_root_global) {
            printf("Initial Energy ==============================\n"); fflush(stdout);
        }

        mSetPotentialVxc();
        mSetPotentialVtot();
        watch.Record(4);

        double E_tot = mGetTotalEnergy(true, false);

        watch.Record(6);



        if (is_root_global) {
            printf("==============================\n\n"); fflush(stdout);
        }

        is_state_initialized = true;
    }

}

inline
void QUMASUN_TD::Evolve(int dynamics_step, int incremental_steps, double time_step_dt) {
    

    if (!is_construction_successful) return ;

    const int proc_id = GetProcessID(m_mpi_comm);
    /*
    //initialize////////////////////
    PrintCondition();
    if (!mCheckConditions()) {
        return ;
    }
    */
    const bool is_root_global = IsRoot(m_mpi_comm);
    watch.Restart();
    

    if (is_root_global) {
        printf("Begin TDDFT Step %d ==============================\n", dynamics_step + 1); fflush(stdout);
    }
    //SCF Loop//////////////////////////////////////////////////////
    for (int tddft_step = 0; tddft_step < incremental_steps; ++tddft_step) {

        watch.Restart();

        mSetPotentialVxc();
        mSetPotentialVtot();
        watch.Record(4);

        mTimeEvolutionH4(time_step_dt);
        watch.Record(5);
        
        is_valid_electron_density = false;


        //calculate electron density without mixing//
        mSetDensityByPsi();     
        watch.Record(3);
        

        //calculate potential in real space//
        mSetPotentialVhart();
        watch.Record(4);
        

        is_valid_electron_density = true;
    }
    //////////////////////////////////////End of SCF Loop//

    if (is_root_global) {
        printf("End TDDFT Step %d ==============================\n\n", dynamics_step + incremental_steps); fflush(stdout);
    }

}


double QUMASUN_TD::GetForce(vec3d* forces) {

    const bool is_root_global = IsRoot(m_mpi_comm);

    watch.Restart();

    if (!is_state_initialized) {
        mInitializeElectrons();
        is_valid_electron_density = true;
    }

    if (!is_valid_core_density) {

        mPrepareCore();
        watch.Record(1);

        is_valid_core_density = true;
    }

    mSetPotentialVxc();
    mSetPotentialVtot();
    watch.Record(4);


    //calculate total energy//////////////////////////////

    if (!is_state_initialized) {
        if (is_root_global) {
            printf("Initial Energy ==============================\n"); fflush(stdout);
        }
    }


    double E_tot = mGetTotalEnergy(true, true);
    watch.Record(6);

    if (!is_state_initialized) {
        if (is_root_global) {
            printf("==============================\n\n"); fflush(stdout);
        }
        is_state_initialized = true;
    }

    DEBUG_PRINTF("[%d]mGetForce:before\n", proc_id);
	mGetForce(true);
    watch.Record(8);
    DEBUG_PRINTF("[%d]mGetForce:after\n", proc_id);
    //MPI_Barrier(m_mpi_comm);
	//mPrintTime();

    for (int i = 0; i < m_num_nuclei; ++i) {
        forces[i] = m_nucl_forces[i];
    }

    return E_tot;
}


inline
void QUMASUN_TD::MoveNuclei(int num_nuclei, const Nucleus* next_nucleis) {

    //set flag to invalid//
    is_valid_core_density = false;

    mMoveNuclei(num_nuclei, next_nucleis);
}
