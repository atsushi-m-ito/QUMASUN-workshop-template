#ifdef USE_MPI
#pragma once
#include "qumasun_base1.h"


class QUMASUN_TD : public QUMASUN_BASE1
{

public:
	QUMASUN_TD(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]);
	~QUMASUN_TD();

    void Initialize();

	void Evolve(int dynamics_step, int incremental_steps, double time_step_dt);
	
private:

    bool is_valid_electron_density = false;
    bool is_valid_core_density = false;
    bool is_state_initialized = false;

private:

    void mTimeEvolutionH4(double time_step_dt);
    size_t mWorkSizeTimeEvoH4(int local_size, int num_solution);
    void mInitializeElectrons();
private:

	void mPrintTime();

public:

	
    void PrintTime();

    void MoveNuclei(int num_nuclei, const Nucleus* next_nucleis);
    double GetForce(vec3d* forces);
};

#include "qumasun_td.constructor.h"
#include "qumasun_td.time_evo_H4.h"
#include "qumasun_td.evolution.h"
#include "qumasun_td.print.h"


#endif
