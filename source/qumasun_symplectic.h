#ifdef USE_MPI
#pragma once
#include "qumasun_base1.h"
#include "DDMGatherScatter.h"


class QUMASUN_SYMPLECTIC : public QUMASUN_BASE1
{

public:
    QUMASUN_SYMPLECTIC(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]);
    ~QUMASUN_SYMPLECTIC();

    
    double EvolveKinetic(double dt);
    void EvolveHartree(double dt);
    void EvolvePPlocal(int num_nuclei, double* nucl_F, double dt);
    void EvolveCoreCore(int num_nuclei, double* nucl_F, double dt);
    void EvolvePotential1(int num_nuclei, double* nucl_F, double dt);
    void EvolvePotentialNonlocal_v2(int num_nuclei, double* nucl_F, double dt, int forward_or_backward);
    void Initialize(int num_nuclei, const Nucleus* next_nucleis);
    double GetEnergy(bool use_kinetic_energy_evolveK);

    void PrintTime();
    void MoveNuclei(int num_nuclei, const Nucleus* next_nucleis);

    //[deprecated]
    void EvolvePotentialAll(int num_nuclei, double* nucl_F, double dt);
    void EvolvePotentialNonlocal(int num_nuclei, double* nucl_F, double dt);

    //for debug
    double GetKineticEnergy() {        
        m_Ekin = mGetEnergyKineticKspace();
        return m_Ekin;
    };

private:

private:

    void mInitializeElectrons();
private:

    void mPrintTime();
    void mEvolveK4(double dt);
    double mEvolveK_FFT(double dt);

    void mProductExpToV(SoAComplex* l_psi_set, const OneComplex* exp_v);

public:
    //test
    void TestEvolveNL4(double dt);
    void TestEvolveNL8(double dt);
    double GetForce(vec3d* forces);

};

#include "qumasun_symplectic.constructor.h"
#include "qumasun_symplectic.time_evo_K4.h"
#include "qumasun_symplectic.evolution.h"
#include "qumasun_symplectic.print.h"
#include "qumasun_symplectic.kinetic_fft.h"

#endif
