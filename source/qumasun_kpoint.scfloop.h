#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qumasun_kpoint.h"
#include "mpi_helper.h"
#include "qumasun_note.h"


inline
void QUMASUN_KPOINT::Initialize() {

    //initialize////////////////////
    PrintCondition();
    MPI_Barrier(m_mpi_comm);
    if (!mCheckConditions()) {
        return;
    }

    watch.Restart();

    if (!m_is_state_loaded) {
        mInitializeState();
        watch.Record(0);
        
        mSetOccupancyInitial();
        watch.Record(2);        
    }

    //calculate electron density in real space///////////////
    mInitializeDensity();
    mSymmetrizeDensity();
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    watch.Record(7);
    //MPI_Barrier(m_mpi_comm);
    //printf("[%d] test3-2\n", proc_id); fflush(stdout);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);

    mPrepareCore();
    watch.Record(1);
    //printf("[%d] test3\n", proc_id); fflush(stdout);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
#if 0
    /*test   */
    {
        const size_t local_size = ml_grid.Size3D();
        if (IsRoot(m_same_ddm_place_comm)) {
            for (size_t i = 0; i < local_size; ++i) {
                ml_rho[i] = ml_pcc_rho[i];
                //ml_rho[i] = ml_hr_nucl_rho[i];
            }
        }
        MPI_Bcast(&ml_rho[0], local_size, MPI_DOUBLE, 0, ml_grid.mpi_comm);
    }

#endif

}

inline
void QUMASUN_KPOINT::Execute(int dynamics_step = 0) {
	
	if (!is_construction_successful) return;
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
	const int proc_id = GetProcessID(m_mpi_comm);
    const bool is_root_global = IsRoot(m_mpi_comm);

    if (IsRoot(m_same_ddm_place_comm)) {
        m_mixer->Initialize(ml_rho, ml_rho_diff);
#ifdef TEST_DENSITY_Z
        m_time_rho.push_back(ml_rho[0]);
#endif
    }

    //clear the fixed state for LOBPCG solver//////////////////////
    if (m_solver == SOLVER::LOBPCG) {
        for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
            m_num_fix_state_sk[sk] = 0;
        }
    }
	
	//calculate potential in real space//////////////////////
	mSetPotentialVhart();
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    mSetPotentialVxc();  //required every after mPrepareCore() because PCC is changed//
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
	mSetPotentialVtot();
	watch.Record(4);
	//MPI_Barrier(m_mpi_comm);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);

	double E_tot = 0.0;

	

	//SCF Loop//////////////////////////////////////////////////////
	for (int scf_step = 0; scf_step < LIMIT_SCF_STEP; ++scf_step) {

		if (is_root_global) {
			printf("Begin SCF Step %d ==============================\n", scf_step + 1); fflush(stdout);
		}

		//Solve the Kohn-Sham equation ////////////////
		switch (m_solver) {
		case SOLVER::Lanczos:
			if (is_root_global) {
				printf("ERROR: Lanczos solver is not supported in mpi run.\n");
			}				
				
			break;
		case SOLVER::LOBPCG:		
		{
			bool is_first_step = ((scf_step == 0) && (dynamics_step==0));
			mSolveLOBPCG(is_first_step ? LOBPCG_STEP_FIRST : LOBPCG_STEP_PER_SCF, scf_step);

			break;
		}
        case SOLVER::DIIS:
        {
            bool is_first_step = ((scf_step == 0) && (dynamics_step == 0));
            mSolveDIIS(is_first_step ? LOBPCG_STEP_FIRST : LOBPCG_STEP_PER_SCF, scf_step);

            break;
        }
		}

		

		watch.Record(5);
		DEBUG_PRINTF("[%d] test5\n", proc_id); gyCheckError(gyGetLastError(), "DEV-ERROR130");
		//calculate occupancy of states//////////////////////
		mSetOccupancy();
		watch.Record(2);
        DEBUG_PRINTF("[%d] test6\n", proc_id); gyCheckError(gyGetLastError(), "DEV-ERROR134");


        mSetDensityByPsi();
        mSymmetrizeDensity();
        double diff_rho = 0.0;
        watch.Record(3);
        DEBUG_PRINTF("[%d] test7\n", proc_id); gyCheckError(gyGetLastError(), "DEV-ERROR141");

        if (IsRoot(m_same_ddm_place_comm)) {

            
            diff_rho = m_mixer->Predict(ml_rho, ml_rho_diff, m_mixing_params);
            diff_rho *= m_dx * m_dy * m_dz / (double)num_electrons;
#ifdef TEST_DENSITY_Z
            m_time_rho.push_back(ml_rho[0]);
#endif
            watch.Record(30);
        }

		
		//calculate potential in real space//////////////////////
		mSetPotentialVhart();
        mSetPotentialVxc();
		mSetPotentialVtot();	
		watch.Record(4);
        DEBUG_PRINTF("[%d] test8\n", proc_id); gyCheckError(gyGetLastError(), "DEV-ERROR160");
		
		const double prev_E_tot = E_tot;
		//calculate total energy//////////////////////////////
		E_tot = mGetTotalEnergy(true, false);
        m_E_tot = E_tot;
		watch.Record(6);
        DEBUG_PRINTF("[%d] test9\n", proc_id);  gyCheckError(gyGetLastError(), "DEV-ERROR167");


		
		//judgement convergence
		bool is_convergence = false;
		if (is_root_global) {
            if (scf_step > 0) {
                printf("delta E_tot = %.15f\n", prev_E_tot - E_tot);
            }
            
			printf("\\int |rho-rho_prev| dr / Ne = %.15f\n", diff_rho); fflush(stdout);
			

			//Check convergence//////////////////////////////////

            switch (m_convergence_mode) {
            case CONVERGENCE_MODE::RHO:
            {
                if (diff_rho < m_threshold_density) {
                    is_convergence = true;
                    printf("SCF is convergence because of delta rho is smaller than threshold, %.5g.\n", m_threshold_density);
                }
            }
            break;
            case CONVERGENCE_MODE::ETOT:
            {
                if (fabs(prev_E_tot - E_tot) < m_threshold_energy) {
                    is_convergence = true;
                    printf("SCF is convergence because of delta Etot is smaller than threshold, %.5g.\n", m_threshold_energy);
                }
            }
            break;
            case CONVERGENCE_MODE::RHO_ETOT:
            {
                if ((diff_rho < m_threshold_density) &&
                    (fabs(prev_E_tot - E_tot) < m_threshold_energy)) {
                    is_convergence = true;
                    printf("SCF is convergence because of delta rho and Etot are smaller than threshold, %.5g and %.5g.\n", m_threshold_density, m_threshold_energy);
                }
            }
            break;
            }

			printf("End SCF Step %d ==============================\n\n", scf_step + 1); fflush(stdout);
		}

		MPI_Bcast(&is_convergence, 1, MPI_C_BOOL, m_root_id, m_mpi_comm);
		if(is_convergence){
			break;
		}

        if (scf_step + 2 == LIMIT_SCF_STEP) {
            OutputEigenValue("test_prev_eigenvalue.txt");
        }
	}
	//////////////////////////////////////End of SCF Loop//

	//print result/////////////////////////////////////
    if (LIMIT_SCF_STEP == 0) {
        E_tot = mGetTotalEnergy(true, false);
        m_E_tot = E_tot;
        watch.Record(6);
    }
	if (is_root_global) {
		printf("%s\n", QUMASUN::note_def_energy); fflush(stdout);
	}
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    watch.Restart();
	mGetForce(false);
    watch.Record(8);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    gyCheckError(gyGetLastError(), "DEV-ERROR239");
	mPrintTime();
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);

#ifdef TEST_DENSITY_Z
    //test///////////////////
    if (IsRoot(m_same_ddm_place_comm)) {
        if (IsRoot(m_ddm_comm)) {
            FILE* fp = fopen("time_rho.txt", "w");
            for (auto&& a : m_time_rho) {
                fprintf(fp, "%.15f\n", a);
            }
            fclose(fp);
        }
    }
#endif

	
	//end of DFT////////////////////////////////////////
}



void QUMASUN_KPOINT::GetForce(vec3d* forces) {
    for (int i = 0; i < m_num_nuclei; ++i) {
        forces[i] = m_nucl_forces[i];
    }
}

/*
* 原子核の座標を変更するときに呼ぶ
* boxサイズは変更しない
* 
*/
inline
void QUMASUN_KPOINT::MoveNuclei(int num_nuclei, const Nucleus* next_nucleis) {
    
    mMoveNuclei(num_nuclei, next_nucleis);
    mPrepareCore();
    watch.Record(1);
    
}


/*
* 原子核の座標だけを変更して、電子状態と電子密度はkeepしたままエネルギーを再計算するときに呼ぶ
* 事前にMoveNucleiは呼んでおく//
* 主にrelaxationで使う
* boxサイズは変更しない
*
*/
inline
double QUMASUN_KPOINT::RecalculateEnergy() {

    watch.Restart();

    mSetPotentialVxc();
    //mSetPotentialVtot();
    watch.Record(4);

    //To skip kinetic term, 1st argument is false.
    //To calculate not only nonlocal energy but also force, 2nd argument is true.
    double E_tot = mGetTotalEnergy(false, true);
    watch.Record(6);
    m_E_tot = E_tot;

    //Since nonlocal force is calculated already, argument is true.
    mGetForce(true);
    watch.Record(8);

    return E_tot;
}


/*
* 原子核の座標を変更するときに呼ぶ
* boxサイズも変更したいときにMoveNucleiではなくこちらを呼ぶ
*
*/
inline
void QUMASUN_KPOINT::ResetBox(double box_x, double box_y, double box_z,
    int num_nuclei, const Nucleus* next_nucleis){

    const double volume_factor = (m_box_x * m_box_y * m_box_z) / (box_x * box_y * box_z);
    const double sqrt_V = sqrt(volume_factor);

    double box_axis[9]{
        box_x, 0.0, 0.0,
        0.0, box_y, 0.0,
        0.0, 0.0, box_z
    };
    mResetBoxSize(box_axis);

    //normilize wave function//
    //空間スケーリングによってdxが変わったので波動関数のnormが1からずれてしまう。
    const size_t local_size = ml_grid.Size3D();
    if (IsRoot(m_same_ddm_place_comm)) {
        

        double sum_rho_up_dn[2]{ 0.0,0.0 };
        for (size_t i = 0; i < local_size; ++i) {
            ml_rho[i] *= volume_factor;
            sum_rho_up_dn[0] += ml_rho[i];
        }
        
        if (is_spin_on) { //spin polarization//
            for (size_t i = 0; i < local_size; ++i) {
                ml_rho_diff[i] *= volume_factor;
                sum_rho_up_dn[1] += ml_rho_diff[i];
            }
        }
        double total_rho_up_dn[2];
        MPI_Reduce(&sum_rho_up_dn[0], &total_rho_up_dn[0], 2, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
        
        if (IsRoot(m_ddm_comm)) {
            total_rho_up_dn[0] *= m_dx * m_dy * m_dz;
            total_rho_up_dn[1] *= m_dx * m_dy * m_dz;
            printf("[Scaling]total_rho = %f + %f = %.15f\n", total_rho_up_dn[0], total_rho_up_dn[1], total_rho_up_dn[0] + total_rho_up_dn[1]);
        }
    }

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        for (int n = 0; n < num_solution; ++n) {
            auto& psi_re = ml_wave_set[sk].l_psi_set[n].re;
            auto& psi_im = ml_wave_set[sk].l_psi_set[n].im;
            for (int i = 0; i < local_size; ++i) {
                psi_re[i] *= sqrt_V;
            }
            for (int i = 0; i < local_size; ++i) {
                psi_im[i] *= sqrt_V;
            }
        }
    }


    for (int ni = 0; ni < num_nuclei; ++ni) {
        m_nuclei[ni] = next_nucleis[ni];
    }
    mInitializePP(true);
    mPrepareCore();
    
}
