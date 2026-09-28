#ifdef USE_MPI
#pragma once
#include "qumasun_base1.h"
#include "qumasun_input.h"
#include "mixing_simple.h"
#include "mixing_LBFGS.h"
#include "mixing_DIIS.h"
#include "diis_z_mpi.h"



class QUMASUN_KPOINT : public QUMASUN_BASE1
{

    using SOLVER = QUMASUN::SOLVER;

public:
	QUMASUN_KPOINT(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]);
	~QUMASUN_KPOINT();

    void Initialize();

    /*
    dynamics_stepに1以上の値をセットした場合は電子状態を引き続き使う
    さらに、LOBPCG最初の初回だけのstep数は用いない
    */
	void Execute(int dynamics_step);
	
    enum CONVERGENCE_MODE : int {
        RHO = 1,
        ETOT = 2,
        RHO_ETOT = 3
    };

private:

    bool is_valid_electron_density = false;
    //bool is_valid_core_density = false;

    const SOLVER m_solver;

	const int LIMIT_SCF_STEP;
	const int LOBPCG_STEP_PER_SCF;
	const int LOBPCG_STEP_FIRST;
	
    double m_mixing_params[4]{ 0.0 };
    double m_mixing_max_params0 = 0.0;
    double m_prev_diff_rho = 0.0;

    std::vector<double> m_time_rho;

    double m_E_tot = 0.0;

    std::unique_ptr<int[]> m_num_fix_state_sk;
    double m_residual_convergence_threshold = 0.0;

    inline static const double DEFAULT_THRESHOLD = 1.0e-6;
    double m_threshold_density = 0.0;
    double m_threshold_energy = 0.0;
    
    int m_convergence_mode = CONVERGENCE_MODE::RHO;


    CONVERGENCE_MODE Mode(const Input& input) {
        int mode = 0;
        if (input.scf_threshold_density > 0.0) mode |= 1;
        if (input.scf_threshold_energy > 0.0) mode |= 2;
        return static_cast<CONVERGENCE_MODE>(mode);
    }

private:

	void mSolveLOBPCG(int steps, int SCF_current_step);
    void mSolveDIIS(int steps, int SCF_current_step);

    const int m_diis_history_count = 4;
    std::vector< DIIS_Psi> m_diis_executer;
	
private:

	//LOBPCG方を利用するための保存領域
	//WaveSetと対応関係があり、同じ数存在する。
	struct KeepLOBPCGSet {
		double* keep_p = nullptr;
		OneComplex* keep_S_matrix = nullptr;
		

		void Allocate(size_t local_size, size_t num_solution) {
			keep_p = gy::AlignedAlloc<double>(num_solution * local_size * 2); //2 means complex number//
			keep_S_matrix = gy::AlignedAlloc<OneComplex>(num_solution * num_solution * 3);
		}

		~KeepLOBPCGSet() {
            gy::AlignedFree(keep_p);
            gy::AlignedFree(keep_S_matrix);
		}
        KeepLOBPCGSet() = default;
        KeepLOBPCGSet(const KeepLOBPCGSet&) = delete;
        KeepLOBPCGSet(KeepLOBPCGSet&&) = delete;
	};

	KeepLOBPCGSet* ml_keep_lobpcg = nullptr;
    double* m_work_lobpcg = nullptr;

#ifdef USE_SCALAPACK
    BlacsGridInfo* m_blacs_grid = nullptr;
#endif

private:

    MixingBase* m_mixer = nullptr;

	//for benchmark/////////////////////////////////////////////
	//StopWatch<80, true> watch;
	void mPrintTime();

public:

    void ResetBox(double box_x, double box_y, double box_z, int num_nuclei, const Nucleus* next_nucleis);
	void MoveNuclei(int num_nuclei, const Nucleus* next_nucleis);
    double RecalculateEnergy();
    void GetForce(vec3d* forces);
    double GetTotalEnergy() { return m_E_tot;};
};

#include "qumasun_kpoint.constructor.h"
#include "qumasun_kpoint.lobpcg.h"
#include "qumasun_kpoint.scfloop.h"
#include "qumasun_kpoint.print.h"


#endif
