#ifdef USE_MPI
#pragma once
#include <complex>
#include <mpi.h>
#include "wave_function.h"
#include "nucleus.h"
#include "vps_loader.h"
//#include "SubspaceField.h"
#include "qumasun_input.h"
#include "GridRange.h"
#include "soacomplex.h"
#include "physical_param.h"
#include "vec3.h"
#include "StopWatch.h"
#include "fftw_executor.h"
#include "LagrangeInterpolation.h"
#include "gyield/gyield.h"
#include "gyield/gy_memory.h"

//#define COMPARISON_INTERPOLATION_MODEL
#ifdef COMPARISON_INTERPOLATION_MODEL
#include "CubicHermiteSpline.h"
#include "FifthSpline.h"
#endif

//Vextの微分を差分ではなくFFTで行う//
#define GRAD_VEXT_FFT 

//Vhartの微分を差分ではなくFFTで行う//
#define GRAD_VHART_FFT     

//Vxcの微分を差分ではなくFFTで行う//
#define GRAD_VXC_FFT


//原子核間の補正項の楕円積分を毎回せずにLagrange補間で代用することで精度向上//
#define INTERPOLATION_CORECORE_CORRECTION

//[TEST用] 原子核間の補正項の楕円積分を無効にする場合(2つ同時に有効無効を切り替えるべし)//
//#define IGNORE_CORECORE_CORRECTION
//#define IGNORE_TF_CORRECTION

//[TEST用] 
//#define IGNORE_XC
//#define IGNORE_EXCHANGE
//#define IGNORE_CORRELATION
//#define IGNORE_GGA_PBE_CORRELATION_H
#define XC_IMPLE_VER2

//[TEST用]
//#define IGNORE_NONLOCAL

//VhartやVlocalのPoissonのFFTのDDMExchangeを利用して並列処理//
#define DDM_FFT

#ifdef DDM_FFT
#include "DDMExchange.h"
#endif

//DDMExchangeと合わせてFFT_manyを使う//
#define USE_FFT_MANY 

#include "DDMGatherScatter.h"

//SCF計算でも運動エネルギー項計算にFFTを使う//
//1の場合はDDMGatherScatterを使う
//2ではLaplacianDDMFFT関数を使う
//#define KINETIC_FFT_SCF    1
//input file の Test.Algorithm.LAplacian に移行した

//VextのDownConvert時にRealSpaceではなくk空間で行う
// 原理的にはDownConvertをRealSpaceでやるよりも精度は良いはず
//TDDFTでEnergyのぶれ方が変わるが精度は同等
//#define DDM_FFT_DOWNCONVERT_CORE

//PCCとXCを高解像度で計算//
//期待に反して精度が落ちる(電子密度やpccが負になるところが出るため)
//#define USE_PCC_HR

//[TEST用] 
//#define IGNORE_PCC


//擬ポテンシャルや内核chargeの原子核座標へのシフトを逆空間で行って引き戻すことで精度向上//
#include "PseudoPotNonlocal_SvF2_kpoint.h"


class QUMASUN_BASE1
{
protected:
	using HAMILTONIAN = QUMASUN::HAMILTONIAN;
    using XC_TYPE = QUMASUN::XC_TYPE;
    using XC_MODEL = QUMASUN::XC_MODEL;
	using Input = QUMASUN::Input;
	using PseudoPotSet = QUMASUN::PseudoPotSet;
	using AtomicWaveSet = QUMASUN::AtomicWaveSet;
	//using SOLVER = QUMASUN::SOLVER;
	//using lapack_complex_double = std::complex<double>;
	
	

public:
    QUMASUN_BASE1(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]);
    virtual ~QUMASUN_BASE1();

	//void Evolve(int dynamics_step, int incremental_steps, double time_step_dt);

protected:
	bool is_construction_successful = false;

protected:
	const HAMILTONIAN m_hamiltonian_type;
    const XC_TYPE m_xc_type;
    const XC_MODEL m_xc_model;


	//parameters for system size and resolution///////////////////////////////////////
	const int m_size_x;//
	const int m_size_y;//
	const int m_size_z;//
	double m_box_x;
	double m_box_y;
	double m_box_z;

	//the following definitions should be written after m_size_x and m_box_x//
	const int m_size_3d;
	double m_dx;
	double m_dy;
	double m_dz;
	//const double m_volume;

	double m_dkx;
	double m_dky;
	double m_dkz;

    //High resolution grid for nuclear charge//
    const int m_HR_ratio_x = 2;
    const int m_HR_ratio_y = 2;
    const int m_HR_ratio_z = 2;


	//parameters for atoms and electrons/////////
    const int m_total_state;
    int m_begin_state = 0;
    int num_solution = 0;
    int m_initial_spin_differnce = 0;
    const bool is_spin_on;
	//int m_num_spin = 1;  //if spin polarization is calculated, m_num_spin becomes 2//
	const int num_electrons;
    
	//Variables: nuclei and energy by core//
	const int m_num_nuclei;

    Nucleus* m_nuclei = nullptr;
	double* m_nuclei_valence_elecron = nullptr;
	double m_diff_Coulomb_Vlocal; //Zi*Zk/r - Zi*Vlocal_k(r)を事前に計算して格納する//
    double m_Enn_close_correction;//Zi*Zk/r - \int rho*Vlocal_k(r)を事前に計算して格納する//
	double m_Ecore_self;          //自己相互作用 Zi*Vlocal_i(r=0)を事前に計算して格納する//
    double m_Ecore_self_v2;       //自己相互作用 int nulc_rho_i*Vlocal_i(r=0)を事前に計算して格納する//
    double m_Ecore_self_HR;       //HR版:自己相互作用 Zi*Vlocal_i(r=0)を事前に計算して格納する//
    double m_E_x;       //Exchange energy, mSetPotentialVtotでポテンシャルと同時に計算//
    double m_E_c;       //Correlation energy, mSetPotentialVtotでポテンシャルと同時に計算//

    struct EnnCorrect {
        double E_nn_cutoff;
        double E_tf_cutoff;
        LagrangeInterpolator<48> E_nn_interpolator;   //24 to 48 are better, 16 or less and 64 or more are bad accuracy//
#ifdef COMPARISON_INTERPOLATION_MODEL
        CubicHermiteSpline E_nn_cubic_spline;
        CubicHermiteSpline_diff2nd E_nn_cubic_spline2;
        FifthSpline E_nn_fifth_spline;
#endif
        double optimal_cutoff_r;  //F_tf - F_correctoon がzeroに近くなるcutoff
        double scale_x_r; //補間用のスケーリングファクター, r in [0,cutoff] -> x in[-1,1]
    };
    std::map<std::pair<int, int>, EnnCorrect> m_const_Enn_close_pairlist;  //CoreCoreEnergyCorrectionとE_TFがカットオフ境界で微妙に異なる(定数)分を補正//


	//parameter for SCF-DFT//////////////////////////
	double m_kbT = 300.0 * KbHartree;// 0.03;

    //Data common to all spins and k-points//////////////////////////////
	gy::unique_aligned_ptr<double[]> m_Vext;
    gy::unique_aligned_ptr<double[]> m_rho;
    gy::unique_aligned_ptr<double[]> m_rho_diff;
    gy::unique_aligned_ptr<double[]> m_rho_prev;
    gy::unique_aligned_ptr<double[]> m_Vhart;
    gy::unique_aligned_ptr<double[]> m_hr_Vext;
    gy::unique_aligned_ptr<double[]> m_hr_rho;
    gy::unique_aligned_ptr<double[]> m_hr_Vhart;

	//data for the case with Pseudo Potential///
	////////////////////////////////////////////

	size_t m_work_size = 0;
	double* m_work = nullptr;
    void mInitializeWorkBuffer(size_t work_size);

    //with FFTW//////////
    FFTW_Executor* m_fftw = nullptr;   //if R2C mdoe, it is used for shift of wave //
    FFTW_Executor* m_HR_fftw = nullptr;
#ifdef DDM_FFT
    DDMExchange* m_ddm_exchanger = nullptr;
#ifdef USE_FFT_MANY
    FFTW_ExecutorMany1D m_HR_fft_many_x;
    FFTW_ExecutorMany1D m_HR_fft_many_y;
    FFTW_ExecutorMany1D m_HR_fft_many_z;
    FFTW_ExecutorMany1D m_fft_many_x;
    FFTW_ExecutorMany1D m_fft_many_y;
    FFTW_ExecutorMany1D m_fft_many_z;
    FFTW_ExecutorMany3D m_fft_many_3d;
#else
    FFTW_Executor1D m_HR_fft_x;
    FFTW_Executor1D m_HR_fft_y;
    FFTW_Executor1D m_HR_fft_z;
#endif
    FFTW_Executor1D m_fft_x;
    FFTW_Executor1D m_fft_y;
    FFTW_Executor1D m_fft_z;
#endif



	//for Pseudo Potential//	
	PseudoPotSet m_pseudo_pot_set;
	AtomicWaveSet m_atomic_wave_set;

    PseudoPotNonlocal_SvF2_kpoint m_pp_SvF;

	//way to set initial wave function//
    //初期値だけなのでクラスメンバから外した方がいいかも//
	std::string m_initial_density;
    std::string m_initial_density_difference;
	std::vector<std::string> m_initial_state_files;
    std::vector<double> m_initial_atomic_spin_difference;
    int m_initial_state_mode = 0;
    std::vector<QUMASUN::OccupancyInfo> m_initial_occupancy;
    bool m_initial_expand_from_kpoint = false;


    std::map<std::string, QUMASUN::AddedVelocityForWave> m_velocity_for_wave;
    vec3d m_initial_shift_grid{ 0.0,0.0,0.0 };
    const int m_algorithm_Laplacian = 1;

    //k-point sampling and state parallelization///////////////////////////////////
	int m_all_kinds_spin_kpoint = 1;  //all spin and k sampling points(Born and von Karman numbers)//
	int m_kpoint_sampling[3]{ 1,1,1 }; //(Born and von Karman numbers)

	uint32_t kpoint_symmetry;

    //force///////////////
    std::unique_ptr<vec3d[]> m_nucl_forces = nullptr;
    std::unique_ptr<vec3d[]> m_force_nonlocal = nullptr;
    std::unique_ptr<vec3d[]> m_force_nn_correction = nullptr;
    std::unique_ptr<vec3d[]> m_force_nn_TF = nullptr;



    static constexpr int NL_FORCE_BANDLE_WIDTH = 8;


private:
    void mInitializeBuffer(const Input& input);

protected:
	bool mCheckConditions();
	void mInitializeState();
private:
	void mInitializeStateRandom(const double* mask);
    void mInitializeStateWave(const double* mask);
    //void mInitializeStateVirtualAtomic(int mode);
	void mInitializeStateFile();

protected:
	void mInitializeDensity();
	
    void mInitializePP(bool is_pp_reset);

	//core and pseudo potential//
	void mPrepareCore();
    void mCoreCoreCorrection();

    //up and down-convert
    void mPoissonDDMFFT_HR(const double* l_hr_rho, double* l_hr_V, double* l_hr_dVdx, double* l_hr_dVdy, double* l_hr_dVdz);
    void mPoissonDDMFFT_Upconvert(const double* l_rho, double* l_hr_Vout, double* l_hr_dVdx_out, double* l_hr_dVdy_out, double* l_hr_dVdz_out, double* l_hr_rho);
    void mDDMFFT_Upconvert(const double* l_rho, double* l_hr_rho);
    void mDDMFFT_Downconvert(const double* l_hr_V, double* l_V);
    //void mDDMFFT_Downconvert_SvF(const double* l_hr_V, double* l_V);
    
private:
	void mSetPotentialVext(double* V, const Nucleus* nuclei, int num_nuclei);
protected:
    void mMoveNuclei(int num_nuclei, const Nucleus* next_nucleis);
    void mResetBoxSize(const double* box_axis);
//private:
    //double mSetDensity(bool is_mixing);
protected:
    void mSetDensityByPsi();
    void mSymmetrizeDensity();
    //double mMixDensity(bool is_mixing);

    //static void mSetDensityOne(GridRange& l_grid, double* l_rho, const SoAComplex* l_psi, const double* occupancy, int num_solution);
	void mSetPotentialVhart();
    void mSetPotentialVxc();
    void mSetPotentialVtot();
private:
	//void mSetPotentialVhart(RspaceFunc<double>& Vhart, RspaceFunc<double>& rho);          //common between PW & RS//
	void mSetPotentialVxc_LDA(double* Vx, double* Vc, const double* rho, const double* pcc_rho); //common between PW & RS//
	void mSetPotentialVxc_LSDA(double* Vx_up, double* Vc_up, double* Vx_down, double* Vc_down, const double* rho, const double* pcc_rho, const double* rho_diff);
    void mSetPotentialVxc_LDA_local(const GridRange& l_grid, double* Vx, double* Vc, const double* rho, const double* pcc_rho);
    void mSetPotentialVxc_LSDA_local(const GridRange& l_grid, double* Vx_up, double* Vc_up, double* Vx_down, double* Vc_down, const double* rho, const double* pcc_rho, const double* rho_diff);


protected:
	void mHamiltonianMatrix_ddm(SoAComplex& l_Hp, const double* l_Vtot, const SoAComplex& l_phi, int kpoint_x, int kpoint_y, int kpoint_z, int id_spin_kpoint);
	void mKineticMatrixAdd_ddm_kpint(SoAComplex& Kp, const SoAComplex& p, int kpoint_x, int kpoint_y, int kpoint_z);
	void mPotentialMatrix_ddm(double* Vp, const double* V, const double* p);

public:  //public is necessary for just CUDA with extended-lambda
    void mHamiltonianMatrix_ddm_bundle(SoAComplex* l_Hp, const double* l_Vtot, const SoAComplex* l_phi, int num_bundle, int kpoint_x, int kpoint_y, int kpoint_z, int id_spin_kpoint);
    void mPotentialMatrixAdd_ddm_bundle(double* Vp, const double* V, const double* p, int num_bundle);


protected:
	void mSetOccupancy();
    void mSetOccupancyInitial();
private:
    void mSetOccupancyZero();
    void mSetOccupancyTemperature();
    void mSetOccupancyIndicate();

protected:
	double mGetTotalEnergy(bool with_kinetic, bool with_nonlocal_force);
    struct EnergyInfo {
        double Ekin;
        double Eext;
        double Ehart;
        double Exc;
        double E_pp;
        double E_corecore_HR;
        double Eext_HR;
        double Ex;
        double Ec;
        double VxRho;
        double VcRho;
        double VxRho_down;
        double VcRho_down;
    };
	void PrintEnergyAll(const int& proc_id, const EnergyInfo& ene);

    //Kinetic energy via FFT///////////////////////////////
    //trueにするとmGetTotalEnergy()関数内で運動エネルギーを計算しない//
    //bool is_already_kinetic_energy = false; 
    double m_Ekin = 0.0;
    double mGetEnergyKineticKspace_algorithm1();
    double mGetEnergyKineticKspace_algorithm2();
    double mGetEnergyKineticKspace() {
        if (m_algorithm_Laplacian == 2) {
            return mGetEnergyKineticKspace_algorithm2();
        } else if (m_algorithm_Laplacian == 1) {
            return mGetEnergyKineticKspace_algorithm1();
        } else {
            return mGetEnergyKinetic_8th_diff();
        }
    }

private:
    double mGetEnergyKinetic_8th_diff();



	double mGetEnergy_V_rho(const double* V, const double* rho);
    double mGetEnergy_V_rho_HR(const double* V, const double* rho);
	double mGetEnergyExtByVextRho();
    double mGetEnergyExtByVextRho_HR(); 
	double mGetEnergyHartree(); 
	double mGetEnergyExtByVhartNuclRho();
    double mGetEnergyExtByVhartNuclRho_HR();
	double mGetEnergyExtByVhartAtPoint();
	double mGetEnergyXC(double* pEx, double* pEc, double* pVxRho, double* pVcRho);
	double mGetEnergyXCSpin(double* pEx, double* pEc, double* pVxRho_up, double* pVcRho_up, double* pVxRho_down, double* pVcRho_down);
#ifdef USE_PCC_HR
    double mGetEnergyXCSpin_HR(double* pEx, double* pEc, double* pVxRho_up, double* pVcRho_up, double* pVxRho_down, double* pVcRho_down);
#endif
    double mGetEnergyPseudoNonlocal();

	double mGetEnergyCoreCore_direct();
	double mGetEnergyPotentialAtNucl(const double* V);
	double mGetEnergyCoreCoreByVextAtPoint();
	double mGetEnergyCoreCoreNuclDensity();
    double mGetEnergyCoreCoreNuclDensity_HR();

protected:
	void mGetForce(bool is_nonlocal_already);
protected:
    void mGetForceOnNuclRho_HR(double* force, const double* l_hr_Vpot);
    void mGetForceOnNuclRho_gradV_HR(double* force, const double* l_dV_dx, double* l_dV_dy, double* l_dV_dz);
    void mGetForceOnNuclRho_diff_proj_HR(double* force, const double* l_hr_Vpot);
    void mGetForceHartreeNuclRho_HR(double* force);
    void mGetForceCoreCoreByVextNuclRho_HR(double* force);
	double mGetForcePseudoNonlocal(double* force);
    void mGetForceXcPcc(double* force);
#ifdef USE_PCC_HR
    void mGetForceXcPcc_HR(double* force);
#endif

protected:
    // DDMにおいて分散したローカルデータはml_で始まる変数とする//
    MPI_Comm m_mpi_comm;

    const int m_root_id = 0;
    MPI_Comm m_ddm_comm;
	MPI_Comm m_same_ddm_place_comm;
	GridRange m_global_grid;      //common to all spins and k-points
	GridRangeMPI ml_grid;         //common to all spins and k-points
	//int m_num_procs_ddm = 1;
	int m_having_spin_kpoint_begin = 0;
	int m_num_having_spin_kpoint = 0;

	enum class SPIN : int {
		UP, DOWN
	};

	
	//kpoinおよびspinごとに保持するデータ構造
	struct WaveSet {		//data for each spin and k-point//
		SoAComplex* l_psi_set = nullptr;		//local grid data split by ddm//
		double* eigen_values = nullptr;
		double* occupancy = nullptr;
		int kpoint_x = 0;
		int kpoint_y = 0;
		int kpoint_z = 0;
		SPIN spin = SPIN::UP;			//1 or -1//
		int kpoint_weight = 1;

		void Allocate(size_t local_size, size_t num_solution, int kx, int ky, int kz, SPIN spin_, int k_weight) {

            kpoint_x = kx;
            kpoint_y = ky;
            kpoint_z = kz;
            spin = spin_;
            kpoint_weight = k_weight;
            

            if (num_solution <= 0)return;

			l_psi_set = new SoAComplex[num_solution];
			//BLAS利用の為に連続領域としてallocateするのが必須//
			//double* buffer = new double[num_solution * local_size * 2]; //2 means complex numebr//
            double* buffer = gy::AlignedAlloc<double>(num_solution * local_size * 2); //2 means complex numebr//
#ifdef SOA_ORBITAL_ORDER2
            for (size_t s = 0; s < num_solution; ++s) {
                l_psi_set[s].re = buffer + s * local_size;
                l_psi_set[s].im = buffer + (s + num_solution) * local_size;
            }
#else
            for (size_t s = 0; s < num_solution; ++s) {
				l_psi_set[s].re = buffer + (s * 2) * local_size;
				l_psi_set[s].im = buffer + (s * 2 + 1) * local_size;
			}
#endif

			eigen_values = gy::AlignedAlloc<double>(num_solution * 2);
			occupancy = eigen_values + num_solution;

			
		}

		~WaveSet() {
            if (l_psi_set != nullptr) gy::AlignedFree(l_psi_set[0].re);
			delete[] l_psi_set;
            gy::AlignedFree(eigen_values);
		}
	};

	WaveSet* ml_wave_set = nullptr;


	//Variables: potential and charge for local region of DDM//
	double* ml_rho = nullptr;       //common to all spins and k-points, used by calculation of density and energy .
    double* ml_rho_diff = nullptr;  //common to all spins and k-points, used by calculation of energy.
    double* ml_Vtot = nullptr;      //depends on spin//
    double* ml_Vtot_down = nullptr; //depends on spin//
    double* ml_Vx = nullptr;        //depends on spin//
    double* ml_Vc = nullptr;        //depends on spin//	
    double* ml_Vx_down = nullptr;        //depends on spin//
    double* ml_Vc_down = nullptr;        //depends on spin//	
    double* ml_Vext = nullptr;      //common to all spins and k-points
    double* ml_pcc_rho = nullptr;   //common to all spins and k-points, used by calculation of energy.
    double* ml_Vhart = nullptr;     //common to all spins and k-points
    double* ml_hr_nucl_rho = nullptr;  //common to all spins and k-points, used by calculation of energy.
    double* ml_hr_Vext = nullptr;      //common to all spins and k-points
    double* ml_hr_rho = nullptr;      //common to all spins and k-points
    double* ml_hr_Vhart = nullptr;      //common to all spins and k-points
#ifdef USE_PCC_HR
    double* ml_hr_rho_diff = nullptr;      //common to all spins and k-points
    double* ml_hr_pcc = nullptr;      //common to all spins and k-points
    double* ml_hr_Vx = nullptr;      //common to all spins and k-points
    double* ml_hr_Vc = nullptr;      //common to all spins and k-points
    double* ml_hr_Vx_down = nullptr;      //common to all spins and k-points
    double* ml_hr_Vc_down = nullptr;      //common to all spins and k-points
#endif
#ifdef GRAD_VEXT_FFT
    double* ml_hr_dVext_dx = nullptr;      //common to all spins and k-points
    double* ml_hr_dVext_dy = nullptr;      //common to all spins and k-points
    double* ml_hr_dVext_dz = nullptr;      //common to all spins and k-points
#endif
#ifdef GRAD_VHART_FFT
    double* ml_hr_dVhart_dx = nullptr;      //common to all spins and k-points
    double* ml_hr_dVhart_dy = nullptr;      //common to all spins and k-points
    double* ml_hr_dVhart_dz = nullptr;      //common to all spins and k-points
#endif
#ifdef GRAD_VXC_FFT
    double* ml_hr_dVxc_dx = nullptr;      //common to all spins and k-points
    double* ml_hr_dVxc_dy = nullptr;      //common to all spins and k-points
    double* ml_hr_dVxc_dz = nullptr;      //common to all spins and k-points
#endif
    //for benchmark/////////////////////////////////////////////
    StopWatch<100, true> watch;

protected:
	//Functions for DDM//
	void mScatterField(double* local_dest, const double* global_src);
	void mGatherField(double* global_dest, const double* local_src);
	void mHierarchyScatterField(double* local_dest, const double* global_src);
	void mHierarchyGatherField(double* global_dest, const double* local_src);
		
    void mScatterField_HR(double* local_dest, const double* global_src);
    void mGatherField_HR(double* global_dest, const double* local_src);
    
    void mReduceFieldInSamePlace(double* local_src_dest);


protected:
    void mShiftRealFieldViaFFT(double* rho, double Rx, double Ry, double Rz);
    void mShiftComplexFieldViaFFT(SoAComplex& psi, double Rx, double Ry, double Rz);
    


public:
	void OutputDensity(QUMASUN::OUTPUT_TARGET target, const char* filepath);
	void OutputEigenValue(const char* filepath);
	void OutputEigenVector(int mode, const char* name_header, const char* list_path, const char* dir_path);
    void OutputEigenVectorList(int mode, const char* filepath_head, std::set<int>& target_states, const char* filepath_tail);
    void OutputOverlapMatrix(const char* filepath, const char* filepath_tail);
    void PrintCondition();

public:
    //計算結果をメモリ上で引き継ぐための機能//
    struct StateMemInfo {
        int total_state;
        int begin_state;
        int num_solution;
        bool is_spin_on;
        int having_spin_kpoint_begin;
        int num_having_spin_kpoint;
        int global_size_x;
        int global_size_y;
        int global_size_z;
        size_t local_size;
        size_t required_byte_size;
        GridRange grid_mem;
    };
     StateMemInfo GetStateMemInfo();
     size_t SaveStateToMemory(double* state_buffer, double* eigen_values, double* occpancy);

     //load用//     
     void LoadStateFromMemory(const StateMemInfo& info, double* load_buffer, const double* eigen_values, const double* occpancy);
protected:
     bool m_is_state_loaded = false;

    ////////////////////////////////////////
    // Error 処理
    // どうしても全プロセスでMPI通信による同期が必要となり、
    // ボトルネックになるので、error信号を常に関数の戻り値で返す使用にはあえてしない
    // 時々CheckError()関数を呼び出して確認するだけにする//
    // CheckError()関数を呼んだ時だけMPI通信する
    //////////////////////////////////
protected:
    
    uint32_t m_error_flag = 0;
public:
    uint32_t CheckError();

};

#include "qumasun_base1.constructor.h"
#include "qumasun_base1.initialize.h"
#include "qumasun_base1.core.h"
#include "qumasun_base1.scatter.h"
#include "qumasun_base1.density.h"
#include "qumasun_base1.potential.h"
#include "qumasun_base1.energy.h"
#include "qumasun_base1.force.h"
#include "qumasun_base1.matrix.h"
#include "qumasun_base1.occupancy.h"
#include "qumasun_base1.move.h"
#include "qumasun_base1.output.h"
#include "qumasun_base1.print.h"
#include "qumasun_base1.error.h"
#include "qumasun_base1.mem_load.h"
#include "qumasun_base1.mem_save.h"

#include "qumasun_base1.ddmfft.poisson_hr.h"
#include "qumasun_base1.ddmfft.poisson_upconvert.h"
#include "qumasun_base1.ddmfft.upconvert.h"
#include "qumasun_base1.ddmfft.downconvert.h"
#include "qumasun_base1.ddmfft.downconvert_svf.h"


#endif
