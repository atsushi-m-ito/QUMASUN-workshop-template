#pragma once
#include <complex>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <string>
#include "wave_function.h"
#include "PseudoPotOperator.h"
#include "vps_loader.h"
#include "nucleus.h"
#include "symmetry_checker.h"



namespace QUMASUN {

	using PseudoPotSet = std::map<int, std::string>;
	using AtomicWaveSet = std::map<int, std::string>;
	enum class SOLVER {
		Lanczos, LOBPCG, DIIS
	};

	enum class HAMILTONIAN {
		Unsupported, Schrodinger, KohnSham_AE, KohnSham_PP, KohnSham_PAW,
	};


	static inline
	HAMILTONIAN ToHamiltonianType(const std::string& word) {
		if (word == "Schrodinger") return HAMILTONIAN::Schrodinger;
		if (word == "KohnSham_AE") return HAMILTONIAN::KohnSham_AE;
		if (word == "KohnSham_PP") return HAMILTONIAN::KohnSham_PP;
		if (word == "KohnSham_PAW") return HAMILTONIAN::KohnSham_PAW;
		return HAMILTONIAN::Unsupported;
	}

    enum class XC_TYPE {
        LDA, GGA, //METAGGA,
    };

    enum class XC_MODEL {
        LDA_CA, GGA_PBE, //METAGGA,
    };




    enum DynamicsMode : int {
        //SCF-DFT////////////////////////////
        None = 0,
        Test = 99,
        Relaxation1 = 101,
        Relaxation2 = 102,
        Relaxation3 = 103,
        Relaxation4 = 104,
        Scaling1 = 51,            //boxsizeを一様にscalingしてエネルギーを比較
        
        //TD-DFT////////////////////////////
        //note: time dependent Khon-Sham is sed when value is 1000 or higher//
        TDDFT = 1000,            //without motion of nuclei
        EhrenfestMDv1 = 2001,
        SymplecticEhrenfestMD = 3002,   //TDDFTと核の運動が共にSymplectic(NLはforward/backward双方向)
        SemiSymplecticTest1 = 3101,  //TDDFT部だけsymplectic(NLがforward片方向のみ).核の運動とは独立//
        SemiSymplecticTest2 = 3102,  //TDDFT部だけsymplectic(NLがforward片方向のみ).核の運動とは独立//
        SemiSymplecticTest3 = 3103   //TDDFT部だけsymplectic(NLはforward/backward双方向).核の運動とは独立//
    };

    struct AddedVelocityForWave {//原子と共に動く場合など、波動関数に初速度を与えるためのもの//
        double center_x;
        double center_y;
        double center_z;
        double velocity_x;
        double velocity_y;
        double velocity_z;
    };
    

    enum MixingMode : int {
        Simple = 0,
        LBFGS = 1,
        SimpleKerker = 2,
        LBFGSKerker = 3,
		DIIS = 4,
    };

    struct OccupancyInfo {
        int num_state;         //この占有率を持つ電子状態の数
        double occupancy_up;   //up-spinの占有率
        double occupancy_down; //down-spinの占有率
    };
    
	struct Input {
		//condition of SCF solver///////////////
		HAMILTONIAN hamiltonian_type;
        XC_TYPE xc_type;
        XC_MODEL xc_model;
        PseudoPotSet pseudo_pot_set;
		AtomicWaveSet atomic_wave_set;
		//condition of system size///////////////
		int grid_size[3];
		double box_axis[9];
		//condition of electrons and nuclei///////////////
		int num_solutions;
		int num_electrons = 0;    //if 0 is set, num_electrons is automatically estimated by PP files.
        int num_nuclei;
		std::vector<Nucleus> nuclei;
        std::map<int, int> ve_for_Z;
		int spin_polarization = 0;
        int initial_spin_difference = 0;
        std::vector<double> initial_atomic_spin_difference;
        int HR_ratio = 2;
		int kpoint_sample[3]{ 1,1,1 };
		uint32_t kpoint_symmetry= KPOINT_SYMMETRY::NONE;
		double temperature_K = 300.0;

        int test_algorithm_Laplacian = 1;        //1 is default, 2 is test//

        std::string initial_density;
        std::string initial_density_difference;
        std::vector < std::string> initial_state;
        int initial_state_mode = 0;
        std::vector < OccupancyInfo > initial_occupancy;
        int initial_expand_from_kpoint=0;

        double test_shift_grid[3]{ 0.0,0.0,0.0 };

        //for SCF //////////////////////////////////////
        int scf_step;
        int eigen_step_per_scf;
        int eigen_step_initial;
        double scf_mixing_ratio = 0.25;
        double scf_mixing_kerker_factor = 2.0;
        MixingMode scf_mixing_mode = MixingMode::DIIS;
        int scf_mixing_num_history = 9;
        double scf_residual_convergence_threshold = 0.0;
        double scf_threshold_density = 0.0;
        double scf_threshold_energy = 0.0;

        //for SCF precalculation in small grid for initial state
        int scf_pre_step = 0; //SCF.Pre.Step       20
        int scf_pre_eigen_step = 0;// SCF.Pre.EigenSolver.Step    2
        int scf_pre_eigen_step_initial = 0;// SCF.Pre.EigenSolver.InitialStep    30
        int scf_pre_grid_size[3]{ 0 };//    SCF.Pre.SpaceGrid    auto
            //SCF.Pre.SpaceGrid.CutoffEnergyRy   50


        //for Dynamics ///////////////////////////////////
        DynamicsMode dynamics_mode = DynamicsMode::None;
        int dynamics_step = 1;
        int dynamics_output_step = 1;
        std::set<int> dynamics_output_orbital_density;
        std::vector<int> save_point;  //saveするstepを指定.複数指定可能//
        int dynamics_start_step = 0;
        double dynamics_timestep_dt = 0.01;
        double dynamics_force_threshold = 0.001;
        std::string output_state_vector;
        std::string output_orbital_density;
        std::vector<double> mass;
        std::vector<double> velocity;
        std::map<std::string, AddedVelocityForWave> velocity_for_wave;
        std::map<std::string, std::string> optional_parameters;
	};


	enum OUTPUT_TARGET {
		Density, Vhart, /*DensityHR,*/ DiffDensity
	};

}

