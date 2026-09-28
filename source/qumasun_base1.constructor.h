#pragma once
#ifdef USE_MPI
#include "mpi_helper.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "qumasun_base1.h"
#include "vps_loader.h"
#include "pao_loader.h"
#include "cube_reader3.h"
//#include "GramSchmidt_mpi.h"
#include "actual_kpoint.h"




inline
QUMASUN_BASE1::QUMASUN_BASE1(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]) :
	//condition of SCF solver///////////////
    m_hamiltonian_type(HAMILTONIAN::KohnSham_PP),    //Schrodinger, KohnSham_AE, KohnSham_PAW, KohnSham_PP//
    m_xc_type(input.xc_type),
    m_xc_model(input.xc_model),
	//condition of system size///////////////
	m_size_x(input.grid_size[0]), m_size_y(input.grid_size[1]), m_size_z(input.grid_size[2]),
	m_size_3d(m_size_x* m_size_y* m_size_z),
	m_box_x(input.box_axis[0]), m_box_y(input.box_axis[4]), m_box_z(input.box_axis[8]),
	m_dx(m_box_x / (double)m_size_x), m_dy(m_box_y / (double)m_size_y), m_dz(m_box_z / (double)m_size_z),
	//m_volume(m_box_x* m_box_y* m_box_z),
    m_HR_ratio_x(input.HR_ratio), m_HR_ratio_y(input.HR_ratio), m_HR_ratio_z(input.HR_ratio),
	//kpoint///////////////////////////////////////
	m_dkx(2.0 * M_PI / (input.box_axis[0] * (double)(input.kpoint_sample[0]))),
	m_dky(2.0 * M_PI / (input.box_axis[4] * (double)(input.kpoint_sample[1]))),
	m_dkz(2.0 * M_PI / (input.box_axis[8] * (double)(input.kpoint_sample[2]))),
    //condition of electrons and nuclei///////////////
    m_total_state(input.num_solutions),
    num_solution(input.num_solutions),//state並列時は書き換わる//
    is_spin_on(input.spin_polarization > 0),
    num_electrons(input.num_electrons),
    m_num_nuclei(input.num_nuclei),
    m_pseudo_pot_set(input.pseudo_pot_set),
	m_atomic_wave_set(input.atomic_wave_set),
	m_kbT(input.temperature_K* KbHartree),	
    //initialconditions/////////////////////////
	m_initial_density(input.initial_density),
    m_initial_density_difference(input.initial_density_difference),
    m_initial_atomic_spin_difference(input.initial_atomic_spin_difference),
	m_initial_state_files(input.initial_state),
    m_initial_state_mode(input.initial_state_mode),
    m_initial_occupancy(input.initial_occupancy),
    //optional and test////////////////////////////////////////
    m_initial_spin_differnce(input.spin_polarization > 0 ? input.initial_spin_difference : 0),
    kpoint_symmetry(input.kpoint_symmetry),
    m_initial_expand_from_kpoint(input.initial_expand_from_kpoint != 0),
    m_velocity_for_wave(input.velocity_for_wave),
    m_initial_shift_grid(input.test_shift_grid[0], input.test_shift_grid[1], input.test_shift_grid[2]),
    m_algorithm_Laplacian(input.test_algorithm_Laplacian),
    //mpi////////////////////////////
	m_mpi_comm(mpi_comm_)
{


	const int num_procs = GetNumProcess(m_mpi_comm);
	const int proc_id = GetProcessID(m_mpi_comm);
    const int num_procs_ddm = ddm_num[0] * ddm_num[1] * ddm_num[2];//ddm: domain decomposition method//
    const int num_procs_sd = ddm_num[3];  //sd: state decomposition//
	const int num_groups_spin_kpoint = num_procs / (num_procs_ddm * num_procs_sd);

	if (num_groups_spin_kpoint * num_procs_ddm * num_procs_sd != num_procs) {
		if (proc_id == m_root_id) {
			printf("The number of total MPI processes must be a multiplier of the ddm parallel number.\n"); fflush(stdout);
		}
		return;
	}

	m_kpoint_sampling[0] = input.kpoint_sample[0];
	m_kpoint_sampling[1] = input.kpoint_sample[1];
	m_kpoint_sampling[2] = input.kpoint_sample[2];
	

	std::vector<Kpoint3D> kpoint_list;
	int all_kinds_kpoint = ListupKpoints(m_kpoint_sampling[0], m_kpoint_sampling[1], m_kpoint_sampling[2], kpoint_symmetry, kpoint_list);


	//	int all_kinds_spin_kpoint = input.kpoint_sample[0] * input.kpoint_sample[1] * input.kpoint_sample[2];
	m_all_kinds_spin_kpoint = all_kinds_kpoint;
	if (is_spin_on) {
		m_all_kinds_spin_kpoint *= 2;
	}

	if (proc_id == m_root_id) {
		printf("kpoint-sample: %d, %d, %d\n", m_kpoint_sampling[0], m_kpoint_sampling[1], m_kpoint_sampling[2]); fflush(stdout);
		printf("Number of active k-point = %d\n", all_kinds_kpoint);
		for (const auto& a : kpoint_list) {
			printf("k-point(%d,%d,%d): weight=%d\n", a.kx, a.ky, a.kz, a.weight);
		}
		fflush(stdout);
		printf("all-spin-kpoint-sample: %d, %d\n", m_all_kinds_spin_kpoint, num_groups_spin_kpoint); fflush(stdout);
	}
	
	if (m_all_kinds_spin_kpoint < num_groups_spin_kpoint) {
		if (proc_id == m_root_id) {
			printf("ERROR: The number of total MPI processes is greater than the required processes.\n");
			printf("Please check the process balance of DDM and k-point parallelization.\n"); 
			fflush(stdout);
		}
		MPI_Barrier(m_mpi_comm);
		return;
	}
	
	if ((m_all_kinds_spin_kpoint / num_groups_spin_kpoint) * num_groups_spin_kpoint != m_all_kinds_spin_kpoint) {
		if (proc_id == m_root_id) {
			printf("Recommendation: it is better performance\n"
				"  when the number of total MPI processes is a multiplier of the kpoint and spin parallel number.\n"); fflush(stdout);
		}		
	}
	
	{
		
		int key_ddm_place = proc_id % num_procs_ddm;
        const int color_ddm = proc_id / num_procs_ddm;
        MPI_Comm_split(m_mpi_comm, color_ddm, key_ddm_place, &m_ddm_comm);
        //m_mpi_split_color = color_ddm;
        
        int key_state_spin_kpoint = proc_id / num_procs_ddm;
		const int color_place = key_ddm_place;
		MPI_Comm_split(m_mpi_comm, color_place, key_state_spin_kpoint, &m_same_ddm_place_comm);

		m_global_grid = MakeRange(m_size_x, m_size_y, m_size_z);
		ml_grid = MakeRange(m_size_x, m_size_y, m_size_z, m_ddm_comm, ddm_num);
		const size_t local_size = ml_grid.Size3D();

		
        int key_sd_rank = (proc_id / num_procs_ddm) % num_procs_sd;
        int key_spin_kpoint = proc_id / (num_procs_ddm * num_procs_sd);

		const int begin_spin_kpoint = (m_all_kinds_spin_kpoint * key_spin_kpoint) / num_groups_spin_kpoint;
		const int end_spin_kpoint = (m_all_kinds_spin_kpoint * (key_spin_kpoint+1)) / num_groups_spin_kpoint;
		

        m_begin_state = (m_total_state * key_sd_rank) / num_procs_sd;
        const int end_state = (m_total_state * (key_sd_rank + 1)) / num_procs_sd;
        num_solution = end_state - m_begin_state;

		m_having_spin_kpoint_begin = begin_spin_kpoint;
		m_num_having_spin_kpoint = end_spin_kpoint - begin_spin_kpoint;
		ml_wave_set = new WaveSet[m_num_having_spin_kpoint];

		for (int sk = begin_spin_kpoint; sk < end_spin_kpoint; ++sk) {
			int spin = (sk >= all_kinds_kpoint) ? 1 : 0;
			const auto& a = kpoint_list[sk % all_kinds_kpoint];
			

			ml_wave_set[sk - begin_spin_kpoint].Allocate(local_size, num_solution, a.kx, a.ky, a.kz, ((spin == 0) ? SPIN::UP : SPIN::DOWN), a.weight);
			//printf("[%d] kpoint = %d,%d,%d, spin=%s\n", proc_id, a.kx, a.ky, a.kz, (spin == 0) ? "up" : "down"); fflush(stdout);

		}

	}
	

    if (IsRoot(m_ddm_comm)) {
        m_fftw = new FFTW_Executor;
        m_fftw->Initialize(m_size_x, m_size_y, m_size_z, FFTW_ESTIMATE);

    }
    if (IsRoot(m_mpi_comm)) {
        m_HR_fftw = new FFTW_Executor;
        m_HR_fftw->Initialize(m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z, FFTW_ESTIMATE);
    }
#ifdef DDM_FFT
    m_ddm_exchanger = new DDMExchange(m_ddm_comm, ddm_num[0], ddm_num[1], ddm_num[2]);
    {
        const int local_size_x = ml_grid.end_x - ml_grid.begin_x;
        const int local_size_y = ml_grid.end_y - ml_grid.begin_y;
        const int local_size_z = ml_grid.end_z - ml_grid.begin_z;

        if (IsRoot(m_same_ddm_place_comm)) {
#ifdef USE_FFT_MANY
            const int num_lines_x = m_ddm_exchanger->GetNumLinesExchangedX(local_size_y * m_HR_ratio_y * local_size_z * m_HR_ratio_z);
            const int num_lines_y = m_ddm_exchanger->GetNumLinesExchangedY(local_size_z * m_HR_ratio_z * local_size_x * m_HR_ratio_x);
            const int num_lines_z = m_ddm_exchanger->GetNumLinesExchangedZ(local_size_x * m_HR_ratio_x * local_size_y * m_HR_ratio_y);
            m_HR_fft_many_x.Initialize(m_size_x * m_HR_ratio_x, num_lines_x);
            m_HR_fft_many_y.Initialize(m_size_y * m_HR_ratio_y, num_lines_y);
            m_HR_fft_many_z.Initialize(m_size_z * m_HR_ratio_z, num_lines_z);
#else
            m_HR_fft_x.Initialize(m_size_x * m_HR_ratio_x, FFTW_ESTIMATE);
            m_HR_fft_y.Initialize(m_size_y * m_HR_ratio_y, FFTW_ESTIMATE);
            m_HR_fft_z.Initialize(m_size_z * m_HR_ratio_z, FFTW_ESTIMATE);
#endif
        }
#ifdef USE_FFT_MANY
        const int num_lines_x = m_ddm_exchanger->GetNumLinesExchangedX(local_size_y * local_size_z);
        const int num_lines_y = m_ddm_exchanger->GetNumLinesExchangedY(local_size_z * local_size_x);
        const int num_lines_z = m_ddm_exchanger->GetNumLinesExchangedZ(local_size_x * local_size_y);
        m_fft_many_x.Initialize(m_size_x, num_lines_x);
        m_fft_many_y.Initialize(m_size_y, num_lines_y);
        m_fft_many_z.Initialize(m_size_z, num_lines_z);

        const int num_gather_scatter = DDMGatherScatter::GetNumDistributed(num_solution, m_ddm_comm);
        m_fft_many_3d.Initialize(m_size_x, m_size_y, m_size_z, num_gather_scatter);
#endif
        m_fft_x.Initialize(m_size_x, FFTW_ESTIMATE);
        m_fft_y.Initialize(m_size_y, FFTW_ESTIMATE);
        m_fft_z.Initialize(m_size_z, FFTW_ESTIMATE);
    }
#endif

	mInitializeBuffer(input);



};

inline
QUMASUN_BASE1::~QUMASUN_BASE1() {

    
	//delete[] ml_psi[0];
	//delete[] ml_psi;
	delete[] ml_wave_set;

	

	gy::AlignedFree(ml_rho);
    gy::AlignedFree(ml_Vtot);
    gy::AlignedFree(ml_Vext);
    gy::AlignedFree(ml_Vhart);
    gy::AlignedFree(ml_Vx);
    gy::AlignedFree(ml_Vc);
    gy::AlignedFree(ml_pcc_rho);

    //for spin//
    gy::AlignedFree(ml_rho_diff);
    gy::AlignedFree(ml_Vtot_down);
    gy::AlignedFree(ml_Vx_down);
    gy::AlignedFree(ml_Vc_down);


	
    gy::AlignedFree(ml_hr_nucl_rho);
    gy::AlignedFree(ml_hr_Vext);
    gy::AlignedFree(ml_hr_rho);
    gy::AlignedFree(ml_hr_Vhart);

#ifdef USE_PCC_HR    
    gy::AlignedFree(ml_hr_rho_diff);
    gy::AlignedFree(ml_hr_pcc);
    gy::AlignedFree(ml_hr_Vx);
    gy::AlignedFree(ml_hr_Vc);
    gy::AlignedFree(ml_hr_Vx_down);
    gy::AlignedFree(ml_hr_Vc_down);
#endif

#ifdef GRAD_VEXT_FFT
    gy::AlignedFree(ml_hr_dVext_dx);
    gy::AlignedFree(ml_hr_dVext_dy);
    gy::AlignedFree(ml_hr_dVext_dz);
#endif
#ifdef GRAD_VHART_FFT
    gy::AlignedFree(ml_hr_dVhart_dx);
    gy::AlignedFree(ml_hr_dVhart_dy);
    gy::AlignedFree(ml_hr_dVhart_dz);
#endif
#ifdef GRAD_VXC_FFT
#ifdef USE_PCC_HR
    gy::AlignedFree(ml_hr_dVxc_dx);
    gy::AlignedFree(ml_hr_dVxc_dy);
    gy::AlignedFree(ml_hr_dVxc_dz);
#endif
#endif
	delete[] m_nuclei;
    delete[] m_nuclei_valence_elecron;
    gy::AlignedFree(m_work);


    delete m_fftw;
    delete m_HR_fftw;
	
#ifdef DDM_FFT 
    delete m_ddm_exchanger;
#endif

};


/*
* MPI並列中にcallしてよい
* process independent
*/
inline
bool QUMASUN_BASE1::mCheckConditions() {
	if (IsRoot(m_mpi_comm)) {
		//printf("sizeof(fftw_complex) = %d\n", (int)sizeof(fftw_complex));
		printf("sizeof(std::complex<double>) = %d\n", (int)sizeof(std::complex<double>));
		fflush(stdout);
	}
    /*
	if (sizeof(fftw_complex) != sizeof(std::complex<double>)) {
		if (IsRoot(m_mpi_comm)) {
			printf("Error: different size, sizeof(fftw_complex) != sizeof(std::complex<double>)\n");
		}
		return false;
	}
    */
	return true;
}

/*
* 粒子の初期位置のセットもここで行っている
* なので、移動しない場合は計算(Execute())前にMoveNuclei()をコールしなくても良い
*/
inline
void QUMASUN_BASE1::mInitializeBuffer(const Input& input) {

    const bool is_root_spin = IsRoot(m_mpi_comm);
    const bool is_root_ddm = IsRoot(m_ddm_comm);

	//memory allocation for domain decomposed sim. on MPI//
	const size_t local_size = ml_grid.Size3D();


	ml_rho = gy::AlignedAlloc<double>(local_size);
	gy::ZeroClear(ml_rho, local_size);
	ml_Vtot = gy::AlignedAlloc<double>(local_size);
	ml_Vext = gy::AlignedAlloc<double>(local_size);
	ml_Vhart = gy::AlignedAlloc<double>(local_size);
	ml_Vx = gy::AlignedAlloc<double>(local_size);
	ml_Vc = gy::AlignedAlloc<double>(local_size);
	ml_pcc_rho = gy::AlignedAlloc<double>(local_size);

    if (is_spin_on) {
        ml_rho_diff = gy::AlignedAlloc<double>(local_size);
        gy::ZeroClear(ml_rho_diff, local_size);
        ml_Vtot_down = gy::AlignedAlloc<double>(local_size);
        ml_Vx_down = gy::AlignedAlloc<double>(local_size);
        ml_Vc_down = gy::AlignedAlloc<double>(local_size);
    }

    const size_t hr_ratio = m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    ml_hr_nucl_rho = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_Vext = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_rho = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_Vhart = gy::AlignedAlloc<double>(local_size * hr_ratio);
#ifdef USE_PCC_HR
    ml_hr_rho_diff = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_pcc = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_Vx = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_Vc = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_Vx_down = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_Vc_down = gy::AlignedAlloc<double>(local_size * hr_ratio);
#endif

#ifdef GRAD_VEXT_FFT
    ml_hr_dVext_dx = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_dVext_dy = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_dVext_dz = gy::AlignedAlloc<double>(local_size * hr_ratio);
#endif
#ifdef GRAD_VHART_FFT
    ml_hr_dVhart_dx = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_dVhart_dy = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_dVhart_dz = gy::AlignedAlloc<double>(local_size * hr_ratio);
#endif

#ifdef GRAD_VXC_FFT
#ifdef USE_PCC_HR
     //HRでないときは追加のバッファは必要ない実装
    ml_hr_dVxc_dx = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_dVxc_dy = gy::AlignedAlloc<double>(local_size * hr_ratio);
    ml_hr_dVxc_dz = gy::AlignedAlloc<double>(local_size * hr_ratio);
#endif
#endif


	m_nuclei = new Nucleus[m_num_nuclei];
	for (int i = 0; i < m_num_nuclei; ++i) {
		m_nuclei[i] = input.nuclei[i];
	}
	
    //test mode////////////////////////////////////
    if (Abs(m_initial_shift_grid) > 1.0e-12) {
        for (int i = 0; i < m_num_nuclei; ++i) {
            m_nuclei[i].Rx += m_initial_shift_grid.x * m_dx;
            m_nuclei[i].Ry += m_initial_shift_grid.y * m_dy;
            m_nuclei[i].Rz += m_initial_shift_grid.z * m_dz;
        }
    }
    ////////////////////////////////////test mode//



	if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
        mInitializePP(false);
	}

    {//Valence Electron/////////////////////////////////////////
        m_nuclei_valence_elecron = new double[m_num_nuclei];
        double ve_total = 0.0;
        for (int i = 0; i < m_num_nuclei; ++i) {
            m_nuclei_valence_elecron[i] = m_pp_SvF.NumValenceElectron(m_nuclei[i].Z);
            ve_total += m_nuclei_valence_elecron[i];
        }

    }



	if (!is_root_ddm) return;//////////////////////////////////////////////

	//initialize////////////////////////////////////////////

	m_Vext = gy::make_unique_aligned<double[]>(m_size_3d);
	m_rho = gy::make_unique_aligned<double[]>(m_size_3d);
	m_rho_prev = gy::make_unique_aligned<double[]>(m_size_3d);
	m_Vhart = gy::make_unique_aligned<double[]>(m_size_3d);

    m_hr_Vext = gy::make_unique_aligned<double[]>(m_size_3d* hr_ratio);
    m_hr_rho = gy::make_unique_aligned<double[]>(m_size_3d * hr_ratio);
    m_hr_Vhart = gy::make_unique_aligned<double[]>(m_size_3d * hr_ratio);

	//for spin polarization///////////////////
	if (is_spin_on) {
		m_rho_diff = gy::make_unique_aligned<double[]>(m_size_3d);
		
	}

}

inline
void QUMASUN_BASE1::mInitializeWorkBuffer(size_t work_size)
{
    size_t min_size, max_size;
    MPI_Reduce(&work_size, &min_size, 1, MPI_UINT64_T, MPI_MIN, 0, m_mpi_comm);
    MPI_Reduce(&work_size, &max_size, 1, MPI_UINT64_T, MPI_MAX, 0, m_mpi_comm);
    if (IsRoot(m_mpi_comm)) {
        printf("NOTE: working memory size(min)= %zd byte\n", min_size * sizeof(double)); 
        printf("NOTE: working memory size(max)= %zd byte\n", max_size * sizeof(double)); fflush(stdout);
    }

    m_work_size = work_size;
    m_work = gy::AlignedAlloc<double>(m_work_size);

}



inline
void QUMASUN_BASE1::mInitializePP(bool is_pp_reset)
{


    m_pp_SvF.InitializeGrid(GridInfo{ (size_t)m_size_3d, m_size_x, m_size_y, m_size_z }, m_dx, m_dy, m_dz);
    m_pp_SvF.SetHighResolution(m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);


    if (is_pp_reset) {
        
        m_pp_SvF.ResetLocal();
        m_pp_SvF.ResetNonlocal();
    } else {
        for (const auto& p : m_pseudo_pot_set) {
            m_pp_SvF.Load(p.first, p.second.c_str(), m_mpi_comm);
        }

        m_pp_SvF.InitializeLocal(m_nuclei, m_num_nuclei, m_mpi_comm);
        m_pp_SvF.InitializeNonlocal(m_nuclei, m_num_nuclei, m_mpi_comm);
        //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    }
    

    //printf("[%d]POS1\n", GetProcessID(m_mpi_comm)); fflush(stdout);

    m_pp_SvF.UpdatePosition(m_nuclei, m_num_nuclei, ml_grid);

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {

        const int kpoint_x = ml_wave_set[sk].kpoint_x;
        const int kpoint_y = ml_wave_set[sk].kpoint_y;
        const int kpoint_z = ml_wave_set[sk].kpoint_z;

        const double gx_dx = (double)kpoint_x * m_dkx * m_dx;
        const double gy_dy = (double)kpoint_y * m_dky * m_dy;
        const double gz_dz = (double)kpoint_z * m_dkz * m_dz;

        m_pp_SvF.UpdateNonlocalBlochExp(m_nuclei, m_num_nuclei, ml_grid, sk, gx_dx, gy_dy, gz_dz);

    }


    
}


/*
* これを
*/
inline 
void QUMASUN_BASE1::mResetBoxSize(const double* box_axis) {
    m_box_x = (box_axis[0]);
    m_box_y = (box_axis[4]);
    m_box_z = (box_axis[8]);
    m_dx = (m_box_x / (double)m_size_x);
    m_dy = (m_box_y / (double)m_size_y);
    m_dz = (m_box_z / (double)m_size_z);
 
    //kpoint///////////////////////////////////////
    m_dkx = (2.0 * M_PI / (m_box_x * (double)(m_kpoint_sampling[0])));
    m_dky = (2.0 * M_PI / (m_box_y * (double)(m_kpoint_sampling[1])));
    m_dkz = (2.0 * M_PI / (m_box_z * (double)(m_kpoint_sampling[2])));


}

#endif
