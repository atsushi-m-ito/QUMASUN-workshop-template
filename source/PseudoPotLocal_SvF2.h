/***********************************************
* 
* Pseudo Potential for Vlocal, which is shifted via Fourier (SvF) Space
* 
*************************************************/

#ifdef USE_MPI
#pragma once
#include <mpi.h>
#include <valarray>
#include "mpi_helper.h"
#include "PseudoPotOperator.h"
#include "vps_mpi_broadcast.h"
//#include "GridSubgrid_arithmetic.h"
#include "soacomplex.h"
#include "comm4atom.h"
#include "SetBlockYlm.h"
#include "shift_via_fft.h"
#include "fftw_executor.h"
#include "shift_to_half_point.h"
#include "pp_test_diagonal.h"
#include "qps_loader.h"
#include "upf_oncv_loader.h"
#include "convertHR.h"


/********************************
* note: subgridとblock
* 擬ポテンシャルはlocal項もnonlocal項も有効カットオフ半径を持つ。カットオフ半径の外では値は0となる。
* そこで、全空間グリッド[0,N)に対して、GridRangeで指定された部分空間グリッド(subgrid)を用意し、
* その内部に擬ポテンシャルを設置すればよい。この段階ではDDMは考慮せず、また、subgridの範囲は
* 全空間グリッドの境界を跨いでいてもよい(overlap)があればよい
*
* 一方で、全空間グリッドをDDMで分割した場合に、各プロセスの担当領域と、
* subgridのオーバーラップする領域をSubgridBlockもしくは単にblockと呼ぶことにする。
* blockは周期境界条件下で、subgridのミラーとのoverlapも想定する。
* そのため、1つのsubgridから生成されるblockは複数存在することがある。
* blockはオーバーラップする領域はGridRange型で表現する(range_blocks)。
* 加えて、周期境界を条件によるミラーとのオーバーラップの場合には、
* subgridをx,y,zのそれぞれの方向にbox幅だけ平行移動したかどうかの情報が必要になる。
* それをShiftedGrid型で表現する(shifted_grids)。
* このようなSubgridBlock型は次のように定義されている
* struct SubgridBlock{
*   std::vector<GridRange> range_blocks;
*	std::vector<ShiftedGrid> shifted_grids;//ブロックが周期境界を跨いで切り取られたときに、空間中の座標として何周期分シフトしているかをしめしたもの//
*	int grid_sizes;//blockの総グリッド数//
*   and additional data
* }
* 
* また、subgridで定義された場のデータを、block上のデータに焼き直す関数が存在する
* 
* note: SvF 
* 擬ポテンシャルは原子核の位置を中心とする中心力ポテンシャルであるが、
* 原子核の位置とグリッドの位置関係に依存して積分値が変わってしまう問題がある。
* 解像度を上げる以外の解決方法として、
* 原子核の移動にともなうポテンシャル中心のシフトをフーリエ変換を介しk空間で行うことで、
* 積分値や内積を原理的に不変とする方法をここでは採用する.
* シフト操作は、全空間グリッドで行う方が精度がよいが(exact)、FFTのコストはかかる。
* subgridでシフト操作を行えば、FFTのコストは下がり、精度もそこそこでる(経験則)。
* 
* 
*****************************/

/*
Note 20240921: 
localグリッド数の偶数／奇数に関する問題
奇数グリッドの問題：
再現ファイル: only_nuclei_tddft.txt
データ: QUMASUN\benchmark\energy_conservation\HeHe_gga_pbe_x_lda_pw_c\time_vs_E_HeHe_xc_check_small4.eps
説明：電子を抜きにしてcoreだけの等速直線運動をさせたとき、奇数グリッドだと不自然な段差ができる

偶数グリッドの問題：
再現ファイル: He_singleB_tddft.txt
説明：nonlocal項のprojectorのノルム<p|p>がシフト時にも保存するために奇数グリッドが必要
偶数グリッドだと、k空間で k*2 == grid_size の時にシフトしてしまうとノルムがずれる。一方でこの場合だけシフトしないのも問題

*/

#if 0
//読み込んだ実数場(核の電荷密度など)の原点を、格子点から半格子ずらす。
//これにより、偶数グリッドでもk*2==grid_sizeの波数の成分がゼロになる
#define SvF2_SUBGRID_EVEN_HALF



#elif 0
//原子核のVlocal相当のchargeをファイルから読み込み(SvF2_SUBGRID_TEST_GAUSSIANと併用不可//
//現状のSvFの実装では偶数の方が奇数よりも精度が出る//
//なぜなら、ちょうどグリッド範囲が切り替わるときに、shift量が0に近いため、
//範囲から外れるグリッド及び範囲に新規にはいるグリッドにおける値も非常に小さくなり、積分値の変化が出にくくなる.
//一方で奇数グリッドや、偶数グリッドでも分布関数のセンターがグリッド点ではなくセル中心になる(SvF2_SUBGRID_EVEN_HALF)場合は
//ちょうどグリッド範囲が切り替わるときに、shift量が最大(グリッド幅の半分)になるため、
//範囲から外れるグリッド及び範囲に新規にはいるグリッドの値が最も大きく、積分値の変化が大きくなりやすい.
//問題点2: SCF-DFTでのnonlocal forceに誤差が乗り易い(diamond8.txt)
#define SvF2_SUBGRID_EVEN2

// k空間で k*2 == grid_sizeとなる波数の係数をzeroにクリア. 
// これによりSvFをしてもノルムが保存される
// ただし、中心からの対称性が崩れてしまい力が釣り合わなくなる//
//#define SvF2_SUBGRID_MID_K_CLEAR
//20241101コードは削除済み上記のコメントを記録として残す//

#elif 1
//v19.16(stable:20240924)
//Nonlocalのprojectorのノルム<p|p>の保存を精度良く満たすためには奇数グリッドが推奨される//
//奇数グリッドではグリッド範囲を切り替えるタイミングはODD1と同じ(Rx+0.5)/dxだが、
//読み込んだ実数場(核の電荷密度など)の原点を、格子点から半格子ずらす(0.5,0.5,0.5に原点がある)。
//これにより、切り替え時はグリッド範囲の境界で明確にzeroになるため、ノイズが減る.
//また、SCF-DFTでのnonlocal forceに誤差がのらない(diamond8.txt)
#define SvF2_SUBGRID_ODD_HALF    1
//#define SvF2_SUBGRID_ODD_HALF_v2

#elif 1
//Nonlocalのprojectorのノルム<p|p>の保存を精度良く満たすためには奇数グリッドが推奨される//
//奇数グリッドではグリッド範囲を切り替えるタイミングを(Rx+0.5)/dxとすることで左右のレンジが対象になるが
//敢えてそうしないことで、切り替え時のノイズを減らす//
#define SvF2_SUBGRID_ODD2

#elif 1
//FFT_r2cをSvFで使うには理想的には奇数が良さそうだが精度が出なかった
//ただし、FORCE_DIFF_PROJと合わせて有効にする場合はSvF2_SUBGRID_EVEN2よりgrad(p)の精度が出る
//Nonlocalのprojectorのノルム<p|p>の保存を精度良く満たすためには奇数グリッドが推奨される//
#define SvF2_SUBGRID_ODD

#else
//原子核のVlocal相当のchargeをGaussianとして強制貼り付け//
#define SvF2_SUBGRID_TEST_GAUSSIAN

#endif

//#define TEST_DIVISOR_RANGE_SvF2
//#define TEST_ALLRANGE_SvF2

#define SvF2_USE_INTERPOLATION  2
inline constexpr int SvF_SHR_ratio = 2;   //this is used when SvF2_USE_INTERPOLATION == 3

#ifdef USE_PCC_HR
#define SvF2_USE_PCC_HR         1
#endif

//PCC経由のxc項のforceが著しく精度が下がることがある//
//#define SvF2_TEST_FULL_BOX
#ifdef SvF2_TEST_FULL_BOX
inline const int svf_box_size_x = 26;
inline const int svf_box_size_y = 44;
inline const int svf_box_size_z = 28;
#endif

//#define DIFF_NUCL_RHO_LOCAL




/*
* VlocalおよびPCC-charge関連の基本的な処理
* また、原子ごとのMPI通信も含む
* このクラスを継承してnonlocalを含む実数関数版(mpi版)および複素関数版(kpoint版)を作成
*/
class PseudoPotLocal_SvF2 : public PseudoPotIntegrator {
protected:
    //貼り付け領域(BlockRange)の大きさを拡大する比率(vlocalに使う)
    //いたずらに大きくしてもノイズが乗るだけ.
    //20240922: もうオラ, マージン無しでもいいんだ.
    static constexpr int SvF2_HR_MARGIN = 0;   //0:v19.16(stable:20240924)
    //貼り付け領域(BlockRange)の大きさを拡大する比率(nonlocal と pccに使う)
    static constexpr int SvF2_MARGIN = 0;     //2 is better for collision of TDDFT(v19.16(stable:20240924), and 3 to 4 is better for SCF-DFT when shifting grid.
    

    std::vector< SubgridBlock> m_hr_vlocal_block;//block化されたVlocalと対応する電荷, high resolution//
    std::vector< SubgridBlock> m_pcc_block;//block化されたVlocalと対応する電荷//
#ifdef SvF2_USE_PCC_HR
    std::vector< SubgridBlock> m_hr_pcc_block;//block化されたVlocalと対応する電荷//
#endif

    std::vector< double*> m_hr_rho_vlocal;  //temporal buffer for nucl charge on block which generated by SvF
#ifdef DIFF_NUCL_RHO_LOCAL
    std::vector< double*> m_hr_rho_vlocal_diff;
#endif

    std::vector< double*> m_pcc_charge;     //temporal buffer for pcc on block which generated by SvF
#ifdef SvF2_USE_PCC_HR
    std::vector< double*> m_hr_pcc_charge;     //temporal buffer for pcc on block which generated by SvF
#endif
    size_t m_max_block_size = 0;
    size_t m_hr_max_block_size = 0;

    //High resolution grid for nuclear charge//
    int m_HR_ratio_x = 1;
    int m_HR_ratio_y = 1;
    int m_HR_ratio_z = 1;

    bool is_root = false;


    std::map<int, double> m_E_self_core_list;


    struct LocalKspace2 {
        OneComplex* charge_kspace = nullptr;
        double cutoff_r;
        GridRange subgrid;
        FFTW_Executor* fftw = nullptr;
    };

    //波数空間度保持された部分密度.SvFする前のデータ//
    std::map<int, LocalKspace2> m_core_charge_kspace;   //元素の数だけ要素がある//
    std::map<int, LocalKspace2> m_pcc_kspace;   //元素の数だけ要素がある//
#ifdef SvF2_USE_PCC_HR
    std::map<int, LocalKspace2> m_hr_pcc_kspace;   //元素の数だけ要素がある//
#endif

    std::vector<GridI3D> m_nucl_int_pos;   //離散化(整数化)された原子位置,要素数は原子核の数//
    std::vector<bool> m_is_update_int_pos;   //離散化(整数化)された原子位置が更新されたか//
    std::vector<GridI3D> m_nucl_int_pos_hr;   //離散化(整数化)された原子位置,要素数は原子核の数//
    std::vector<bool> m_is_update_int_pos_hr;   //離散化(整数化)された原子位置が更新されたか//

    CommForAtoms m_comm4atoms_local;



public:
    virtual ~PseudoPotLocal_SvF2() {
        for (auto&& a : m_core_charge_kspace) {
            gy::AlignedFree( a.second.charge_kspace );
            delete a.second.fftw;
        }

        for (auto&& a : m_pcc_kspace) {
            gy::AlignedFree( a.second.charge_kspace );
            delete a.second.fftw;
        }
#ifdef SvF2_USE_PCC_HR
        for (auto&& a : m_hr_pcc_kspace) {
            gy::AlignedFree( a.second.charge_kspace );
            delete a.second.fftw;
        }
#endif

        for (auto&& a : m_hr_rho_vlocal) {
            gy::AlignedFree(a);
        }

#ifdef DIFF_NUCL_RHO_LOCAL
        for (auto&& density : m_hr_rho_vlocal_diff) {
            gy::AlignedFree(density);
        }
#endif

        for (auto&& a : m_pcc_charge) {
            gy::AlignedFree(a);
        }
#ifdef SvF2_USE_PCC_HR
        for (auto&& a : m_hr_pcc_charge) {
            gy::AlignedFree(a);
        }
#endif

    }

    
public:

    /*
    * mpi並列時に全データで呼ばれる関数
    * [note] PseudoPotIntegrator_vlocalと同じ実装
    */
    void Load(int Z, const char* filepath, MPI_Comm& mpi_comm) {
        if (m_pp_list.find(Z) != m_pp_list.end()) return;

        PseudoPot_MBK* pp = nullptr;
        int mode = 0;
        const int length = std::strlen(filepath);
        if (std::strncmp(filepath + length - 4, ".vps", 4) == 0) {
            mode = 10;
        } else if (std::strncmp(filepath + length - 4, ".qps", 4) == 0) {
            mode = 1;
        } else if (std::strncmp(filepath + length - 4, ".upf", 4) == 0) {
            mode = 20;
        } else {
            return;
        }

        if (IsRoot(mpi_comm)) {
            if (mode == 10) {
                pp = LoadVPS(filepath);
            } else if (mode == 20) {
                pp = LoadUPF_ONCV(filepath);
            } else if (mode == 1) {
                pp = LoadQPS(filepath);
            }
            if (pp == nullptr) {
                printf("ERROR: PP-file cannot be found:%s\n", filepath);
            }
#ifdef IGNORE_PCC
            pp->has_pcc_charge = 0;
#endif
        } else {
            pp = new PseudoPot_MBK;
        }

        //broadcast data//
        BroadcastPseudoPot(pp, mpi_comm, 0);

        //register//
        m_pp_list.emplace(Z, pp);

        is_root = IsRoot(mpi_comm);
        if (is_root) {
            TestDiagonal(pp);
        }
    }



    /*
    * 粒子のk空間でのシフト操作やFFTをMPIで分割して担当するため
    * 担当プロセスを登録する
    * プロセス数より担当粒子数が少ないときは、担当しない場合もある(将来的にFFTもMPI分割するときは再検討)
    */
    void InitializeLocal(const Nucleus* nuclei, int num_nuclei, MPI_Comm& mpi_comm) {
        const int num_procs = GetNumProcess(mpi_comm);
        const int proc_id = GetProcessID(mpi_comm);


        for (int ni = 0; ni < num_nuclei; ++ni) {
            const int Z = nuclei[ni].Z;

            auto it = m_core_charge_kspace.find(Z);
            if (it == m_core_charge_kspace.end()) {
                //nonlocalの新規生成//
                const PseudoPot_MBK* pp = mFindPseudoPot(Z);

                {
                    LocalKspace2 core_charge_kspace;
                    double sum_int_V_rho = mInitializeChargeVlocal(Z, pp, &core_charge_kspace);
                    m_core_charge_kspace.emplace(Z, core_charge_kspace);
                    m_E_self_core_list.emplace(Z, sum_int_V_rho);
                }
                {
                    LocalKspace2 pcc_kspace;
                    mInitializePCC_2(Z, pp, &pcc_kspace, false);
                    m_pcc_kspace.emplace(Z, pcc_kspace);
                }
#ifdef SvF2_USE_PCC_HR
                {
                    LocalKspace2 hr_pcc_kspace;
                    mInitializePCC_2(Z, pp, &hr_pcc_kspace, true);
                    m_hr_pcc_kspace.emplace(Z, hr_pcc_kspace);
                }
#endif
            }
        }

    }

    //gridの幅m_dxが更新されたときに呼ぶ必要がある//
    // InitializeLocalとの違いは空のコンテナに追加するか、すでにあるコンテナを上書きするか
    //
    void ResetLocal() {
        

        for (auto&& a : m_core_charge_kspace) {
            const int Z = a.first;

            //nonlocalの新規生成//
            const PseudoPot_MBK* pp = mFindPseudoPot(Z);
            {
                gy::AlignedFree( a.second.charge_kspace );
                delete a.second.fftw;        
                
                //LocalKspace2 core_charge_kspace;
                double sum_int_V_rho = mInitializeChargeVlocal(Z, pp, &(a.second));
                
                m_E_self_core_list[Z] = sum_int_V_rho;                
            }
            {

                auto& b = m_pcc_kspace[Z];
                gy::AlignedFree( b.charge_kspace);
                delete b.fftw;

                //LocalKspace2 pcc_kspace;
                mInitializePCC_2(Z, pp, &b, false);
                //m_pcc_kspace.emplace(Z, pcc_kspace);                
            }
#ifdef SvF2_USE_PCC_HR
            {

                auto& b = m_hr_pcc_kspace[Z];
                gy::AlignedFree( b.charge_kspace );
                delete b.fftw;

                //LocalKspace2 pcc_kspace;
                mInitializePCC_2(Z, pp, &b, true);
                //m_pcc_kspace.emplace(Z, pcc_kspace);
            }
#endif
        }

        //これによってsubridからrange_blockの再計算がUpdatePosition()において行われるようになる。//
        //これを行わないと、range_blockの再計算が行われないのでバグる//
        m_nucl_int_pos.clear();
        m_nucl_int_pos_hr.clear();
        
    }


protected:
    //Vlocalのためのnucl_chargeに関するデータを擬ポテンシャルからkspaceグリッドデータに変換して保持する//
    //同時に、Vlocalとnucl_rhoの自己相互作用エネルギーも計算して返す//
    double mInitializeChargeVlocal(int Z, const PseudoPot_MBK* pp, LocalKspace2* core_charge_kspace) {
        
            //LocalKspace2 core_charge_kspace;
            const double cutoff_vlocal = pp->cutoff_vlocal;;
            core_charge_kspace->cutoff_r = cutoff_vlocal;
#ifdef SvF2_TEST_FULL_BOX
            GridRange& subgrid = core_charge_kspace->subgrid;
            subgrid.end_x = (svf_box_size_x + 1) / 2;
            subgrid.end_y = (svf_box_size_y + 1) / 2;
            subgrid.end_z = (svf_box_size_z + 1) / 2;
            subgrid.begin_x = subgrid.end_x - svf_box_size_x;
            subgrid.begin_y = subgrid.end_y - svf_box_size_y;
            subgrid.begin_z = subgrid.end_z - svf_box_size_z;

#else
            GridRange& subgrid = core_charge_kspace->subgrid = SubgridFromCutoffHR(cutoff_vlocal);
#endif
            auto* fftw = new FFTW_Executor;
            const int64_t sub_size = subgrid.Size3D();
            OneComplex* k_rho_core = gy::AlignedAlloc<OneComplex>(sub_size);
            OneComplex* rho_core = k_rho_core;
            core_charge_kspace->charge_kspace = k_rho_core;

            //printf("TEST: %s: %d\n", __FILE__,__LINE__);

            fftw->Initialize(subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), FFTW_ESTIMATE);
            core_charge_kspace->fftw = fftw;
            //printf("TEST2: %s: %d\n", __FILE__, __LINE__);
#ifdef SvF2_USE_INTERPOLATION
#ifndef SvF2_SUBGRID_ODD_HALF 
            printf("ERROR: SvF2_USE_INTERPOLATION requires SvF2_SUBGRID_ODD_HALF flag\n");
#endif

            const double hr_dx = m_dx / (double)m_HR_ratio_x;
            const double hr_dy = m_dy / (double)m_HR_ratio_y;
            const double hr_dz = m_dz / (double)m_HR_ratio_z;

            memset(k_rho_core, 0, sizeof(OneComplex) * sub_size);
            //printf("TEST3: %s: %d\n", __FILE__, __LINE__);

            auto rho_core_sub = gy::make_unique_aligned<OneComplex[]>(sub_size);
            double sum_int_rho = 0.0;
            double sum_int_V_rho = 0.0;

            //interpolation for grid-point and half-point//
            constexpr int csize = 2;
            constexpr double half = 0.5;
            double weight = 1.0 / (double)(csize * csize * csize);

            auto fold_pos = [&half, &csize](int cx) {
                return half - (double)(cx * 2 > csize ? cx - csize : cx) / (double)csize;
                };

            for (int cz = 0; cz < csize; ++cz) {
                for (int cy = 0; cy < csize; ++cy) {
                    for (int cx = 0; cx < csize; ++cx) {

                        const double center_x = fold_pos(cx);
                        const double center_y = fold_pos(cy);
                        const double center_z = fold_pos(cz);

                        MyDouble2 res2 = mSetChargeVlocalAtCenter_half_v6(Z, &rho_core_sub[0], subgrid, hr_dx, hr_dy, hr_dz,
                            center_x, center_y, center_z);
                        sum_int_rho += res2.val0;
                        sum_int_V_rho += res2.val1;

                        fftw->ForwardDirect(&rho_core_sub[0]);
#if 0
                        ShiftToHalfPoint(&rho_core_sub[0], cx, cy, cz, csize, fftw, m_dx, m_dy, m_dz, subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ());
                        for (int i = 0; i < sub_size; ++i) {
                            k_rho_core[i].r += rho_core_sub[i].r;
                            k_rho_core[i].i += rho_core_sub[i].i;
                        }
#else

                        ShiftInKspace_update(&rho_core_sub[0], (half - center_x) * m_dx, (half - center_y) * m_dy, (half - center_z) * m_dz, subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), m_dx, m_dy, m_dz);
#endif
                        for (int i = 0; i < sub_size; ++i) {
                            k_rho_core[i].r += rho_core_sub[i].r * weight;
                            k_rho_core[i].i += rho_core_sub[i].i * weight;
                        }
                    }
                }
            }
            sum_int_rho *= weight;
            sum_int_V_rho *= weight;

#if 0
            for (int i = 0; i < sub_size; ++i) {
                k_rho_core[i].r *= weight;
                k_rho_core[i].i *= weight;
            }
#endif
            std::string path("local_cor_Z");
            path += std::to_string(Z) + ".txt";
            CorrectEdgeZeroByWave_3(k_rho_core, fftw, m_dx, m_dy, m_dz, subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), IsRoot(MPI_COMM_WORLD) ? path.c_str() : nullptr);


#else
#ifdef SvF2_SUBGRID_TEST_GAUSSIAN
            auto sum_int_rho = mSetChargeVlocalAtCenter_gaussian(Z, rho_core, subgrid, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
            double sum_int_V_rho = 0.0;
#elif defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)
            MyDouble2 res2 = mSetChargeVlocalAtCenter_half_v6(Z, rho_core, subgrid, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z, 0.5, 0.5, 0.5);
            //                    MyDouble2 res2 = mSetChargeVlocalAtCenter_half_v5(Z, rho_core, subgrid, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
            double sum_int_rho = res2.val0;
            double sum_int_V_rho = res2.val1;
#elif 1 //defined(SvF2_SUBGRID_EVEN2) || defined(SvF2_SUBGRID_ODD) || defined(SvF2_SUBGRID_ODD2)
            MyDouble2 res2 = mSetChargeVlocalAtCenter_v4(Z, rho_core, subgrid, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
            double sum_int_rho = res2.val0;
            double sum_int_V_rho = res2.val1;
#else
            MyDouble2 res2 = mSetChargeVlocalAtCenter_v2(Z, rho_core, subgrid, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
            double sum_int_rho = res2.val0;
            double sum_int_V_rho = res2.val1;
#endif

            //core chargeをk空間に変換//
            fftw->ForwardExecute((fftw_complex*)rho_core, (fftw_complex*)k_rho_core);
            core_charge_kspace.charge_kspace = k_rho_core;
            m_core_charge_kspace.emplace(Z, core_charge_kspace);


            m_E_self_core_list.emplace(Z, sum_int_V_rho);
#endif
        
            //printf("TEST4: %s: %d\n", __FILE__, __LINE__);

            return sum_int_V_rho;
    }


    //PCCに関するデータを擬ポテンシャルからkspaceグリッドデータに変換して保持する
    void mInitializePCC_2(int Z, const PseudoPot_MBK* pp, LocalKspace2* pcc_kspace, bool is_HR) {

        if (pp->has_pcc_charge == 0) {
            //m_pcc_kspace.emplace(Z, LocalKspace2());//empty pcc
        } else {
            
            const double dx = (is_HR) ? m_dx / (double)m_HR_ratio_x : m_dx;
            const double dy = (is_HR) ? m_dy / (double)m_HR_ratio_y : m_dy;
            const double dz = (is_HR) ? m_dz / (double)m_HR_ratio_z : m_dz;
            /*
            const double dx = m_dx;
            const double dy = m_dy;
            const double dz = m_dz;
            */
            //LocalKspace2 pcc_kspace;
            //const double cutoff_pcc = pp->MaxCutoffLength();
            const double cutoff_pcc = pp->cutoff_pcc;
            pcc_kspace->cutoff_r = cutoff_pcc;
#ifdef SvF2_TEST_FULL_BOX
            GridRange& subgrid_pcc = pcc_kspace->subgrid;
            subgrid_pcc.end_x = (svf_box_size_x + 1) / 2;
            subgrid_pcc.end_y = (svf_box_size_y + 1) / 2;
            subgrid_pcc.end_z = (svf_box_size_z + 1) / 2;
            subgrid_pcc.begin_x = subgrid_pcc.end_x - svf_box_size_x;
            subgrid_pcc.begin_y = subgrid_pcc.end_y - svf_box_size_y;
            subgrid_pcc.begin_z = subgrid_pcc.end_z - svf_box_size_z;      
#else
            GridRange& subgrid_pcc = pcc_kspace->subgrid = (is_HR) ? SubgridFromCutoffHR(cutoff_pcc) : SubgridFromCutoff(cutoff_pcc, SvF2_MARGIN);
#endif       

            //printf("TEST1-1: %s: %d\n", __FILE__, __LINE__);
            const double rr_min = (pp->radius[0]) * (pp->radius[0]);

            auto* fftw_pcc = new FFTW_Executor;
            const int64_t sub_size_pcc = subgrid_pcc.Size3D();
            OneComplex* k_pcc_charge = gy::AlignedAlloc<OneComplex>(sub_size_pcc);
            OneComplex* rho_pcc = k_pcc_charge;
            pcc_kspace->charge_kspace = k_pcc_charge;
            //printf("TEST1-2: %s: %d\n", __FILE__, __LINE__);

            fftw_pcc->Initialize(subgrid_pcc.SizeX(), subgrid_pcc.SizeY(), subgrid_pcc.SizeZ(), FFTW_ESTIMATE);
            pcc_kspace->fftw = fftw_pcc;
            //printf("TEST1-3: %s: %d\n", __FILE__, __LINE__);

            auto SetPccOnSubgrid = [&pp, &rr_min](OneComplex* rho_pcc, GridRange& subgrid_pcc, double cutoff_pcc,
                double dx, double dy, double dz,
                double center_x, double center_y, double center_z) {

                    const double rr_max_pcc = cutoff_pcc * cutoff_pcc;
                    ForXYZ(subgrid_pcc,
                        [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {

                            double x = ((double)ix - center_x) * dx;
                            double y = ((double)iy - center_y) * dy;
                            double z = ((double)iz - center_z) * dz;

                            double rr = x * x + y * y + z * z;
                            if (rr < rr_min) {
                                const double p = pp->pcc_charge[0];
                                rho_pcc[i] = p;
                            } else if (rr > rr_max_pcc) {
                                rho_pcc[i] = 0.0;
                            } else {
                                //NOTE: if it is not relative DFT with spin-orbit interaction, projectors (j+1/2) and (j-1/2) should be averaged.
                                const double p = RadialGrid2::GetValueBySquare(rr, pp->pcc_charge, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                rho_pcc[i] = p;
                            }
                        });
                };

#ifdef SvF2_USE_INTERPOLATION
#if SvF2_USE_INTERPOLATION == 3
            memset(k_pcc_charge, 0, sizeof(OneComplex) * sub_size_pcc);

            
            const double center_x = 0.5 * (double)SvF_SHR_ratio;
            const double center_y = 0.5 * (double)SvF_SHR_ratio;
            const double center_z = 0.5 * (double)SvF_SHR_ratio;

            double shr_dx = m_dx / (double)SvF_SHR_ratio;
            double shr_dy = m_dy / (double)SvF_SHR_ratio;
            double shr_dz = m_dz / (double)SvF_SHR_ratio;
            const int64_t shr_size_pcc = sub_size_pcc * SvF_SHR_ratio * SvF_SHR_ratio * SvF_SHR_ratio;
            auto rho_pcc_shr = gy::make_unique_aligned<OneComplex[]>(shr_size_pcc);
            //auto rho_pcc_sub = gy::make_unique_aligned<OneComplex[]>(sub_size_pcc);

            auto shr_subgrid = ScaleRange(subgrid_pcc, SvF_SHR_ratio, SvF_SHR_ratio, SvF_SHR_ratio);
            SetPccOnSubgrid(&rho_pcc_shr[0], shr_subgrid, cutoff_pcc, shr_dx, shr_dy, shr_dz,
                center_x, center_y, center_z);

            FFTW_Executor fftw_shr_pcc;
            fftw_shr_pcc.Initialize(shr_subgrid.SizeX(), shr_subgrid.SizeY(), shr_subgrid.SizeZ(), FFTW_ESTIMATE);
            fftw_shr_pcc.ForwardDirect((fftw_complex*)&rho_pcc_shr[0]);
            DownConvert_Kspace(&rho_pcc_shr[0], subgrid_pcc.SizeX(), subgrid_pcc.SizeY(), subgrid_pcc.SizeZ(), &k_pcc_charge[0], SvF_SHR_ratio, SvF_SHR_ratio, SvF_SHR_ratio);


            ShiftToHalfPoint(&k_pcc_charge[0], 0, 0, 0, 1, fftw_pcc, m_dx, m_dy, m_dz, subgrid_pcc.SizeX(), subgrid_pcc.SizeY(), subgrid_pcc.SizeZ());
            
#else
            memset(k_pcc_charge, 0, sizeof(OneComplex) * sub_size_pcc);
            //printf("TEST1-4: %s: %d\n", __FILE__, __LINE__);

            auto rho_pcc_sub = gy::make_unique_aligned<OneComplex[]>(sub_size_pcc);

            constexpr int csize = 2;
            constexpr double half = 0.5;
            double weight = 1.0 / (double)(csize * csize * csize);

            for (int cz = 0; cz < csize; ++cz) {
                for (int cy = 0; cy < csize; ++cy) {
                    for (int cx = 0; cx < csize; ++cx) {
                        auto fold = [&csize](int cx) {
                            return cx * 2 > csize ? cx - csize : cx;
                            };
                        const double center_x = 0.5 - (double)fold(cx) / (double)csize;
                        const double center_y = 0.5 - (double)fold(cy) / (double)csize;
                        const double center_z = 0.5 - (double)fold(cz) / (double)csize;

                        SetPccOnSubgrid(&rho_pcc_sub[0], subgrid_pcc, cutoff_pcc, dx, dy, dz,
                            center_x, center_y, center_z);

                        fftw_pcc->ForwardDirect(&rho_pcc_sub[0]);
#if 0
                        ShiftToHalfPoint(&rho_pcc_sub[0], cx, cy, cz, csize, fftw_pcc, dx, dy, dz, subgrid_pcc.SizeX(), subgrid_pcc.SizeY(), subgrid_pcc.SizeZ());
#else
                        ShiftInKspace_update(&rho_pcc_sub[0], (half - center_x) * dx, (half - center_y) * dy, (half - center_z) * dz, subgrid_pcc.SizeX(), subgrid_pcc.SizeY(), subgrid_pcc.SizeZ(), dx, dy, dz);
#endif
                        for (int i = 0; i < sub_size_pcc; ++i) {
                            k_pcc_charge[i].r += rho_pcc_sub[i].r * weight;
                            k_pcc_charge[i].i += rho_pcc_sub[i].i * weight;
                        }
                    }
                }
            }
            //printf("TEST1-5: %s: %d\n", __FILE__, __LINE__);
            std::string path("pcc_cor_Z");
            path += std::to_string(Z) + ".txt";
            CorrectEdgeZeroByWave_3(k_pcc_charge, fftw_pcc, m_dx, m_dy, m_dz, subgrid_pcc.SizeX(), subgrid_pcc.SizeY(), subgrid_pcc.SizeZ(), IsRoot(MPI_COMM_WORLD) ? path.c_str() : nullptr);

            //printf("TEST1-6: %s: %d\n", __FILE__, __LINE__);
            
#endif   //SvF2_USE_INTERPOLATION != 3
#else

#if defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)
            SetPccOnSubgrid(rho_pcc, subgrid_pcc, cutoff_pcc, dx, dy, dz, 0.5, 0.5, 0.5);
#else
            SetPccOnSubgrid(rho_pcc, subgrid_pcc, cutoff_pcc, dx, dy, dz, 0.0, 0.0, 0.0);
#endif

            //pcc_chargeをk空間に変換//
            fftw_pcc->ForwardExecute((fftw_complex*)rho_pcc, (fftw_complex*)k_pcc_charge);
            pcc_kspace->charge_kspace = k_pcc_charge;
            
#endif
        }
    }

public:
    double GetEnergySelfCore(const Nucleus* nuclei, int num_nuclei) {
        //const int num_procs = GetNumProcess(mpi_comm);
        //const int proc_id = GetProcessID(mpi_comm);


        double l_E_self_core = 0.0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            const int Z = nuclei[ni].Z;
            l_E_self_core += m_E_self_core_list[Z];
        }

        return l_E_self_core;
    }

protected:

    //約数を求める//
    int Divisor(int val, int mother) {
        for (int i = val; i < mother; ++i) {
            if ((mother / i) * i == mother) return i;
        }
        return mother;
    }

    GridRange SubgridFromCutoffHR(double cutoff) {
#if SvF2_SUBGRID_ODD_HALF == 2
        const int half_x = ((int)ceil(cutoff / (m_dx / (double)m_HR_ratio_x) + 0.5) + SvF2_HR_MARGIN);
        const int half_y = ((int)ceil(cutoff / (m_dy / (double)m_HR_ratio_y) + 0.5) + SvF2_HR_MARGIN);
        const int half_z = ((int)ceil(cutoff / (m_dz / (double)m_HR_ratio_z) + 0.5) + SvF2_HR_MARGIN);
#else
        const int half_x = ((int)ceil(cutoff / (m_dx / (double)m_HR_ratio_x)) + SvF2_HR_MARGIN);
        const int half_y = ((int)ceil(cutoff / (m_dy / (double)m_HR_ratio_y)) + SvF2_HR_MARGIN);
        const int half_z = ((int)ceil(cutoff / (m_dz / (double)m_HR_ratio_z)) + SvF2_HR_MARGIN);
#endif

#ifdef TEST_ALLRANGE_SvF2
        return GridRange{ -half_x, -half_y, -half_z, m_grid.size_x* m_HR_ratio_x - half_x, m_grid.size_y * m_HR_ratio_y - half_y, m_grid.size_z * m_HR_ratio_z - half_z };
#elif defined( TEST_DIVISOR_RANGE_SvF2)
        const int divisor_x = Divisor(half_x * 2, m_grid.size_x * m_HR_ratio_x);
        const int divisor_y = Divisor(half_y * 2, m_grid.size_y * m_HR_ratio_y);
        const int divisor_z = Divisor(half_z * 2, m_grid.size_z * m_HR_ratio_z);
        return GridRange{ -(divisor_x / 2), -(divisor_y/2), -(divisor_z/2), divisor_x - (divisor_x / 2), divisor_y -(divisor_y / 2), divisor_z -(divisor_z / 2) };
#elif defined(SvF2_SUBGRID_EVEN2) || defined(SvF2_SUBGRID_TEST_GAUSSIAN) || defined(SvF2_SUBGRID_EVEN_HALF)
        return GridRange{ -half_x, -half_y, -half_z, half_x + 2, half_y + 2, half_z + 2 };
#elif defined(SvF2_SUBGRID_ODD) || defined(SvF2_SUBGRID_ODD2) || defined(SvF2_SUBGRID_ODD_HALF)
        return GridRange{ -half_x, -half_y, -half_z, half_x + 1, half_y + 1, half_z + 1};
#else
        return GridRange{ -half_x, -half_y, -half_z, half_x, half_y, half_z };
#endif
    }


    GridRange SubgridFromCutoff(double cutoff, int margin) {

#if SvF2_SUBGRID_ODD_HALF == 2
        const int half_x = (int)ceil(cutoff / m_dx + 0.5) + margin;
        const int half_y = (int)ceil(cutoff / m_dy + 0.5) + margin;
        const int half_z = (int)ceil(cutoff / m_dz + 0.5) + margin;
#else
        const int half_x = (int)ceil(cutoff / m_dx) + margin;
        const int half_y = (int)ceil(cutoff / m_dy) + margin;
        const int half_z = (int)ceil(cutoff / m_dz) + margin;
#endif

#ifdef TEST_ALLRANGE_SvF2
        return GridRange{ -half_x, -half_y, -half_z, m_grid.size_x - half_x, m_grid.size_y - half_y, m_grid.size_z - half_z };
#elif defined( TEST_DIVISOR_RANGE_SvF2)
        //const int divisor_x = Divisor(half_x * 2, m_grid.size_x );
        //const int divisor_y = Divisor(half_y * 2, m_grid.size_y );
        //const int divisor_z = Divisor(half_z * 2, m_grid.size_z );
        const int divisor_x = Divisor(half_x, m_grid.size_x ) * 2;
        const int divisor_y = Divisor(half_y, m_grid.size_y ) * 2;
        const int divisor_z = Divisor(half_z, m_grid.size_z ) * 2;
        return GridRange{ -(divisor_x / 2), -(divisor_y / 2), -(divisor_z / 2), divisor_x - (divisor_x / 2), divisor_y - (divisor_y / 2), divisor_z - (divisor_z / 2) };
#elif defined(SvF2_SUBGRID_EVEN2) || defined(SvF2_SUBGRID_TEST_GAUSSIAN) ||defined(SvF2_SUBGRID_EVEN_HALF)
        return GridRange{ -half_x, -half_y, -half_z, half_x + 2, half_y + 2, half_z + 2 };
        //return GridRange{ -half_x, -half_y, -half_z, half_x + 1, half_y + 1, half_z + 1 };
#elif defined(SvF2_SUBGRID_ODD)||defined(SvF2_SUBGRID_ODD2)||defined(SvF2_SUBGRID_ODD_HALF)
        return GridRange{ -half_x, -half_y, -half_z, half_x + 1, half_y + 1, half_z + 1 };
#else
        return GridRange{ -half_x, -half_y, -half_z, half_x, half_y, half_z };
#endif
    }


    /*
    * Use it to determine the integer grid point from position(floating point)
    *
    */
    static
    int GridPosForSvF(double x, double dx) {
#if defined(SvF2_SUBGRID_EVEN2)||defined(SvF2_SUBGRID_ODD2) || defined(SvF2_SUBGRID_TEST_GAUSSIAN)||defined(SvF2_SUBGRID_EVEN_HALF)
        return (int)floor(x / dx );
#elif defined(SvF2_SUBGRID_ODD)||defined(SvF2_SUBGRID_ODD_HALF)
        return (int)floor(x / dx + 0.5);
#elif 1
        return (int)floor(x / dx + 0.5);
#else
        return (int)ceil(x / dx);
#endif
    }

protected:
    /*
    * 粒子位置から離散的なgrid位置を計算する.
    */
    static
        bool UpdateGridPosition(const MPI_Comm& mpi_comm, const Nucleus* nuclei, int num_nuclei,
            std::vector<GridI3D>& nucl_int_pos,
            std::vector<bool>& is_update_int_pos,
            double dx, double dy, double dz)
    {
        bool is_reset_at_least_one = false;
        {
            bool is_reset = false;
            if (nucl_int_pos.size() != num_nuclei) {
                nucl_int_pos.resize(num_nuclei);   //離散化(整数化)された原子位置,要素数は原子核の数//
                is_update_int_pos.resize(num_nuclei);   //離散化(整数化)された原子位置が更新されたか//


                is_reset = true;
                is_reset_at_least_one = true;
            }

            auto int_pos = std::make_unique<GridI3D[]>(num_nuclei);
            for (int ni = 0; ni < num_nuclei; ++ni) {
                //discrete grid point of nuclei//
                const int irx = GridPosForSvF(nuclei[ni].Rx, dx);
                const int iry = GridPosForSvF(nuclei[ni].Ry, dy);
                const int irz = GridPosForSvF(nuclei[ni].Rz, dz);
                //printf("INTPOS-up %d: %d, %d, %d\n", ni, irx, iry, irz);
                int_pos[ni] = { irx, iry, irz };
            }
            MPI_Bcast(&int_pos[0], num_nuclei * sizeof(GridI3D), MPI_BYTE, 0, mpi_comm);
            for (int ni = 0; ni < num_nuclei; ++ni) {
                const int irx = int_pos[ni].x;
                const int iry = int_pos[ni].y;
                const int irz = int_pos[ni].z;

                if (is_reset) {
                    is_update_int_pos[ni] = true;
                } else {
                    is_update_int_pos[ni] = ((nucl_int_pos[ni].x != irx) || (nucl_int_pos[ni].y != iry) || (nucl_int_pos[ni].z != irz));

                    is_reset_at_least_one |= is_update_int_pos[ni];
                }

                nucl_int_pos[ni].x = irx;
                nucl_int_pos[ni].y = iry;
                nucl_int_pos[ni].z = irz;
            }

        }
        return is_reset_at_least_one;
    }

private:


    /*
    * 原子核が原点(0,0,0)にあるとしてデカルトグリッドの離散場にCoreChargeを貼り付け
    * ただし、
    * コアchargeと同じ量の電荷をガウシアンとして張り付ける
    * テストのための実装
    */
    template<class T>
    double mSetChargeVlocalAtCenter_gaussian(const int Z, T* rho_core,
        GridRange& grid,
        double dx, double dy, double dz) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        double cutoff_vlocal = pp->cutoff_vlocal;
        double ix_cutoff = ceil(cutoff_vlocal / dx);
        double iy_cutoff = ceil(cutoff_vlocal / dy);
        double iz_cutoff = ceil(cutoff_vlocal / dz);
        double valence_charge = pp->valence_electron;
        double sigma_rho = cutoff_vlocal / 5.0;

        const int64_t begin_ix = grid.begin_x;
        const int64_t begin_iy = grid.begin_y;
        const int64_t begin_iz = grid.begin_z;
        const int64_t end_ix = grid.end_x;
        const int64_t end_iy = grid.end_y;
        const int64_t end_iz = grid.end_z;
        const int64_t grid_x = grid.SizeX();
        const int64_t grid_y = grid.SizeY();
        const int64_t grid_z = grid.SizeZ();


        //バッファのクリア//
        std::memset(rho_core, 0, sizeof(T) * grid_x * grid_y * grid_z);

        double sum_int_rho = 0.0;
        

        for (int iz = begin_iz; iz < end_iz; ++iz) {
            const double z = dz * (double)iz;
            const double zz = z * z;

            for (int iy = begin_iy; iy < end_iy; ++iy) {
                const double y = dy * (double)iy;
                const double yy_zz = y * y + zz;

                for (int ix = begin_ix; ix < end_ix; ++ix) {
                    const double x = dx * (double)ix;
                    const double rr = x * x + yy_zz;

                    const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));

                    double val = exp(-rr / (2.0 * sigma_rho * sigma_rho)) ;
                    rho_core[io] = val;

                    sum_int_rho += val;
                }
            }
        }
        sum_int_rho *= dx * dy * dz;
        
        const double mag = valence_charge / sum_int_rho;
        sum_int_rho = 0.0;
        for (int64_t i = 0; i < grid_x * grid_y * grid_z; ++i) {
            rho_core[i] *= mag;
            sum_int_rho += rho_core[i];
        }

        return sum_int_rho *= dx * dy * dz;
    }

    /*
    * 原子核が原点からハーフグリッドずれた(0.5,0.5,0.5)にあるとしてデカルトグリッドの離散場にCoreChargeを貼り付け
    */
    struct MyDouble2 {
        double val0, val1;
    };

    /*
    * center_x,y,zにはセンターの位置のグリッド点からのずれの量を指定する。
    * 単位はグリッド幅とする。
    */
    template<class T>
    MyDouble2 mSetChargeVlocalAtCenter_half_v6(const int Z, T* rho_core,
        GridRange& grid,
        double dx, double dy, double dz, double center_x, double center_y, double center_z) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        double cutoff_vlocal = pp->cutoff_vlocal;

        const int64_t begin_ix = grid.begin_x;
        const int64_t begin_iy = grid.begin_y;
        const int64_t begin_iz = grid.begin_z;
        const int64_t end_ix = grid.end_x;
        const int64_t end_iy = grid.end_y;
        const int64_t end_iz = grid.end_z;
        const int64_t grid_x = grid.SizeX();
        const int64_t grid_y = grid.SizeY();
        const int64_t grid_z = grid.SizeZ();

        const double r1_min = (pp->radius[0]) + 1.0e-14;
        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const double rr_max = (cutoff_vlocal) * (cutoff_vlocal);


        //vlocalの貼り付け//
        auto vlocal = gy::make_unique_aligned<double[]>(grid_x * grid_y * grid_z);

        for (int iz = (begin_iz); iz < (end_iz); ++iz) {
            const double z = dz * ((double)iz - center_z);
            const double zz = z * z;
            for (int iy = (begin_iy); iy < (end_iy); ++iy) {
                const double y = dy * ((double)iy - center_y);
                const double yy_zz = y * y + zz;
                for (int ix = (begin_ix); ix < (end_ix); ++ix) {
                    const double x = dx * ((double)ix - center_x);
                    const double rr = x * x + yy_zz;
                    const int i = (ix - (begin_ix)) + grid_x * (iy - (begin_iy)+grid_y * (iz - (begin_iz)));

                    if (rr_min > rr) {
                        vlocal[i] = pp->V_local[0];
                    } else if(rr_max>=rr){
                        vlocal[i] = RadialGrid2::GetValueBySquare(rr, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                    } else {
                        vlocal[i] = 0.0;
                    }

                }
            }

        }



        //バッファのクリア//
        //std::memset(rho_core, 0, sizeof(T) * grid_x * grid_y * grid_z);

        //ラプラシアンの演算//        
        
        const auto& p = vlocal;
        double sum_int_rho = 0.0;
        double sum_int_V_rho = 0.0;

        const int size_xy = grid_x * grid_y;


        if (pp->has_nucl_charge) {
            //VPS24Q which has the nuclear charge density//
            for (int iz = begin_iz; iz < end_iz; ++iz) {
                const double z = dz * ((double)iz - center_z);
                const double zz = z * z;

                for (int iy = begin_iy; iy < end_iy; ++iy) {
                    const double y = dy * ((double)iy - center_y);
                    const double yy_zz = y * y + zz;

                    for (int ix = begin_ix; ix < end_ix; ++ix) {
                        const double x = dx * ((double)ix - center_x);
                        const double rr = x * x + yy_zz;

                        //const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                        const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                        const int i = io;

                        double d2Vdx2=0.0;
                        if (rr_min > rr) {
                            d2Vdx2 = pp->nucl_charge[0];
                        } else if (rr_max >= rr) {
                            d2Vdx2 = RadialGrid2::GetValueBySquare(rr, pp->nucl_charge, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        } 

                        rho_core[io] = d2Vdx2;

                        sum_int_rho += d2Vdx2;
                        sum_int_V_rho += d2Vdx2 * p[i];

                    }
                }
            }
        } else {

            for (int iz = begin_iz; iz < end_iz; ++iz) {
                const double z = dz * ((double)iz - center_z);
                const double zz = z * z;

                for (int iy = begin_iy; iy < end_iy; ++iy) {
                    const double y = dy * ((double)iy - center_y);
                    const double yy_zz = y * y + zz;

                    for (int ix = begin_ix; ix < end_ix; ++ix) {
                        const double x = dx * ((double)ix - center_x);
                        const double rr = x * x + yy_zz;

                        //const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                        const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                        const int i = io;

                        double d2Vdx2 = (1.0 / (2.0 * M_PI)) * RadialGrid2::GetLaplacian(std::max(std::sqrt(rr), r1_min), pp->V_local, 0, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        rho_core[io] = d2Vdx2;

                        sum_int_rho += d2Vdx2;
                        sum_int_V_rho += d2Vdx2 * p[i];

                    }
                }
            }
        }

#if 0//def _DEBUG
        {
            FILE* fp = fopen("rho_core.txt", "w");
            double sigma_rho = cutoff_vlocal / 5.0;

            int64_t ixy = (grid_x/2) + grid_x * (grid_y/2);
            const int64_t iz_end = grid_z;
            const int64_t ixy_end = grid_x * grid_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                //double target_exp = exp(-((0.5 + (double)(iz + begin_iz)) * dz) * ((0.5 + (double)(iz + begin_iz)) * dz) / (2.0 * sigma_rho * sigma_rho));
                fprintf(fp, "%zd\t%.15f\t%.15f\n", iz, rho_core[ixy + ixy_end * iz], rho_core[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif

        sum_int_rho *= dx * dy * dz;
        sum_int_V_rho *= dx * dy * dz;


        return { sum_int_rho , sum_int_V_rho };
    }


    template<class T>
    MyDouble2 mSetChargeVlocalAtCenter_half_v5(const int Z, T* rho_core,
        GridRange& grid,
        double dx, double dy, double dz) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        double cutoff_vlocal = pp->cutoff_vlocal;

        const int64_t begin_ix = grid.begin_x;
        const int64_t begin_iy = grid.begin_y;
        const int64_t begin_iz = grid.begin_z;
        const int64_t end_ix = grid.end_x;
        const int64_t end_iy = grid.end_y;
        const int64_t end_iz = grid.end_z;
        const int64_t grid_x = grid.SizeX();
        const int64_t grid_y = grid.SizeY();
        const int64_t grid_z = grid.SizeZ();

        const double r1_min = (pp->radius[0]) + 1.0e-14;
        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const double rr_max = (cutoff_vlocal) * (cutoff_vlocal);


        //vlocalの貼り付け//
        auto vlocal = gy::make_unique_aligned<double[]>(grid_x * grid_y * grid_z);

        for (int iz = (begin_iz); iz < (end_iz); ++iz) {
            const double z = dz * (-0.5+(double)iz);
            const double zz = z * z;
            for (int iy = (begin_iy); iy < (end_iy); ++iy) {
                const double y = dy * (-0.5+(double)iy);
                const double yy_zz = y * y + zz;
                for (int ix = (begin_ix); ix < (end_ix); ++ix) {
                    const double x = dx * (-0.5+(double)ix);
                    const double rr = x * x + yy_zz;
                    const int i = (ix - (begin_ix)) + grid_x * (iy - (begin_iy)+grid_y * (iz - (begin_iz)));

                    if (rr_min > rr) {
                        vlocal[i] = pp->V_local[0];
                    } else {
                        vlocal[i] = RadialGrid2::GetValueBySquare(rr, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                    }

                }
            }

        }



        //バッファのクリア//
        std::memset(rho_core, 0, sizeof(T) * grid_x * grid_y * grid_z);

        //ラプラシアンの演算//        
        //double* rho_vlocal = new double[half_x * half_y * half_z * 8];
        const auto& p = vlocal;
        double sum_int_rho = 0.0;
        double sum_int_V_rho = 0.0;

        const int size_xy = grid_x * grid_y;


        if (pp->has_nucl_charge) {
            //VPS24Q which has the nuclear charge density//
            for (int iz = begin_iz + SvF2_HR_MARGIN; iz <= -(begin_iz + SvF2_HR_MARGIN); ++iz) {
                const double z = dz * (-0.5+(double)iz);
                const double zz = z * z;

                for (int iy = (begin_iy + SvF2_HR_MARGIN); iy <= -(begin_iy + SvF2_HR_MARGIN); ++iy) {
                    const double y = dy * (-0.5+(double)iy);
                    const double yy_zz = y * y + zz;

                    for (int ix = (begin_ix + SvF2_HR_MARGIN); ix <= -(begin_ix + SvF2_HR_MARGIN); ++ix) {
                        const double x = dx * (-0.5+(double)ix);
                        const double rr = x * x + yy_zz;

                        //const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                        const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                        const int i = io;

                        double d2Vdx2;
                        if (rr_min > rr) {
                            d2Vdx2 = pp->nucl_charge[0];
                        } else {
                            d2Vdx2 = RadialGrid2::GetValueBySquare(rr, pp->nucl_charge, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        }

                        rho_core[io] = d2Vdx2;

                        sum_int_rho += d2Vdx2;
                        sum_int_V_rho += d2Vdx2 * p[i];

                    }
                }
            }
        } else {

            for (int iz = begin_iz + SvF2_HR_MARGIN; iz <= -(begin_iz + SvF2_HR_MARGIN); ++iz) {
                const double z = dz * (-0.5+(double)iz);
                const double zz = z * z;

                for (int iy = (begin_iy + SvF2_HR_MARGIN); iy <= -(begin_iy + SvF2_HR_MARGIN); ++iy) {
                    const double y = dy * (-0.5+(double)iy);
                    const double yy_zz = y * y + zz;

                    for (int ix = (begin_ix + SvF2_HR_MARGIN); ix <= -(begin_ix + SvF2_HR_MARGIN); ++ix) {
                        const double x = dx * (-0.5+(double)ix);
                        const double rr = x * x + yy_zz;

                        //const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                        const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                        const int i = io;

                        double d2Vdx2 = (1.0 / (2.0 * M_PI)) * RadialGrid2::GetLaplacian(std::max(std::sqrt(rr), r1_min), pp->V_local, 0, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        rho_core[io] = d2Vdx2;

                        sum_int_rho += d2Vdx2;
                        sum_int_V_rho += d2Vdx2 * p[i];

                    }
                }
            }
        }

#ifdef _DEBUG
        {
            FILE* fp = fopen("rho_core.txt", "w");
            double sigma_rho = cutoff_vlocal / 5.0;

            int64_t ixy = (8) + grid_x * (8);
            const int64_t iz_end = grid_z;
            const int64_t ixy_end = grid_x * grid_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                double target_exp = exp(-((0.5+(double)(iz + begin_iz)) * dz) * ((0.5+(double)(iz + begin_iz)) * dz) / (2.0 * sigma_rho * sigma_rho));
                fprintf(fp, "%zd\t%.15f\t%.15f\n", iz, rho_core[ixy + ixy_end * iz], rho_core[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif

        sum_int_rho *= dx * dy * dz;
        sum_int_V_rho *= dx * dy * dz;


        return { sum_int_rho , sum_int_V_rho };
    }


    /*
    * 原子核が原点(0,0,0)にあるとしてデカルトグリッドの離散場にCoreChargeを貼り付け
    */
    template<class T>
    MyDouble2 mSetChargeVlocalAtCenter_v4(const int Z, T* rho_core,
        GridRange& grid,
        double dx, double dy, double dz) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        double cutoff_vlocal = pp->cutoff_vlocal;

        const int64_t begin_ix = grid.begin_x;
        const int64_t begin_iy = grid.begin_y;
        const int64_t begin_iz = grid.begin_z;
        const int64_t end_ix = grid.end_x;
        const int64_t end_iy = grid.end_y;
        const int64_t end_iz = grid.end_z;
        const int64_t grid_x = grid.SizeX();
        const int64_t grid_y = grid.SizeY();
        const int64_t grid_z = grid.SizeZ();

        const double r1_min = (pp->radius[0]) + 1.0e-14;
        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const double rr_max = (cutoff_vlocal) * (cutoff_vlocal);
        

        //vlocalの貼り付け//
        auto vlocal = gy::make_unique_aligned<double[]>(grid_x * grid_y * grid_z);

        for (int iz = (begin_iz ); iz < (end_iz ); ++iz) {
            const double z = dz * (double)iz;
            const double zz = z * z;
            for (int iy = (begin_iy ); iy < (end_iy ); ++iy) {
                const double y = dy * (double)iy;
                const double yy_zz = y * y + zz;
                for (int ix = (begin_ix ); ix < (end_ix ); ++ix) {
                    const double x = dx * (double)ix;
                    const double rr = x * x + yy_zz;
                    const int i = (ix - (begin_ix )) + grid_x * (iy - (begin_iy ) + grid_y * (iz - (begin_iz )));

                    if (rr_min > rr) {
                        vlocal[i] = pp->V_local[0];
                    } else {
                        vlocal[i] = RadialGrid2::GetValueBySquare(rr, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                    }

                }
            }

        }



        //バッファのクリア//
        std::memset(rho_core, 0, sizeof(T) * grid_x * grid_y * grid_z);

        //ラプラシアンの演算//        
        
        const auto& p = vlocal;
        double sum_int_rho = 0.0;
        double sum_int_V_rho = 0.0;

        const int size_xy = grid_x * grid_y;


        if (pp->has_nucl_charge) {
            //VPS24Q which has the nuclear charge density//
            for (int iz = begin_iz + SvF2_HR_MARGIN; iz <= - (begin_iz + SvF2_HR_MARGIN); ++iz) {
                const double z = dz * (double)iz;
                const double zz = z * z;

                for (int iy = (begin_iy + SvF2_HR_MARGIN); iy <= - (begin_iy + SvF2_HR_MARGIN); ++iy) {
                    const double y = dy * (double)iy;
                    const double yy_zz = y * y + zz;

                    for (int ix = (begin_ix + SvF2_HR_MARGIN); ix <= - (begin_ix + SvF2_HR_MARGIN); ++ix) {
                        const double x = dx * (double)ix;
                        const double rr = x * x + yy_zz;

                        //const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                        const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                        const int i = io;

                        double d2Vdx2;
                        if (rr_min > rr) {
                            d2Vdx2 = pp->nucl_charge[0];
                        } else {
                            d2Vdx2 = RadialGrid2::GetValueBySquare(rr, pp->nucl_charge, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        }
                        
                        rho_core[io] = d2Vdx2;

                        sum_int_rho += d2Vdx2;
                        sum_int_V_rho += d2Vdx2 * p[i];

                    }
                }
            }
        } else {

            for (int iz = begin_iz + SvF2_HR_MARGIN; iz <= -(begin_iz + SvF2_HR_MARGIN); ++iz) {
                const double z = dz * (double)iz;
                const double zz = z * z;

                for (int iy = (begin_iy + SvF2_HR_MARGIN); iy <= -(begin_iy + SvF2_HR_MARGIN); ++iy) {
                    const double y = dy * (double)iy;
                    const double yy_zz = y * y + zz;

                    for (int ix = (begin_ix + SvF2_HR_MARGIN); ix <= -(begin_ix + SvF2_HR_MARGIN); ++ix) {
                        const double x = dx * (double)ix;
                        const double rr = x * x + yy_zz;
                        
                        //const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                        const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                        const int i = io;

                        double d2Vdx2 = (1.0 / (2.0 * M_PI)) * RadialGrid2::GetLaplacian(std::max(std::sqrt(rr), r1_min), pp->V_local, 0, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        rho_core[io] = d2Vdx2;

                        sum_int_rho += d2Vdx2;
                        sum_int_V_rho += d2Vdx2 * p[i];

                    }
                }
            }
        }

#ifdef _DEBUG
        {
            FILE* fp = fopen("rho_core.txt", "w");
            double sigma_rho = cutoff_vlocal / 5.0;

            int64_t ixy = (8) + grid_x * (8);
            const int64_t iz_end = grid_z;
            const int64_t ixy_end = grid_x * grid_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                double target_exp = exp(-((double)(iz + begin_iz) *dz)* ((double)(iz+begin_iz) * dz) / (2.0 * sigma_rho * sigma_rho));
                fprintf(fp, "%zd\t%.15f\t%.15f\n", iz, rho_core[ixy + ixy_end * iz], rho_core[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif

        sum_int_rho *= dx * dy * dz;
        sum_int_V_rho *= dx * dy * dz;

        //delete[] vlocal;

        return { sum_int_rho , sum_int_V_rho };
    }


    /*
    * 原子核が原点(0,0,0)にあるとしてデカルトグリッドの離散場にCoreChargeを貼り付け
    */
    template<class T>
    MyDouble2 mSetChargeVlocalAtCenter_v3(const int Z, T* rho_core,
        GridRange& grid,
        double dx, double dy, double dz) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        double cutoff_vlocal = pp->cutoff_vlocal;

        const int64_t begin_ix = grid.begin_x;
        const int64_t begin_iy = grid.begin_y;
        const int64_t begin_iz = grid.begin_z;
        const int64_t end_ix = grid.end_x;
        const int64_t end_iy = grid.end_y;
        const int64_t end_iz = grid.end_z;
        const int64_t grid_x = grid.SizeX();
        const int64_t grid_y = grid.SizeY();
        const int64_t grid_z = grid.SizeZ();

        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const double rr_max = (cutoff_vlocal) * (cutoff_vlocal);
        constexpr int margin = 4;


        const int size_x = grid_x + margin * 2;
        const int size_y = grid_y + margin * 2;
        const int size_z = grid_z + margin * 2;

        //vlocalの貼り付け//
        auto vlocal_buf = gy::make_unique_aligned<double[]>(size_x * size_y * size_z);
        double* vlocal = vlocal_buf.get();

        for (int iz = (begin_iz - margin); iz < (end_iz + margin); ++iz) {
            const double z = dz * (double)iz;
            const double zz = z * z;
            for (int iy = (begin_iy - margin); iy < (end_iy + margin); ++iy) {
                const double y = dy * (double)iy;
                const double yy_zz = y * y + zz;
                for (int ix = (begin_ix - margin); ix < (end_ix + margin); ++ix) {
                    const double x = dx * (double)ix;
                    const double rr = x * x + yy_zz;
                    const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));

                    if (rr_min > rr) {
                        vlocal[i] = pp->V_local[0];
                    } else {
                        vlocal[i] = RadialGrid2::GetValueBySquare(rr, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                    }

                }
            }

        }

#ifdef _DEBUG
        {
            FILE* fp = fopen("Vlocal.txt", "w");

            int64_t ixy = (8 + margin) + size_x * (8 + margin);
            const int64_t iz_end = size_z;
            const int64_t ixy_end = size_x * size_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                fprintf(fp, "%zd\t%.15f\n", iz, vlocal[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif


        //バッファのクリア//
        std::memset(rho_core, 0, sizeof(T) * grid_x * grid_y * grid_z);

        //ラプラシアンの演算//        
        
        const double* p = vlocal;
        double sum_int_rho = 0.0;
        double sum_int_V_rho = 0.0;

        const int size_xy = size_x * size_y;


        const double coef = -1.0 / (4.0 * M_PI);
        const double c0 = -14350.0 / 5040.0 * coef * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
        const double c4x = -9.0 * coef / (5040.0 * dx * dx);
        const double c3x = 128.0 * coef / (5040.0 * dx * dx);
        const double c2x = -1008.0 * coef / (5040.0 * dx * dx);
        const double c1x = 8064.0 * coef / (5040.0 * dx * dx);
        const double c4y = -9.0 * coef / (5040.0 * dy * dy);
        const double c3y = 128.0 * coef / (5040.0 * dy * dy);
        const double c2y = -1008.0 * coef / (5040.0 * dy * dy);
        const double c1y = 8064.0 * coef / (5040.0 * dy * dy);
        const double c4z = -9.0 * coef / (5040.0 * dz * dz);
        const double c3z = 128.0 * coef / (5040.0 * dz * dz);
        const double c2z = -1008.0 * coef / (5040.0 * dz * dz);
        const double c1z = 8064.0 * coef / (5040.0 * dz * dz);


        for (int iz = begin_iz + SvF2_HR_MARGIN; iz < end_iz - SvF2_HR_MARGIN - 1; ++iz) {
            const double z = dz * (double)iz;
            const double zz = z * z;

            const int idz_m4 = -4 * size_xy;
            const int idz_m3 = -3 * size_xy;
            const int idz_m2 = -2 * size_xy;
            const int idz_m = -1 * size_xy;
            const int idz_p = 1 * size_xy;
            const int idz_p2 = 2 * size_xy;
            const int idz_p3 = 3 * size_xy;
            const int idz_p4 = 4 * size_xy;

            for (int iy = begin_iy + SvF2_HR_MARGIN; iy < end_iy - SvF2_HR_MARGIN - 1; ++iy) {
                const double y = dy * (double)iy;
                const double yy_zz = y * y + zz;

                const int idy_m4 = -4 * size_x;
                const int idy_m3 = -3 * size_x;
                const int idy_m2 = -2 * size_x;
                const int idy_m = -1 * size_x;
                const int idy_p = 1 * size_x;
                const int idy_p2 = 2 * size_x;
                const int idy_p3 = 3 * size_x;
                const int idy_p4 = 4 * size_x;

                for (int ix = begin_ix + SvF2_HR_MARGIN; ix < end_ix - SvF2_HR_MARGIN; ++ix) {
                    const double x = dx * (double)ix;
                    const double rr = x * x + yy_zz;

                    const int idx_m4 = -4;
                    const int idx_m3 = -3;
                    const int idx_m2 = -2;
                    const int idx_m = -1;
                    const int idx_p = 1;
                    const int idx_p2 = 2;
                    const int idx_p3 = 3;
                    const int idx_p4 = 4;
                    const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                    const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                    //((ix >= 0) ? ix : ix + grid_x) + grid_x * (((iy >= 0) ? iy : iy + grid_y) + grid_y * ((iz >= 0) ? iz : iz + grid_z));


                    //if (rr_max >= rr) {


                        const double d2psidx2 = c4x * (p[i + idx_p4] + p[i + idx_m4]) + c3x * (p[i + idx_p3] + p[i + idx_m3]) + c2x * (p[i + idx_p2] + p[i + idx_m2]) + c1x * (p[i + idx_p] + p[i + idx_m]);
                        const double d2psidy2 = c4y * (p[i + idy_p4] + p[i + idy_m4]) + c3y * (p[i + idy_p3] + p[i + idy_m3]) + c2y * (p[i + idy_p2] + p[i + idy_m2]) + c1y * (p[i + idy_p] + p[i + idy_m]);
                        const double d2psidz2 = c4z * (p[i + idz_p4] + p[i + idz_m4]) + c3z * (p[i + idz_p3] + p[i + idz_m3]) + c2z * (p[i + idz_p2] + p[i + idz_m2]) + c1z * (p[i + idz_p] + p[i + idz_m]);

                        /*
                        if constexpr (std::is_same_v< T, OneComplex>) {
                            rho_core[io].r = d2psidx2;
                        } else { //simple double or float//
                        */
                        rho_core[io] = c0 * p[i] + d2psidx2 + d2psidy2 + d2psidz2;
                        //}
                        sum_int_rho += rho_core[io];
                        sum_int_V_rho += rho_core[io] * p[i];
                        /*
                        if (ix == 0 && iy == 0) {
                            double d2Vdx2 = (1.0 / (2.0 * M_PI)) * RadialGrid2::GetLaplacian(std::sqrt(rr), pp->V_local, 0, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                            double d = d2Vdx2 - rho_core[io];
                            d = d;
                        }
                        */
                    //}

                }
            }
        }

#ifdef _DEBUG
        {
            FILE* fp = fopen("rho_core.txt", "w");

            int64_t ixy = (8) + grid_x * (8);
            const int64_t iz_end = grid_z;
            const int64_t ixy_end = grid_x * grid_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                fprintf(fp, "%zd\t%.15f\n", iz, rho_core[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif

        sum_int_rho *= dx * dy * dz;
        sum_int_V_rho *= dx * dy * dz;

        

        return { sum_int_rho , sum_int_V_rho };
    }

    /*
    * 原子核が原点(0,0,0)にあるとしてデカルトグリッドの離散場にCoreChargeを貼り付け
    */
    template<class T>
    MyDouble2 mSetChargeVlocalAtCenter_v2(const int Z, T* rho_core,
        GridRange& grid,
        double dx, double dy, double dz) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        double cutoff_vlocal = pp->cutoff_vlocal;

        const int64_t begin_ix = grid.begin_x;
        const int64_t begin_iy = grid.begin_y;
        const int64_t begin_iz = grid.begin_z;
        const int64_t end_ix = grid.end_x;
        const int64_t end_iy = grid.end_y;
        const int64_t end_iz = grid.end_z;
        const int64_t grid_x = grid.SizeX();
        const int64_t grid_y = grid.SizeY();
        const int64_t grid_z = grid.SizeZ();

        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const double rr_max = (cutoff_vlocal) * (cutoff_vlocal);
        constexpr int margin = 4;


        const int size_x = grid_x + margin * 2;
        const int size_y = grid_y + margin * 2;
        const int size_z = grid_z + margin * 2;

        //vlocalの貼り付け//
        auto vlocal_buf = gy::make_unique_aligned<double[]>(size_x * size_y * size_z);
        double* vlocal = vlocal_buf.get();

        for (int iz = (begin_iz - margin); iz < (end_iz + margin); ++iz) {
            const double z = dz * (double)iz;
            const double zz = z * z;
            for (int iy = (begin_iy - margin); iy < (end_iy + margin); ++iy) {
                const double y = dy * (double)iy;
                const double yy_zz = y * y + zz;
                for (int ix = (begin_ix - margin); ix < (end_ix + margin); ++ix) {
                    const double x = dx * (double)ix;
                    const double rr = x * x + yy_zz;
                    const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));

                    if (rr_min > rr) {
                        vlocal[i] = pp->V_local[0];
                    } else {
                        vlocal[i] = RadialGrid2::GetValueBySquare(rr, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                    }

                }
            }

        }

#ifdef _DEBUG
        {
            FILE* fp = fopen("Vlocal.txt", "w");

            int64_t ixy = (8+ margin) + size_x * (8 + margin);
            const int64_t iz_end = size_z;
            const int64_t ixy_end = size_x * size_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                fprintf(fp, "%zd\t%.15f\n", iz, vlocal[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif


        //バッファのクリア//
        std::memset(rho_core, 0, sizeof(T) * grid_x * grid_y * grid_z);

        //ラプラシアンの演算//        
        
        const double* p = vlocal;
        double sum_int_rho = 0.0;
        double sum_int_V_rho = 0.0;

        const int size_xy = size_x * size_y;


        const double coef = -1.0 / (4.0 * M_PI);
        const double c0 = -14350.0 / 5040.0 * coef * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
        const double c4x = -9.0 * coef / (5040.0 * dx * dx);
        const double c3x = 128.0 * coef / (5040.0 * dx * dx);
        const double c2x = -1008.0 * coef / (5040.0 * dx * dx);
        const double c1x = 8064.0 * coef / (5040.0 * dx * dx);
        const double c4y = -9.0 * coef / (5040.0 * dy * dy);
        const double c3y = 128.0 * coef / (5040.0 * dy * dy);
        const double c2y = -1008.0 * coef / (5040.0 * dy * dy);
        const double c1y = 8064.0 * coef / (5040.0 * dy * dy);
        const double c4z = -9.0 * coef / (5040.0 * dz * dz);
        const double c3z = 128.0 * coef / (5040.0 * dz * dz);
        const double c2z = -1008.0 * coef / (5040.0 * dz * dz);
        const double c1z = 8064.0 * coef / (5040.0 * dz * dz);


        for (int iz = begin_iz; iz < end_iz; ++iz) {
            const double z = dz * (double)iz;
            const double zz = z * z;

            const int idz_m4 = -4 * size_xy;
            const int idz_m3 = -3 * size_xy;
            const int idz_m2 = -2 * size_xy;
            const int idz_m = -1 * size_xy;
            const int idz_p = 1 * size_xy;
            const int idz_p2 = 2 * size_xy;
            const int idz_p3 = 3 * size_xy;
            const int idz_p4 = 4 * size_xy;

            for (int iy = begin_iy; iy < end_iy; ++iy) {
                const double y = dy * (double)iy;
                const double yy_zz = y * y + zz;

                const int idy_m4 = -4 * size_x;
                const int idy_m3 = -3 * size_x;
                const int idy_m2 = -2 * size_x;
                const int idy_m = -1 * size_x;
                const int idy_p = 1 * size_x;
                const int idy_p2 = 2 * size_x;
                const int idy_p3 = 3 * size_x;
                const int idy_p4 = 4 * size_x;

                for (int ix = begin_ix; ix < end_ix; ++ix) {
                    const double x = dx * (double)ix;
                    const double rr = x * x + yy_zz;

                    const int idx_m4 = -4;
                    const int idx_m3 = -3;
                    const int idx_m2 = -2;
                    const int idx_m = -1;
                    const int idx_p = 1;
                    const int idx_p2 = 2;
                    const int idx_p3 = 3;
                    const int idx_p4 = 4;
                    const int i = (ix - (begin_ix - margin)) + size_x * (iy - (begin_iy - margin) + size_y * (iz - (begin_iz - margin)));
                    const int io = (ix - begin_ix) + grid_x * ((iy - begin_iy) + grid_y * (iz - begin_iz));
                    //((ix >= 0) ? ix : ix + grid_x) + grid_x * (((iy >= 0) ? iy : iy + grid_y) + grid_y * ((iz >= 0) ? iz : iz + grid_z));


                    if (rr_max >= rr) {

                        
                        const double d2psidx2 = c4x * (p[i + idx_p4] + p[i + idx_m4]) + c3x * (p[i + idx_p3] + p[i + idx_m3]) + c2x * (p[i + idx_p2] + p[i + idx_m2]) + c1x * (p[i + idx_p] + p[i + idx_m]);
                        const double d2psidy2 = c4y * (p[i + idy_p4] + p[i + idy_m4]) + c3y * (p[i + idy_p3] + p[i + idy_m3]) + c2y * (p[i + idy_p2] + p[i + idy_m2]) + c1y * (p[i + idy_p] + p[i + idy_m]);
                        const double d2psidz2 = c4z * (p[i + idz_p4] + p[i + idz_m4]) + c3z * (p[i + idz_p3] + p[i + idz_m3]) + c2z * (p[i + idz_p2] + p[i + idz_m2]) + c1z * (p[i + idz_p] + p[i + idz_m]);
                        
                        /*
                        if constexpr (std::is_same_v< T, OneComplex>) {
                            rho_core[io].r = d2psidx2;
                        } else { //simple double or float//
                        */
                        rho_core[io] = c0 * p[i] + d2psidx2 + d2psidy2 + d2psidz2;
                        //}
                        sum_int_rho += rho_core[io];
                        sum_int_V_rho += rho_core[io] * p[i];

                        if (ix == 0 && iy == 0) {
                            double d2Vdx2 = (1.0/(2.0*M_PI)) * RadialGrid2::GetLaplacian(std::sqrt(rr), pp->V_local, 0, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                            double d = d2Vdx2 - rho_core[io];
                            d = d;
                        }

                    }

                }
            }
        }

#ifdef _DEBUG
        {
            FILE* fp = fopen("rho_core.txt", "w");

            int64_t ixy = (8 ) + grid_x * (8 );
            const int64_t iz_end = grid_z;
            const int64_t ixy_end = grid_x * grid_y;
            for (int64_t iz = 0; iz < iz_end; ++iz) {
                fprintf(fp, "%zd\t%.15f\n", iz, rho_core[ixy + ixy_end * iz]);
            }
            fclose(fp);

        }
#endif

        sum_int_rho *= dx * dy * dz;
        sum_int_V_rho *= dx * dy * dz;



        return { sum_int_rho , sum_int_V_rho };
    }

public:

    /*
    * Vlocalに相当する原子核の電荷密度を高解像度で解くグリッドの倍率(整数)
    * [note] PseudoPotIntegrator_vlocalと同じ実装
    */
    void SetHighResolution(int ratio_x, int ratio_y, int ratio_z) {
        m_HR_ratio_x = ratio_x;
        m_HR_ratio_y = ratio_y;
        m_HR_ratio_z = ratio_z;
    }

protected:

    /*
    * SubgridとDDM担当領域がオーバーラップする範囲を求める
    * Periodicを跨いでも重なる場合は跨ぐ前後の二つの範囲に分割される
    * つまり、複数の範囲のリストが返る
    * subgrid_org0は Nonlocalのプロジェクターや原子核電荷,pccなど、小領域の場.
    * その中心は(0,0,0)であり、GridRangeとしては-hogeからhoge+1(or2)を想定している.
    * これをnucl_int_posにシフトし、その後に周期境界での折り畳みを計算
    * l_gridはDDMにおける自プロセスの担当グリッド範囲
    */
    size_t mSetSubspaceBlock2(SubgridBlock& nonlocal_blocks, const GridI3D& nucl_int_pos,
        const GridRange& subgrid_org0, const GridRange& l_grid) {

        const int irx = nucl_int_pos.x;
        const int iry = nucl_int_pos.y;
        const int irz = nucl_int_pos.z;
        GridRange subgrid_actual = subgrid_org0;
        subgrid_actual.begin_x += irx;
        subgrid_actual.end_x += irx;
        subgrid_actual.begin_y += iry;
        subgrid_actual.end_y += iry;
        subgrid_actual.begin_z += irz;
        subgrid_actual.end_z += irz;

        const int total_block_size = GetOverlapPeriodic(l_grid, subgrid_actual, m_grid.size_x, m_grid.size_y, m_grid.size_z, nonlocal_blocks.range_blocks, nonlocal_blocks.shifted_grids);

        nonlocal_blocks.grid_sizes = total_block_size;


        return total_block_size;
    }


public:


    /*
    * localデータとして保持している電子密度やPCCをDDMの担当領域内に配置する//
    * Note: DDMの担当領域内に、電子密度やPCCがoverlapしていたらis_overlapにflagを立てる
    *
    **/
    bool UpdatePositionLocal_2(const Nucleus* nuclei, int num_nuclei, const GridRangeMPI& l_grid) {


        for (auto&& density : m_hr_rho_vlocal) {
            gy::AlignedFree(density);
        }
        m_hr_rho_vlocal.resize(num_nuclei);

#ifdef DIFF_NUCL_RHO_LOCAL
        for (auto&& density : m_hr_rho_vlocal_diff) {
            gy::AlignedFree(density);
        }
        m_hr_rho_vlocal_diff.resize(num_nuclei);
#endif

        for (auto&& density : m_pcc_charge) {
            gy::AlignedFree(density);
        }
        m_pcc_charge.resize(num_nuclei);
#ifdef SvF2_USE_PCC_HR
        for (auto&& density : m_hr_pcc_charge) {
            gy::AlignedFree(density);
        }
        m_hr_pcc_charge.resize(num_nuclei);
#endif


        if (m_hr_vlocal_block.size() != num_nuclei) {
        m_hr_vlocal_block.resize(num_nuclei);
        }
        if (m_pcc_block.size() != num_nuclei) {
        m_pcc_block.resize(num_nuclei);
        }
        
#ifdef SvF2_USE_PCC_HR
        if (m_hr_pcc_block.size() != num_nuclei) {
        m_hr_pcc_block.resize(num_nuclei);
        }
#endif
        m_hr_max_block_size = 0;
        m_max_block_size = 0;

        //原子ごとのprojectorのsubgrid領域が更新されているか調べる///////////////
        bool is_reset_at_least_one = UpdateGridPosition(l_grid.mpi_comm, nuclei, num_nuclei, m_nucl_int_pos, m_is_update_int_pos, m_dx, m_dy, m_dz);
        const double hr_dx = m_dx / (double)m_HR_ratio_x;
        const double hr_dy = m_dy / (double)m_HR_ratio_y;
        const double hr_dz = m_dz / (double)m_HR_ratio_z;
        UpdateGridPosition(l_grid.mpi_comm, nuclei, num_nuclei, m_nucl_int_pos_hr, m_is_update_int_pos_hr, hr_dx, hr_dy, hr_dz);

        auto hr_grid = ScaleRange(l_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
        

        auto is_overlap = gy::make_unique_aligned<int[]>(num_nuclei);

        for (int ni = 0; ni < num_nuclei; ++ni) {
            const int Z = nuclei[ni].Z;

            if (m_is_update_int_pos_hr[ni]) {
                m_hr_vlocal_block[ni].range_blocks.clear();
                m_hr_vlocal_block[ni].shifted_grids.clear();
                mSetSubspaceBlockHR(m_hr_vlocal_block[ni], m_nucl_int_pos_hr[ni], m_core_charge_kspace[Z].subgrid, hr_grid);
            }
            if (m_hr_max_block_size < m_hr_vlocal_block[ni].grid_sizes) m_hr_max_block_size = m_hr_vlocal_block[ni].grid_sizes;

#ifdef DIFF_NUCL_RHO_LOCAL
            mCreateDensityOnBlock_SvF(m_core_charge_kspace[Z], m_hr_vlocal_block[ni], &m_hr_rho_vlocal[ni], hr_dx, hr_dy, hr_dz, nuclei[ni], m_nucl_int_pos_hr[ni], hr_grid, &m_hr_rho_vlocal_diff[ni]);
#else
            mCreateDensityOnBlock_SvF(m_core_charge_kspace[Z], m_hr_vlocal_block[ni], &m_hr_rho_vlocal[ni], hr_dx, hr_dy, hr_dz, nuclei[ni], m_nucl_int_pos_hr[ni], hr_grid);
#endif

#ifdef _DEBUG_2
            if (ni == 1) {
                static int step = 0;
                /*{
                    std::string filepath("SvF_rho" + std::to_string(step) + ".txt");
                    FILE* fp = fopen(filepath.c_str(), "w");
                    int64_t iz = 0;
                    const int64_t ixy_end = m_hr_vlocal_block[ni].range_blocks[0].SizeX() * m_hr_vlocal_block[ni].range_blocks[0].SizeY();
                    for (int64_t ixy = 0; ixy < ixy_end; ++ixy) {
                        fprintf(fp, "%zd\t%.10f\n", ixy, m_hr_rho_vlocal[ni][ixy + ixy_end * iz]);
                    }
                    fclose(fp);
                }*/
                {
                    std::string filepath("SvF_rho" + std::to_string(step) + ".txt");
                    FILE* fp = fopen(filepath.c_str(), "w");

                    int64_t ixy = 9 + m_hr_vlocal_block[ni].range_blocks[0].SizeX() * 9;
                    const int64_t iz_end = m_hr_vlocal_block[ni].range_blocks[0].SizeZ();
                    const int64_t ixy_end = m_hr_vlocal_block[ni].range_blocks[0].SizeX() * m_hr_vlocal_block[ni].range_blocks[0].SizeY();
                    for (int64_t iz = 0; iz < iz_end; ++iz) {
                        fprintf(fp, "%zd\t%.15f\n", iz, m_hr_rho_vlocal[ni][ixy + ixy_end * iz]);
                    }
                    fclose(fp);

                }
                ++step;

            }
#endif

            is_overlap[ni] = m_hr_vlocal_block[ni].grid_sizes;

            //pcc charge
            if (m_pcc_kspace[Z].charge_kspace == nullptr) {
                m_pcc_charge[ni] = nullptr;

            } else {
                if (m_is_update_int_pos[ni]) {
                    m_pcc_block[ni].range_blocks.clear();
                    m_pcc_block[ni].shifted_grids.clear();
                    mSetSubspaceBlock2(m_pcc_block[ni], m_nucl_int_pos[ni], m_pcc_kspace[Z].subgrid, l_grid);
                }
                if (m_max_block_size < m_pcc_block[ni].grid_sizes) m_max_block_size = m_pcc_block[ni].grid_sizes;
                mCreateDensityOnBlock_SvF(m_pcc_kspace[Z], m_pcc_block[ni], &m_pcc_charge[ni], m_dx, m_dy, m_dz, nuclei[ni], m_nucl_int_pos[ni], l_grid);

                is_overlap[ni] += m_pcc_block[ni].grid_sizes;
            }

#ifdef SvF2_USE_PCC_HR
            //pcc charge
            if (m_hr_pcc_kspace[Z].charge_kspace == nullptr) {
                m_hr_pcc_charge[ni] = nullptr;

            } else {

                if (m_is_update_int_pos_hr[ni]) {
                    m_hr_pcc_block[ni].range_blocks.clear();
                    m_hr_pcc_block[ni].shifted_grids.clear();
                    mSetSubspaceBlockHR(m_hr_pcc_block[ni], m_nucl_int_pos_hr[ni], m_hr_pcc_kspace[Z].subgrid, hr_grid);
                }
                if (m_hr_max_block_size < m_hr_pcc_block[ni].grid_sizes) m_hr_max_block_size = m_hr_pcc_block[ni].grid_sizes;
                mCreateDensityOnBlock_SvF(m_hr_pcc_kspace[Z], m_hr_pcc_block[ni], &m_hr_pcc_charge[ni], hr_dx, hr_dy, hr_dz, nuclei[ni], m_nucl_int_pos_hr[ni], hr_grid);

                is_overlap[ni] += m_hr_pcc_block[ni].grid_sizes;
            }
#endif
        }


        if (is_reset_at_least_one) {//DDM領域と粒子中心のsubgridのoverlapによるcomm-splitの再設定//
            m_comm4atoms_local.DeleteComms();
            m_comm4atoms_local.CreateCommsDirect(&is_overlap[0], num_nuclei, l_grid.mpi_comm);
        }

        return is_reset_at_least_one;
    }


private:
    void mSetSubspaceBlockHR(SubgridBlock& target_block, const GridI3D& nucl_int_pos, 
        const GridRange& subgrid, const GridRange& hr_grid) {


        const int irx = nucl_int_pos.x;
        const int iry = nucl_int_pos.y;
        const int irz = nucl_int_pos.z;


        GridRange subgrid_actual = subgrid;
        subgrid_actual.begin_x += irx;
        subgrid_actual.end_x += irx;
        subgrid_actual.begin_y += iry;
        subgrid_actual.end_y += iry;
        subgrid_actual.begin_z += irz;
        subgrid_actual.end_z += irz;


        const int total_block_size = GetOverlapPeriodic(hr_grid, subgrid_actual, m_grid.size_x * m_HR_ratio_x, m_grid.size_y * m_HR_ratio_y, m_grid.size_z * m_HR_ratio_z, target_block.range_blocks, target_block.shifted_grids);

        target_block.grid_sizes = total_block_size;

    }


    /*
    * FFT後の波数空間データとして保持してある密度(rho:内殻電子密度やPCC)を、
    * 粒子の実際の座標にシフトしてから
    * 張り付ける前段階の実空間データに変換する.
    * 実空間データはDDMを加味してブロック分割(SubgridBlock)された実空間小領域の集合体である
    */
    void mCreateDensityOnBlock_SvF(LocalKspace2& local_kspace, const SubgridBlock& block,
        double** p_rho, const double dx, const double dy, const double dz,
        const Nucleus nucleus, const GridI3D int_pos, const GridRange& l_grid, double** p_rho_diff = nullptr) {

        if (block.grid_sizes == 0) {//DDMの担当領域と粒子のオーバーラップがない場合はスキップ.
            *p_rho = nullptr;
            return;
        }



        //DDM region//
        const size_t total_block_size = block.grid_sizes;
        const auto& range_blocks = block.range_blocks;
        const auto& shifted_grid = block.shifted_grids;
       


        //Real space projector Shifted via FFT//
        const int irx = int_pos.x;
        const int iry = int_pos.y;
        const int irz = int_pos.z;
#if defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)
        const double drx = nucleus.Rx - (0.5+(double)irx) * dx;
        const double dry = nucleus.Ry - (0.5+(double)iry) * dy;
        const double drz = nucleus.Rz - (0.5+(double)irz) * dz;
#else
        const double drx = nucleus.Rx - (double)irx * dx;
        const double dry = nucleus.Ry - (double)iry * dy;
        const double drz = nucleus.Rz - (double)irz * dz;
#endif
        GridRange subgrid_actual = local_kspace.subgrid;
        subgrid_actual.begin_x += irx;
        subgrid_actual.end_x += irx;
        subgrid_actual.begin_y += iry;
        subgrid_actual.end_y += iry;
        subgrid_actual.begin_z += irz;
        subgrid_actual.end_z += irz;
        int64_t sub_size = subgrid_actual.Size3D();
        const int sub_size_x = subgrid_actual.SizeX();
        const int sub_size_y = subgrid_actual.SizeY();
        const int sub_size_z = subgrid_actual.SizeZ();




        auto*& fftw = local_kspace.fftw;
        OneComplex* shifted_rho_kspace = (OneComplex*)(fftw->GetBuffer());

        
        //std::memset(shifted_rho_kspace, 0, sizeof(OneComplex) * sub_size);

        auto* density_kspace = local_kspace.charge_kspace;
        ShiftInKspace_set(shifted_rho_kspace, drx, dry, drz, density_kspace,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);

        const double invN = 1.0 / (double)(sub_size);


        if (p_rho_diff) {
            double* shifted_dp_dx = gy::AlignedAlloc<double>(total_block_size*3);
            *p_rho_diff = &shifted_dp_dx[0];
            double* shifted_dp_dy = &shifted_dp_dx[0] + total_block_size;
            double* shifted_dp_dz = &shifted_dp_dx[0] + total_block_size*2;

            //OneComplex* shifted_k_proj_mirror = new OneComplex[sub_size];
            auto shifted_k_proj_mirror = gy::make_unique_aligned<OneComplex[]>(sub_size);


            std::memcpy(&shifted_k_proj_mirror[0], shifted_rho_kspace, sizeof(OneComplex) * sub_size);



            //x微分//
            GradientXInKspace(&shifted_rho_kspace[0], &shifted_k_proj_mirror[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


            fftw->BackwardDirect(shifted_rho_kspace);
            //auto* shifted_dp_dx = fftw->GetBuffer();

            
            ForBlocks_in_subspace(subgrid_actual, range_blocks, shifted_grid,
                [&](int i, int sub_i) {                    
                    shifted_dp_dx[i] = shifted_rho_kspace[sub_i].r * invN;
                });


            //y微分//
            GradientYInKspace(&shifted_rho_kspace[0], &shifted_k_proj_mirror[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);

            fftw->BackwardDirect(shifted_rho_kspace);
            

            ForBlocks_in_subspace(subgrid_actual, range_blocks, shifted_grid,
                [&](int i, int sub_i) {
                    shifted_dp_dy[i] = shifted_rho_kspace[sub_i].r * invN;
                });


            //z微分//
            GradientZInKspace(&shifted_rho_kspace[0], &shifted_k_proj_mirror[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


            fftw->BackwardDirect(shifted_rho_kspace);

            ForBlocks_in_subspace(subgrid_actual, range_blocks, shifted_grid,
                [&](int i, int sub_i) {                    
                    shifted_dp_dz[i] = shifted_rho_kspace[sub_i].r * invN;

                });


            //焼き戻し//
            std::memcpy(shifted_rho_kspace, &shifted_k_proj_mirror[0], sizeof(OneComplex) * sub_size);



        }


        fftw->BackwardDirect(shifted_rho_kspace);
        auto* shifted_density = fftw->GetBuffer();
        

        double* rho_vlocal = gy::AlignedAlloc<double>(total_block_size);
        *p_rho = rho_vlocal;

        ForBlocks_in_subspace(subgrid_actual, range_blocks, shifted_grid,
            [&](int i, int sub_i) {
                rho_vlocal[i] = shifted_density[sub_i].r * invN;

            });

    }

public:
    void SetChargeVlocal_HR(double* l_rho, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);


        size_t local_size = l_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
        for (size_t i = 0; i < local_size; ++i) {
            l_rho[i] = 0.0;
        }

        auto hr_grid = ScaleRange(l_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
        for (int ni = 0; ni < num_nuclei; ++ni) {
            AddSubgridByRanges(hr_grid, l_rho, m_hr_vlocal_block[ni].range_blocks, m_hr_rho_vlocal[ni]);
        }
    }


    void SetPccCharge_v2(double* l_rho, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);

        size_t local_size = l_grid.Size3D();
        for (size_t i = 0; i < local_size; ++i) {
            l_rho[i] = 0.0;
        }

        for (int ni = 0; ni < num_nuclei; ++ni) {
            if (m_pcc_charge[ni] == nullptr) continue;
            AddSubgridByRanges(l_grid, l_rho, m_pcc_block[ni].range_blocks, m_pcc_charge[ni]);
        }
    }

#ifdef SvF2_USE_PCC_HR

    void SetPccCharge_HR(double* l_rho, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);

        size_t local_size = l_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
        for (size_t i = 0; i < local_size; ++i) {
            l_rho[i] = 0.0;
        }

        auto hr_grid = ScaleRange(l_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
        for (int ni = 0; ni < num_nuclei; ++ni) {
            if (m_hr_pcc_charge[ni] == nullptr) continue;
            AddSubgridByRanges(hr_grid, l_rho, m_hr_pcc_block[ni].range_blocks, m_hr_pcc_charge[ni]);
        }
    }

#endif


    void InnerForChargeVlocal(double* inner, int stride, const double* l_V, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei, bool is_high_reso = false) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);


        //watch_pp.Restart();

        const int& num_inner = num_nuclei;

        //watch_pp.Record(0);

        auto local_buf = gy::make_unique_aligned<double[]>(num_inner * 2);
        double* l_inner = local_buf.get();
        double* sum_inner = l_inner + num_inner;
        for (int i = 0; i < num_inner; ++i) {
            l_inner[i] = 0.0;
        }

        
        GridRangeMPI hr_grid = ScaleRange(l_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
        auto cut_feild = gy::make_unique_aligned<double[]>(m_hr_max_block_size);
        const double dV = m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
        mInnerChargeVlocal(l_V, hr_grid, m_hr_vlocal_block, m_hr_rho_vlocal, nuclei, num_nuclei, l_inner, dV, cut_feild.get());
        
        

        for (int ni = 0; ni < num_nuclei; ++ni) {
            inner[ni * stride] = sum_inner[ni];
        }
        
    }

    void InnerForChargeVlocalDifferential(double* inner, int stride, const double* l_V, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei, bool is_high_reso = false) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);

#ifdef DIFF_NUCL_RHO_LOCAL

        //watch_pp.Restart();

        const int& num_inner = num_nuclei;

        //watch_pp.Record(0);
        auto local_buf = gy::make_unique_aligned<double[]>(num_inner * 3 * 2);
        double* l_inner = local_buf.get();
        double* sum_inner = l_inner + num_inner*3;
        for (int i = 0; i < num_inner*3; ++i) {
            l_inner[i] = 0.0;
        }


        GridRangeMPI hr_grid = ScaleRange(l_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
        auto cut_feild = gy::make_unique_aligned<double[]>(m_hr_max_block_size);
        const double dV = m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
        mInnerChargeVlocal3(l_V, hr_grid, m_hr_vlocal_block, m_hr_rho_vlocal_diff, nuclei, num_nuclei, l_inner, dV, cut_feild);
        


        for (int ni = 0; ni < num_nuclei; ++ni) {
            inner[ni * stride] = sum_inner[ni*3];
            inner[ni * stride+1] = sum_inner[ni * 3+1];
            inner[ni * stride+2] = sum_inner[ni * 3+2];
        }
        
#endif
    }

    void mInnerChargeVlocal(const double* l_V, const GridRangeMPI& l_grid, std::vector< SubgridBlock>& vlocal_blocks, std::vector< double*>& rho_vlocal, const Nucleus* nuclei, int num_nuclei, double* l_inner, double dV, double* cut_feild) {


        const int num_inner = num_nuclei;
        double* sum_inner = l_inner + num_inner;




        //watch_pp.Record(2);

        //slower than Iallreduce with comm 4 atoms
#define TEST_WITH_REDUCE_TO_ROOT


        int num_valid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms_local.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
#ifdef TEST_WITH_REDUCE_TO_ROOT
                l_inner[ni] = 0.0;
#endif
                continue;
            }
            const int& index = ni;

            const auto& cut_range_block_l = vlocal_blocks[ni].range_blocks;
            const auto grid_size = vlocal_blocks[ni].grid_sizes;
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
            CutSubgridByRanges(cut_range_block_l, cut_feild, l_grid, l_V);
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
            //watch_pp.Record(3);
            int index_PYlm = 0;

            const auto* rho = rho_vlocal[ni];

            double inner = 0.0;
            for (int i = 0; i < grid_size; ++i) {
                inner += rho[i] * cut_feild[i];
            }
            l_inner[index] = inner * dV;




            //watch_pp.Record(4);


#ifndef TEST_WITH_REDUCE_TO_ROOT


            MPI_Iallreduce(l_inner + info_range[ni], sum_inner + info_range[ni],
                info_range[ni + 1] - info_range[ni], MPI_DOUBLE, MPI_SUM, mycomm, &m_request4atoms[num_valid]);
            ++num_valid;

#endif

        }


#ifndef TEST_WITH_REDUCE_TO_ROOT

        MPI_Waitall(num_valid, &m_request4atoms[0], &m_status4atoms[0]);
        //watch_pp.Record(5);

#else

        MPI_Reduce(l_inner, sum_inner, num_inner, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        //watch_pp.Record(5);
#endif

    }


    void mInnerChargeVlocal3(const double* l_V, const GridRangeMPI& l_grid, std::vector< SubgridBlock>& vlocal_blocks, std::vector< double*>& rho_vlocal, const Nucleus* nuclei, int num_nuclei, double* l_inner, double dV, double* cut_feild) {


        const int num_inner = num_nuclei;
        double* sum_inner = l_inner + num_inner*3;




        //watch_pp.Record(2);

        //slower than Iallreduce with comm 4 atoms
#define TEST_WITH_REDUCE_TO_ROOT


        int num_valid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms_local.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
#ifdef TEST_WITH_REDUCE_TO_ROOT
                l_inner[ni] = 0.0;
#endif
                continue;
            }
            const int& index = ni;

            const auto& cut_range_block_l = vlocal_blocks[ni].range_blocks;
            const auto grid_size = vlocal_blocks[ni].grid_sizes;
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
            CutSubgridByRanges(cut_range_block_l, cut_feild, l_grid, l_V);
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
            //watch_pp.Record(3);
            int index_PYlm = 0;

            const auto* rho_x = rho_vlocal[ni];
            const auto* rho_y = rho_vlocal[ni] + grid_size;
            const auto* rho_z = rho_vlocal[ni] + grid_size*2;

            double inner[3] = { 0.0,0.0, 0.0};
            for (int i = 0; i < grid_size; ++i) {
                inner[0] += rho_x[i] * cut_feild[i];
                inner[1] += rho_y[i] * cut_feild[i];
                inner[2] += rho_z[i] * cut_feild[i];
            }
            l_inner[index * 3 + 0] = inner[0] * dV;
            l_inner[index * 3 + 1] = inner[1] * dV;
            l_inner[index * 3 + 2] = inner[2] * dV;




            //watch_pp.Record(4);


#ifndef TEST_WITH_REDUCE_TO_ROOT


            MPI_Iallreduce(l_inner + info_range[ni], sum_inner + info_range[ni],
                info_range[ni + 1] - info_range[ni], MPI_DOUBLE, MPI_SUM, mycomm, &m_request4atoms[num_valid]);
            ++num_valid;

#endif

        }


#ifndef TEST_WITH_REDUCE_TO_ROOT

        MPI_Waitall(num_valid, &m_request4atoms[0], &m_status4atoms[0]);
        //watch_pp.Record(5);

#else

        MPI_Reduce(l_inner, sum_inner, num_inner*3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        //watch_pp.Record(5);
#endif

    }


    void InnerForChargePcc(double* inner, int stride, const double* l_V, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);


        //watch_pp.Restart();

        const int& num_inner = num_nuclei;

        //watch_pp.Record(0);
        auto local_buf = gy::make_unique_aligned<double[]>(num_inner * 2);
        double* l_inner = local_buf.get();
        double* sum_inner = l_inner + num_inner;
        for (int i = 0; i < num_inner; ++i) {
            l_inner[i] = 0.0;
        }


        auto cut_feild = gy::make_unique_aligned<double[]>(m_max_block_size);
        const double dV = m_dx * m_dy * m_dz;
        mInnerChargeVlocal(l_V, l_grid, m_pcc_block, m_pcc_charge, nuclei, num_nuclei, l_inner, dV, cut_feild.get());
        


        for (int ni = 0; ni < num_nuclei; ++ni) {
            inner[ni * stride] = sum_inner[ni];
        }
        
    }

#ifdef SvF2_USE_PCC_HR
    void InnerForChargePcc_HR(double* inner, int stride, const double* l_V, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei) {
        const int proc_id = GetProcessID(l_grid.mpi_comm);


        //watch_pp.Restart();

        const int& num_inner = num_nuclei;

        //watch_pp.Record(0);

        auto local_buf = gy::make_unique_aligned<double[]>(num_inner * 2);
        double* l_inner = local_buf.get();
        double* sum_inner = l_inner + num_inner;
        for (int i = 0; i < num_inner; ++i) {
            l_inner[i] = 0.0;
        }


        GridRangeMPI hr_grid = ScaleRange(l_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
        auto cut_feild = gy::make_unique_aligned<double[]>(m_hr_max_block_size);
        const double dV = m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
        mInnerChargeVlocal(l_V, hr_grid, m_hr_pcc_block, m_hr_pcc_charge, nuclei, num_nuclei, l_inner, dV, cut_feild.get());
        


        for (int ni = 0; ni < num_nuclei; ++ni) {
            inner[ni * stride] = sum_inner[ni];
        }
    }
#endif
};

#endif
