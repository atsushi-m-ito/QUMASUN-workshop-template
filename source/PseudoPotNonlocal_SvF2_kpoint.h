#ifdef USE_MPI
#pragma once



#include <mpi.h>
#include "mpi_helper.h"
#include "PseudoPotLocal_SvF2.h"

#include <cstring>
#include "soacomplex.h"
#include "poisson_fft.h"
#include "SetBlockYlm.h"
#include "inverse_m.h"
#include "w_dsyevd.h"
#include "w_dgemm.h"
#include "StopWatch.h"
#include "fftw_executor.h"
#include "DDMOverlapChecker.h"
#include "GridGradient8th.h"
#include "shift_to_half_point.h"
#include "work_arena.h"


//[必須]これを定義しないときはSvFでのnonlocal貼り付け時にcircle状にする//
//#define SvF_PASTE_RECTANGLE
////20241015defaultでONとしたのでコード削除. 上記のコメントを記録として残す//


//v19.16(stable:20240924)
//(解決)forceの計算にgrad(p)を使い.1e-4の桁の誤差が乗る-->local領域のグリッドを適切にとることで解決(SvF2_SUBGRID_ODD_HALF)
//nonlocalのSymplecticで精度を出すために必要(特に衝突時)
//nonlocalのsymplecticのLiouvillianに<p|dp/dx>の項があるため、それを辻褄を合わせる必要があるためと考えられる
#define FORCE_DIFF_PROJ

#ifdef FORCE_DIFF_PROJ
//FORCE_DIFF_PROJも定義されているときに有効
//projectorのgradientを3次元FFTではなく1次元FFTで行うことで計算量削減
//ただし、nonlocalのsubgridが立方体(x,y,zのグリッド幅が等しい)を仮定してfftw_1dは共通にしている//
//遅くなったのでpending
//#define FORCE_DIFF_PROJ_2     
//20241101コードは削除済み上記のコメントを記録として残す//

//FORCE_DIFF_PROJも定義されているときに有効
//projectorのgradientを差分法で実行する. それでも境界の微分値は0にできない(根本的な奇関数の問題)
//SvFもgrad(p)毎に行うことになるため、計算量は4倍になる//
//#define FORCE_DIFF_PROJ_3     //コードは削除済み
//20241101コードは削除済み上記のコメントを記録として残す/
//グリッドを1つ余分にとってsmoothingする処理(Symplectic保存性が悪いので却下)//
//#define FORCE_DIFF_PROJ_3_V2   //コードは削除済み
//#define SMOOTH_SUBGRID2
//20241101コードは削除済み上記のコメントを記録として残す//
// 
//projectorが奇関数の場合にsubgridの端で微分値がnon-zeroになる問題について、//
//奇関数の場合にはその方向のグリッドを倍にとって、ミラーリングすることで偶関数にさせるテクニック//
//これにより元々の境界だったグリッドでの微分値が0になる//
//ただしコストは倍になる//
//#define ODD_PROJ_MIRROR     //コードは削除済み
//#define FULL_MIRROR     //コードは削除済み
//20241101コードは削除済み上記のコメントを記録として残す//

//<p|do/dx>matrixが反対称からずれる問題を補正
//#define CORRECTION_ANTI_SYMMETRIC_P_GRAD_P  1     // 1 or 2


//projectorが奇関数の場合にFFTによる微分値を実際に行って補正する
//実際に微分して、端の微分値を0にして、積分して戻す//
//原子核がセル中心にいるときにグリッドの端の微分値が非対称になることに起因した精度問題は解決
//v20.9(stable:20241024)この方法で解決できた. (MIRRORでは解決できなかった)
#define ODD_PROJ_CORRECTION

//再び擬ポテンシャルのsubgridのedgeの1グリッド分をsmoothに挑戦
//右端に1グリッド追加して、左端のグリッドを右端にコピー
//ただし1以下の重みを乗じて、左右のグリッドで重みの和が1になるようにする//
//これも上手く行かなかった。
//subgridの切り替え時に境界値が0になるようにすでに作ってあるので
//スムージングのためにグリッドを追加すると、切り替えポイントが半格子ずれて逆に精度が落ちる//
//削除済み//
//#define SMOOTH_SUBGRID  3  //should not use

#endif

//v19.16(stable:20240924) -> FFTを使う//
//Symplecticのprojectorの微分に敢えてFFTを使わない(解像度が足りていないのでnoisyになりすぎる)
////#undef DIFF_PROJ_NO_FFT
////20241015defaultでOFFとしたのでコード削除. 上記のコメントを記録として残す//

//以下の二つを有効にしてProjectionPP_bundleとその中のDGEMMを利用するのが最速//
//#define NONLOCAL_DGEMM1 (default化して削除)


//Nonlocal projectorのSvF処理を, comm4atomsで共有しているprocess間で負荷分散する//
//1,2はあまり速くない//
//3なら速い//
//4はもっと速い//
//5,6は4と同等//
//5を残して7を追加1-4,6は削除//
#define NONLOCAL_SvF_PARALLEL   5

//nonlocal項にもSvFを利用した補間を適用する
// 1: 行列<p|p>の計算はシフトして加算した後に1度だけ行う
// 2: Symplecticようの行列<p|p>の計算をミラーごとに個別に行う(精度は悪い)
// 3: FFTを使わず多項式内挿による係数倍をかけて平均化(精度は悪い.nonlocalエネルギーもずれる)
// 4: (推奨) 1の修正の試行版. 端点での微分値の修正をsinカーブで行う.
// note: 20250516時点で1はTDDFTで精度が低い. 4は精度が高い.
// 5: 奇関数の場合にxで割ってからFFTし、利用時は逆FFTしてからxを乗じる.これによってFFTは偶関数となり境界で値と微分値の両方をゼロにできる
// 6: 4をベースに、保持するprojectorが偶関数になるように、奇関数方向は微分してしまう。使うときは積分して射影する//
// 　6だとp-dpdx行列の対称性が崩れる
#define SvF2_USE_INTERPOLATION_NL  4
constexpr int SvF2_INTERPOLATION_NL_MAGNITUDE = 2;
//次のフラグはSvF2_USE_INTERPOLATION_NL==3の場合だけ有効
//#define SvF_INTERPOLATION_NL_3RD_ORDER   

//nonlocal projectorが低解像度で真のprojectorから離れた時にでも、2次形式を保存するように補正する
//すでにこの補正をしない元もとの実装の方がRitz射影に対して二次形式を保存できてる。
//この補正をすると、むしろ保存しなくなる。
//#define NONLOCAL_RITZ_CORRECTION

//projectorを粗いグリッド上で規格化する(ADPACKではNG)//
//#define NONLOCAL_NORMALIZE


//note:
// M=(P^t P), where projector matrix P=[p_1, p_2, ...] を考えた時、
// Mの最大固有値e_maxと最小固有値e_minの比kappa(M)=e_max/e_min ~ 1なら精度が良い。
// 一方でkappa(M) >> 1なら精度が悪い。
// これは解像度の十分さを表すメジャーにもなるし、補間方法の性能を表すメジャーでもある.


//nonlocal小領域のOrbitalの切り取りを、全ての原子について一気に行う場合
//GPUでカーネルを纏められるので速くなる(ただしメモリアロケートを毎回しないことが重要)
#ifdef GY_WITH_CUDA_OR_HIP
#define PP_NONLOCAL_CUT_ALL
#endif


class PseudoPotNonlocal_SvF2_kpoint
    : public PseudoPotLocal_SvF2 
{
private:
#if  (SvF2_USE_INTERPOLATION_NL == 1)
    static constexpr int SvF2_MARGIN_NL = 2;
#else
    static constexpr int SvF2_MARGIN_NL = 2;
#endif
    //using NonlocalBlocks = std::vector< SubgridBlock>;

    //擬ポテンシャルをk空間にFFTしたデータ//
    //初回に用意して保持し続けるデータ, 原子の位置が変わっても書き換えない//////////////////////
    //原子ごとではなく元素ごとに保持するデータ//
    struct NonlocalKspace2 {
        std::vector<OneComplex*> k_projector_Ylm;  //projector(R * Y)の数だけ要素がある//
        int num_proj_RY = 0;
        double cutoff_l[4]{ 0.0 };
        GridRange subgrid;
        FFTW_Executor* fftw = nullptr;

        gy::unique_aligned_ptr<double[]> projector_energy_RY = nullptr;
        std::vector<int> odd_flag;

    };
    std::map<int, NonlocalKspace2> m_nonlocal_kspace;   //元素の数だけ要素がある//

    //Nonlocal項のSymplectic Integratorに用いるデータ//
    //原子ごとではなく元素ごとに保持するデータ//
    struct NonlocalSymplectic {
        int num_proj_RY;
        gy::unique_aligned_ptr<double[]> matrix_pp = nullptr;  //<p_k|p_m>, projector(R * Y)^2の数だけ要素がある//
        double* matrix_p_dp_dx;  //<p_k|d(p_m)/dx>, projector(R * Y)^2の数だけ要素がある//
        double* matrix_p_dp_dy;  //<p_k|d(p_m)/dy>, projector(R * Y)^2の数だけ要素がある//
        double* matrix_p_dp_dz;  //<p_k|d(p_m)/dz>, projector(R * Y)^2の数だけ要素がある//
        
        
        // Y is (Np x Np) matrix to diagonalize <p_k|p_m>, where Np is number of nonlocal projector//
        // and, Y={y_1, y_2,...,y_Np} is eigen vectors as follows:
        //   Y^{-1} <p|p> Y =  Y^t <p|p> Y = D,
        // and D(=diagD) is diagonal matrix and its diagonal elements are eigen values of <p|p>
        gy::unique_aligned_ptr<double[]> Y = nullptr;
        gy::unique_aligned_ptr<double[]> diagD = nullptr;

        // Here, nonlocal enegry is given by
        //   E_{NL} = sum_{k,m} <psi|p_k> B_{km} <p_m|psi>
        // and projector p is orthonormalized by
        //   |p_k> -> |p'_k> = |p_k> Y D^{-1/2}
        // To maintain the operator
        //   V_{NL} = |p_k>B_{km}<p_m| = |p'_k>Bbar_{km}<p'_m|
        // Bbar is defined as//
        //   Bbar = (Y D^{1/2})^t B (Y D^{1/2})

        // moreover, X is matrix to diagonalize Bbar, and is eigen vectors X = {x_1,x_2, ...,x_Np}
        //   X^{-1} Bbar X = X^t Bbar X = Lambda
        // and Lambda(=diagLambda) is eigen values of Bbar
        gy::unique_aligned_ptr<double[]> X = nullptr;
        gy::unique_aligned_ptr<double[]> diagLambda = nullptr;

        // Consequently, the final form of V_{NL} is 
        //    V_{NL} = |p_k> Y D^{-1/2} X Lambda X^{t} D^{-1/2} Y^{t} <p_m|

        gy::unique_aligned_ptr<double[]> YDmhX = nullptr; // Y D^{-1/2} X
        
        gy::unique_aligned_ptr<double[]> YDmhX_expL_sub_one_XtDmhYt_re = nullptr; // Y D^{-1/2} X (exp(-iLambda/hbar) (Y D^{-1/2} X)^t
        gy::unique_aligned_ptr<double[]> YDmhX_expL_sub_one_XtDmhYt_im = nullptr; // Y D^{-1/2} X (exp(-iLambda/hbar) (Y D^{-1/2} X)^t
    };
    std::map<int, NonlocalSymplectic> m_nonlocal_symplectic;   //元素の数だけ要素がある//


    double m_dt = 0.0; //time step for symplectic integrator//

    static constexpr int MAX_L_SYSTEM = 4;
    size_t m_max_grid_size = 0;
    


    //原子の位置が変わるたびに書き換えるデータ///////////////////////
    //原子ごとに保持する//
    struct PP_Proj_Ylm_each_atom {
        int num_proj_RY;


        double* projector_RY = nullptr; //block化されたprojector(Ylmも作用済み), nonlocal項の数だけ存在, Lの種類ごとにまとまる//
#ifdef FORCE_DIFF_PROJ
        double* d_projector_RY_dx = nullptr; //block化されたprojector(Ylmも作用済み)をk空間で微分して逆FFTしたもの, nonlocal項の数だけ存在, Lの種類ごとにまとまる//
        double* d_projector_RY_dy = nullptr; //block化されたprojector(Ylmも作用済み)をk空間で微分して逆FFTしたもの, nonlocal項の数だけ存在, Lの種類ごとにまとまる//
        double* d_projector_RY_dz = nullptr; //block化されたprojector(Ylmも作用済み)をk空間で微分して逆FFTしたもの, nonlocal項の数だけ存在, Lの種類ごとにまとまる//
#endif

        std::vector<double*> bloch_expikx; //nonlocal項の作用で用いるBlock定理のexp(ikx)ファクターのblock化された場. Lの種類ごとに生成/


        ~PP_Proj_Ylm_each_atom() {
            gy::AlignedFree(projector_RY);
            for (auto&& p : bloch_expikx) {
                gy::AlignedFree(p);
            }
        }
    };

    std::vector<SubgridBlock> m_nonlocal_blocks; //要素数は原子核の数//
    std::vector<PP_Proj_Ylm_each_atom>  m_nonlocal_projectors;//要素数は原子核の数//
    
    
    CommForAtoms m_comm4atoms;

    std::vector<int> m_sorted_proc_ids;
    std::vector<int> m_sorted_count_list;

    int m_num_all_nonlocal = 0;
    std::vector<int> m_num_list_nonlocal;


    std::vector<bool> m_is_gamma_point_list;



    template<bool IS_WATCH>
    void Watch_Barrier(MPI_Comm comm) {
        if constexpr (IS_WATCH) {
            MPI_Barrier(comm);
        }
    }
#ifdef TIME_PP_MPI
    static constexpr bool IS_MEASURE_TIME = true;
    StopWatch<30, IS_MEASURE_TIME> watch_pp;
    int64_t m_bench_flop_nonlocal = 0;
    int64_t m_bench_flop_nonlocal_high = 0;
    int64_t m_bench_flop_nonlocal_f = 0;
    int64_t m_bench_flop_nonlocal_f_high = 0;
    static constexpr int64_t LIMIT_64 = (0x1LL << 60);

#else
    StopWatch<30, false> watch_pp;
#endif


public:
    ~PseudoPotNonlocal_SvF2_kpoint() {

        for (auto&& a : m_nonlocal_kspace) {

            gy::AlignedFree(a.second.k_projector_Ylm[0]);

            delete a.second.fftw;

        }

#ifdef TIME_PP_MPI
        if (is_root) {
            mPrintTime();
        }
        mPrintTimeMax();
#endif

    }

private:
    /*
    * Use it to determine the integer grid point from position(floating point)
    *
    */
    int GridPosForSvF_NL(double x, double dx) {
#if defined(SMOOTH_SUBGRID) || defined(SvF2_SUBGRID_EVEN2)||defined(SvF2_SUBGRID_ODD2) || defined(SvF2_SUBGRID_TEST_GAUSSIAN)||defined(SvF2_SUBGRID_EVEN_HALF)
        return (int)floor(x / dx);
#elif defined(SvF2_SUBGRID_ODD)||defined(SvF2_SUBGRID_ODD_HALF)
        return (int)floor(x / dx + 0.5);
#elif 1
        return (int)floor(x / dx + 0.5);
#else
        return (int)ceil(x / dx);
#endif
    }

    /*
    * Nonlocal項をFFT経由でシフトするための初期化
    * 原点中心でnonlocal projectorを貼り付け、FFTでk空間に移し、それを保存しておく.
    * 粒子の位置がUpdateされる度に、保存したk空間の情報をシフトしてからIFFTで実空間戻し、初付ける.
    * ただし、貼り付けるのは、領域分割で自分の担当領域のみ.
    * ここで、local項はMPIによる粒子分割であり、少なくともIFFT後の貼り付けは領域分割ではない.
    * それに対し、nonlocal項は全てのプロセスがk空間の情報を持っておき、
    * IFFT後の実空間への貼り付けでは領域分割で担当した部分だけを貼り付ける.
    */

public:


    void InitializeNonlocal(const Nucleus* nuclei, int num_nuclei, MPI_Comm& mpi_comm) {
        const int num_procs = GetNumProcess(mpi_comm);
        const int proc_id = GetProcessID(mpi_comm);


        //元素の種類ごとのnonlocal-projectorを原点中心で生成(使うときはk空間でshiftして利用)//
        //かつ、ここでk空間にFFTまでしておく
        for (int ni = 0; ni < num_nuclei; ++ni) {
            const int Z = nuclei[ni].Z;

            auto it = m_nonlocal_kspace.find(Z);
            if (it == m_nonlocal_kspace.end()) {
                //nonlocalの新規生成//

                //保存(利用時はここからShift and 逆FFTして使う)//
                auto it2 = m_nonlocal_kspace.emplace(Z, NonlocalKspace2());
                NonlocalKspace2& nonlocal_kspace = it2.first->second;
                //printf("TEST: %s: %d\n", __FILE__, __LINE__);
                mInitializeNonlocalProjector(Z, &nonlocal_kspace);
#ifdef NONLOCAL_RITZ_CORRECTION
                mRitzCorrectionProjector(Z, &nonlocal_kspace);
#endif
                //printf("TEST: %s: %d\n", __FILE__, __LINE__);
#if SvF2_USE_INTERPOLATION_NL==2

                //こっちは精度が出ない
                {
                    auto it = m_nonlocal_symplectic.find(Z);
                    auto& nonlocal_symplectic = it->second;
                    auto* p = &nonlocal_symplectic.matrix_pp[0];
                    const int i_end = num_proj_RY * num_proj_RY * 4;
                    for (int i = 0; i < i_end; ++i) {
                        p[i] *= weight;
                    }
                }
#else
                mCreateSymplecticProjectorMatrix_PPmatrix(Z, &nonlocal_kspace);
                //printf("TEST: %s: %d\n", __FILE__, __LINE__);
#endif
                mCreateSymplecticProjectorMatrix_YDXmatrix(Z);
                //printf("TEST: %s: %d\n", __FILE__, __LINE__);
            }

        }

        //離散化位置の記録消去//
        //これによりsubgridも再計算//
        //解像度変更時にも必要//
        m_nucl_int_pos.resize(0);
        //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    }

    /*
    * 解像度(m_dx, gridsize)が変更になったときはこちらを呼ぶ
    * InitializeNonlocal()との違いはm_nonlocal_kspaceのコンテナを再利用するかどうかだけ
    * コンテナの中身は消して再構築するので、はっきり言って違いは殆どない
    */
    void ResetNonlocal() {
        
        //元素の種類ごとのnonlocal-projectorを原点中心で生成(使うときはk空間でshiftして利用)//
        //かつ、ここでk空間にFFTまでしておく
        for (auto&& a : m_nonlocal_kspace) {
            const int Z = a.first;
            auto& nonlocal_kspace = a.second;

            //以前のバッファの削除//
            gy::AlignedFree(nonlocal_kspace.k_projector_Ylm[0]);
            nonlocal_kspace.k_projector_Ylm.clear();
            delete nonlocal_kspace.fftw;

            //データの再生成//
            mInitializeNonlocalProjector(Z, &nonlocal_kspace);
#ifdef NONLOCAL_RITZ_CORRECTION
            mRitzCorrectionProjector(Z, &nonlocal_kspace);
#endif

#if SvF2_USE_INTERPOLATION_NL==2

                //こっちは精度が出ない
                {
                    auto it = m_nonlocal_symplectic.find(Z);
                    auto& nonlocal_symplectic = it->second;
                    auto* p = &nonlocal_symplectic.matrix_pp[0];
                    const int i_end = num_proj_RY * num_proj_RY * 4;
                    for (int i = 0; i < i_end; ++i) {
                        p[i] *= weight;
                    }
                }
#else
            mCreateSymplecticProjectorMatrix_PPmatrix(Z, &nonlocal_kspace/*.k_projector_Ylm[0]*/);
#endif
            mCreateSymplecticProjectorMatrix_YDXmatrix(Z);

    
        }
        
        //離散化位置の記録消去//
        //これによりsubgridも再計算//
        //解像度変更時にも必要//
        m_nucl_int_pos.resize(0);
    }

#if SvF2_USE_INTERPOLATION_NL==3

    void mInitializeNonlocalProjector(int Z, NonlocalKspace2* nonlocal_kspace) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const int max_l = pp->MaxProjectorL();
        const double cutoff_max = pp->MaxCutoffLength();
        const GridRange subgrid = SubgridFromCutoff(cutoff_max, SvF2_MARGIN_NL);
        const int64_t sub_size = subgrid.Size3D();
        nonlocal_kspace->subgrid = subgrid;

        //csize倍の高解像度なローカルグリッドを用意して、奇数グリッドの情報を偶数グリッドに係数倍を変えて足す
        //この時の係数は、粗い側の場(波動関数)がグリッドの半格子点をどのように補間するかにより、
        //その補間係数を表す行列の逆行列に相当する//
        constexpr int csize = SvF2_INTERPOLATION_NL_MAGNITUDE;     //3の場合もある//
        const auto hr_subgrid = ScaleRange(subgrid, csize, csize, csize);
        const double hr_dx = m_dx / (double)csize;
        const double hr_dy = m_dy / (double)csize;
        const double hr_dz = m_dz / (double)csize;
        const int64_t hr_size = hr_subgrid.Size3D();


#ifdef SMOOTH_SUBGRID
        nonlocal_kspace.subgrid.end_x += 1;
        nonlocal_kspace.subgrid.end_y += 1;
        nonlocal_kspace.subgrid.end_z += 1;
#endif                


        //(実)球面調和関数用のバッファを確保//                
        double* pYlm[4]{ nullptr,nullptr,nullptr,nullptr };
        for (int l = 0; l <= max_l; ++l) {
            const double cutoff_l = pp->cutoff_r[l];
            nonlocal_kspace->cutoff_l[l] = cutoff_l;
            pYlm[l] = new double[hr_size * (2 * l + 1)];
        }

        //FFTWの準備//
        {
            auto* fftw = new FFTW_Executor;
            fftw->Initialize(subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), FFTW_ESTIMATE);
            nonlocal_kspace->fftw = fftw;

        }


        //動径関数のprojectorを原点中心でデカルトグリッドに焼き直す//
        const int num_projectors_R = pp->num_radial_projectors;
        int num_proj_RY = 0;
        for (int k = 0; k < num_projectors_R; ++k) {
            const int l = pp->projector_quantum_l[k];
            num_proj_RY += 2 * l + 1;
        }
        nonlocal_kspace->num_proj_RY = num_proj_RY;

        nonlocal_kspace->projector_energy_RY.resize(num_proj_RY);


        double* proj_R = new double[hr_size];
        double* hr_RY = new double[hr_size*2];
        OneComplex* proj_RY_all = gy::AlignedAlloc<OneComplex>(num_proj_RY * sub_size);
        OneComplex* tmp_RY_all = proj_RY_all;// new OneComplex[num_proj_RY * sub_size];

        {
            memset(proj_RY_all, 0, sizeof(OneComplex) * num_proj_RY * sub_size);
            auto* p = proj_RY_all;
            for (int k = 0; k < num_projectors_R; ++k) {
                const int l = pp->projector_quantum_l[k];
                for (int m = -l; m <= l; ++m) {
                    nonlocal_kspace->k_projector_Ylm.push_back(p);
                    p += sub_size;
                }
            }
        }


        const int cx = 1;
        const int cy = cx;
        const int cz = cx;
        constexpr double half = 0.5;
        //const double weight = 1.0 / (double)(csize * csize * csize);
        const double weight = 1.0 / (double)(csize);
        auto fold_pos = [&half, &csize](int cx) {
            return half - (double)(cx * 2 > csize ? cx - csize : cx) / (double)csize;
            };
        

        const double center_x = fold_pos(cx) * (double)csize; //高解像度グリッドでは格子点が存在する場合もある
        const double center_y = fold_pos(cy) * (double)csize;
        const double center_z = fold_pos(cz) * (double)csize;


        //(実)球面調和関数を原点中心で生成(元素に共通)//
        for (int l = 0; l <= max_l; ++l) {
            for (int m = -l; m <= l; ++m) {
                auto* pY = pYlm[l] + hr_size * (l + m);
                SetYlmAnyRangeFromOrigin(l, m, pY, hr_subgrid, hr_dx, hr_dy, hr_dz, center_x, center_y, center_z);
            }
        }

        //動径関数を生成//
        int id_proj_RY = 0;
        for (int k = 0; k < num_projectors_R; ++k) {
            const int l = pp->projector_quantum_l[k];
            const double cutoff_l = pp->cutoff_r[l];
            const double rr_max = cutoff_l * cutoff_l;



            ForXYZ(hr_subgrid,
                [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                    const double x = ((double)ix - center_x) * hr_dx;
                    const double y = ((double)iy - center_y) * hr_dy;
                    const double z = ((double)iz - center_z) * hr_dz;

                    const double rr = x * x + y * y + z * z;
                    if (rr < rr_min) {
                        const double proj_spin_orbit_up = pp->projector[k * 2][0];
                        const double proj_spin_orbit_dn = pp->projector[k * 2 + 1][0];
                        const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;
                        proj_R[i] = p;
                    } else if (rr > rr_max) {
                        proj_R[i] = 0.0;
                    } else {
                        //NOTE: if it is not relative DFT with spin-orbit interaction, projectors (j+1/2) and (j-1/2) should be averaged.
                        const double proj_spin_orbit_up = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        const double proj_spin_orbit_dn = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2 + 1], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                        const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;

                        proj_R[i] = p;

                    }
                });


            const double projector_energy_nl = pp->projector_energy_up[k];

            const int sub_size_x = subgrid.SizeX();
            const int sub_size_y = subgrid.SizeY();
            const int sub_size_z = subgrid.SizeZ();
            const int hr_size_x = hr_subgrid.SizeX();
            const int hr_size_y = hr_subgrid.SizeY();
            const int hr_size_z = hr_subgrid.SizeZ();
            //原点中心のprojector(proj_R)と原点中心Ylmを合成する//
            //FFTでk-spaceへの変換も行う//

            for (int m = -l; m <= l; ++m) {
                const auto* pY = pYlm[l] + hr_size * (l + m);
                auto* tmp_RY = tmp_RY_all + sub_size * id_proj_RY;


                ForXYZ(hr_subgrid,
                    [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                        hr_RY[i] = proj_R[i] * pY[i];
                    });
            

                if (csize == 2) {
#ifdef SvF_INTERPOLATION_NL_3RD_ORDER
                    double* hr_RY_2 = hr_RY + hr_size;
                    const int hr_margin = SvF2_MARGIN_NL * csize;

                    //z方向の畳み込み(補間係数の逆行列の作用)
                    memset(hr_RY_2, 0, sizeof(double) * hr_size);
                    const int diz = hr_size_x * hr_size_y;
                    for (int iz = hr_margin; iz < hr_size_z - hr_margin; iz += 2) {
                        for (int iy = hr_margin; iy < hr_size_y - hr_margin; ++iy) {
                            for (int ix = hr_margin; ix < hr_size_x - hr_margin; ++ix) {
                                const int i = ix + hr_size_x * (iy + hr_size_y * iz);
                                hr_RY_2[i] = (-hr_RY[i - 2*diz] + 9.0*(hr_RY[i - diz] + hr_RY[i + diz])- hr_RY[i +2* diz]) /16.0;
                            }
                        }
                    }
                    for (int i = 0; i < hr_size; ++i) {
                        hr_RY[i] += hr_RY_2[i];
                    }
/*
                    //y方向の畳み込み(補間係数の逆行列の作用)
                    memset(hr_RY_2, 0, sizeof(double)* hr_size);
                    const int diy = hr_size_x;
                    for (int iz = hr_margin; iz < hr_size_z - hr_margin; iz += 2) {
                        for (int iy = hr_margin; iy < hr_size_y - hr_margin; iy += 2) {
                            for (int ix = hr_margin; ix < hr_size_x - hr_margin; ++ix) {
                                const int i = ix + hr_size_x * (iy + hr_size_y * iz);
                                hr_RY_2[i] = (-hr_RY[i - 2 * diy] + 9.0 * (hr_RY[i - diy] + hr_RY[i + diy]) - hr_RY[i + 2 * diy]) / 16.0;
                            }
                        }
                    }
                    for (int i = 0; i < hr_size; ++i) {
                        hr_RY[i] += hr_RY_2[i];
                    }

                    //x方向の畳み込み(補間係数の逆行列の作用)        
                    memset(hr_RY_2, 0, sizeof(double) * hr_size);
                    for (int iz = hr_margin; iz < hr_size_z - hr_margin; iz += 2) {
                        for (int iy = hr_margin; iy < hr_size_y - hr_margin; iy += 2) {
                            for (int ix = hr_margin; ix < hr_size_x - hr_margin; ix += 2) {
                                const int i = ix + hr_size_x * (iy + hr_size_y * iz);
                                hr_RY_2[i] = (-hr_RY[i - 2] + 9.0 * (hr_RY[i - 1] + hr_RY[i + 1]) - hr_RY[i + 2]) / 16.0;
                            }
                        }
                    }
                    for (int i = 0; i < hr_size; ++i) {
                        hr_RY[i] += hr_RY_2[i];
                    }
*/


#else  //1st order (linear)
                    const int hr_margin = SvF2_MARGIN_NL * csize;
                    //z方向の畳み込み(補間係数の逆行列の作用)
                    const int diz = hr_size_x * hr_size_y;
                    for (int iz = hr_margin; iz < hr_size_z - hr_margin; iz += 2) {
                        for (int iy = hr_margin; iy < hr_size_y - hr_margin; ++iy) {
                            for (int ix = hr_margin; ix < hr_size_x - hr_margin; ++ix) {
                                const int i = ix + hr_size_x * (iy + hr_size_y * iz);
                                if (iz == 0) {
                                    hr_RY[i] += hr_RY[i + diz] * 0.5;
                                } else {
                                    hr_RY[i] += (hr_RY[i - diz] + hr_RY[i + diz]) * 0.5;
                                }
                            }
                        }
                    }

                    //y方向の畳み込み(補間係数の逆行列の作用)
                    const int diy = hr_size_x;
                    for (int iz = hr_margin; iz < hr_size_z - hr_margin; iz += 2) {
                        for (int iy = hr_margin; iy < hr_size_y - hr_margin; iy += 2) {
                            for (int ix = hr_margin; ix < hr_size_x - hr_margin; ++ix) {
                                const int i = ix + hr_size_x * (iy + hr_size_y * iz);
                                if (iy == 0) {
                                    hr_RY[i] += hr_RY[i + diy] * 0.5;
                                } else {
                                    hr_RY[i] += (hr_RY[i - diy] + hr_RY[i + diy]) * 0.5;
                                }
                            }
                        }
                    }

                    //x方向の畳み込み(補間係数の逆行列の作用)                
                    for (int iz = hr_margin; iz < hr_size_z - hr_margin; iz += 2) {
                        for (int iy = hr_margin; iy < hr_size_y - hr_margin; iy += 2) {
                            for (int ix = hr_margin; ix < hr_size_x - hr_margin; ix += 2) {
                                const int i = ix + hr_size_x * (iy + hr_size_y * iz);
                                if (iy == 0) {
                                    hr_RY[i] += hr_RY[i + 1] * 0.5;
                                } else {
                                    hr_RY[i] += (hr_RY[i - 1] + hr_RY[i + 1]) * 0.5;
                                }
                            }
                        }
                    }
#endif
                }

                //const double weight = 1.0 / (double)(csize * csize * csize);
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    for (int iy = 0; iy < sub_size_y; ++iy) {
                        for (int ix = 0; ix < sub_size_x; ++ix) {
                            const int i = ix + sub_size_x * (iy + sub_size_y * iz);
                            const int ih = csize * (ix + sub_size_x * (csize * (iy + sub_size_y * (csize * iz))));

                            tmp_RY[i].r = hr_RY[ih] * weight;
                            tmp_RY[i].i = 0.0;
                        }
                    }
                }


                auto& fftw = nonlocal_kspace->fftw;
                fftw->ForwardDirect((fftw_complex*)tmp_RY);


                //x,y,zの各方向ごとに独立にシフトし、シフトによる境界点での補正も行う//
                const int sub_size_x = subgrid.SizeX();
                const int sub_size_y = subgrid.SizeY();
                const int sub_size_z = subgrid.SizeZ();

                ShiftToHalfPoint(&tmp_RY[0], cx, cy, cz, csize, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z);



#ifdef ODD_PROJ_CORRECTION
                //微分値がsubgridの端で0になるように補正する//

                const int proj_odd_flag = CheckYlmOddForDirection(l, m);
                if (proj_odd_flag) {

                    int sub_size_x = subgrid.SizeX();
                    int sub_size_y = subgrid.SizeY();
                    int sub_size_z = subgrid.SizeZ();
                    const double invN = 1.0 / (double)sub_size;

                    const double coef1_x = (2.0 * M_PI / (m_dx * (double)sub_size_x));
                    const double coef1_y = (2.0 * M_PI / (m_dy * (double)sub_size_y));
                    const double coef1_z = (2.0 * M_PI / (m_dz * (double)sub_size_z));

                    auto grad_p = gy::make_unique_aligned<OneComplex[]>(sub_size);


                    if (proj_odd_flag & ODD_YLM_X)
                        //if(false)
                    {
                        GradientXInKspace((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                        fftw->BackwardDirect((fftw_complex*)&grad_p[0]);

                        for (int iz = 0; iz < sub_size_z; ++iz) {
                            for (int iy = 0; iy < sub_size_y; ++iy) {
                                const int64_t i = sub_size_x * (iy + sub_size_y * iz);


                                //SvF2_MARGIN_NL
                                double sum = grad_p[i].r / 2.0;
                                grad_p[i].r = 0.0;

                                for (int ix = 1; ix < SvF2_MARGIN_NL; ++ix) {
                                    sum += grad_p[i + ix].r;
                                    grad_p[i + ix].r = 0.0;
                                    grad_p[i + sub_size_x - ix].r = 0.0;
                                }
                                {
                                    int ix = SvF2_MARGIN_NL;
                                    grad_p[i + ix].r += sum;
                                    grad_p[i + sub_size_x - ix].r += sum;
                                }


                            }
                        }

                        for (int i = 0; i < sub_size; ++i) {
                            grad_p[i].r *= invN;
                            grad_p[i].i *= invN;
                        }

                        fftw->ForwardExecute((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY);

                        for (int iz = 0; iz < sub_size_z; ++iz) {
                            for (int iy = 0; iy < sub_size_y; ++iy) {
                                for (int ix = 1; ix < sub_size_x; ++ix) {
                                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                                    const double kx1 = (coef1_x * (double)(ix * 2 > sub_size_x ? ix - sub_size_x : ix));
                                    double ore = tmp_RY[i].r;
                                    double oim = tmp_RY[i].i;
                                    tmp_RY[i].r = oim / kx1;
                                    tmp_RY[i].i = -ore / kx1;
                                }
                            }
                        }
                    }

                    if (proj_odd_flag & ODD_YLM_Y)
                        //if (false)
                    {
                        GradientYInKspace((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                        fftw->BackwardDirect((fftw_complex*)&grad_p[0]);

                        for (int iz = 0; iz < sub_size_z; ++iz) {
                            for (int ix = 0; ix < sub_size_x; ++ix) {

                                const int64_t i = ix + sub_size_x * (0 + sub_size_y * iz);

                                //SvF2_MARGIN_NL
                                double sum = grad_p[i].r / 2.0;
                                grad_p[i].r = 0.0;

                                for (int iy = 1; iy < SvF2_MARGIN_NL; ++iy) {
                                    sum += grad_p[i + sub_size_x * iy].r;
                                    grad_p[i + sub_size_x * iy].r = 0.0;
                                    grad_p[i + sub_size_x * (sub_size_y - iy)].r = 0.0;
                                }
                                {
                                    int iy = SvF2_MARGIN_NL;
                                    grad_p[i + sub_size_x * iy].r += sum;
                                    grad_p[i + sub_size_x * (sub_size_y - iy)].r += sum;
                                }

                            }
                        }


                        for (int i = 0; i < sub_size; ++i) {
                            grad_p[i].r *= invN;
                            grad_p[i].i *= invN;
                        }

                        fftw->ForwardExecute((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY);


                        for (int iz = 0; iz < sub_size_z; ++iz) {
                            for (int iy = 1; iy < sub_size_y; ++iy) {
                                const double ky1 = (coef1_y * (double)(iy * 2 > sub_size_y ? iy - sub_size_y : iy));

                                for (int ix = 0; ix < sub_size_x; ++ix) {
                                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                                    double ore = tmp_RY[i].r;
                                    double oim = tmp_RY[i].i;
                                    tmp_RY[i].r = oim / ky1;
                                    tmp_RY[i].i = -ore / ky1;
                                }
                            }
                        }
                    }

                    if (proj_odd_flag & ODD_YLM_Z)
                        //if (false)
                    {
                        GradientZInKspace((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                        fftw->BackwardDirect((fftw_complex*)&grad_p[0]);

                        for (int iy = 0; iy < sub_size_y; ++iy) {
                            for (int ix = 0; ix < sub_size_x; ++ix) {

                                const int64_t i = ix + sub_size_x * (iy + sub_size_y * 0);


                                //SvF2_MARGIN_NL
                                double sum = grad_p[i].r / 2.0;
                                grad_p[i].r = 0.0;

                                for (int iz = 1; iz < SvF2_MARGIN_NL; ++iz) {
                                    sum += grad_p[i + sub_size_x * sub_size_y * iz].r;
                                    grad_p[i + sub_size_x * sub_size_y * iz].r = 0.0;
                                    grad_p[i + sub_size_x * sub_size_y * (sub_size_z - iz)].r = 0.0;
                                }
                                {
                                    int iz = SvF2_MARGIN_NL;
                                    grad_p[i + sub_size_x * sub_size_y * iz].r += sum;
                                    grad_p[i + sub_size_x * sub_size_y * (sub_size_z - iz)].r += sum;
                                }

                            }
                        }


                        for (int i = 0; i < sub_size; ++i) {
                            grad_p[i].r *= invN;
                            grad_p[i].i *= invN;
                        }
                        fftw->ForwardExecute((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY);

                        for (int iz = 1; iz < sub_size_z; ++iz) {
                            const double kz1 = (coef1_z * (double)(iz * 2 > sub_size_z ? iz - sub_size_z : iz));

                            for (int iy = 0; iy < sub_size_y; ++iy) {

                                for (int ix = 0; ix < sub_size_x; ++ix) {
                                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                                    double ore = tmp_RY[i].r;
                                    double oim = tmp_RY[i].i;
                                    tmp_RY[i].r = oim / kz1;
                                    tmp_RY[i].i = -ore / kz1;
                                }
                            }
                        }

                    }
                }

#endif


                nonlocal_kspace->projector_energy_RY[id_proj_RY] = projector_energy_nl;
                ++id_proj_RY;

            }


        }//end of k//


        //Ylmのバッファ開放//
        for (int l = 0; l <= max_l; ++l) {
            delete[] pYlm[l];
        }
        delete[] proj_R;
        delete[] hr_RY;
        
    }

#elif (SvF2_USE_INTERPOLATION_NL == 4) || (SvF2_USE_INTERPOLATION_NL == 6)

    void mInitializeNonlocalProjector(int Z, NonlocalKspace2* nonlocal_kspace) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const int max_l = pp->MaxProjectorL();
        const double cutoff_max = pp->MaxCutoffLength();
        const GridRange subgrid = SubgridFromCutoff(cutoff_max, SvF2_MARGIN_NL);
        const int64_t sub_size = subgrid.Size3D();
        nonlocal_kspace->subgrid = subgrid;
    

        //(実)球面調和関数用のバッファを確保//                
        double* pYlm[4]{ nullptr,nullptr,nullptr,nullptr };
        pYlm[0] = gy::AlignedAlloc<double>(sub_size * (max_l+1)* (max_l + 1));
        for (int l = 0; l <= max_l; ++l) {
            const double cutoff_l = pp->cutoff_r[l];
            nonlocal_kspace->cutoff_l[l] = cutoff_l;
            pYlm[l] = pYlm[0] + sub_size * (l * l);
        }

        //FFTWの準備//
        {
            auto* fftw = new FFTW_Executor;
            fftw->Initialize(subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), FFTW_ESTIMATE);
            nonlocal_kspace->fftw = fftw;

        }


        //動径関数のprojectorを原点中心でデカルトグリッドに焼き直す//
        const int num_projectors_R = pp->num_radial_projectors;
        int num_proj_RY = 0;
        for (int k = 0; k < num_projectors_R; ++k) {
            const int l = pp->projector_quantum_l[k];
            num_proj_RY += 2 * l + 1;
        }
        nonlocal_kspace->num_proj_RY = num_proj_RY;

        nonlocal_kspace->projector_energy_RY = gy::make_unique_aligned<double[]>(num_proj_RY);
        nonlocal_kspace->odd_flag.clear();


        double* proj_R = gy::AlignedAlloc<double>(sub_size);
        OneComplex* proj_RY_all = gy::AlignedAlloc<OneComplex>(num_proj_RY * sub_size);
        OneComplex* tmp_RY_all = gy::AlignedAlloc<OneComplex>(num_proj_RY * sub_size);

        {
            memset(proj_RY_all, 0, sizeof(OneComplex) * num_proj_RY * sub_size);
            auto* p = proj_RY_all;
            for (int k = 0; k < num_projectors_R; ++k) {
                const int l = pp->projector_quantum_l[k];
                for (int m = -l; m <= l; ++m) {
                    nonlocal_kspace->k_projector_Ylm.push_back(p);
                    p += sub_size;
                }
            }
        }


    #ifndef SvF2_SUBGRID_ODD_HALF 
        printf("ERROR: SvF2_USE_INTERPOLATION_NL requires SvF2_SUBGRID_ODD_HALF flag\n");
    #endif
        constexpr int csize = SvF2_INTERPOLATION_NL_MAGNITUDE;
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


                    //(実)球面調和関数を原点中心で生成(元素に共通)//
                    for (int l = 0; l <= max_l; ++l) {
                        for (int m = -l; m <= l; ++m) {
                            auto* pY = pYlm[l] + sub_size * (l + m);
                            SetYlmAnyRangeFromOrigin(l, m, pY, subgrid, m_dx, m_dy, m_dz, center_x, center_y, center_z);
                        }
                    }

                    //動径関数を生成//
                    int id_proj_RY = 0;
                    for (int k = 0; k < num_projectors_R; ++k) {
                        const int l = pp->projector_quantum_l[k];
                        const double cutoff_l = pp->cutoff_r[l];
                        const double rr_max = cutoff_l * cutoff_l;



                        ForXYZ(subgrid,
                            [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                                const double x = ((double)ix - center_x) * m_dx;
                                const double y = ((double)iy - center_y) * m_dy;
                                const double z = ((double)iz - center_z) * m_dz;

                                const double rr = x * x + y * y + z * z;
                                if (rr < rr_min) {
                                    const double proj_spin_orbit_up = pp->projector[k * 2][0];
                                    const double proj_spin_orbit_dn = pp->projector[k * 2 + 1][0];
                                    const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;
                                    proj_R[i] = p;
                                } else if (rr > rr_max) {
                                    proj_R[i] = 0.0;
                                } else {
                                    //NOTE: if it is not relative DFT with spin-orbit interaction, projectors (j+1/2) and (j-1/2) should be averaged.
                                    const double proj_spin_orbit_up = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                    const double proj_spin_orbit_dn = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2 + 1], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                    const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;

                                    proj_R[i] = p;

                                }
                            });


                        const double projector_energy_nl = pp->projector_energy_up[k];


                        //原点中心のprojector(proj_R)と原点中心Ylmを合成する//
                        //FFTでk-spaceへの変換も行う//

                        for (int m = -l; m <= l; ++m) {
                            const auto* pY = pYlm[l] + sub_size * (l + m);
                            auto* tmp_RY = tmp_RY_all + sub_size * id_proj_RY;

                            for (int i = 0; i < sub_size; ++i) {
                                tmp_RY[i].r = proj_R[i] * pY[i];
                                tmp_RY[i].i = 0.0;
                            }

                            auto& fftw = nonlocal_kspace->fftw;
                            fftw->ForwardDirect(tmp_RY);


                            //x,y,zの各方向ごとに独立にシフトし、シフトによる境界点での補正も行う//
                            const int sub_size_x = subgrid.SizeX();
                            const int sub_size_y = subgrid.SizeY();
                            const int sub_size_z = subgrid.SizeZ();

#if 0
                            std::string path("pp_beta_");
                            path += std::to_string(k) + "_" + std::to_string(l) + "_" + std::to_string(m) + ".txt";
                            ShiftToHalfPoint_2(&tmp_RY[0], cx, cy, cz, csize, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z, IsRoot(MPI_COMM_WORLD) ? path.c_str() : nullptr);
#elif 0
                            ShiftToHalfPoint(&tmp_RY[0], cx, cy, cz, csize, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z);
#else
                            ShiftInKspace_update(&tmp_RY[0], (half - center_x) * m_dx, (half - center_y) * m_dy, (half - center_z) * m_dz, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                            //CorrectEdgeZeroByWave(tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z);
                            //CorrectEdgeZeroByWave_2(tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z);

#endif

                            nonlocal_kspace->projector_energy_RY[id_proj_RY] = projector_energy_nl;
                            ++id_proj_RY;

                        }


                    }//end of k//

    #if SvF2_USE_INTERPOLATION_NL==2
                    mCreateSymplecticProjectorMatrix_PPmatrix(Z, tmp_RY_all);
    #endif
                    
                    for (int i = 0; i < num_proj_RY * sub_size; ++i) {
                        proj_RY_all[i].r += tmp_RY_all[i].r * weight;
                        proj_RY_all[i].i += tmp_RY_all[i].i * weight;
                    }
                }//end of loop for cx
            }//end of loop for cy
        }//end of loop for cz

//correction////////////////////////////////
        int id_proj_RY = 0;
        for (int k = 0; k < num_projectors_R; ++k) {
            const int l = pp->projector_quantum_l[k];
            const double cutoff_l = pp->cutoff_r[l];
            const double rr_max = cutoff_l * cutoff_l;

            for (int m = -l; m <= l; ++m) {
                const auto* pY = pYlm[l] + sub_size * (l + m);
                auto* tmp_RY = proj_RY_all + sub_size * id_proj_RY;

                auto& fftw = nonlocal_kspace->fftw;

                int sub_size_x = subgrid.SizeX();
                int sub_size_y = subgrid.SizeY();
                int sub_size_z = subgrid.SizeZ();
                const double invN = 1.0 / (double)sub_size;
                const int proj_odd_flag = CheckYlmOddForDirection(l, m);
                nonlocal_kspace->odd_flag.push_back(proj_odd_flag);

#if (SvF2_USE_INTERPOLATION_NL == 6)
                //保持するものが偶関数になるように微分してしまう//

#if 1
                if (proj_odd_flag & ODD_YLM_X) {
                    GradientXInKspace_inplace((fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
                if (proj_odd_flag & ODD_YLM_Y) {
                    GradientYInKspace_inplace((fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
                if (proj_odd_flag & ODD_YLM_Z) {
                    GradientZInKspace_inplace((fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
#else
                if (proj_odd_flag == ODD_YLM_X) {
                    GradientXInKspace_inplace((fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
#endif
#endif

                /*
                * 境界点での値がゼロになるように補正を加える                
                * ただし、tmp_RYはkspaceへFFTされた後の場を受け取る
                */
                std::string path("pp_beta_");
                path += std::to_string(k) + "_" + std::to_string(l) + "_" + std::to_string(m) + ".txt";
                CorrectEdgeZeroByWave_3(tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z, IsRoot(MPI_COMM_WORLD) ? path.c_str() : nullptr);


#if (SvF2_USE_INTERPOLATION_NL != 6)
#ifdef ODD_PROJ_CORRECTION
                //微分値がsubgridの端で0になるように補正する//

                std::string path2("pp_beta_diff");
                path2 += std::to_string(k) + "_" + std::to_string(l) + "_" + std::to_string(m) + ".txt";
                
                CorrectEdgeDifferenceZeroByWave_2(proj_odd_flag, tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z, IsRoot(MPI_COMM_WORLD) ? path2.c_str() : nullptr);

#endif

#ifdef NONLOCAL_NORMALIZE
                {
                    double norm = 0.0;
                    for (int i = 0; i < sub_size; ++i) {
                        norm += tmp_RY[i].r * tmp_RY[i].r + tmp_RY[i].i * tmp_RY[i].i;
                    }
                    norm = 1.0 / sqrt(norm * m_dx*m_dy*m_dz / (double)sub_size);
                    for (int i = 0; i < sub_size; ++i) {
                        tmp_RY[i].r *= norm;
                        tmp_RY[i].i *= norm;
                    }
                }
#endif

#endif



                ++id_proj_RY;
            }//end of m//

        }//end of k//

        //Ylmのバッファ開放//
        gy::AlignedFree(pYlm[0]);
        
        gy::AlignedFree(proj_R);
        gy::AlignedFree(tmp_RY_all);

    }


#elif SvF2_USE_INTERPOLATION_NL == 5

void mInitializeNonlocalProjector(int Z, NonlocalKspace2* nonlocal_kspace) {

    const PseudoPot_MBK* pp = mFindPseudoPot(Z);
    const double rr_min = (pp->radius[0]) * (pp->radius[0]);
    const int max_l = pp->MaxProjectorL();
    const double cutoff_max = pp->MaxCutoffLength();
    const GridRange subgrid = SubgridFromCutoff(cutoff_max, SvF2_MARGIN_NL);
    const int64_t sub_size = subgrid.Size3D();
    const int sub_size_x = subgrid.SizeX();
    const int sub_size_y = subgrid.SizeY();
    const int sub_size_z = subgrid.SizeZ();

    nonlocal_kspace->subgrid = subgrid;


    //(実)球面調和関数用のバッファを確保//                
    double* pYlm[4]{ nullptr,nullptr,nullptr,nullptr };
    for (int l = 0; l <= max_l; ++l) {
        const double cutoff_l = pp->cutoff_r[l];
        nonlocal_kspace->cutoff_l[l] = cutoff_l;
        pYlm[l] = new double[sub_size * (2 * l + 1)];
    }

    //FFTWの準備//
    {
        auto* fftw = new FFTW_Executor;
        fftw->Initialize(subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), FFTW_ESTIMATE);
        nonlocal_kspace->fftw = fftw;

    }


    //動径関数のprojectorを原点中心でデカルトグリッドに焼き直す//
    const int num_projectors_R = pp->num_radial_projectors;
    int num_proj_RY = 0;
    for (int k = 0; k < num_projectors_R; ++k) {
        const int l = pp->projector_quantum_l[k];
        num_proj_RY += 2 * l + 1;
    }
    nonlocal_kspace->num_proj_RY = num_proj_RY;

    nonlocal_kspace->projector_energy_RY = gy::make_unique_aligned<double[]>(num_proj_RY);
    nonlocal_kspace->odd_flag.clear();

    double* proj_R = new double[sub_size];
    OneComplex* proj_RY_all = new OneComplex[num_proj_RY * sub_size];
    OneComplex* tmp_RY_all = new OneComplex[num_proj_RY * sub_size];

    {
        memset(proj_RY_all, 0, sizeof(OneComplex) * num_proj_RY * sub_size);
        auto* p = proj_RY_all;
        for (int k = 0; k < num_projectors_R; ++k) {
            const int l = pp->projector_quantum_l[k];
            for (int m = -l; m <= l; ++m) {
                nonlocal_kspace->k_projector_Ylm.push_back(p);
                const int proj_odd_flag = CheckYlmOddForDirection(l, m);
                nonlocal_kspace->odd_flag.push_back(proj_odd_flag);
                p += sub_size;
            }
        }
    }


#ifndef SvF2_SUBGRID_ODD_HALF 
    printf("ERROR: SvF2_USE_INTERPOLATION_NL requires SvF2_SUBGRID_ODD_HALF flag\n");
#endif
    constexpr int csize = SvF2_INTERPOLATION_NL_MAGNITUDE;
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


                //(実)球面調和関数を原点中心で生成(元素に共通)//
                for (int l = 0; l <= max_l; ++l) {
                    for (int m = -l; m <= l; ++m) {
                        auto* pY = pYlm[l] + sub_size * (l + m);
                        SetYlmAnyRangeFromOrigin(l, m, pY, subgrid, m_dx, m_dy, m_dz, center_x, center_y, center_z);
                    }
                }

                //動径関数を生成//
                int id_proj_RY = 0;
                for (int k = 0; k < num_projectors_R; ++k) {
                    const int l = pp->projector_quantum_l[k];
                    const double cutoff_l = pp->cutoff_r[l];
                    const double rr_max = cutoff_l * cutoff_l;



                    ForXYZ(subgrid,
                        [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                            const double x = ((double)ix - center_x) * m_dx;
                            const double y = ((double)iy - center_y) * m_dy;
                            const double z = ((double)iz - center_z) * m_dz;

                            const double rr = x * x + y * y + z * z;
                            if (rr < rr_min) {
                                const double proj_spin_orbit_up = pp->projector[k * 2][0];
                                const double proj_spin_orbit_dn = pp->projector[k * 2 + 1][0];
                                const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;
                                proj_R[i] = p;
                            } else if (rr > rr_max) {
                                proj_R[i] = 0.0;
                            } else {
                                //NOTE: if it is not relative DFT with spin-orbit interaction, projectors (j+1/2) and (j-1/2) should be averaged.
                                const double proj_spin_orbit_up = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                const double proj_spin_orbit_dn = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2 + 1], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;

                                proj_R[i] = p;

                            }
                        });


                    const double projector_energy_nl = pp->projector_energy_up[k];


                    //原点中心のprojector(proj_R)と原点中心Ylmを合成する//
                    //FFTでk-spaceへの変換も行う//

                    for (int m = -l; m <= l; ++m) {
                        const auto* pY = pYlm[l] + sub_size * (l + m);
                        auto* tmp_RY = tmp_RY_all + sub_size * id_proj_RY;


                        for (int i = 0; i < sub_size; ++i) {
                            tmp_RY[i].r = proj_R[i] * pY[i];
                            tmp_RY[i].i = 0.0;
                        }

        

                        auto& fftw = nonlocal_kspace->fftw;
                        fftw->ForwardDirect((fftw_complex*)tmp_RY);


                        //x,y,zの各方向ごとに独立にシフトし、シフトによる境界点での補正も行う//
                        const int sub_size_x = subgrid.SizeX();
                        const int sub_size_y = subgrid.SizeY();
                        const int sub_size_z = subgrid.SizeZ();


                        ShiftInKspace_update(&tmp_RY[0], (half - center_x) * m_dx, (half - center_y) * m_dy, (half - center_z) * m_dz, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


                        nonlocal_kspace->projector_energy_RY[id_proj_RY] = projector_energy_nl;
                        ++id_proj_RY;

                    }


                }//end of k//


                for (int i = 0; i < num_proj_RY * sub_size; ++i) {
                    proj_RY_all[i].r += tmp_RY_all[i].r * weight;
                    proj_RY_all[i].i += tmp_RY_all[i].i * weight;
                }
            }//end of loop for cx
        }//end of loop for cy
    }//end of loop for cz

//correction////////////////////////////////
    int id_proj_RY = 0;
    for (int k = 0; k < num_projectors_R; ++k) {
        const int l = pp->projector_quantum_l[k];
        const double cutoff_l = pp->cutoff_r[l];
        const double rr_max = cutoff_l * cutoff_l;

        for (int m = -l; m <= l; ++m) {
            const auto* pY = pYlm[l] + sub_size * (l + m);
            auto* tmp_RY = proj_RY_all + sub_size * id_proj_RY;

            auto& fftw = nonlocal_kspace->fftw;

            int sub_size_x = subgrid.SizeX();
            int sub_size_y = subgrid.SizeY();
            int sub_size_z = subgrid.SizeZ();
            const double invN = 1.0 / (double)sub_size;

            {
                const double center_x = 0.5;
                const double center_y = 0.5;
                const double center_z = 0.5;

                fftw->BackwardDirect((fftw_complex*)tmp_RY);

                const int proj_odd_flag = CheckYlmOddForDirection(l, m);
                auto SetRY = [&](auto filter) {

                    ForXYZ(subgrid,
                        [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                            const double x = ((double)ix - center_x) * m_dx;
                            const double y = ((double)iy - center_y) * m_dy;
                            const double z = ((double)iz - center_z) * m_dz;
                            const double f = filter(x, y, z);
                            if (fabs(f) < 1.e-10) {
                                tmp_RY[i].r *= invN;
                                tmp_RY[i].i *= invN;
                            } else {
                                tmp_RY[i].r *= invN / f;
                                tmp_RY[i].i *= invN / f;
                            }
                        });
                    };

                if (proj_odd_flag & ODD_YLM_X) {
                    if (proj_odd_flag & ODD_YLM_Y) {
                        if (proj_odd_flag & ODD_YLM_Z) {
                            SetRY([&](double x, double y, double z) {
                                return x * y * z;
                                });

                        } else {
                            SetRY([&](double x, double y, double z) {
                                return x * y;
                                });
                        }
                    } else {
                        if (proj_odd_flag & ODD_YLM_Z) {
                            SetRY([&](double x, double y, double z) {
                                return x * z;
                                });

                        } else {
                            SetRY([&](double x, double y, double z) {
                                return x;
                                });

                        }
                    }
                } else {
                    if (proj_odd_flag & ODD_YLM_Y) {
                        if (proj_odd_flag & ODD_YLM_Z) {
                            SetRY([&](double x, double y, double z) {
                                return y * z;
                                });
                        } else {
                            SetRY([&](double x, double y, double z) {
                                return y;
                                });
                        }
                    } else {
                        if (proj_odd_flag & ODD_YLM_Z) {
                            SetRY([&](double x, double y, double z) {
                                return z;
                                });
                        } else {
                            SetRY([&](int ix, int iy, int iz) {
                                return 1.0;
                                });
                        }
                    }
                }

                fftw->ForwardDirect((fftw_complex*)tmp_RY);
            }

            /*
            * 境界点での値がゼロになるように補正を加える
            * ただし、tmp_RYはkspaceへFFTされた後の場を受け取る
            */
            std::string path("pp_beta_");
            path += std::to_string(k) + "_" + std::to_string(l) + "_" + std::to_string(m) + ".txt";
            CorrectEdgeZeroByWave_3(tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z, IsRoot(MPI_COMM_WORLD) ? path.c_str() : nullptr);
            //CorrectEdgeZeroByWave_2(tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z, IsRoot(MPI_COMM_WORLD) ? path.c_str() : nullptr);
            //CorrectEdgeZeroByWave(tmp_RY, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z);



            ++id_proj_RY;
        }//end of m//

    }//end of k//

    //Ylmのバッファ開放//
    for (int l = 0; l <= max_l; ++l) {
        delete[] pYlm[l];
    }
    delete[] proj_R;
    delete[] tmp_RY_all;

}

#else   //SvF2_USE_INTERPOLATION_NL == 1

    void mInitializeNonlocalProjector(int Z, NonlocalKspace2* nonlocal_kspace) {

        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        const double rr_min = (pp->radius[0]) * (pp->radius[0]);
        const int max_l = pp->MaxProjectorL();
        const double cutoff_max = pp->MaxCutoffLength();
        const GridRange subgrid = SubgridFromCutoff(cutoff_max, SvF2_MARGIN_NL);
        const int64_t sub_size = subgrid.Size3D();
        nonlocal_kspace->subgrid = subgrid;
#ifdef SMOOTH_SUBGRID
        nonlocal_kspace.subgrid.end_x += 1;
        nonlocal_kspace.subgrid.end_y += 1;
        nonlocal_kspace.subgrid.end_z += 1;
#endif                


        //(実)球面調和関数用のバッファを確保//                
        double* pYlm[4]{ nullptr,nullptr,nullptr,nullptr };
        for (int l = 0; l <= max_l; ++l) {
            const double cutoff_l = pp->cutoff_r[l];
            nonlocal_kspace->cutoff_l[l] = cutoff_l;
            pYlm[l] = new double[sub_size * (2 * l + 1)];
        }

        //FFTWの準備//
        {
            auto* fftw = new FFTW_Executor;
            fftw->Initialize(subgrid.SizeX(), subgrid.SizeY(), subgrid.SizeZ(), FFTW_ESTIMATE);
            nonlocal_kspace->fftw = fftw;

        }


        //動径関数のprojectorを原点中心でデカルトグリッドに焼き直す//
        const int num_projectors_R = pp->num_radial_projectors;
        int num_proj_RY = 0;
        for (int k = 0; k < num_projectors_R; ++k) {
            const int l = pp->projector_quantum_l[k];
            num_proj_RY += 2 * l + 1;
        }
        nonlocal_kspace->num_proj_RY = num_proj_RY;

        nonlocal_kspace->projector_energy_RY = gy::make_unique_aligned<double[]>(num_proj_RY);


        double* proj_R = new double[sub_size];
        OneComplex* proj_RY_all = new OneComplex[num_proj_RY * sub_size];
        OneComplex* tmp_RY_all = new OneComplex[num_proj_RY * sub_size];

        {
            memset(proj_RY_all, 0, sizeof(OneComplex) * num_proj_RY * sub_size);
            auto* p = proj_RY_all;
            for (int k = 0; k < num_projectors_R; ++k) {
                const int l = pp->projector_quantum_l[k];
                for (int m = -l; m <= l; ++m) {
                    nonlocal_kspace->k_projector_Ylm.push_back(p);
                    p += sub_size;
                    const int proj_odd_flag = CheckYlmOddForDirection(l, m);
                    nonlocal_kspace->odd_flag.push_back(proj_odd_flag);
                }
            }
        }


#ifdef SvF2_USE_INTERPOLATION_NL
#ifndef SvF2_SUBGRID_ODD_HALF 
        printf("ERROR: SvF2_USE_INTERPOLATION_NL requires SvF2_SUBGRID_ODD_HALF flag\n");
#endif
        constexpr int csize = SvF2_INTERPOLATION_NL_MAGNITUDE;
        constexpr double half = 0.5;
        double weight = 1.0 / (double)(csize * csize * csize);
        auto fold_pos = [&half, &csize](int cx) {
            return half - (double)(cx * 2 > csize ? cx - csize : cx) / (double)csize;
            };

#elif defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)

        constexpr int csize = 1;
        constexpr double half = 0.5;
        double weight = 1.0;
        auto fold_pos = [](int cx) {
            return 0.5;
            };
#else

        constexpr int csize = 1;
        constexpr double half = 0.5;
        double weight = 1.0;
        auto fold_pos = [](int cx) {
            return 0.0;
            };
#endif
        for (int cz = 0; cz < csize; ++cz) {
            for (int cy = 0; cy < csize; ++cy) {
                for (int cx = 0; cx < csize; ++cx) {

                    const double center_x = fold_pos(cx);
                    const double center_y = fold_pos(cy);
                    const double center_z = fold_pos(cz);


                    //(実)球面調和関数を原点中心で生成(元素に共通)//
                    for (int l = 0; l <= max_l; ++l) {
                        for (int m = -l; m <= l; ++m) {
                            auto* pY = pYlm[l] + sub_size * (l + m);
                            SetYlmAnyRangeFromOrigin(l, m, pY, subgrid, m_dx, m_dy, m_dz, center_x, center_y, center_z);
                        }
                    }

                    //動径関数を生成//
                    int id_proj_RY = 0;
                    for (int k = 0; k < num_projectors_R; ++k) {
                        const int l = pp->projector_quantum_l[k];
                        const double cutoff_l = pp->cutoff_r[l];
                        const double rr_max = cutoff_l * cutoff_l;



                        ForXYZ(subgrid,
                            [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                                const double x = ((double)ix - center_x) * m_dx;
                                const double y = ((double)iy - center_y) * m_dy;
                                const double z = ((double)iz - center_z) * m_dz;

                                const double rr = x * x + y * y + z * z;
                                if (rr < rr_min) {
                                    const double proj_spin_orbit_up = pp->projector[k * 2][0];
                                    const double proj_spin_orbit_dn = pp->projector[k * 2 + 1][0];
                                    const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;
                                    proj_R[i] = p;
                                } else if (rr > rr_max) {
                                    proj_R[i] = 0.0;
                                } else {
                                    //NOTE: if it is not relative DFT with spin-orbit interaction, projectors (j+1/2) and (j-1/2) should be averaged.
                                    const double proj_spin_orbit_up = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                    const double proj_spin_orbit_dn = RadialGrid2::GetValueBySquare(rr, pp->projector[k * 2 + 1], pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                                    const double p = (proj_spin_orbit_up + proj_spin_orbit_dn) / 2.0;

                                    proj_R[i] = p;

                                }
                            });


                        const double projector_energy_nl = pp->projector_energy_up[k];


                        //原点中心のprojector(proj_R)と原点中心Ylmを合成する//
                        //FFTでk-spaceへの変換も行う//

                        for (int m = -l; m <= l; ++m) {
                            const auto* pY = pYlm[l] + sub_size * (l + m);
                            auto* tmp_RY = tmp_RY_all + sub_size * id_proj_RY;

                            for (int i = 0; i < sub_size; ++i) {
                                tmp_RY[i].r = proj_R[i] * pY[i];
                                tmp_RY[i].i = 0.0;
                            }

                            auto& fftw = nonlocal_kspace->fftw;
                            fftw->ForwardDirect((fftw_complex*)tmp_RY);

                            //auto* projRYlm = nonlocal_kspace.k_projector_Ylm[id_proj_RY];



                            //x,y,zの各方向ごとに独立にシフトし、シフトによる境界点での補正も行う//
                            const int sub_size_x = subgrid.SizeX();
                            const int sub_size_y = subgrid.SizeY();
                            const int sub_size_z = subgrid.SizeZ();

#if 1
                            ShiftToHalfPoint(&tmp_RY[0], cx, cy, cz, csize, fftw, m_dx, m_dy, m_dz, sub_size_x, sub_size_y, sub_size_z);
#else
                            ShiftInKspace_update(&tmp_RY[0], (half - center_x) * m_dx, (half - center_y) * m_dy, (half - center_z) * m_dz, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
#endif


#ifdef ODD_PROJ_CORRECTION
                            //微分値がsubgridの端で0になるように補正する//


                            const int proj_odd_flag = CheckYlmOddForDirection(l, m);
                            if (proj_odd_flag) {

                                int sub_size_x = subgrid.SizeX();
                                int sub_size_y = subgrid.SizeY();
                                int sub_size_z = subgrid.SizeZ();
                                const double invN = 1.0 / (double)sub_size;

                                const double coef1_x = (2.0 * M_PI / (m_dx * (double)sub_size_x));
                                const double coef1_y = (2.0 * M_PI / (m_dy * (double)sub_size_y));
                                const double coef1_z = (2.0 * M_PI / (m_dz * (double)sub_size_z));

                                auto grad_p = gy::make_unique_aligned<OneComplex[]>(sub_size);


                                if (proj_odd_flag & ODD_YLM_X)
                                    //if(false)
                                {
                                    GradientXInKspace((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                                    fftw->BackwardDirect((fftw_complex*)&grad_p[0]);

                                    for (int iz = 0; iz < sub_size_z; ++iz) {
                                        for (int iy = 0; iy < sub_size_y; ++iy) {
                                            const int64_t i = sub_size_x * (iy + sub_size_y * iz);


                                            //SvF2_MARGIN_NL
                                            double sum = grad_p[i].r / 2.0;
                                            grad_p[i].r = 0.0;

                                            for (int ix = 1; ix < SvF2_MARGIN_NL; ++ix) {
                                                sum += grad_p[i + ix].r;
                                                grad_p[i + ix].r = 0.0;
                                                grad_p[i + sub_size_x - ix].r = 0.0;
                                            }
                                            {
                                                int ix = SvF2_MARGIN_NL;
                                                grad_p[i + ix].r += sum;
                                                grad_p[i + sub_size_x - ix].r += sum;
                                            }


                                        }
                                    }

                                    for (int i = 0; i < sub_size; ++i) {
                                        grad_p[i].r *= invN;
                                        grad_p[i].i *= invN;
                                    }

                                    fftw->ForwardExecute((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY);

                                    for (int iz = 0; iz < sub_size_z; ++iz) {
                                        for (int iy = 0; iy < sub_size_y; ++iy) {
                                            for (int ix = 1; ix < sub_size_x; ++ix) {
                                                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                                                const double kx1 = (coef1_x * (double)(ix * 2 > sub_size_x ? ix - sub_size_x : ix));
                                                double ore = tmp_RY[i].r;
                                                double oim = tmp_RY[i].i;
                                                tmp_RY[i].r = oim / kx1;
                                                tmp_RY[i].i = -ore / kx1;
                                            }
                                        }
                                    }
                                }

                                if (proj_odd_flag & ODD_YLM_Y)
                                    //if (false)
                                {
                                    GradientYInKspace((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                                    fftw->BackwardDirect((fftw_complex*)&grad_p[0]);

                                    for (int iz = 0; iz < sub_size_z; ++iz) {
                                        for (int ix = 0; ix < sub_size_x; ++ix) {

                                            const int64_t i = ix + sub_size_x * (0 + sub_size_y * iz);

                                            //SvF2_MARGIN_NL
                                            double sum = grad_p[i].r / 2.0;
                                            grad_p[i].r = 0.0;

                                            for (int iy = 1; iy < SvF2_MARGIN_NL; ++iy) {
                                                sum += grad_p[i + sub_size_x * iy].r;
                                                grad_p[i + sub_size_x * iy].r = 0.0;
                                                grad_p[i + sub_size_x * (sub_size_y - iy)].r = 0.0;
                                            }
                                            {
                                                int iy = SvF2_MARGIN_NL;
                                                grad_p[i + sub_size_x * iy].r += sum;
                                                grad_p[i + sub_size_x * (sub_size_y - iy)].r += sum;
                                            }

                                        }
                                    }


                                    for (int i = 0; i < sub_size; ++i) {
                                        grad_p[i].r *= invN;
                                        grad_p[i].i *= invN;
                                    }

                                    fftw->ForwardExecute((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY);


                                    for (int iz = 0; iz < sub_size_z; ++iz) {
                                        for (int iy = 1; iy < sub_size_y; ++iy) {
                                            const double ky1 = (coef1_y * (double)(iy * 2 > sub_size_y ? iy - sub_size_y : iy));

                                            for (int ix = 0; ix < sub_size_x; ++ix) {
                                                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                                                double ore = tmp_RY[i].r;
                                                double oim = tmp_RY[i].i;
                                                tmp_RY[i].r = oim / ky1;
                                                tmp_RY[i].i = -ore / ky1;
                                            }
                                        }
                                    }
                                }

                                if (proj_odd_flag & ODD_YLM_Z)
                                    //if (false)
                                {
                                    GradientZInKspace((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                                    fftw->BackwardDirect((fftw_complex*)&grad_p[0]);

                                    for (int iy = 0; iy < sub_size_y; ++iy) {
                                        for (int ix = 0; ix < sub_size_x; ++ix) {

                                            const int64_t i = ix + sub_size_x * (iy + sub_size_y * 0);


                                            //SvF2_MARGIN
                                            double sum = grad_p[i].r / 2.0;
                                            grad_p[i].r = 0.0;

                                            for (int iz = 1; iz < SvF2_MARGIN_NL; ++iz) {
                                                sum += grad_p[i + sub_size_x * sub_size_y * iz].r;
                                                grad_p[i + sub_size_x * sub_size_y * iz].r = 0.0;
                                                grad_p[i + sub_size_x * sub_size_y * (sub_size_z - iz)].r = 0.0;
                                            }
                                            {
                                                int iz = SvF2_MARGIN_NL;
                                                grad_p[i + sub_size_x * sub_size_y * iz].r += sum;
                                                grad_p[i + sub_size_x * sub_size_y * (sub_size_z - iz)].r += sum;
                                            }

                                        }
                                    }


                                    for (int i = 0; i < sub_size; ++i) {
                                        grad_p[i].r *= invN;
                                        grad_p[i].i *= invN;
                                    }
                                    fftw->ForwardExecute((fftw_complex*)&grad_p[0], (fftw_complex*)tmp_RY);

                                    for (int iz = 1; iz < sub_size_z; ++iz) {
                                        const double kz1 = (coef1_z * (double)(iz * 2 > sub_size_z ? iz - sub_size_z : iz));

                                        for (int iy = 0; iy < sub_size_y; ++iy) {

                                            for (int ix = 0; ix < sub_size_x; ++ix) {
                                                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                                                double ore = tmp_RY[i].r;
                                                double oim = tmp_RY[i].i;
                                                tmp_RY[i].r = oim / kz1;
                                                tmp_RY[i].i = -ore / kz1;
                                            }
                                        }
                                    }

                                }
                            }

#endif


                            nonlocal_kspace->projector_energy_RY[id_proj_RY] = projector_energy_nl;
                            ++id_proj_RY;

                        }


                    }//end of k//

#if SvF2_USE_INTERPOLATION_NL==2
                    mCreateSymplecticProjectorMatrix_PPmatrix(Z, tmp_RY_all);
#endif

                    for (int i = 0; i < num_proj_RY * sub_size; ++i) {
                        proj_RY_all[i].r += tmp_RY_all[i].r * weight;
                        proj_RY_all[i].i += tmp_RY_all[i].i * weight;
                    }
                }//end of loop for cx
            }//end of loop for cy
        }//end of loop for cz


        //Ylmのバッファ開放//
        for (int l = 0; l <= max_l; ++l) {
            delete[] pYlm[l];
        }
        delete[] proj_R;
        delete[] tmp_RY_all;

    }
#endif

private:

    void mSetNonlocalProjectorInfo(PP_Proj_Ylm_each_atom& nonlocal_projectors, const Nucleus nucleus) {

        const PseudoPot_MBK* pp = mFindPseudoPot(nucleus.Z);
        if (pp == nullptr) {
            printf("ERROR: VPS is not loaded: Z = %d\n", nucleus.Z);
            return;
        }

    }

    


    void CountNonlocalProjector(const Nucleus* nuclei, int num_nuclei){
        
        int num_all_nonlocal = 0;
        m_num_list_nonlocal.clear();
        m_num_list_nonlocal.push_back(0);
        for (int ni = 0; ni < num_nuclei; ++ni) {
            const PseudoPot_MBK* pp = mFindPseudoPot(nuclei[ni].Z);
            if (pp == nullptr) {
                printf("ERROR: VPS is not loaded: Z = %d\n", nuclei[ni].Z);
                return;
            }

            num_all_nonlocal += pp->TotalProjectorLM();
            m_num_list_nonlocal.push_back(num_all_nonlocal);
        }
        
        m_num_all_nonlocal = num_all_nonlocal;
    }


    

#if NONLOCAL_SvF_PARALLEL==5

#if SvF2_USE_INTERPOLATION_NL != 5

    //実空間におけるnonlocal projectorとその微分を求めてバッファに格納する//
    void MakeProjectorInRealspace(const OneComplex* src_kspace, int sub_size_x, int sub_size_y, int sub_size_z, int proj_odd_flag, 
        double* dest_proj, double* dest_dpdx, double* dest_dpdy, double* dest_dpdz,
        FFTW_Executor* fftw, OneComplex* work) {

        auto* temp = work;
        const int sub_size = sub_size_x * sub_size_y * sub_size_z;
        const double invN = 1.0 / (double)sub_size;

        auto CopySubgrid_ZtoR = [](double* dest, OneComplex* src, double coef, int ksub_size_x, int ksub_size_y, int ksub_size_z) {
            for (int iz = 0; iz < ksub_size_z; ++iz) {
                for (int iy = 0; iy < ksub_size_y; ++iy) {
                    for (int ix = 0; ix < ksub_size_x; ++ix) {
                        const int i = ix + ksub_size_x * (iy + ksub_size_y * iz);
                        dest[i] = src[i].r * coef;
                    }
                }
            }
            };

        std::memcpy(temp, src_kspace, sizeof(OneComplex) * sub_size);

#if (SvF2_USE_INTERPOLATION_NL == 6)


        //const auto proj_odd_flag = nonlocal_kspace.odd_flag[n];
        switch (proj_odd_flag) {
        case ODD_YLM_X:
        {

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            std::memcpy(shifted_dp_dx, temp, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dx);

            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralXInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            GradientYInKspace(shifted_dp_dy, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);

            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            GradientZInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);


        }
        break;
#if 1
        case ODD_YLM_Y:
        {

            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            std::memcpy(shifted_dp_dy, temp, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralYInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            GradientXInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dx);
            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            GradientZInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);

        }
        break;
        case ODD_YLM_Z:
        {
            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            std::memcpy(shifted_dp_dz, temp, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dz);

            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralZInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            GradientXInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dx);
            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            GradientYInKspace(shifted_dp_dy, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);


        }
        break;
        case ODD_YLM_X | ODD_YLM_Y:
        {

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            IntegralYInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_dp_dx, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dx);
            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralXInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            GradientYInKspace(shifted_dp_dy, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);

            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            GradientZInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);


        }
        break;
        case ODD_YLM_Y | ODD_YLM_Z:
        {
            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            IntegralZInKspace(shifted_dp_dy, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_dp_dy, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralYInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            GradientXInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dx);
            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            GradientZInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);

        }
        break;
        case ODD_YLM_X | ODD_YLM_Z:
        {

            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            IntegralXInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_dp_dz, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralZInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            GradientXInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dx);
            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            GradientYInKspace(shifted_dp_dy, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);

        }
        break;
        case ODD_YLM_X | ODD_YLM_Y | ODD_YLM_Z:
        {

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            IntegralZInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_dp_dx, sizeof(OneComplex) * sub_size);
            IntegralYInKspace(shifted_dp_dx, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_dp_dx, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_dp_dx);
            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);

            //元の関数//
            auto* shifted_p_org = fftw->GetBuffer();
            IntegralXInKspace(shifted_p_org, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            std::memcpy(temp, shifted_p_org, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_p_org);
            CopySubgrid_ZtoR(dest_proj, shifted_p_org, invN, sub_size_x, sub_size_y, sub_size_z);

            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            GradientYInKspace(shifted_dp_dy, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);

            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            GradientZInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);

        }
        break;
#endif
        default:
#endif

        {
#ifdef FORCE_DIFF_PROJ

            //x微分//
            auto* shifted_dp_dx = fftw->GetBuffer();
            GradientXInKspace(shifted_dp_dx, (OneComplex*)temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);

            fftw->BackwardDirect(shifted_dp_dx);

            CopySubgrid_ZtoR(dest_dpdx, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);


            //y微分//
            auto* shifted_dp_dy = fftw->GetBuffer();
            GradientYInKspace(shifted_dp_dy, (OneComplex*)temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(dest_dpdy, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);
            //for (int i = 0; i < actual_sub_size; ++i) {
            //    dest_dpdy[i] = shifted_dp_dy[i][0] * invN;
            //}


            //z微分//
            auto* shifted_dp_dz = fftw->GetBuffer();
            GradientZInKspace(shifted_dp_dz, temp, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(dest_dpdz, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);
            //for (int i = 0; i < actual_sub_size; ++i) {
            //    dest_dpdz[i] = shifted_dp_dz[i][0] * invN;
            //}

#endif

            auto* shifted_proj = fftw->GetBuffer();
            //焼き戻し//
            std::memcpy(shifted_proj, temp, sizeof(OneComplex) * sub_size);
            fftw->BackwardDirect(shifted_proj);
            CopySubgrid_ZtoR(dest_proj, shifted_proj, invN, sub_size_x, sub_size_y, sub_size_z);

        }

#if (SvF2_USE_INTERPOLATION_NL == 6)
        }
#endif


    }


//すでにFFTされているk_projector_Yを、シフトしてから実空間に引き戻す
//mpi並列版: aomm4atomsで原子を共有しているprocess間で計算を分担//
void mCreateProjectorRYOnBlock_SvF_gamma_mpi5(const Nucleus* nuclei, int num_nuclei, const GridRangeMPI& l_grid) {
    /*SubgridBlock& nonlocal_blocks,
    PP_Proj_Ylm_each_atom& nonlocal_projectors,
    const Nucleus nucleus, const GridRangeMPI& l_grid, MPI_Comm& atom_comm) {
    */

#ifdef FORCE_DIFF_PROJ
    constexpr int DATA_FACTOR = 4;
#else
    constexpr int DATA_FACTOR = 1;
#endif

    //MPI送受信用バッファ確保//
    int all_num_procs = 0;
    for (int ni = 0; ni < num_nuclei; ++ni) {
        MPI_Comm atom_comm = m_comm4atoms.GetComm(ni);
        if (atom_comm == MPI_COMM_NULL) continue;
        all_num_procs += GetNumProcess(atom_comm);

    }

    auto all_send_heads = std::make_unique<int[]>(all_num_procs * 5);
    auto send_buffer_list = std::make_unique<double* []>(num_nuclei);

    int64_t send_heads_offset = 0;
    int num_valid = 0;
    auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei * 4);
    auto status4atoms = std::make_unique<MPI_Status[]>(num_nuclei * 4);


    for (const auto& ni : m_sorted_proc_ids) {
        send_buffer_list[ni] = nullptr;
        MPI_Comm atom_comm = m_comm4atoms.GetComm(ni);
        if (atom_comm == MPI_COMM_NULL)continue;

        const Nucleus& nucleus = nuclei[ni];
        SubgridBlock& nonlocal_blocks = m_nonlocal_blocks[ni];
        PP_Proj_Ylm_each_atom& nonlocal_projectors = m_nonlocal_projectors[ni];


        auto& nonlocal_kspace = m_nonlocal_kspace[nucleus.Z];
        const int num_proj_RY = nonlocal_kspace.num_proj_RY;
        nonlocal_projectors.num_proj_RY = num_proj_RY;
        const int64_t total_buf_size = nonlocal_blocks.grid_sizes * num_proj_RY;

#ifdef FORCE_DIFF_PROJ
        auto* buffer = gy::AlignedAlloc<double>(total_buf_size * 4);
        auto* buf_dp_dx = buffer + total_buf_size;
        auto* buf_dp_dy = buffer + total_buf_size * 2;
        auto* buf_dp_dz = buffer + total_buf_size * 3;
        nonlocal_projectors.projector_RY = buffer;
        nonlocal_projectors.d_projector_RY_dx = buf_dp_dx;
        nonlocal_projectors.d_projector_RY_dy = buf_dp_dy;
        nonlocal_projectors.d_projector_RY_dz = buf_dp_dz;


#else
        nonlocal_projectors.projector_RY = gy::AlignedAlloc<double>(total_buf_size);
        auto*& buffer = nonlocal_projectors.projector_RY;
#endif


        //Real space projector Shifted via FFT//
        const int irx = m_nucl_int_pos[ni].x;
        const int iry = m_nucl_int_pos[ni].y;
        const int irz = m_nucl_int_pos[ni].z;
        
#if (defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)) && !defined(SMOOTH_SUBGRID)
        const double drx = nucleus.Rx - (0.5 + (double)irx) * m_dx;
        const double dry = nucleus.Ry - (0.5 + (double)iry) * m_dy;
        const double drz = nucleus.Rz - (0.5 + (double)irz) * m_dz;
#else
        const double drx = nucleus.Rx - (double)irx * m_dx;
        const double dry = nucleus.Ry - (double)iry * m_dy;
        const double drz = nucleus.Rz - (double)irz * m_dz;
#endif


        const size_t total_block_size = nonlocal_blocks.grid_sizes;
        const auto& range_blocks = nonlocal_blocks.range_blocks;
        const auto& shifted_grid = nonlocal_blocks.shifted_grids;


        GridRange subgrid_actual = nonlocal_kspace.subgrid;
        subgrid_actual.begin_x += irx;
        subgrid_actual.end_x += irx;
        subgrid_actual.begin_y += iry;
        subgrid_actual.end_y += iry;
        subgrid_actual.begin_z += irz;
        subgrid_actual.end_z += irz;
#ifdef SMOOTH_SUBGRID
        const int sub_size_x = subgrid_actual.SizeX() - 1;
        const int sub_size_y = subgrid_actual.SizeY() - 1;
        const int sub_size_z = subgrid_actual.SizeZ() - 1;
#else
        const int sub_size_x = subgrid_actual.SizeX();
        const int sub_size_y = subgrid_actual.SizeY();
        const int sub_size_z = subgrid_actual.SizeZ();
#endif
        const int64_t ksub_size = sub_size_x * sub_size_y * sub_size_z;
        const int64_t actual_sub_size = subgrid_actual.Size3D();


        auto shifted_buffer = gy::make_unique_aligned<double[]>(actual_sub_size * DATA_FACTOR);
        send_buffer_list[ni] = gy::AlignedAlloc<double>(num_proj_RY * actual_sub_size * DATA_FACTOR);
        double* send_buffer = send_buffer_list[ni];
#ifdef FORCE_DIFF_PROJ
        double* send_buffer_x = &send_buffer[num_proj_RY * actual_sub_size];
        double* send_buffer_y = &send_buffer[num_proj_RY * actual_sub_size * 2];
        double* send_buffer_z = &send_buffer[num_proj_RY * actual_sub_size * 3];
        auto shifted_k_proj_mirror = gy::make_unique_aligned<OneComplex[]>(ksub_size);
#endif


        const int proc_id = GetProcessID(atom_comm);
        const int num_procs_on_atom = GetNumProcess(atom_comm);
        const int n_begin = (num_proj_RY * proc_id) / num_procs_on_atom;
        const int n_end = (num_proj_RY * (proc_id + 1)) / num_procs_on_atom;
#ifdef _DEBUG
        printf("[%d]TEST:k_begin,end: %d, %d\n", proc_id, n_begin, n_end);

        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif

        auto list = DDMOverlapChecker::GetOverlapProc3D(subgrid_actual, GridRange{ 0,0,0,m_grid.size_x,m_grid.size_y,m_grid.size_z },
            l_grid.num_split_x, l_grid.num_split_y, l_grid.num_split_z);
#ifdef _DEBUG
        printf("[%d]TEST:list before sort, %zd\n", proc_id, list.size());
        for (const auto& l : list) {
            printf("[%d]TEST:range, proc=%d, [%d,%d)x[%d,%d)x[%d,%d)\n", proc_id, l.proc_id, l.range.begin_x, l.range.end_x, l.range.begin_y, l.range.end_y, l.range.begin_z, l.range.end_z);
        }
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif

        DDMOverlapChecker::Sort(list);
#ifdef _DEBUG
        printf("[%d]TEST:list after sort, %zd\n", proc_id, list.size());
        for (const auto& l : list) {
            printf("[%d]TEST:range, proc=%d, [%d,%d)x[%d,%d)x[%d,%d)\n", proc_id, l.proc_id, l.range.begin_x, l.range.end_x, l.range.begin_y, l.range.end_y, l.range.begin_z, l.range.end_z);
        }
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif


        int* each_proc_sizes = &all_send_heads[send_heads_offset];
        int* send_heads = &all_send_heads[send_heads_offset + num_procs_on_atom];
        int* send_sizes = &all_send_heads[send_heads_offset + num_procs_on_atom * 2];
        int* recv_heads = &all_send_heads[send_heads_offset + num_procs_on_atom * 3];
        int* recv_sizes = &all_send_heads[send_heads_offset + num_procs_on_atom * 4];
        send_heads_offset += num_procs_on_atom * 5;
        {
            memset(each_proc_sizes, 0, sizeof(int) * num_procs_on_atom);
            int index = 0;
            int currect_id = list[0].proc_id;
            for (const auto& l : list) {
                if (l.proc_id > currect_id) {
                    ++index;
                    currect_id = l.proc_id;
                }
                each_proc_sizes[index] += l.range.Size3D();
            }
            int num = 0;
            for (int i = 0; i < num_procs_on_atom; ++i) {
                send_heads[i] = num;
                send_sizes[i] = each_proc_sizes[i] * (n_end - n_begin);
                num += send_sizes[i];
            }

#ifdef _DEBUG
            for (int i = 0; i < num_procs_on_atom; ++i) {
                printf("[%d]TESTmpi2: %d, %d, %d\n", proc_id, i, send_heads[i], send_sizes[i]);
            }
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif
        }

        watch_pp.Record(21);


        for (int n = n_begin; n < n_end; ++n) {

#ifdef FORCE_DIFF_PROJ
            double* shifted_proj_k = &shifted_buffer[0];
            double* shifted_dpdx_k = shifted_proj_k + actual_sub_size;
            double* shifted_dpdy_k = shifted_proj_k + actual_sub_size * 2;
            double* shifted_dpdz_k = shifted_proj_k + actual_sub_size * 3;
#else
            double* shifted_proj_k = &shifted_buffer[0];
#endif
            auto* k_proj_Y = nonlocal_kspace.k_projector_Ylm[n];



            auto*& fftw = nonlocal_kspace.fftw;
            OneComplex* shifted_k_proj = (OneComplex*)(fftw->GetBuffer());

            ShiftInKspace_set(
                shifted_k_proj, drx, dry, drz, k_proj_Y,
                sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


            MakeProjectorInRealspace(shifted_k_proj, sub_size_x, sub_size_y, sub_size_z,
                nonlocal_kspace.odd_flag[n],
                shifted_proj_k, shifted_dpdx_k, shifted_dpdy_k, shifted_dpdz_k, fftw, &shifted_k_proj_mirror[0]);

            watch_pp.Record(22);



            auto CopyRangeListFrom = [&](double* send_buffer, double* shifted_proj_k) {
                int index = 0;
                int currect_id = list[0].proc_id;
                int offset = each_proc_sizes[0] * (n - n_begin);
                for (const auto& l : list) {
                    if (l.proc_id > currect_id) {
                        ++index;
                        currect_id = l.proc_id;
                        offset = send_heads[index] + each_proc_sizes[index] * (n - n_begin);
                    }

                    GridRange org_range{ l.range.begin_x - l.shifted_grid.x,
                        l.range.begin_y - l.shifted_grid.y,
                        l.range.begin_z - l.shifted_grid.z,
                        l.range.end_x - l.shifted_grid.x,
                        l.range.end_y - l.shifted_grid.y,
                        l.range.end_z - l.shifted_grid.z };
                    CutSubgrid(org_range, &send_buffer[offset], subgrid_actual, shifted_proj_k);
                    offset += l.range.Size3D();
                }
                };
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
            CopyRangeListFrom(&send_buffer[0], shifted_proj_k);
#ifdef FORCE_DIFF_PROJ
            CopyRangeListFrom(&send_buffer_x[0], shifted_dpdx_k);
            CopyRangeListFrom(&send_buffer_y[0], shifted_dpdy_k);
            CopyRangeListFrom(&send_buffer_z[0], shifted_dpdz_k);
#endif
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);


            watch_pp.Record(24);
        }

#ifdef _DEBUG
        printf("[%d]TEST:after shift and cut\n", proc_id);
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif


        //MPI通信//分担して求めたshift座標を共有//
        {
            int head = 0;
            for (int p = 0; p < num_procs_on_atom; ++p) {
                recv_heads[p] = head;
                head = ((num_proj_RY * (p + 1)) / num_procs_on_atom) * each_proc_sizes[proc_id];
                recv_sizes[p] = head - recv_heads[p];
            }
        }
#ifdef _DEBUG
        for (int i = 0; i < num_procs_on_atom; ++i) {
            printf("[%d]TEST:recv_heads,sizes: %d, %d, %d\n", proc_id, i, recv_heads[i], recv_sizes[i]);
        }
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif
#ifdef FORCE_DIFF_PROJ
#if 0
        MPI_Alltoallv(&send_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buffer[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm);
        MPI_Alltoallv(&send_buffer_x[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dx[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm);
        MPI_Alltoallv(&send_buffer_y[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dy[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm);
        MPI_Alltoallv(&send_buffer_z[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dz[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm);

#else
        MPI_Ialltoallv(&send_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buffer[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid]);
        MPI_Ialltoallv(&send_buffer_x[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dx[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid + 1]);
        MPI_Ialltoallv(&send_buffer_y[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dy[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid + 2]);
        MPI_Ialltoallv(&send_buffer_z[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dz[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid + 3]);
#endif
        num_valid += 4;
#else
        MPI_Ialltoallv(&send_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buffer[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid]);
        ++num_valid;
#endif
        watch_pp.Record(23);
        ////////////////
#ifdef _DEBUG
        printf("[%d]TEST:after Alltoall\n", proc_id);
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif
    }
#if 1
    MPI_Waitall(num_valid, &request4atoms[0], &status4atoms[0]);
    watch_pp.Record(23);
#endif
    for (int ni = 0; ni < num_nuclei; ++ni) {
        gy::AlignedFree(send_buffer_list[ni]);
    }
}

#else


//すでにFFTされているk_projector_Yを、シフトしてから実空間に引き戻す
//mpi並列版: aomm4atomsで原子を共有しているprocess間で計算を分担//
void mCreateProjectorRYOnBlock_SvF_gamma_mpi5(const Nucleus* nuclei, int num_nuclei, const GridRangeMPI& l_grid) {
    /*SubgridBlock& nonlocal_blocks,
    PP_Proj_Ylm_each_atom& nonlocal_projectors,
    const Nucleus nucleus, const GridRangeMPI& l_grid, MPI_Comm& atom_comm) {
    */

#ifdef FORCE_DIFF_PROJ
    constexpr int DATA_FACTOR = 4;
#else
    constexpr int DATA_FACTOR = 1;
#endif

    //MPI送受信用バッファ確保//
    int all_num_procs = 0;
    for (int ni = 0; ni < num_nuclei; ++ni) {
        MPI_Comm atom_comm = m_comm4atoms.GetComm(ni);
        if (atom_comm == MPI_COMM_NULL) continue;
        all_num_procs += GetNumProcess(atom_comm);

    }

    auto all_send_heads = std::make_unique<int[]>(all_num_procs * 5);
    auto send_buffer_list = std::make_unique<double* []>(num_nuclei);

    int64_t send_heads_offset = 0;
    int num_valid = 0;
    auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei * 4);
    auto status4atoms = std::make_unique<MPI_Status[]>(num_nuclei * 4);


    for (const auto& ni : m_sorted_proc_ids) {
        send_buffer_list[ni] = nullptr;
        MPI_Comm atom_comm = m_comm4atoms.GetComm(ni);
        if (atom_comm == MPI_COMM_NULL)continue;

        const Nucleus& nucleus = nuclei[ni];
        SubgridBlock& nonlocal_blocks = m_nonlocal_blocks[ni];
        PP_Proj_Ylm_each_atom& nonlocal_projectors = m_nonlocal_projectors[ni];


        auto& nonlocal_kspace = m_nonlocal_kspace[nucleus.Z];
        const int num_proj_RY = nonlocal_kspace.num_proj_RY;
        nonlocal_projectors.num_proj_RY = num_proj_RY;
        const int64_t total_buf_size = nonlocal_blocks.grid_sizes * num_proj_RY;

#ifdef FORCE_DIFF_PROJ
        auto* buffer = newgy::AlignedAlloc<double>(total_buf_size * 4);
        auto* buf_dp_dx = buffer + total_buf_size;
        auto* buf_dp_dy = buffer + total_buf_size * 2;
        auto* buf_dp_dz = buffer + total_buf_size * 3;
        nonlocal_projectors.projector_RY = buffer;
        nonlocal_projectors.d_projector_RY_dx = buf_dp_dx;
        nonlocal_projectors.d_projector_RY_dy = buf_dp_dy;
        nonlocal_projectors.d_projector_RY_dz = buf_dp_dz;


#else
        nonlocal_projectors.projector_RY = gy::AlignedAlloc<double>(total_buf_size);
        auto*& buffer = nonlocal_projectors.projector_RY;
#endif


        //Real space projector Shifted via FFT//
        const int irx = m_nucl_int_pos[ni].x;
        const int iry = m_nucl_int_pos[ni].y;
        const int irz = m_nucl_int_pos[ni].z;
        

#if (defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)) && !defined(SMOOTH_SUBGRID)
        const double drx = nucleus.Rx - (0.5 + (double)irx) * m_dx;
        const double dry = nucleus.Ry - (0.5 + (double)iry) * m_dy;
        const double drz = nucleus.Rz - (0.5 + (double)irz) * m_dz;
#else
        const double drx = nucleus.Rx - (double)irx * m_dx;
        const double dry = nucleus.Ry - (double)iry * m_dy;
        const double drz = nucleus.Rz - (double)irz * m_dz;
#endif


        const size_t total_block_size = nonlocal_blocks.grid_sizes;
        const auto& range_blocks = nonlocal_blocks.range_blocks;
        const auto& shifted_grid = nonlocal_blocks.shifted_grids;


        GridRange subgrid_actual = nonlocal_kspace.subgrid;
        subgrid_actual.begin_x += irx;
        subgrid_actual.end_x += irx;
        subgrid_actual.begin_y += iry;
        subgrid_actual.end_y += iry;
        subgrid_actual.begin_z += irz;
        subgrid_actual.end_z += irz;
#ifdef SMOOTH_SUBGRID
        const int sub_size_x = subgrid_actual.SizeX() - 1;
        const int sub_size_y = subgrid_actual.SizeY() - 1;
        const int sub_size_z = subgrid_actual.SizeZ() - 1;
#else
        const int sub_size_x = subgrid_actual.SizeX();
        const int sub_size_y = subgrid_actual.SizeY();
        const int sub_size_z = subgrid_actual.SizeZ();
#endif
        const int64_t ksub_size = sub_size_x * sub_size_y * sub_size_z;
        const int64_t actual_sub_size = subgrid_actual.Size3D();


        auto shifted_buffer = gy::make_unique_aligned<double[]>(actual_sub_size * DATA_FACTOR);
        send_buffer_list[ni] = new double[num_proj_RY * actual_sub_size * DATA_FACTOR];
        double* send_buffer = send_buffer_list[ni];
#ifdef FORCE_DIFF_PROJ
        double* send_buffer_x = &send_buffer[num_proj_RY * actual_sub_size];
        double* send_buffer_y = &send_buffer[num_proj_RY * actual_sub_size * 2];
        double* send_buffer_z = &send_buffer[num_proj_RY * actual_sub_size * 3];
        auto shifted_k_proj_mirror = gy::make_unique_aligned<OneComplex[]>(ksub_size);
#endif


        const int proc_id = GetProcessID(atom_comm);
        const int num_procs_on_atom = GetNumProcess(atom_comm);
        const int n_begin = (num_proj_RY * proc_id) / num_procs_on_atom;
        const int n_end = (num_proj_RY * (proc_id + 1)) / num_procs_on_atom;
#ifdef _DEBUG
        printf("[%d]TEST:k_begin,end: %d, %d\n", proc_id, n_begin, n_end);

        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif

        auto list = DDMOverlapChecker::GetOverlapProc3D(subgrid_actual, GridRange{ 0,0,0,m_grid.size_x,m_grid.size_y,m_grid.size_z },
            l_grid.num_split_x, l_grid.num_split_y, l_grid.num_split_z);
#ifdef _DEBUG
        printf("[%d]TEST:list before sort, %zd\n", proc_id, list.size());
        for (const auto& l : list) {
            printf("[%d]TEST:range, proc=%d, [%d,%d)x[%d,%d)x[%d,%d)\n", proc_id, l.proc_id, l.range.begin_x, l.range.end_x, l.range.begin_y, l.range.end_y, l.range.begin_z, l.range.end_z);
        }
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif

        DDMOverlapChecker::Sort(list);
#ifdef _DEBUG
        printf("[%d]TEST:list after sort, %zd\n", proc_id, list.size());
        for (const auto& l : list) {
            printf("[%d]TEST:range, proc=%d, [%d,%d)x[%d,%d)x[%d,%d)\n", proc_id, l.proc_id, l.range.begin_x, l.range.end_x, l.range.begin_y, l.range.end_y, l.range.begin_z, l.range.end_z);
        }
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif


        int* each_proc_sizes = &all_send_heads[send_heads_offset];
        int* send_heads = &all_send_heads[send_heads_offset + num_procs_on_atom];
        int* send_sizes = &all_send_heads[send_heads_offset + num_procs_on_atom * 2];
        int* recv_heads = &all_send_heads[send_heads_offset + num_procs_on_atom * 3];
        int* recv_sizes = &all_send_heads[send_heads_offset + num_procs_on_atom * 4];
        send_heads_offset += num_procs_on_atom * 5;
        {
            memset(each_proc_sizes, 0, sizeof(int) * num_procs_on_atom);
            int index = 0;
            int currect_id = list[0].proc_id;
            for (const auto& l : list) {
                if (l.proc_id > currect_id) {
                    ++index;
                    currect_id = l.proc_id;
                }
                each_proc_sizes[index] += l.range.Size3D();
            }
            int num = 0;
            for (int i = 0; i < num_procs_on_atom; ++i) {
                send_heads[i] = num;
                send_sizes[i] = each_proc_sizes[i] * (n_end - n_begin);
                num += send_sizes[i];
            }

#ifdef _DEBUG
            for (int i = 0; i < num_procs_on_atom; ++i) {
                printf("[%d]TESTmpi2: %d, %d, %d\n", proc_id, i, send_heads[i], send_sizes[i]);
            }
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif
        }

        watch_pp.Record(21);

        auto filter_xyz = gy::make_unique_aligned<double[]>(actual_sub_size * 4);
        double* filter_x = &filter_xyz[0];
        double* filter_y = &filter_xyz[actual_sub_size];
        double* filter_z = &filter_xyz[actual_sub_size * 2];
        double* p_real = &filter_xyz[actual_sub_size * 3];
        {
            auto SetRY = [&](auto filter, double* buf) {
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    for (int iy = 0; iy < sub_size_y; ++iy) {
                        for (int ix = 0; ix < sub_size_x; ++ix) {
                            const int i = ix + sub_size_x * (iy + sub_size_y * iz);
                            const double f = filter(ix + subgrid_actual.begin_x, iy + subgrid_actual.begin_y, iz + subgrid_actual.begin_z);
                            buf[i] = f;
                        }
                    }
                }
                };

            SetRY([&](int ix, int iy, int iz) {
                return ((double)ix * m_dx - nucleus.Rx);
                }, filter_x);

            SetRY([&](int ix, int iy, int iz) {
                return ((double)iy * m_dy - nucleus.Ry);
                }, filter_y);

            SetRY([&](int ix, int iy, int iz) {
                return ((double)iz * m_dz - nucleus.Rz);
                }, filter_z);

        }


        for (int n = n_begin; n < n_end; ++n) {

#ifdef FORCE_DIFF_PROJ
            double* shifted_proj_k = &shifted_buffer[0];
            double* shifted_dpdx_k = shifted_proj_k + actual_sub_size;
            double* shifted_dpdy_k = shifted_proj_k + actual_sub_size * 2;
            double* shifted_dpdz_k = shifted_proj_k + actual_sub_size * 3;
#else
            double* shifted_proj_k = &shifted_buffer[0];
#endif
            auto* k_proj_Y = nonlocal_kspace.k_projector_Ylm[n];
            const int proj_odd_flag = nonlocal_kspace.odd_flag[n];


            auto CopySubgrid_ZtoR = [](double* dest, fftw_complex* src, double coef, int ksub_size_x, int ksub_size_y, int ksub_size_z) {
                for (int iz = 0; iz < ksub_size_z; ++iz) {
                    for (int iy = 0; iy < ksub_size_y; ++iy) {
                        for (int ix = 0; ix < ksub_size_x; ++ix) {
                            const int i = ix + ksub_size_x * (iy + ksub_size_y * iz);
                            dest[i] = src[i][0] * coef;
                        }
                    }
                }
                };

            auto*& fftw = nonlocal_kspace.fftw;
            OneComplex* shifted_k_proj = (OneComplex*)(fftw->GetBuffer());

            ShiftInKspace_set(
                shifted_k_proj, drx, dry, drz, k_proj_Y,
                sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


            std::memcpy(&shifted_k_proj_mirror[0], shifted_k_proj, sizeof(OneComplex) * ksub_size);

            const double invN = 1.0 / (double)(ksub_size);

            fftw->BackwardDirect(shifted_k_proj);
            auto* shifted_proj = fftw->GetBuffer();
            CopySubgrid_ZtoR(p_real, shifted_proj, invN, sub_size_x, sub_size_y, sub_size_z);


            auto AddSubgrid_ZtoR = [](double* dest, const double* src, double coef, int ksub_size) {
                for (int i = 0; i < ksub_size; ++i) {
                    dest[i] += src[i] * coef;
                }
                };


            auto MulSubgrid_R = [](double* dest, const double* fact, int ksub_size) {
                for (int i = 0; i < ksub_size; ++i) {
                    dest[i] *= fact[i];
                }
                };

#ifdef FORCE_DIFF_PROJ




            //x微分//
            GradientXInKspace((fftw_complex*)&shifted_k_proj[0], (fftw_complex*)&shifted_k_proj_mirror[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


            auto* shifted_dp_dx = fftw->GetBuffer();
            fftw->BackwardDirect(shifted_dp_dx);




            //q(x) = x * dp/dx + p;
            CopySubgrid_ZtoR(shifted_dpdx_k, shifted_dp_dx, invN, sub_size_x, sub_size_y, sub_size_z);
            if (proj_odd_flag & ODD_YLM_X) {
                MulSubgrid_R(shifted_dpdx_k, filter_x, actual_sub_size);
                AddSubgrid_ZtoR(shifted_dpdx_k, p_real, 1.0, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_Y) {
                MulSubgrid_R(shifted_dpdx_k, filter_y, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_Z) {
                MulSubgrid_R(shifted_dpdx_k, filter_z, actual_sub_size);
            }





            //y微分//
            GradientYInKspace((fftw_complex*)&shifted_k_proj[0], (fftw_complex*)&shifted_k_proj_mirror[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);

            auto* shifted_dp_dy = fftw->GetBuffer();
            fftw->BackwardDirect(shifted_dp_dy);
            CopySubgrid_ZtoR(shifted_dpdy_k, shifted_dp_dy, invN, sub_size_x, sub_size_y, sub_size_z);
            if (proj_odd_flag & ODD_YLM_Y) {
                MulSubgrid_R(shifted_dpdy_k, filter_y, actual_sub_size);
                AddSubgrid_ZtoR(shifted_dpdy_k, p_real, 1.0, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_X) {
                MulSubgrid_R(shifted_dpdy_k, filter_x, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_Z) {
                MulSubgrid_R(shifted_dpdy_k, filter_z, actual_sub_size);
            }



            //z微分//
            GradientZInKspace((fftw_complex*)&shifted_k_proj[0], (fftw_complex*)&shifted_k_proj_mirror[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


            auto* shifted_dp_dz = fftw->GetBuffer();
            fftw->BackwardDirect(shifted_dp_dz);
            CopySubgrid_ZtoR(shifted_dpdz_k, shifted_dp_dz, invN, sub_size_x, sub_size_y, sub_size_z);
            if (proj_odd_flag & ODD_YLM_Z) {
                MulSubgrid_R(shifted_dpdz_k, filter_z, actual_sub_size);
                AddSubgrid_ZtoR(shifted_dpdz_k, p_real, 1.0, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_X) {
                MulSubgrid_R(shifted_dpdz_k, filter_x, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_Y) {
                MulSubgrid_R(shifted_dpdz_k, filter_y, actual_sub_size);
            }

#endif


            std::memcpy(shifted_proj_k, p_real, sizeof(double) * ksub_size);
            if (proj_odd_flag & ODD_YLM_X) {
                MulSubgrid_R(shifted_proj_k, filter_x, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_Y) {
                MulSubgrid_R(shifted_proj_k, filter_y, actual_sub_size);
            }
            if (proj_odd_flag & ODD_YLM_Z) {
                MulSubgrid_R(shifted_proj_k, filter_z, actual_sub_size);
            }


            watch_pp.Record(22);





            auto CopyRangeListFrom = [&](double* send_buffer, double* shifted_proj_k) {
                int index = 0;
                int currect_id = list[0].proc_id;
                int offset = each_proc_sizes[0] * (n - n_begin);
                for (const auto& l : list) {
                    if (l.proc_id > currect_id) {
                        ++index;
                        currect_id = l.proc_id;
                        offset = send_heads[index] + each_proc_sizes[index] * (n - n_begin);

                    }

                    GridRange org_range{ l.range.begin_x - l.shifted_grid.x,
                        l.range.begin_y - l.shifted_grid.y,
                        l.range.begin_z - l.shifted_grid.z,
                        l.range.end_x - l.shifted_grid.x,
                        l.range.end_y - l.shifted_grid.y,
                        l.range.end_z - l.shifted_grid.z };
                    CutSubgrid(org_range, &send_buffer[offset], subgrid_actual, shifted_proj_k);
                    offset += l.range.Size3D();
                }
                };

            CopyRangeListFrom(&send_buffer[0], shifted_proj_k);
#ifdef FORCE_DIFF_PROJ
            CopyRangeListFrom(&send_buffer_x[0], shifted_dpdx_k);
            CopyRangeListFrom(&send_buffer_y[0], shifted_dpdy_k);
            CopyRangeListFrom(&send_buffer_z[0], shifted_dpdz_k);
#endif



            watch_pp.Record(24);
        }

#ifdef _DEBUG
        printf("[%d]TEST:after shift and cut\n", proc_id);
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif


        //MPI通信//分担して求めたshift座標を共有//
        {
            int head = 0;
            for (int p = 0; p < num_procs_on_atom; ++p) {
                recv_heads[p] = head;
                head = ((num_proj_RY * (p + 1)) / num_procs_on_atom) * each_proc_sizes[proc_id];
                recv_sizes[p] = head - recv_heads[p];
            }
        }
#ifdef _DEBUG
        for (int i = 0; i < num_procs_on_atom; ++i) {
            printf("[%d]TEST:recv_heads,sizes: %d, %d, %d\n", proc_id, i, recv_heads[i], recv_sizes[i]);
        }
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif
#ifdef FORCE_DIFF_PROJ
        MPI_Ialltoallv(&send_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buffer[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid]);
        MPI_Ialltoallv(&send_buffer_x[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dx[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid + 1]);
        MPI_Ialltoallv(&send_buffer_y[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dy[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid + 2]);
        MPI_Ialltoallv(&send_buffer_z[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buf_dp_dz[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid + 3]);
        num_valid += 4;
#else
        MPI_Ialltoallv(&send_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buffer[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid]);
        ++num_valid;
#endif
        watch_pp.Record(23);
        ////////////////
#ifdef _DEBUG
        printf("[%d]TEST:after Alltoall\n", proc_id);
        fflush(stdout);
        MPI_Barrier(atom_comm);
#endif
    }

    MPI_Waitall(num_valid, &request4atoms[0], &status4atoms[0]);
    watch_pp.Record(23);

    for (int ni = 0; ni < num_nuclei; ++ni) {
        delete[] send_buffer_list[ni];
    }
}

#endif   //SvF2_USE_INTERPOLATION_NL
#endif  //NONLOCAL_SvF_PARALLEL


    

#if NONLOCAL_SvF_PARALLEL==7

    //すでにFFTされているk_projector_Yを、シフトしてから実空間に引き戻す
    //mpi並列版: aomm4atomsで原子を共有しているprocess間で計算を分担//    
    void mCreateProjectorRYOnBlock_SvF_gamma_mpi7(const Nucleus* nuclei, int num_nuclei, const GridRangeMPI& l_grid) {
        /*SubgridBlock& nonlocal_blocks,
        PP_Proj_Ylm_each_atom& nonlocal_projectors,
        const Nucleus nucleus, const GridRangeMPI& l_grid, MPI_Comm& atom_comm) {
        */

#ifdef FORCE_DIFF_PROJ
        constexpr int DATA_FACTOR = 4;
#else
        constexpr int DATA_FACTOR = 1;
#endif

        //MPI送受信用バッファ確保//
        int all_num_procs = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            MPI_Comm atom_comm = m_comm4atoms.GetComm(ni);
            if (atom_comm == MPI_COMM_NULL) continue;
            all_num_procs += GetNumProcess(atom_comm);

        }

        auto all_send_heads = std::make_unique<int[]>(all_num_procs * 5);
        auto send_buffer_list = std::unique_array<double*[]>(num_nuclei);

        int64_t send_heads_offset = 0;
        int num_valid = 0;
        auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei);
        auto status4atoms = std::make_unique<MPI_Status[]>(num_nuclei);


        for (const auto& ni : m_sorted_proc_ids) {
            send_buffer_list[ni] = nullptr;
            MPI_Comm atom_comm = m_comm4atoms.GetComm(ni);
            if (atom_comm == MPI_COMM_NULL)continue;

            const Nucleus& nucleus = nuclei[ni];
            SubgridBlock& nonlocal_blocks = m_nonlocal_blocks[ni];
            PP_Proj_Ylm_each_atom& nonlocal_projectors = m_nonlocal_projectors[ni];


            auto& nonlocal_kspace = m_nonlocal_kspace[nucleus.Z];
            const int num_proj_RY = nonlocal_kspace.num_proj_RY;
            nonlocal_projectors.num_proj_RY = num_proj_RY;
            const int64_t total_buf_size = nonlocal_blocks.grid_sizes * num_proj_RY;

#ifdef FORCE_DIFF_PROJ
            auto* buffer = gy::AlignedAlloc<double>(total_buf_size * 4);
            auto* buf_dp_dx = buffer + total_buf_size;
            auto* buf_dp_dy = buffer + total_buf_size * 2;
            auto* buf_dp_dz = buffer + total_buf_size * 3;
            nonlocal_projectors.projector_RY = buffer;
            nonlocal_projectors.d_projector_RY_dx = buf_dp_dx;
            nonlocal_projectors.d_projector_RY_dy = buf_dp_dy;
            nonlocal_projectors.d_projector_RY_dz = buf_dp_dz;


#else
            nonlocal_projectors.projector_RY = gy::AlignedAlloc<double>(total_buf_size);
            auto*& buffer = nonlocal_projectors.projector_RY;
#endif


            //Real space projector Shifted via FFT//
            const int irx = m_nucl_int_pos[ni].x;
            const int iry = m_nucl_int_pos[ni].y;
            const int irz = m_nucl_int_pos[ni].z;
            /*
                        const int irx = GridPosForSvF_NL(nucleus.Rx, m_dx);
                        const int iry = GridPosForSvF_NL(nucleus.Ry, m_dy);
                        const int irz = GridPosForSvF_NL(nucleus.Rz, m_dz);
                        */
                        //printf("INTPOS-cp %d: %d, %d, %d\n", ni, irx, iry, irz);

#if (defined(SvF2_SUBGRID_EVEN_HALF) || defined(SvF2_SUBGRID_ODD_HALF)) && !defined(SMOOTH_SUBGRID)
            const double drx = nucleus.Rx - (0.5 + (double)irx) * m_dx;
            const double dry = nucleus.Ry - (0.5 + (double)iry) * m_dy;
            const double drz = nucleus.Rz - (0.5 + (double)irz) * m_dz;
#else
            const double drx = nucleus.Rx - (double)irx * m_dx;
            const double dry = nucleus.Ry - (double)iry * m_dy;
            const double drz = nucleus.Rz - (double)irz * m_dz;
#endif


            const size_t total_block_size = nonlocal_blocks.grid_sizes;
            const auto& range_blocks = nonlocal_blocks.range_blocks;
            const auto& shifted_grid = nonlocal_blocks.shifted_grids;


            GridRange subgrid_actual = nonlocal_kspace.subgrid;
            subgrid_actual.begin_x += irx;
            subgrid_actual.end_x += irx;
            subgrid_actual.begin_y += iry;
            subgrid_actual.end_y += iry;
            subgrid_actual.begin_z += irz;
            subgrid_actual.end_z += irz;
            const int sub_size_x = subgrid_actual.SizeX();
            const int sub_size_y = subgrid_actual.SizeY();
            const int sub_size_z = subgrid_actual.SizeZ();

            const int64_t sub_size = sub_size_x * sub_size_y * sub_size_z;


            auto shifted_buffer = gy::make_unique_aligned<double[]>(sub_size);
            send_buffer_list[ni] = new double[num_proj_RY * sub_size * DATA_FACTOR];
            double* send_buffer = send_buffer_list[ni];


            const int proc_id = GetProcessID(atom_comm);
            const int num_procs_on_atom = GetNumProcess(atom_comm);
            const int k_begin = (num_proj_RY * DATA_FACTOR * proc_id) / num_procs_on_atom;
            const int k_end = (num_proj_RY * DATA_FACTOR * (proc_id + 1)) / num_procs_on_atom;

#ifdef _DEBUG
            printf("[%d]TEST:k_begin,end: %d, %d\n", proc_id, k_begin, k_end);

            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif

            auto list = DDMOverlapChecker::GetOverlapProc3D(subgrid_actual, GridRange{ 0,0,0,m_grid.size_x,m_grid.size_y,m_grid.size_z },
                l_grid.num_split_x, l_grid.num_split_y, l_grid.num_split_z);
#ifdef _DEBUG
            printf("[%d]TEST:list before sort, %zd\n", proc_id, list.size());
            for (const auto& l : list) {
                printf("[%d]TEST:range, proc=%d, [%d,%d)x[%d,%d)x[%d,%d)\n", proc_id, l.proc_id, l.range.begin_x, l.range.end_x, l.range.begin_y, l.range.end_y, l.range.begin_z, l.range.end_z);
            }
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif

            DDMOverlapChecker::Sort(list);
#ifdef _DEBUG
            printf("[%d]TEST:list after sort, %zd\n", proc_id, list.size());
            for (const auto& l : list) {
                printf("[%d]TEST:range, proc=%d, [%d,%d)x[%d,%d)x[%d,%d)\n", proc_id, l.proc_id, l.range.begin_x, l.range.end_x, l.range.begin_y, l.range.end_y, l.range.begin_z, l.range.end_z);
            }
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif


            int* each_proc_sizes = &all_send_heads[send_heads_offset];
            int* send_heads = &all_send_heads[send_heads_offset + num_procs_on_atom];
            int* send_sizes = &all_send_heads[send_heads_offset + num_procs_on_atom * 2];
            int* recv_heads = &all_send_heads[send_heads_offset + num_procs_on_atom * 3];
            int* recv_sizes = &all_send_heads[send_heads_offset + num_procs_on_atom * 4];
            send_heads_offset += num_procs_on_atom * 5;
            {
                memset(each_proc_sizes, 0, sizeof(int) * num_procs_on_atom);
                int index = 0;
                int currect_id = list[0].proc_id;
                for (const auto& l : list) {
                    if (l.proc_id > currect_id) {
                        ++index;
                        currect_id = l.proc_id;
                    }
                    each_proc_sizes[index] += l.range.Size3D();
                }
                int num = 0;
                for (int i = 0; i < num_procs_on_atom; ++i) {
                    send_heads[i] = num;
                    send_sizes[i] = each_proc_sizes[i] * (k_end - k_begin);
                    num += send_sizes[i];
                }

#ifdef _DEBUG
                for (int i = 0; i < num_procs_on_atom; ++i) {
                    printf("[%d]TESTmpi2: %d, %d, %d\n", proc_id, i, send_heads[i], send_sizes[i]);
                }
                fflush(stdout);
                MPI_Barrier(atom_comm);
#endif
            }

            watch_pp.Record(21);

            for (int k = k_begin; k < k_end; ++k) {
                const int k_mod = k % num_proj_RY;

                double* shifted_proj_k = &shifted_buffer[0];
                auto* k_proj_Y = nonlocal_kspace.k_projector_Ylm[k_mod];



                auto*& fftw = nonlocal_kspace.fftw;
                auto* shifted_proj = fftw->GetBuffer();

                ShiftInKspace_set(
                    (OneComplex*)shifted_proj, drx, dry, drz, k_proj_Y,
                    sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);


                const double invN = 1.0 / (double)(sub_size);

                switch (k / num_proj_RY) {
                case 1: 
                {
                    //x微分//
                    GradientXInKspace_inplace((fftw_complex*)shifted_proj, sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
                break;
                case 2:
                {
                    //y微分//
                    GradientYInKspace_inplace((fftw_complex*)&shifted_proj[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
                break;
                case 3:
                {
                    //z微分//
                    GradientZInKspace_inplace((fftw_complex*)&shifted_proj[0], sub_size_x, sub_size_y, sub_size_z, m_dx, m_dy, m_dz);
                }
                default:
                    //nothing to do//
                break;
                }


                fftw->BackwardDirect(shifted_proj);
                for (int i = 0; i < sub_size; ++i) {
                    shifted_proj_k[i] = shifted_proj[i][0] * invN;
                }



#ifdef _DEBUG

                {
                    auto proj_pre = gy::make_unique_aligned<double[]>(sub_size_x * sub_size_y);
                    for (int i = 0; i < sub_size_x * sub_size_y; ++i) {
                        proj_pre[i] = shifted_proj_k[i + sub_size_x * sub_size_y * (sub_size_z / 2 + 1)];
                    }
                    std::string filename3("cut_" + std::to_string(ni) + "-" + std::to_string(k) + "_zhalf.txt");
                    OutputMatrix(&proj_pre[0], sub_size_x, sub_size_y, filename3.c_str());
                }

#endif




                auto CopyRangeListFrom = [&](double* send_buffer, double* shifted_proj_k) {
                    int index = 0;
                    int currect_id = list[0].proc_id;
                    int offset = each_proc_sizes[0] * (k - k_begin);
                    for (const auto& l : list) {
                        if (l.proc_id > currect_id) {
                            ++index;
                            currect_id = l.proc_id;
                            offset = send_heads[index] + each_proc_sizes[index] * (k - k_begin);

                        }

                        GridRange org_range{ l.range.begin_x - l.shifted_grid.x,
                            l.range.begin_y - l.shifted_grid.y,
                            l.range.begin_z - l.shifted_grid.z,
                            l.range.end_x - l.shifted_grid.x,
                            l.range.end_y - l.shifted_grid.y,
                            l.range.end_z - l.shifted_grid.z };
                        CutSubgrid(org_range, &send_buffer[offset], subgrid_actual, shifted_proj_k);
                        offset += l.range.Size3D();
                    }
                    };

                CopyRangeListFrom(&send_buffer[0], shifted_proj_k);


                watch_pp.Record(24);
            }

#ifdef _DEBUG
            printf("[%d]TEST:after shift and cut\n", proc_id);
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif


            //MPI通信//分担して求めたshift座標を共有//
            {
                int head = 0;
                for (int p = 0; p < num_procs_on_atom; ++p) {
                    recv_heads[p] = head;
                    head = ((num_proj_RY * DATA_FACTOR * (p + 1)) / num_procs_on_atom) * each_proc_sizes[proc_id];
                    recv_sizes[p] = head - recv_heads[p];
                }
            }
#ifdef _DEBUG
            for (int i = 0; i < num_procs_on_atom; ++i) {
                printf("[%d]TEST:recv_heads,sizes: %d, %d, %d\n", proc_id, i, recv_heads[i], recv_sizes[i]);
            }
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif

            MPI_Ialltoallv(&send_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, &buffer[0], recv_sizes, &recv_heads[0], MPI_DOUBLE, atom_comm, &request4atoms[num_valid]);
            ++num_valid;

            watch_pp.Record(23);
            ////////////////
#ifdef _DEBUG
            printf("[%d]TEST:after Alltoall: %d\n", proc_id, ni);
            fflush(stdout);
            MPI_Barrier(atom_comm);
#endif
        }

        MPI_Waitall(num_valid, &request4atoms[0], &status4atoms[0]);
        watch_pp.Record(23);

        for (int ni = 0; ni < num_nuclei; ++ni) {
            delete[] send_buffer_list[ni];
        }
    }

#endif


public:

    /*
    * 擬ポテンシャルを原子核位置に合わせて必要な情報を用意.
    * 
    * 呼び出しスキーム
    * qumasun_hogeクラス
    * --> UpdatePosition: この関数
    *   --> GridPosForSvF_NL: 粒子位置から離散グリッド点を算出
    *   --> MPI_Bcastで全プロセスで同期(これをしないと稀にグリッドがプロセス間でズレてbugる. 一度遭遇.)
    *   --> UpdatePositionLocal_2: Vlocalの核電荷とPCCの射影関数の更新
    *   ** 以降はnonlocal用の処理
    *   --> 離散グリッドが更新されていたとき、nonlocal用のサブグリッドBlockを生成
    *   --> mSetNonlocalProjectorInfo: 実質空の関数
    *   --> m_comm4atomsの更新. nonlocal projectorのサブグリッドとオーバーラップするプロセスだけを束ねたmpi_comm
    *   --> mCreateNonoverlapList: nonlocal projectorのサブグリッドが重なっている原子同士で束ねる. symplecticに必要
    *   --> mCreateProjectorRYOnBlock_SvF_gamma_mpi5: projectorをブロック状の格子点に貼り付け
    */
    void UpdatePosition(const Nucleus* nuclei, int num_nuclei, const GridRangeMPI& l_grid) {
        watch_pp.Restart();
        const int proc_id = GetProcessID(l_grid.mpi_comm);

        m_max_grid_size = 0;
        //m_hr_max_grid_size = 0;


        CountNonlocalProjector(nuclei, num_nuclei);

        Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
        watch_pp.Record(14);

        m_nonlocal_projectors.clear();
        m_nonlocal_projectors.resize(num_nuclei);

        if (m_nonlocal_blocks.size() != num_nuclei) {
            m_nonlocal_blocks.resize(num_nuclei);
        }

               
        //local termのsubgridと場の値の更新//
        const bool is_reset_at_least_one = UpdatePositionLocal_2(nuclei, num_nuclei, l_grid);
        //is_reset_at_least_one には原子ごとのprojectorのsubgrid領域が更新されているかが返る///////////////

        Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
        watch_pp.Record(15);

        auto is_nucl_overlap = std::make_unique<int[]>(num_nuclei);
        
        //nonlocal termのsubgridと場の値の更新//
        for (int ni = 0; ni < num_nuclei; ++ni) {

            //set m_vlocal_block and m_nonlocal_block//
            if (m_is_update_int_pos[ni]) {
                m_nonlocal_blocks[ni].range_blocks.clear();
                m_nonlocal_blocks[ni].shifted_grids.clear();
                mSetSubspaceBlock2(m_nonlocal_blocks[ni], m_nucl_int_pos[ni], m_nonlocal_kspace[nuclei[ni].Z].subgrid, l_grid);
            }

            size_t grid_size = m_nonlocal_blocks[ni].grid_sizes;
            if (m_max_grid_size < grid_size) {
                m_max_grid_size = grid_size;
            }


            mSetNonlocalProjectorInfo(m_nonlocal_projectors[ni], nuclei[ni]);

            //DDMの担当領域とこの原子のBlockがoverlapしていたらnon-zero//
            is_nucl_overlap[ni] = grid_size;


            //Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
            //watch_pp.Record(17);
        }
        Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
        watch_pp.Record(16);

        //原子核ごとのMPI_Commを用意する(cutoff長の範囲内で担当グリッドとoverlapするプロセスだけのcomm)

        if (is_reset_at_least_one)
        {

            m_comm4atoms.DeleteComms();
            m_comm4atoms.CreateCommsDirect(&is_nucl_overlap[0], num_nuclei, l_grid.mpi_comm);
            const int num_valid_comm4atoms = m_comm4atoms.CountValidComms();
            //m_request4atoms.resize(num_valid_comm4atoms);
            //m_status4atoms.resize(num_valid_comm4atoms);

            //printf("[%d]DDM-num-overlap-nucl: %d\n", proc_id, m_num_valid_comm4atoms); fflush(stdout);

            Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
            watch_pp.Record(18);

            mCreateNonoverlapList(nuclei, num_nuclei);
            
        }

#if NONLOCAL_SvF_PARALLEL==7
        mCreateProjectorRYOnBlock_SvF_gamma_mpi7(nuclei, num_nuclei, l_grid);
        Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
        watch_pp.Record(20);
#elif NONLOCAL_SvF_PARALLEL==5
        mCreateProjectorRYOnBlock_SvF_gamma_mpi5(nuclei, num_nuclei, l_grid);
        Watch_Barrier<IS_MEASURE_TIME>(l_grid.mpi_comm);
        watch_pp.Record(20);
#else
        print("ERROR\n");
#endif


    }

    //subgrid領域がoverlapしている場合はNonlocal projectorを同時に作用できない//
    //そのため、overlapしてない原子の組み合わせを事前に作る//
    void mCreateNonoverlapList(const Nucleus* nuclei, int num_nuclei) {

        auto FoldDiffGridPos = [](int diff, int size_x) {
            int idiff_x = ((diff)+size_x * 4) % size_x; //4 means margin//
            return std::min(idiff_x, size_x - idiff_x);
            };

        auto is_overlap_list = std::make_unique<bool[]>(num_nuclei * num_nuclei);
        for (int ni = 0; ni < num_nuclei; ++ni) {            
            auto& ipos_i = m_nucl_int_pos[ni];
            auto& subgrid_i = m_nonlocal_kspace[nuclei[ni].Z].subgrid;
            is_overlap_list[ni + num_nuclei * ni] = false;

            for (int nk = ni + 1; nk < num_nuclei; ++nk) {                
                auto& ipos_k = m_nucl_int_pos[nk];
                auto& subgrid_k = m_nonlocal_kspace[nuclei[nk].Z].subgrid;

                //立方体なのでこれが共通したカットオフ長になるはず//
                const int cutoff = subgrid_i.end_x - subgrid_k.begin_x;
                int idiff_x = FoldDiffGridPos(ipos_k.x - ipos_i.x, m_grid.size_x); //4 means margin//
                int idiff_y = FoldDiffGridPos(ipos_k.y - ipos_i.y, m_grid.size_y); //4 means margin//
                int idiff_z = FoldDiffGridPos(ipos_k.z - ipos_i.z, m_grid.size_z); //4 means margin//
                /*
                if (is_root) {
                    printf("distance(%d, %d): %d, %d, %d\n",ni,nk, idiff_x, idiff_y, idiff_z);
                }
                */
                bool is_overlap = (cutoff > idiff_x) && (cutoff > idiff_y) && (cutoff > idiff_z);
                is_overlap_list[nk + num_nuclei * ni] = is_overlap;
                is_overlap_list[ni + num_nuclei * nk] = is_overlap;
            }
        }

        auto is_used = std::make_unique<bool[]>(num_nuclei);
        memset(&is_used[0], 0, sizeof(bool) * num_nuclei);
        m_sorted_proc_ids.resize(num_nuclei);
        m_sorted_count_list.clear();
        {
            int head = 0;
            for (int phase = 0; head < num_nuclei; ++phase) {
                int count = 0;
                for (int ni = 0; ni < num_nuclei; ++ni) {
                    if (is_used[ni] == false) {
                        bool is_overlap = false;
                        for (int k = head; k < head + count; ++k) {
                            int nk = m_sorted_proc_ids[k];
                            if (is_overlap_list[nk + num_nuclei * ni] == true) {
                                is_overlap = true;
                                break;
                            }
                        }
                        if (is_overlap == false) {
                            m_sorted_proc_ids[head + count] = ni;
                            is_used[ni] = true;
                            ++count;
                        }
                    }
                }
                m_sorted_count_list.push_back(count);
                head += count;
            }
        }

#ifdef _DEBUG
        if (is_root) {
            printf("nonlocal, non-overlap nuclei list=======\n");
            int head = 0;
            for (const auto& count : m_sorted_count_list) {
                printf("["); 
                for (int k = head; k < head + count-1; ++k) {
                    printf("%d,", m_sorted_proc_ids[k]);
                }
                printf("%d]\n", m_sorted_proc_ids[head + count - 1]);
                head += count;
            }
        }
#endif
    }

    /*
    * 実質的にBloch定理用のexpファクターをk点ごとに用意する
    */
    void UpdateNonlocalBlochExp(const Nucleus* nuclei, int num_nuclei, const GridRangeMPI& l_grid,
        int id_spin_kpoint, double gx_dx, double gy_dy, double gz_dz) {

        if (m_is_gamma_point_list.size() <= id_spin_kpoint) {
            m_is_gamma_point_list.resize(id_spin_kpoint + 1);
        }



        if ((gx_dx == 0.0) && (gy_dy == 0.0) && (gz_dz == 0.0)) {

            m_is_gamma_point_list[id_spin_kpoint] = true;

            //nothing to do//

            for (int ni = 0; ni < num_nuclei; ++ni) {
                auto mycomm = m_comm4atoms.GetComm(ni);
                if (mycomm == MPI_COMM_NULL) {
                    continue;
                }
                m_nonlocal_projectors[ni].bloch_expikx.emplace_back(nullptr);
            }
        } 
        else 
        {
            m_is_gamma_point_list[id_spin_kpoint] = false;

            for (int ni = 0; ni < num_nuclei; ++ni) {
                auto mycomm = m_comm4atoms.GetComm(ni);
                if (mycomm == MPI_COMM_NULL) {
                    continue;
                }

                mCreateBlochExp(m_nonlocal_blocks[ni], m_nonlocal_projectors[ni], nuclei[ni], l_grid, id_spin_kpoint, gx_dx, gy_dy, gz_dz);
                

            }
        }


    }



    //Blochの定理で用いるexp(-ikx)をグリッドデータに割り付けて保存する//
    void mCreateBlochExp(SubgridBlock& nonlocal_blocks,
        PP_Proj_Ylm_each_atom& nonlocal_projectors,
        const Nucleus nucleus, const GridRange& l_grid,
        int id_spin_kpoint, double gx_dx, double gy_dy, double gz_dz) {


        if (nonlocal_projectors.num_proj_RY == 0) {
            return;
        }




        int64_t total_buf_size = nonlocal_blocks.grid_sizes;
        

        auto& bloch_expikx_list = nonlocal_projectors.bloch_expikx;
        double* bloch_expikx = gy::AlignedAlloc<double>(total_buf_size * 2);
        bloch_expikx_list.emplace_back(bloch_expikx);


        auto& nonlocal_kspace = m_nonlocal_kspace[nucleus.Z];

        int64_t offset = 0;

        int index = 0;
        {
            //DDM region//
            const size_t grid_sizes = nonlocal_blocks.grid_sizes;
            const auto& range_blocks = nonlocal_blocks.range_blocks;
            const auto& shifted_grid = nonlocal_blocks.shifted_grids;
            size_t num_blocks = range_blocks.size();

        

            SoAComplex expikx_l{ bloch_expikx + offset, bloch_expikx + offset + grid_sizes };

            {
                int offset_i = 0;
                for (size_t ib = 0; ib < num_blocks; ++ib) {
                    const int ix_begin = range_blocks[ib].begin_x - shifted_grid[ib].x;//global grid内の座標値//
                    const int iy_begin = range_blocks[ib].begin_y - shifted_grid[ib].y;
                    const int iz_begin = range_blocks[ib].begin_z - shifted_grid[ib].z;
                    const int ix_end = range_blocks[ib].end_x - shifted_grid[ib].x;
                    const int iy_end = range_blocks[ib].end_y - shifted_grid[ib].y;
                    const int iz_end = range_blocks[ib].end_z - shifted_grid[ib].z;

                    const int size_x = ix_end - ix_begin;
                    const int size_y = iy_end - iy_begin;

                    for (int iz = iz_begin; iz < iz_end; ++iz) {
                        const double kzz = gz_dz * (double)(iz);
                        for (int iy = iy_begin; iy < iy_end; ++iy) {
                            const double kyy = gy_dy * (double)(iy);
                            for (int ix = ix_begin; ix < ix_end; ++ix) {
                                const double kxx = gx_dx * (double)(ix);
                                const int i = offset_i + (ix - ix_begin) + size_x * ((iy - iy_begin) + size_y * (iz - iz_begin));

                                double cosikx = cos(kxx + kyy + kzz);
                                double sinikx = sin(kxx + kyy + kzz);

                                expikx_l.re[i] = cosikx;
                                expikx_l.im[i] = sinikx;

                            }
                        }
                    }

                    offset_i += range_blocks[ib].Size3D();

                }

            };

            offset += 2 * grid_sizes;
            
        }




    }









#ifdef FORCE_DIFF_PROJ
    double ForceNonlocal_bundle(double* forces, const SoAComplex* l_psi,
        const double* occupancies, int num_bundle,
        const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, double gx, double gy, double gz, std::byte* work_buffer) {
        //note: gx, gy, gz is not used in this mode in which grad(p) is used and grad(psi) is not used//


        watch_pp.Restart();

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const int local_size = l_grid.Size3D();

        const double dV = m_dx * m_dy * m_dz;


        const int num_all_nonlocal = m_num_all_nonlocal;


        double* l_inner = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(num_all_nonlocal * num_bundle * 2 * 4);// gy::AlignedAlloc<double>(num_all_nonlocal * num_bundle * 4 * 4);
        //sum_innerはMPI_Iallreduceのためにホストメモリである必要がある//
        auto sum_inner = std::make_unique<double[]>(num_all_nonlocal * num_bundle * 2 * 4);

        auto cut_Hp_0 = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(m_max_grid_size * 2 * num_bundle);// gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle);

        auto& info_range = m_num_list_nonlocal;

        BlockCutInfo2* block_info = (BlockCutInfo2*)work_buffer; work_buffer += gy::AlignedBytesize<BlockCutInfo2>(8);//  gy::AlignedAlloc<BlockCutInfo2>(8);
        watch_pp.Record(0);
        mInnerNonlocalPsi_diff_bundle(l_psi, 0, num_bundle, l_grid, nuclei, num_nuclei, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);


        double l_ene = 0.0;
        double* l_forces = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(num_nuclei * 3 * 2);// gy::AlignedAlloc<double>(num_nuclei * 3 * 2);
        double* sum_forces = l_forces + num_nuclei * 3;
        memset(l_forces, 0, sizeof(double) * num_nuclei * 3);


        for (int ni = 0; ni < num_nuclei; ++ni) {

            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            if (!IsRoot(mycomm)) continue; //required this conddition to sum only for host//

            const int index = info_range[ni] * num_bundle * 2 * 4; //2 means Complex, and 4 means p and grad(p)//

            auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];

            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;
            //const int width = (m_nonlocal_projectors[ni].head_projector_l[MAX_L_SYSTEM] - m_nonlocal_projectors[ni].head_projector_l[0]);
            //const int width = m_nonlocal_projectors[ni].num_proj_RY;
            const int width = nonlocal_kspace.num_proj_RY;

            for (int n = 0; n < num_bundle; ++n) {  //2 means Complex number, enen and odd of n is real-part and imaginary-part, respectively//
                double ene_n = 0.0;


                const int begin_w = 0;
                const int end_w = width;
                for (int w = begin_w; w < end_w; ++w) {                    

                    const double E_proj_occ = nonlocal_kspace.projector_energy_RY[w] * occupancies[n];                    

                    OneComplex inner{ sum_inner[index + width * (8 * n) + w], sum_inner[index + width * (8 * n + 4) + w] };
                    ene_n += (inner.r * inner.r + inner.i * inner.i) * E_proj_occ;


                    OneComplex diff_x{ sum_inner[index + width * (8 * n + 1) + w], sum_inner[index + width * (8 * n + 5) + w] };
                    OneComplex diff_y{ sum_inner[index + width * (8 * n + 2) + w], sum_inner[index + width * (8 * n + 6) + w] };
                    OneComplex diff_z{ sum_inner[index + width * (8 * n + 3) + w], sum_inner[index + width * (8 * n + 7) + w] };


                    l_forces[ni * 3 + 0] += 2.0 * (inner.r * diff_x.r + inner.i * diff_x.i) * E_proj_occ;
                    l_forces[ni * 3 + 1] += 2.0 * (inner.r * diff_y.r + inner.i * diff_y.i) * E_proj_occ;
                    l_forces[ni * 3 + 2] += 2.0 * (inner.r * diff_z.r + inner.i * diff_z.i) * E_proj_occ;

                }
                
                l_ene += ene_n;
            }


        }

        watch_pp.Record(9);

        //gy::AlignedFree(l_inner);
        //gy::AlignedFree(block_info);


        double ene = 0.0;
        MPI_Reduce(&l_ene, &ene, 1, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        MPI_Reduce(l_forces, sum_forces, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);

        for (int ni = 0; ni < num_nuclei; ++ni) {
            forces[ni * 3 + 0] += sum_forces[ni * 3 + 0];
            forces[ni * 3 + 1] += sum_forces[ni * 3 + 1];
            forces[ni * 3 + 2] += sum_forces[ni * 3 + 2];
        }

        //gy::AlignedFree(l_forces);
        return ene;
    }

#else
    //with grad(psi)//
    double ForceNonlocal_bundle_withGradPsi(double* forces, const SoAComplex* l_psi_and_grad_psi, 
        const double* occupancies, int num_bundle,
        const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, double gx, double gy, double gz) {



        watch_pp.Restart();

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const int local_size = l_grid.Size3D();

        const double dV = m_dx * m_dy * m_dz;


        const int num_all_nonlocal = m_num_all_nonlocal;

        
        double* l_inner = new double[num_all_nonlocal * num_bundle * 4 * 4];
        double* sum_inner = l_inner + num_all_nonlocal * num_bundle * 2 * 4;

        
        auto& info_range = m_num_list_nonlocal;
        
        mInnerNonlocalPsi_v4_bundle(l_psi_and_grad_psi, 0, num_bundle*4, l_grid, nuclei, num_nuclei, l_inner, id_spin_kpoint, info_range, dV);


        double l_ene = 0.0;
        double* l_forces = new double[num_nuclei * 3*2];
        double* sum_forces = l_forces + num_nuclei * 3;
        memset(l_forces, 0, sizeof(double) * num_nuclei * 3);


        for (int ni = 0; ni < num_nuclei; ++ni) {

            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            if (!IsRoot(mycomm)) continue; //required this conddition to sum only for host//

            const int index = info_range[ni] * num_bundle * 2 * 4; //2 means Complex, and 4 means psi and its derivatives//


            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;


            auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];
            const int width = nonlocal_kspace.num_proj_RY;

            

            for (int n = 0; n < num_bundle ; ++n) {  //2 means Complex number, enen and odd of n is real-part and imaginary-part, respectively//
                double ene_n = 0.0;
                int k_offset = 0;

                {                        
                    //const size_t grid_size = blocks.grid_sizes;                    
                    const int begin_w = 0;
                    const int end_w = width;
                    for (int w = begin_w; w < end_w; ++w) {
                        
                        const double E_proj_occ = nonlocal_kspace.projector_energy_RY[w] * occupancies[n];

                        OneComplex inner{ sum_inner[index + width * (8 * n) + w], sum_inner[index + width * (8*n + 1) + w] };
                        ene_n += (inner.r * inner.r + inner.i * inner.i) * E_proj_occ; 
                            

                        OneComplex diff_x{ sum_inner[index + width * (8 * n + 2) + w], sum_inner[index + width * (8 * n + 3) + w] };
                        OneComplex diff_y{ sum_inner[index + width * (8 * n + 4) + w], sum_inner[index + width * (8 * n + 5) + w] };
                        OneComplex diff_z{ sum_inner[index + width * (8 * n + 6) + w], sum_inner[index + width * (8 * n + 7) + w] };
                        diff_x.i += gx * inner.r;
                        diff_x.r += -gx * inner.i;
                        diff_y.i += gy * inner.r;
                        diff_y.r += -gy * inner.i;
                        diff_z.i += gz * inner.r;
                        diff_z.r += -gz * inner.i;

                        constexpr double coef_for_cc = -2.0; //to be c.c.// 
                        l_forces[ni * 3 + 0] += coef_for_cc * (inner.r * diff_x.r + inner.i * diff_x.i) * E_proj_occ;
                        l_forces[ni * 3 + 1] += coef_for_cc * (inner.r * diff_y.r + inner.i * diff_y.i) * E_proj_occ;
                        l_forces[ni * 3 + 2] += coef_for_cc * (inner.r * diff_z.r + inner.i * diff_z.i) * E_proj_occ;

                    }
                    
                }

                l_ene += ene_n;
            }


        }

        watch_pp.Record(9);
        
        delete[] l_inner;


        double ene = 0.0;
        MPI_Reduce(&l_ene, &ene, 1, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        MPI_Reduce(l_forces, sum_forces, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);

        for (int ni = 0; ni < num_nuclei; ++ni) {
            forces[ni * 3 + 0] += sum_forces[ni * 3 + 0];
            forces[ni * 3 + 1] += sum_forces[ni * 3 + 1];
            forces[ni * 3 + 2] += sum_forces[ni * 3 + 2];
        }

        delete[] l_forces;
        return ene;
    }



#endif


















 public:

     size_t GetWorkSize(int num_bundle, int num_nuclei) {
         size_t work_size = 0;
#ifdef PP_NONLOCAL_CUT_ALL

         int num_nuclei_effective = 0;
         int64_t total_grid = 0;
         int64_t total_blocks = 0;
         for (int ni = 0; ni < num_nuclei; ++ni) {
             auto mycomm = m_comm4atoms.GetComm(ni);
             if (mycomm != MPI_COMM_NULL) {
                 ++num_nuclei_effective;

                 const auto& blocks = m_nonlocal_blocks[ni];
                 const int64_t grid_size = blocks.grid_sizes;
                 total_grid += grid_size;
                 total_blocks += blocks.range_blocks.size();
             }
         }

         //for ProjectionPP_bundle// 
         work_size += gy::AlignedBytesize<double>(total_grid * 2 * num_bundle);  //to cut small region from orbital//         
         work_size += gy::AlignedBytesize<BlockCutInfo2>( 8 * total_blocks);
#else
         //for ProjectionPP_bundle//
         work_size += gy::AlignedBytesize<double>(m_max_grid_size * 2 * num_bundle);  //to cut small region from orbital//         
         work_size += gy::AlignedBytesize<BlockCutInfo2>( 8);
#endif
         //for ForceNonlocal_bundle (greater than ProjectionPP_bundle) //
         work_size += gy::AlignedBytesize<double>(m_num_all_nonlocal * num_bundle * 4 * 4);
         //for SymplecticIntegration_bundle_ids (greater than ForceNonlocal_bundle)//
         work_size += gy::AlignedBytesize<double>(num_nuclei * 3 * 4);

         work_size += gy::AlignedBytesize<int>(num_nuclei); //for offset of add memory(overlap)
         return work_size;
     }


     /*
     * 粒子ごとにばらして計算
     * cpuなどのキャッシュ向け
     * 
     */
    void ProjectionPP_bundle_1(SoAComplex* Hp, const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, std::byte* work_buffer) {

        watch_pp.Restart();
        

        const int proc_id = GetProcessID(l_grid.mpi_comm);

        const double coef = (m_dx * m_dy * m_dz);
        const double dV = coef;
       
        const int num_all_nonlocal = m_num_all_nonlocal;
        auto& info_range = m_num_list_nonlocal;

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];

        

        const int num_bundle = end_n - begin_n;
        //double* l_inner = gy::AlignedAlloc<double>(num_all_nonlocal * num_bundle * 4);
        double* l_inner = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(num_all_nonlocal * num_bundle * 2);
        //sum_innerはMPI_Iallreduceのためにホストメモリである必要がある//
        auto sum_inner = std::make_unique<double[]>(num_all_nonlocal * num_bundle*2);

#ifdef PP_NONLOCAL_CUT_ALL
        int64_t total_grid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm != MPI_COMM_NULL) {                
                const auto& blocks = m_nonlocal_blocks[ni];
                const int64_t grid_size = blocks.grid_sizes;
                total_grid += grid_size;
            }
        }

        //working buffer used for local(cut) |psi> and the operated H|psi>. 
        auto cut_Hp_0 = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(2 * num_bundle * total_grid);// gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle * num_nuclei_effective);
        //gy::ZeroClear(cut_Hp_0.get(), m_max_grid_size * 2 * num_bundle * num_nuclei_effective);
        BlockCutInfo2* block_info = (BlockCutInfo2*)work_buffer;//gy::AlignedAlloc<BlockCutInfo2>(8* num_nuclei_effective);
        watch_pp.Record(0);

        //calculate < p_{ n,j } | psi_i >
        mInnerNonlocalPsi_v6_bundle(l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);

#else
        //working buffer used for local(cut) |psi> and the operated H|psi>. 
        auto cut_Hp_0 = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(m_max_grid_size * 2 * num_bundle);// gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle);
        BlockCutInfo2* block_info = (BlockCutInfo2*)work_buffer;//gy::AlignedAlloc<BlockCutInfo2>(8);     
        //gy::ZeroClear(&cut_Hp_0[0], m_max_grid_size * 2 * num_bundle);
        watch_pp.Record(0);

        //calculate < p_{ n,j } | psi_i >
        mInnerNonlocalPsi_v4_bundle(l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);
#endif

        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            const int index = info_range[ni] * num_bundle * 2;

            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;


            const auto& blocks = m_nonlocal_blocks[ni];

            
            const int64_t grid_size = blocks.grid_sizes;



            auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];
            const int width = nonlocal_kspace.num_proj_RY;

            // c_{j,i} = e_j < p_{ n,j } | psi_i >
            // where e_j = projector energy

            for (int n = 0; n < num_bundle * 2; ++n) {  //2 means bundle SoAComplex

                const int begin_w = 0;
                const int end_w = width;
                for (int w = begin_w; w < end_w; ++w) {
                    //hostメモリのsum_innerからdeviceメモリのl_innerへコピー//
                    l_inner[index + width * n + w] = sum_inner[index + width * n + w] * nonlocal_kspace.projector_energy_RY[w];
                }
            }
            watch_pp.Record(2);

            // |\delta psi_i > = c_{j,i} |p_{ n,j }>
            // ただし、i \in Nbのバンドルであり、かつ、nonlocal-projectorのグリッド数Ngがあるので、
            // 左辺は Ng x Nb の行列である//
            blas_DGEMM_n(grid_size, 2 * num_bundle, width, proj_p, &l_inner[index], &cut_Hp_0[0], grid_size, 1.0, 0.0);
            gy::Synchronize();
            watch_pp.Record(6);
        


            if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
                for (int n = 0; n < num_bundle; ++n) {
                    double* cut_Hp_l_re = &cut_Hp_0[0] + grid_size * n * 2;
                    double* cut_Hp_l_im = &cut_Hp_0[0] + grid_size * (n * 2 + 1);

                    const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                    const double* expikx_im = expikx_re + grid_size;
                    for (int64_t i = 0; i < grid_size; ++i) {
                        const double p_re = cut_Hp_l_re[i];
                        const double p_im = cut_Hp_l_im[i];

                        cut_Hp_l_re[i] = expikx_re[i] * p_re + expikx_im[i] * p_im;
                        cut_Hp_l_im[i] = expikx_re[i] * p_im - expikx_im[i] * p_re;

                    }
                }
            }

            

#ifdef GY_WITH_CUDA_OR_HIP


            AddSubgrid2_bundle(num_bundle, Hp[begin_n].re, l_grid, blocks, &cut_Hp_0[0], block_info);

#else
            
            for (int n = begin_n; n < end_n; ++n) {
                AddSubgridByRanges(l_grid, Hp[n].re, blocks.range_blocks, &cut_Hp_0[0] + grid_size * 2 * (n - begin_n));
                AddSubgridByRanges(l_grid, Hp[n].im, blocks.range_blocks, &cut_Hp_0[0] + grid_size * (2 * (n - begin_n) + 1));
            }
#endif
            gy::Synchronize();
            watch_pp.Record(7);
        }


        //gy::AlignedFree(block_info);
        //gy::AlignedFree(l_inner);
    }


    /*
    * 全原子を纏めて計算
    * メモリが必要になるがGPUなどでkernelの発行数を抑えたいとき
    */
    void ProjectionPP_bundle_2(SoAComplex* Hp, const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, std::byte* work_buffer) {

        watch_pp.Restart();


        const int proc_id = GetProcessID(l_grid.mpi_comm);

        const double coef = (m_dx * m_dy * m_dz);
        const double dV = coef;

        const int num_all_nonlocal = m_num_all_nonlocal;
        auto& info_range = m_num_list_nonlocal;

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];



        const int num_bundle = end_n - begin_n;
        //double* l_inner = gy::AlignedAlloc<double>(num_all_nonlocal * num_bundle * 4);
        double* l_inner = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(num_all_nonlocal * num_bundle * 2);
        //sum_innerはMPI_Iallreduceのためにホストメモリである必要がある//
        auto sum_inner = std::make_unique<double[]>(num_all_nonlocal * num_bundle * 2);
        

        int* nucl_grid_offset = (int*)work_buffer; work_buffer += gy::AlignedBytesize<int>(num_nuclei);

        int64_t total_grid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            nucl_grid_offset[ni] = total_grid;

            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm != MPI_COMM_NULL) {
                const auto& blocks = m_nonlocal_blocks[ni];
                const int64_t grid_size = blocks.grid_sizes;
                total_grid += grid_size;
            }
            
        }
        //working buffer used for local(cut) |psi> and the operated H|psi>. 
        auto cut_Hp_0 = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(2 * num_bundle * total_grid);// gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle * num_nuclei_effective);
        //gy::ZeroClear(cut_Hp_0.get(), m_max_grid_size * 2 * num_bundle * num_nuclei_effective);
        BlockCutInfo2* block_info = (BlockCutInfo2*)work_buffer;//gy::AlignedAlloc<BlockCutInfo2>(8* num_nuclei_effective);
        watch_pp.Record(0);

        //calculate < p_{ n,j } | psi_i >
        mInnerNonlocalPsi_v6_bundle(l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);


        double* cut_psi_n;
        cut_psi_n = cut_Hp_0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            const int index = info_range[ni] * num_bundle * 2;

            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;


            const auto& blocks = m_nonlocal_blocks[ni];


            const int64_t grid_size = blocks.grid_sizes;



            auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];
            const int width = nonlocal_kspace.num_proj_RY;

            // c_{j,i} = e_j < p_{ n,j } | psi_i >
            // where e_j = projector energy

            for (int n = 0; n < num_bundle * 2; ++n) {  //2 means bundle SoAComplex

                const int begin_w = 0;
                const int end_w = width;
                for (int w = begin_w; w < end_w; ++w) {
                    //const int k = k_offset + (w - begin_w) / (2 * l + 1);

                    l_inner[index + width * n + w] = sum_inner[index + width * n + w] * nonlocal_kspace.projector_energy_RY[w];
                }
            }
            //watch_pp.Record(2);

            // |\delta psi_i > = c_{j,i} |p_{ n,j }>
            // ただし、i \in Nbのバンドルであり、かつ、nonlocal-projectorのグリッド数Ngがあるので、
            // 左辺は Ng x Nb の行列である//
            blas_DGEMM_n(grid_size, 2 * num_bundle, width, proj_p, &l_inner[index], cut_psi_n, grid_size, 1.0, 0.0);

            cut_psi_n += grid_size * 2 * num_bundle;
        }
        gy::Synchronize();
        watch_pp.Record(6);



        if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
            cut_psi_n = cut_Hp_0;
            for (int ni = 0; ni < num_nuclei; ++ni) {
                auto mycomm = m_comm4atoms.GetComm(ni);
                if (mycomm == MPI_COMM_NULL) continue;




                const auto& blocks = m_nonlocal_blocks[ni];
                const int64_t grid_size = blocks.grid_sizes;



                auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];
                const int width = nonlocal_kspace.num_proj_RY;


                for (int n = 0; n < num_bundle; ++n) {
                    double* cut_Hp_l_re = &cut_psi_n[0] + grid_size * n * 2;
                    double* cut_Hp_l_im = &cut_psi_n[0] + grid_size * (n * 2 + 1);

                    const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                    const double* expikx_im = expikx_re + grid_size;
                    for (int64_t i = 0; i < grid_size; ++i) {
                        const double p_re = cut_Hp_l_re[i];
                        const double p_im = cut_Hp_l_im[i];

                        cut_Hp_l_re[i] = expikx_re[i] * p_re + expikx_im[i] * p_im;
                        cut_Hp_l_im[i] = expikx_re[i] * p_im - expikx_im[i] * p_re;

                    }

                }


                cut_psi_n += grid_size * 2 * num_bundle;
            }
            gy::Synchronize();
            watch_pp.Record(1);
        }

#ifdef GY_WITH_CUDA_OR_HIP

        AddSubgrid3_bundle(num_bundle, Hp[begin_n].re, l_grid, cut_Hp_0, nucl_grid_offset, block_info);

#else
        cut_psi_n = cut_Hp_0;
        int block_offset = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            


            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;



            auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];
            const int width = nonlocal_kspace.num_proj_RY;


#ifdef GY_WITH_CUDA_OR_HIP

            AddSubgrid2_bundle(num_bundle, Hp[begin_n].re, l_grid, blocks, &cut_psi_n[0], block_info + block_offset);


#else

            for (int n = begin_n; n < end_n; ++n) {
                AddSubgridByRanges(l_grid, Hp[n].re, blocks.range_blocks, &cut_psi_n[0] + grid_size * 2 * (n - begin_n));
                AddSubgridByRanges(l_grid, Hp[n].im, blocks.range_blocks, &cut_psi_n[0] + grid_size * (2 * (n - begin_n) + 1));
            }
#endif



            cut_psi_n += grid_size * 2 * num_bundle;
            const int num_blocks = (int)(blocks.range_blocks.size());
            block_offset += num_blocks;
        }
#endif
        gy::Synchronize();//necessary because memory region is overlap for nuclei//
        watch_pp.Record(7);

    }


    void ProjectionPP_bundle(SoAComplex* Hp, const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, std::byte* work_buffer) {
#ifdef PP_NONLOCAL_CUT_ALL
        ProjectionPP_bundle_2(Hp, l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, id_spin_kpoint, work_buffer);
#else
        ProjectionPP_bundle_1(Hp, l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, id_spin_kpoint, work_buffer);
#endif

    }


public:  //public is necessary for just CUDA with extended-lambda


#ifdef GY_WITH_CUDA_OR_HIP

    void CutSubgrid2_bundle(int num_bundle, const double* l_psi_re, const GridRangeMPI& l_grid, const SubgridBlock& blocks, double* cut_psi_0, BlockCutInfo2* block_info)
    {
        int num_blocks = (int)(blocks.range_blocks.size());
        int offset = 0;
        for (int ib = 0; ib < num_blocks; ++ib) {
            const auto& subgrid = blocks.range_blocks[ib];

            block_info[ib].ix_end = subgrid.SizeX();
            block_info[ib].iy_end = subgrid.SizeY();
            block_info[ib].iz_end = subgrid.SizeZ();
            block_info[ib].ix_offset = subgrid.begin_x - l_grid.begin_x;
            block_info[ib].iy_offset = subgrid.begin_y - l_grid.begin_y;
            block_info[ib].iz_offset = subgrid.begin_z - l_grid.begin_z;
            block_info[ib].small_offset = offset;

            offset += subgrid.Size3D();

        }

        const int l_size = l_grid.Size3D();

        dim3 cu_threads;
        cu_threads.x = 256;// GY_MAX_THREADS_PER_BLOCK;
        cu_threads.y = 1;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = num_blocks;
        cu_blocks.y = num_bundle * 2;
        cu_blocks.z = 1;

        const int grid_size = blocks.grid_sizes;
        gy::kernel_CutSubgrid2 <<<cu_blocks, cu_threads >>> (block_info, &cut_psi_0[0], grid_size, l_psi_re, l_grid.SizeX(), l_grid.SizeX() * l_grid.SizeY(), l_size);


    }


    void CutSubgrid3_bundle(int num_bundle, const double* l_psi_re, const GridRangeMPI& l_grid, 
        int num_nuclei, double* cut_psi_0, BlockCutInfo2* block_info)
    {
        
        int dest_offset = 0;
        int total_block = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }


            const auto& blocks = m_nonlocal_blocks[ni];
            const int grid_size = blocks.grid_sizes;


            const int num_blocks = (int)(blocks.range_blocks.size());
            int offset = 0;
            for (int ib = 0; ib < num_blocks; ++ib) {
                const auto& subgrid = blocks.range_blocks[ib];

                block_info[ib + total_block].ix_end = subgrid.SizeX();
                block_info[ib + total_block].iy_end = subgrid.SizeY();
                block_info[ib + total_block].iz_end = subgrid.SizeZ();
                block_info[ib + total_block].ix_offset = subgrid.begin_x - l_grid.begin_x;
                block_info[ib + total_block].iy_offset = subgrid.begin_y - l_grid.begin_y;
                block_info[ib + total_block].iz_offset = subgrid.begin_z - l_grid.begin_z;
                block_info[ib + total_block].small_offset = offset + dest_offset;
                block_info[ib + total_block].total_size = grid_size;

                offset += subgrid.Size3D();
            }


            dest_offset += grid_size * 2 * num_bundle;
            total_block += num_blocks;

        }

        const int l_size = l_grid.Size3D();

        dim3 cu_threads;
        cu_threads.x = 256;// GY_MAX_THREADS_PER_BLOCK;
        cu_threads.y = 1;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = total_block;
        cu_blocks.y = num_bundle * 2;
        cu_blocks.z = 1;
        
        gy::kernel_CutSubgrid3 <<<cu_blocks, cu_threads >>> (block_info, &cut_psi_0[0], l_psi_re, l_grid.SizeX(), l_grid.SizeX() * l_grid.SizeY(), l_size);


    }

    void AddSubgrid2_bundle(int num_bundle, double* dest_Hp_re, const GridRangeMPI& l_grid, const SubgridBlock& blocks, const double* cut_Hp_0, BlockCutInfo2* block_info) {


        int num_blocks = (int)(blocks.range_blocks.size());

        int offset = 0;
        for (int ib = 0; ib < num_blocks; ++ib) {
            const auto& subgrid = blocks.range_blocks[ib];

            block_info[ib].ix_end = subgrid.SizeX();
            block_info[ib].iy_end = subgrid.SizeY();
            block_info[ib].iz_end = subgrid.SizeZ();
            block_info[ib].ix_offset = subgrid.begin_x - l_grid.begin_x;
            block_info[ib].iy_offset = subgrid.begin_y - l_grid.begin_y;
            block_info[ib].iz_offset = subgrid.begin_z - l_grid.begin_z;
            block_info[ib].small_offset = offset;

            offset += subgrid.Size3D();

        }

        const int l_size = l_grid.Size3D();

        dim3 cu_threads;

        cu_threads.x = 256;// GY_MAX_THREADS_PER_BLOCK;
        cu_threads.y = 1;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = num_blocks;
        cu_blocks.y = num_bundle * 2;
        cu_blocks.z = 1;

        if (cu_blocks.x == 0) {
            printf("ERROR\n");
            exit(0);
        }

        const int grid_size = blocks.grid_sizes;
        gy::kernel_AddSubgrid2 <<<cu_blocks, cu_threads >>> (block_info, dest_Hp_re, l_grid.SizeX(), l_grid.SizeX() * l_grid.SizeY(), l_size, &cut_Hp_0[0], grid_size);

    }


    void AddSubgrid3_bundle(int num_bundle, double* dest_Hp_re, const GridRangeMPI& l_grid, 
        const double* cut_Hp_0, const int* nucl_grid_offset, BlockCutInfo2* block_info) {

        int head = 0;
        int total_block = 0;
        const int l_size = l_grid.Size3D();
        int head_block = 0;
        for (const auto& count : m_sorted_count_list) {            
            for (int nn = head; nn < head + count; ++nn) {
                int ni = m_sorted_proc_ids[nn];

                const auto& blocks = m_nonlocal_blocks[ni];
                const int64_t grid_size = blocks.grid_sizes;

                int num_blocks = (int)(blocks.range_blocks.size());
                const int dest_offset = nucl_grid_offset[ni] * 2 * num_bundle;

                int offset = 0;
                for (int ib = 0; ib < num_blocks; ++ib) {
                    const auto& subgrid = blocks.range_blocks[ib];

                    block_info[ib + total_block].ix_end = subgrid.SizeX();
                    block_info[ib + total_block].iy_end = subgrid.SizeY();
                    block_info[ib + total_block].iz_end = subgrid.SizeZ();
                    block_info[ib + total_block].ix_offset = subgrid.begin_x - l_grid.begin_x;
                    block_info[ib + total_block].iy_offset = subgrid.begin_y - l_grid.begin_y;
                    block_info[ib + total_block].iz_offset = subgrid.begin_z - l_grid.begin_z;
                    block_info[ib + total_block].small_offset = offset + dest_offset;
                    block_info[ib + total_block].total_size = grid_size;

                    offset += subgrid.Size3D();

                }
                
                total_block += num_blocks;
            }            
            head += count;

            if (total_block > head_block) {//DDMの場合にnonlocal領域がDDM領域とオーバーラップしないものがあり得る//

                dim3 cu_threads;
                cu_threads.x = 256;
                cu_threads.y = 1;
                cu_threads.z = 1;
                dim3 cu_blocks;
                cu_blocks.x = total_block - head_block;
                cu_blocks.y = num_bundle * 2;
                cu_blocks.z = 1;


                gy::kernel_AddSubgrid3 <<<cu_blocks, cu_threads >>> (&block_info[head_block], dest_Hp_re, l_grid.SizeX(), l_grid.SizeX() * l_grid.SizeY(), l_size, &cut_Hp_0[0]);
                
                head_block = total_block;
            }
        }
    }


#endif

    /*
    * 原子をn, 原子毎のnonlocal-projectorをp_{n,j}としたとき、where j \in [0, M-1]
    * <p_{n,j}|psi_i>
    * を計算する
    * この量は、{j,i} \in M x Nbの二次元行列になる。(Nb is num of bundle)
    * さらに原子nの数だけ存在する。
    * 
    * 関数内でMPI_Iallreduceする
    * この関数ではprojectorのsubgrid内での微分(grad p)を使わず、波動関数の微分grad psiを前提としている
    * cut_psi_0 の サイズは(m_max_grid_size * num_bundle * 2)が必要//
    */
    void mInnerNonlocalPsi_v4_bundle(const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei, 
        double* l_inner, double* sum_inner, int id_spin_kpoint, std::vector<int>& info_range, double dV, double* cut_psi_0, BlockCutInfo2* block_info)
    {


        const int num_bundle = end_n - begin_n;
        const int num_all_nonlocal = info_range[num_nuclei];
        //double* sum_inner = l_inner + num_all_nonlocal * num_bundle * 2; //2 means degree of complex//

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];

        
        
        auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei);
        
        int64_t flop = 0;

        int num_valid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }

            const int index = info_range[ni] * num_bundle * 2;

            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;
            const int width = m_nonlocal_projectors[ni].num_proj_RY;

            
            const auto& blocks = m_nonlocal_blocks[ni];
            const int grid_size = blocks.grid_sizes;
            
            if (grid_size == 0) {

                gy::ZeroClear(&l_inner[index], width * num_bundle * 2);
                watch_pp.Record(1);
            } else {

                watch_pp.Record(1);

#ifdef GY_WITH_CUDA_OR_HIP
                CutSubgrid2_bundle(num_bundle, l_psi[begin_n].re, l_grid, blocks, cut_psi_0, block_info);
                gy::Synchronize();
#else
                for (int n = begin_n; n < end_n; ++n) {
                    CutSubgridByRanges(blocks.range_blocks, cut_psi_0 + grid_size * 2 * (n - begin_n), l_grid, l_psi[n].re);
                    CutSubgridByRanges(blocks.range_blocks, cut_psi_0 + grid_size * (2 * (n - begin_n) + 1), l_grid, l_psi[n].im);
                }
#endif
                watch_pp.Record(3);


                if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
                    for (int n = begin_n; n < end_n; ++n) {
                        double* psi_re = cut_psi_0 + grid_size * 2 * (n - begin_n);
                        double* psi_im = cut_psi_0 + grid_size * (2 * (n - begin_n) + 1);
                        const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                        const double* expikx_im = expikx_re + grid_size;
                        for (int64_t i = 0; i < grid_size; ++i) {
                            const double p_re = psi_re[i];
                            const double p_im = psi_im[i];

                            psi_re[i] = expikx_re[i] * p_re - expikx_im[i] * p_im;
                            psi_im[i] = expikx_re[i] * p_im + expikx_im[i] * p_re;

                        }
                    }
                }


                blas_DGEMM_t(width, 2 * num_bundle, grid_size, proj_p, cut_psi_0, &l_inner[index], width, dV, 0.0);
                gy::Synchronize();



                flop += width * grid_size * 4 * num_bundle; //4 meand fma (x2) and psi of complex(x2)//
                watch_pp.Record(10);
            }
#if 1
            gy::CopyMemoryToHost(sum_inner + info_range[ni] * num_bundle * 2, l_inner + info_range[ni] * num_bundle * 2, (info_range[ni + 1] - info_range[ni]) * num_bundle * 2 * sizeof(double));
            MPI_Iallreduce(MPI_IN_PLACE, sum_inner + info_range[ni] * num_bundle * 2,
                (info_range[ni + 1] - info_range[ni]) * num_bundle * 2, MPI_DOUBLE, MPI_SUM, mycomm, &request4atoms[num_valid]);

#elif 0
            //GPUではMPI_Iallreduceとdevice/managedメモリは未サポートで動作不良
            MPI_Iallreduce(l_inner + info_range[ni] * num_bundle*2, sum_inner + info_range[ni] * num_bundle*2,
                (info_range[ni + 1] - info_range[ni]) * num_bundle*2, MPI_DOUBLE, MPI_SUM, mycomm, &request4atoms[num_valid]);
#else
            MPI_Allreduce(l_inner + info_range[ni] * num_bundle * 2, sum_inner + info_range[ni] * num_bundle * 2,
                (info_range[ni + 1] - info_range[ni]) * num_bundle * 2, MPI_DOUBLE, MPI_SUM, mycomm);
#endif


            ++num_valid;


        }
        watch_pp.Record(1);

        auto status4atoms = std::make_unique<MPI_Status[]>(num_valid);
        MPI_Waitall(num_valid, &request4atoms[0], &status4atoms[0]);
        watch_pp.Record(5);

#ifdef TIME_PP_MPI
        m_bench_flop_nonlocal += flop;
        if (m_bench_flop_nonlocal > LIMIT_64) {
            m_bench_flop_nonlocal -= LIMIT_64;
            m_bench_flop_nonlocal_high++;
        }
#endif


    }

    /*
    * 原子をn, 原子毎のnonlocal-projectorをp_{n,j}としたとき、where j \in [0, M-1]
    * <p_{n,j}|psi_i>
    * を計算する
    * この量は、{j,i} \in M x Nbの二次元行列になる。(Nb is num of bundle)
    * さらに原子nの数だけ存在する。
    */
    void mInnerNonlocalPsi_v6_bundle(const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        double* l_inner, double* sum_inner, int id_spin_kpoint, std::vector<int>& info_range, double dV, double* cut_psi_0, BlockCutInfo2* block_info)
    {


        const int num_bundle = end_n - begin_n;
        const int num_all_nonlocal = info_range[num_nuclei];
        //double* sum_inner = l_inner + num_all_nonlocal * num_bundle * 2; //2 means degree of complex//

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];



        auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei);

        watch_pp.Record(1);

        double* cut_psi_n;

#ifdef GY_WITH_CUDA_OR_HIP
        CutSubgrid3_bundle(num_bundle, l_psi[begin_n].re, l_grid, num_nuclei, cut_psi_0, block_info);

#else
        int block_offset = 0;
        cut_psi_n = cut_psi_0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }


            const auto& blocks = m_nonlocal_blocks[ni];
            const int grid_size = blocks.grid_sizes;


            if (grid_size > 0) {


#ifdef GY_WITH_CUDA_OR_HIP
                CutSubgrid2_bundle(num_bundle, l_psi[begin_n].re, l_grid, blocks, cut_psi_n, block_info + block_offset);
                //gy::Synchronize();
#else
                for (int n = begin_n; n < end_n; ++n) {
                    CutSubgridByRanges(blocks.range_blocks, cut_psi_n + grid_size * 2 * (n - begin_n), l_grid, l_psi[n].re);
                    CutSubgridByRanges(blocks.range_blocks, cut_psi_n + grid_size * (2 * (n - begin_n) + 1), l_grid, l_psi[n].im);
                }
#endif

            }


            cut_psi_n += grid_size * 2 * num_bundle;

            const int num_blocks = (int)(blocks.range_blocks.size());
            block_offset += num_blocks;

        }
#endif

        gy::Synchronize();
        watch_pp.Record(3);

        if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
            int block_offset = 0;
            cut_psi_n = cut_psi_0;
            for (int ni = 0; ni < num_nuclei; ++ni) {
                auto mycomm = m_comm4atoms.GetComm(ni);
                if (mycomm == MPI_COMM_NULL) {
                    continue;
                }


                const auto& blocks = m_nonlocal_blocks[ni];
                const int grid_size = blocks.grid_sizes;


                if (grid_size > 0) {


                    for (int n = begin_n; n < end_n; ++n) {
                        double* psi_re = cut_psi_n + grid_size * 2 * (n - begin_n);
                        double* psi_im = cut_psi_n + grid_size * (2 * (n - begin_n) + 1);
                        const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                        const double* expikx_im = expikx_re + grid_size;
                        for (int64_t i = 0; i < grid_size; ++i) {
                            const double p_re = psi_re[i];
                            const double p_im = psi_im[i];

                            psi_re[i] = expikx_re[i] * p_re - expikx_im[i] * p_im;
                            psi_im[i] = expikx_re[i] * p_im + expikx_im[i] * p_re;

                        }
                    }
                }

                cut_psi_n += grid_size * 2 * num_bundle;

                const int num_blocks = (int)(blocks.range_blocks.size());
                block_offset += num_blocks;

            }


            gy::Synchronize();
            watch_pp.Record(1);
        }


        
        int64_t flop = 0;
        int num_nuclei_valid = 0;
        cut_psi_n = cut_psi_0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }

            const int index = info_range[ni] * num_bundle * 2;
            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;
            const int width = m_nonlocal_projectors[ni].num_proj_RY;
            const auto& blocks = m_nonlocal_blocks[ni];
            const int grid_size = blocks.grid_sizes;

            //double* cut_psi_n = cut_psi_0 + m_max_grid_size * 2 * num_bundle * num_nuclei_valid;

            if (grid_size == 0) {
                gy::ZeroClear(&l_inner[index], width * num_bundle * 2);

            } else {
                blas_DGEMM_t(width, 2 * num_bundle, grid_size, proj_p, cut_psi_n, &l_inner[index], width, dV, 0.0);
                //gy::Synchronize();
                flop += width * grid_size * 4 * num_bundle; //4 meand fma (x2) and psi of complex(x2)//                
            }
            ++num_nuclei_valid;
            cut_psi_n += grid_size * 2 * num_bundle;
        }
        
        gy::Synchronize();
        watch_pp.Record(10);

        num_nuclei_valid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }
            
            gy::CopyMemoryToHost(sum_inner + info_range[ni] * num_bundle * 2, l_inner + info_range[ni] * num_bundle * 2, (info_range[ni + 1] - info_range[ni])* num_bundle * 2 * sizeof(double));
            MPI_Iallreduce(MPI_IN_PLACE, sum_inner + info_range[ni] * num_bundle * 2,
                (info_range[ni + 1] - info_range[ni]) * num_bundle * 2, MPI_DOUBLE, MPI_SUM, mycomm, &request4atoms[num_nuclei_valid]);

            ++num_nuclei_valid;


        }
        //watch_pp.Record(1);


        auto status4atoms = std::make_unique<MPI_Status[]>(num_nuclei_valid);
        MPI_Waitall(num_nuclei_valid, &request4atoms[0], &status4atoms[0]);
        watch_pp.Record(5);

#ifdef TIME_PP_MPI
        m_bench_flop_nonlocal += flop;
        if (m_bench_flop_nonlocal > LIMIT_64) {
            m_bench_flop_nonlocal -= LIMIT_64;
            m_bench_flop_nonlocal_high++;
        }
#endif


    }



#ifdef FORCE_DIFF_PROJ
    /*
    * 原子をn, 原子毎のnonlocal-projector p_{n,j}と、そのx,y,z方向の導関数dp/dx_{n,j}をまとめて
    * p'_{n,j} = { p_{n,j}      if (j < M), 
    *             dp/dx_{n,j-M} if ( M <= j < 2M), 
    *             dp/dy_{n,j-M} if (2M <= j < 3M), 
    *             dp/dz_{n,j-M} if (3M <= j < 4M), 
    * としたとき、
    * <p'_{n,j}|psi_i>
    * を計算する
    * この量は、{j,i} \in 4M x Nbの二次元行列になる。(Nb is num of bundle)
    * さらに原子nの数だけ存在する。
    */
    void mInnerNonlocalPsi_diff_bundle(const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid,
            const Nucleus* nuclei, int num_nuclei, double* l_inner, double* sum_inner, int id_spin_kpoint,
        std::vector<int>& info_range, double dV, double* cut_psi_0, BlockCutInfo2* block_info) {


        const int num_bundle = end_n - begin_n;
        const int num_all_nonlocal = info_range[num_nuclei];
        //double* sum_inner = l_inner + num_all_nonlocal * 4 * num_bundle * 2; //2 means degree of complex, 4 means pand grad(p)//

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];


        auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei);

        int64_t flop = 0;

        int num_valid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }

            const int index = info_range[ni] * num_bundle * 2 * 4;//4 meand p nad grad(p)//

            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;
            //4 means p and grad(p) //
            const int width = 4 * m_nonlocal_projectors[ni].num_proj_RY;

            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;            
            

            if (grid_size == 0) {

                gy::ZeroClear(&l_inner[index], width * num_bundle * 2);

                watch_pp.Record(1);

            }else{
                watch_pp.Record(1);

#ifdef GY_WITH_CUDA_OR_HIP
                CutSubgrid2_bundle(num_bundle, l_psi[begin_n].re, l_grid, blocks, cut_psi_0, block_info);
                gy::Synchronize();
#else
                for (int n = begin_n; n < end_n; ++n) {
                    CutSubgridByRanges(blocks.range_blocks, cut_psi_0 + grid_size * 2 * (n - begin_n), l_grid, l_psi[n].re);
                    CutSubgridByRanges(blocks.range_blocks, cut_psi_0 + grid_size * (2 * (n - begin_n) + 1), l_grid, l_psi[n].im);
                }
#endif
                watch_pp.Record(11);
            
                if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
                    for (int n = begin_n; n < end_n; ++n) {
                        double* psi_re = cut_psi_0 + grid_size * 2 * (n - begin_n);
                        double* psi_im = cut_psi_0 + grid_size * (2 * (n - begin_n) + 1);
                        const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                        const double* expikx_im = expikx_re + grid_size;
                        for (int64_t i = 0; i < grid_size; ++i) {
                            const double p_re = psi_re[i];
                            const double p_im = psi_im[i];

                            psi_re[i] = expikx_re[i] * p_re - expikx_im[i] * p_im;
                            psi_im[i] = expikx_re[i] * p_im + expikx_im[i] * p_re;

                        }
                    }
                }


                //Using BLAS DGEMM(faster than bottom because 1st size (size of A) is larger)
                blas_DGEMM_t(width, 2 * num_bundle, grid_size, proj_p, cut_psi_0, &l_inner[index], width, dV, 0.0);
                gy::Synchronize();


                flop += width * grid_size * 4 * num_bundle; //4 meand fma (x2) and psi of complex(x2)//
                
                watch_pp.Record(12);
            }
            
            gy::CopyMemoryToHost(sum_inner + info_range[ni] * 4 * num_bundle * 2, l_inner + info_range[ni] * 4 * num_bundle * 2, (info_range[ni + 1] - info_range[ni]) * 4 * num_bundle * 2 * sizeof(double));

            MPI_Iallreduce(MPI_IN_PLACE, sum_inner + info_range[ni] * 4 * num_bundle * 2,
                (info_range[ni + 1] - info_range[ni]) * 4 * num_bundle * 2, MPI_DOUBLE, MPI_SUM, mycomm, &request4atoms[num_valid]);

            ++num_valid;


        }
        watch_pp.Record(1);


        auto status4atoms = std::make_unique<MPI_Status[]>(num_valid);
        MPI_Waitall(num_valid, &request4atoms[0], &status4atoms[0]);
        watch_pp.Record(5);

#ifdef TIME_PP_MPI
        m_bench_flop_nonlocal += flop;
        if (m_bench_flop_nonlocal > LIMIT_64) {
            m_bench_flop_nonlocal -= LIMIT_64;
            m_bench_flop_nonlocal_high++;
        }
#endif


    }
#endif


    /*
    * 関数内でMPI_Iallreduceする
    * projectorの幅はgrad(p)も含んだ四倍になる
    */
    void mInnerNonlocalPsi_v5_one_bundle(int id_nucl, const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, 
        double* l_inner, int id_spin_kpoint, int num_proj, double dV, double* cut_psi_0, BlockCutInfo2* block_info) {


        const int num_bundle = end_n - begin_n;
        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];

        int64_t flop = 0;

        
        const int& ni = id_nucl;
        {
            
            const int index = 0;

            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;
#ifdef FORCE_DIFF_PROJ
            const int width = 4 * num_proj;
            //const int width = 4 * (m_nonlocal_projectors[ni].head_projector_l[MAX_L_SYSTEM] - m_nonlocal_projectors[ni].head_projector_l[0]);
#else
            const int width = num_proj;
            //const int width = (m_nonlocal_projectors[ni].head_projector_l[MAX_L_SYSTEM] - m_nonlocal_projectors[ni].head_projector_l[0]);
#endif

            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;

            if (grid_size == 0) {

                gy::ZeroClear(&l_inner[index], width * num_bundle * 2);
                    
                watch_pp.Record(1);
            } else {

                watch_pp.Record(1);


#ifdef GY_WITH_CUDA_OR_HIP
                CutSubgrid2_bundle(num_bundle, l_psi[begin_n].re, l_grid, blocks, cut_psi_0, block_info);
                gy::Synchronize();
#else
                for (int n = begin_n; n < end_n; ++n) {
                    CutSubgridByRanges(blocks.range_blocks, &cut_psi_0[grid_size * 2 * (n - begin_n)], l_grid, l_psi[n].re);
                    CutSubgridByRanges(blocks.range_blocks, &cut_psi_0[grid_size * (2 * (n - begin_n) + 1)], l_grid, l_psi[n].im);
                }
#endif
                watch_pp.Record(13);

                if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
                    for (int n = begin_n; n < end_n; ++n) {
                        double* psi_re = &cut_psi_0[grid_size * 2 * (n - begin_n)];
                        double* psi_im = &cut_psi_0[grid_size * (2 * (n - begin_n) + 1)];
                        const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                        const double* expikx_im = expikx_re + grid_size;
                        for (int64_t i = 0; i < grid_size; ++i) {
                            const double p_re = psi_re[i];
                            const double p_im = psi_im[i];

                            psi_re[i] = expikx_re[i] * p_re - expikx_im[i] * p_im;
                            psi_im[i] = expikx_re[i] * p_im + expikx_im[i] * p_re;

                        }
                    }
                }



                //Using BLAS DGEMM(faster than bottom because 1st size (size of A) is larger)
                blas_DGEMM_t(width, 2 * num_bundle, grid_size, proj_p, &cut_psi_0[0], &l_inner[index], width, dV, 0.0);
                gy::Synchronize();


                flop += width * grid_size * 4 * num_bundle; //4 meand fma (x2) and psi of complex(x2)//

                watch_pp.Record(4);
            }
            

        }

        watch_pp.Record(1);

#ifdef TIME_PP_MPI
        m_bench_flop_nonlocal += flop;
        if (m_bench_flop_nonlocal > LIMIT_64) {
            m_bench_flop_nonlocal -= LIMIT_64;
            m_bench_flop_nonlocal_high++;
        }
#endif

    }

    void mInnerNonlocalPsi_v5_bundle_ids(const SoAComplex* l_psi, int begin_n, int end_n, const GridRangeMPI& l_grid, 
        const Nucleus* nuclei, int num_nuclei, const int* nucl_ids, double* l_inner, double* sum_inner, int id_spin_kpoint,
        std::vector<int>& info_range, double dV, double* cut_psi_0, BlockCutInfo2* block_info)
    {
        const int num_bundle = end_n - begin_n;
        const int num_all_nonlocal = m_num_all_nonlocal;
/*
#ifdef FORCE_DIFF_PROJ
        double* sum_inner = l_inner + num_all_nonlocal * num_bundle * 2 * 4; //4 means p and grad(p), 2 means degree of complex//
#else
        double* sum_inner = l_inner + num_all_nonlocal * num_bundle * 2; //2 means degree of complex//
#endif
*/        
        auto request4atoms = std::make_unique<MPI_Request[]>(num_nuclei);


        int num_valid = 0;
        for (int nn = 0; nn < num_nuclei; ++nn) {
            const int ni = nucl_ids[nn];

            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) {
                continue;
            }

            const int num_proj = info_range[ni + 1] - info_range[ni];
#ifdef FORCE_DIFF_PROJ
            const int offset = info_range[ni] * num_bundle * 2 * 4;
            const int reduce_size = num_proj * num_bundle * 2 * 4;
#else
            const int offset = info_range[ni] * num_bundle * 2;
            const int reduce_size = num_proj * num_bundle * 2;
#endif
            mInnerNonlocalPsi_v5_one_bundle(ni, l_psi, begin_n, end_n, l_grid, l_inner + offset, id_spin_kpoint, num_proj, dV, cut_psi_0, block_info);
            gy::CopyMemoryToHost(sum_inner + offset, l_inner + offset, sizeof(double) * reduce_size);
            MPI_Iallreduce(MPI_IN_PLACE, sum_inner + offset,
                reduce_size, MPI_DOUBLE, MPI_SUM, mycomm, &request4atoms[num_valid]);

            ++num_valid;
        }

        auto status4atoms = std::make_unique<MPI_Status[]>(num_valid);
        MPI_Waitall(num_valid, &request4atoms[0], &status4atoms[0]);
        watch_pp.Record(5);

    }

public:
    /*
    * Nonlocal Energyを返す
    * bundle版
    */
    double EnergyPPnonlocal_bundle(const SoAComplex* l_psi, const double* occupancies, int begin_n, int end_n,
        const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, std::byte *work_buffer)
    {

        const int proc_id = GetProcessID(l_grid.mpi_comm);

        const double coef = (m_dx * m_dy * m_dz);
        const double dV = coef;
        

        watch_pp.Restart();


        const int num_all_nonlocal = m_num_all_nonlocal;
        auto& info_range = m_num_list_nonlocal;


        const int num_bundle = end_n - begin_n;
        double* l_inner = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(num_all_nonlocal * num_bundle * 2);
        //sum_innerはMPI_Iallreduceのためにホストメモリである必要がある//
        auto sum_inner = std::make_unique<double[]>(num_all_nonlocal * num_bundle * 2);



#ifdef PP_NONLOCAL_CUT_ALL
        
        int64_t total_grid = 0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm != MPI_COMM_NULL) {
                
                const auto& blocks = m_nonlocal_blocks[ni];
                const int64_t grid_size = blocks.grid_sizes;
                total_grid += grid_size;
            }
        }
        

        //working buffer used for local(cut) |psi> and the operated H|psi>. 
        auto cut_Hp_0 = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(2 * num_bundle * total_grid);//gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle * num_nuclei_effective);
        //gy::ZeroClear(cut_Hp_0.get(), m_max_grid_size * 2 * num_bundle * num_nuclei_effective);
        BlockCutInfo2* block_info = (BlockCutInfo2*)work_buffer;//gy::AlignedAlloc<BlockCutInfo2>(8 * num_nuclei_effective);
        watch_pp.Record(0);

        //calculate < p_{ n,j } | psi_i >
        mInnerNonlocalPsi_v6_bundle(l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);

        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
#else
        //working buffer used for local(cut) |psi> and the operated H|psi>. 
        auto cut_Hp_0 = (double*)work_buffer; work_buffer += gy::AlignedBytesize<double>(m_max_grid_size * 2 * num_bundle);// gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle);
        //gy::ZeroClear(cut_Hp_0.get(), m_max_grid_size * 2 * num_bundle);
        BlockCutInfo2* block_info = (BlockCutInfo2*)work_buffer;// gy::AlignedAlloc<BlockCutInfo2>(8);
        watch_pp.Record(0);

        //calculate < p_{ n,j } | psi_i >
        mInnerNonlocalPsi_v4_bundle(l_psi, begin_n, end_n, l_grid, nuclei, num_nuclei, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);
#endif

        double l_ene = 0.0;
        for (int ni = 0; ni < num_nuclei; ++ni) {
            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            if (!IsRoot(mycomm)) continue; //required this conddition to sum only for host//
            
            const int index = info_range[ni] * num_bundle * 2;

            //auto* proj_p = m_nonlocal_projectors[ni].projector_RY;


            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;


            auto& nonlocal_kspace = m_nonlocal_kspace[nuclei[ni].Z];
            const int width = nonlocal_kspace.num_proj_RY;

            // sum_{i,j} = < psi_i | p_{ n,j } > e_j < p_{ n,j } | psi_i >
            // where e_j = projector energy
            for (int n = 0; n < num_bundle * 2; ++n) {  //2 means Complex number, enen and odd of n is real-part and imaginary-part, respectively//
                double ene_n = 0.0;
                const int begin_w = 0;
                const int end_w = width;
                for (int w = begin_w; w < end_w; ++w) {
                    double& inner = sum_inner[index + width * n + w];
                    ene_n += inner * inner * nonlocal_kspace.projector_energy_RY[w];
                }

                l_ene += ene_n * occupancies[n / 2];
            }

            watch_pp.Record(8);
        }

        //gy::AlignedFree(l_inner);
        //gy::AlignedFree(block_info);


        double ene = 0.0;
        MPI_Reduce(&l_ene, &ene, 1, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        return ene;
    }



 public:

private:

    /*
    * projectorの将来的な補正のために作った関数
    * Rayleigh–Ritz methodを使って、低解像度の場合の補正を行う。
    */
    void mRitzCorrectionProjector(int Z, NonlocalKspace2* p_nonlocal_kspace) {
        const auto& nonlocal_kspace = *p_nonlocal_kspace;
        const auto& k_projector_Ylm = nonlocal_kspace.k_projector_Ylm;
        const int num_proj_RY = nonlocal_kspace.num_proj_RY;
        auto& subgrid_actual = nonlocal_kspace.subgrid;

        const int sub_size_x = subgrid_actual.SizeX();
        const int sub_size_y = subgrid_actual.SizeY();
        const int sub_size_z = subgrid_actual.SizeZ();
        const int64_t sub_size = sub_size_x * sub_size_y * sub_size_z;

        // P = <p_i|p_j>
        auto P = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);

        for (int m = 0; m < num_proj_RY; ++m) {
            const OneComplex* proj_RY_m = k_projector_Ylm[m];
            
            for (int k = 0; k < num_proj_RY; ++k) {
                const int il = k + num_proj_RY * m;

                const OneComplex* proj_RY_k = k_projector_Ylm[k]; 

                double sum = 0;
                for (int i = 0; i < sub_size; ++i) {
                    sum += proj_RY_k[i].r * proj_RY_m[i].r;
                    sum += proj_RY_k[i].i * proj_RY_m[i].i;
                }
                sum *= m_dx * m_dy * m_dz / (double)sub_size;

                P[il] = sum;

            }
        }

#define _DEBUG_3
#ifdef _DEBUG_3
        //ここからテスト1//////////////////////////////
        {
            std::string filename("matrix-P-diag3_Z" + std::to_string(Z) + ".txt");
            OutputMatrix(&P[0], num_proj_RY, num_proj_RY, filename.c_str());
        }
#endif
    
        //逆行列を求める
        auto P_inv = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
        InverseM(P.get(), num_proj_RY, P_inv.get());

#ifdef _DEBUG_3
        {
            std::string filename("matrix-invP-diag3_Z" + std::to_string(Z) + ".txt");
            OutputMatrix(&P_inv[0], num_proj_RY, num_proj_RY, filename.c_str());
        }
#endif
        
        //対角化//
        // |\tilde{p}_i> = |p_i> P^{-1}
        auto proj_tilde = gy::make_unique_aligned<OneComplex[]>(num_proj_RY * sub_size);
#if 1

        blas_DGEMM_n(sub_size*2, num_proj_RY, num_proj_RY, (double*)k_projector_Ylm[0], &P_inv[0], (double*)&proj_tilde[0], sub_size * 2, 1.0, 0.0);
        gy::Synchronize();
#else
        memset(proj_tilde.get(), 0, sizeof(OneComplex) * num_proj_RY * sub_size);
        
        for (int m = 0; m < num_proj_RY; ++m) {
            OneComplex* proj_tgt = &proj_tilde[m * sub_size];

            for (int k = 0; k < num_proj_RY; ++k) {
                const int il = k + num_proj_RY * m;

                const OneComplex* proj_RY_k = k_projector_Ylm[k];

                double sum = 0;
                for (int i = 0; i < sub_size; ++i) {
                    proj_tgt[i].r += proj_RY_k[i].r * P_inv[il];
                    proj_tgt[i].i += proj_RY_k[i].i * P_inv[il];
                }
            }
        }
#endif

        memcpy(k_projector_Ylm[0], proj_tilde.get(), sizeof(OneComplex) * num_proj_RY * sub_size);

    }


    //SymplecticIntegrator用に、projectorのグラム行列を準備する//
    // 具体的にはnonlocal_symplectic.matrix_ppに内積の行列をセット//    
    void mCreateSymplecticProjectorMatrix_PPmatrix(int Z, NonlocalKspace2* p_nonlocal_kspace) {

        auto& nonlocal_kspace = *p_nonlocal_kspace;

        auto it = m_nonlocal_symplectic.find(Z);
        if (it == m_nonlocal_symplectic.end()) {
            m_nonlocal_symplectic.emplace(std::make_pair(Z, NonlocalSymplectic{}));
            it = m_nonlocal_symplectic.find(Z);
        }
        auto& nonlocal_symplectic = it->second;

        //auto& nonlocal_kspace = m_nonlocal_kspace[Z];
        const auto& k_projector_Ylm = nonlocal_kspace.k_projector_Ylm;

        const int num_proj_RY = nonlocal_kspace.num_proj_RY;
        auto& subgrid_actual = nonlocal_kspace.subgrid;

#ifdef SMOOTH_SUBGRID
        const int sub_size_x = subgrid_actual.SizeX() - 1;
        const int sub_size_y = subgrid_actual.SizeY() - 1;
        const int sub_size_z = subgrid_actual.SizeZ() - 1;
        const int64_t sub_size = sub_size_x * sub_size_y * sub_size_z;

#else
        const int sub_size_x = subgrid_actual.SizeX();
        const int sub_size_y = subgrid_actual.SizeY();
        const int sub_size_z = subgrid_actual.SizeZ();
        const int64_t sub_size = sub_size_x * sub_size_y * sub_size_z;
#endif

        const int64_t total_buf_size = num_proj_RY * sub_size;
        nonlocal_symplectic.num_proj_RY = num_proj_RY;



        double* buffer = gy::AlignedAlloc<double>(total_buf_size * 4);
        double* proj_RY = buffer;
        auto* buf_dp_dx = buffer + total_buf_size;
        auto* buf_dp_dy = buffer + total_buf_size * 2;
        auto* buf_dp_dz = buffer + total_buf_size * 3;

#if SvF2_USE_INTERPOLATION_NL==5
        auto filter_xyz = gy::make_unique_aligned<double[]>(sub_size * 4);
        double* filter_x = &filter_xyz[0];
        double* filter_y = &filter_xyz[sub_size];
        double* filter_z = &filter_xyz[sub_size * 2];
        double* p_real = &filter_xyz[sub_size * 3];
        {
            auto SetRY = [&](auto filter, double* buf) {
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    for (int iy = 0; iy < sub_size_y; ++iy) {
                        for (int ix = 0; ix < sub_size_x; ++ix) {
                            const int i = ix + sub_size_x * (iy + sub_size_y * iz);
                            const double f = filter(ix + subgrid_actual.begin_x, iy + subgrid_actual.begin_y, iz + subgrid_actual.begin_z);
                            buf[i] = f;
                        }
                    }
                }
                };

            SetRY([&](int ix, int iy, int iz) {
                return ((double)ix - 0.5) * m_dx;
                }, filter_x);

            SetRY([&](int ix, int iy, int iz) {
                return ((double)iy - 0.5) * m_dy;
                }, filter_y);

            SetRY([&](int ix, int iy, int iz) {
                return ((double)iz - 0.5) * m_dz;
                }, filter_z);

        }

        auto MulSubgrid_R = [&](double* dest, const double* src, int count) {
            for (int i = 0; i < count; ++i) {
                dest[i] *= src[i];
            }
            };
        auto AddSubgrid_R = [&](double* dest, const double* src, int count) {
            for (int i = 0; i < count; ++i) {
                dest[i] += src[i];
            }
            };
#endif   //SvF2_USE_INTERPOLATION_NL==5



        auto temp = gy::make_unique_aligned<OneComplex[]>(sub_size);

        int64_t offset = 0;

        int index = 0;
        for (int k = 0; k < num_proj_RY; ++k) {

            auto* k_proj_Y = nonlocal_kspace.k_projector_Ylm[k];
            //auto* k_proj_Y = k_projector_Ylm + k * sub_size;
            auto*& fftw = nonlocal_kspace.fftw;



            MakeProjectorInRealspace(k_proj_Y, sub_size_x, sub_size_y, sub_size_z,
                nonlocal_kspace.odd_flag[k],
                &proj_RY[k * sub_size], &buf_dp_dx[k * sub_size], &buf_dp_dy[k * sub_size], &buf_dp_dz[k * sub_size], fftw, temp.get());




        }



        {//Symplectic用のmatrix<p_k|p_m>の生成
            if (nonlocal_symplectic.matrix_pp.get() == nullptr) {
                nonlocal_symplectic.matrix_pp = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY * 4);
                memset(&nonlocal_symplectic.matrix_pp[0], 0, sizeof(double)* num_proj_RY* num_proj_RY * 4);
            }
            auto& Mpp = nonlocal_symplectic.matrix_pp;
            auto& Mp_dp_dx = nonlocal_symplectic.matrix_p_dp_dx = &Mpp[num_proj_RY * num_proj_RY];
            auto& Mp_dp_dy = nonlocal_symplectic.matrix_p_dp_dy = &Mpp[num_proj_RY * num_proj_RY * 2];
            auto& Mp_dp_dz = nonlocal_symplectic.matrix_p_dp_dz = &Mpp[num_proj_RY * num_proj_RY * 3];


            const int64_t one_size = sub_size;

            for (int m = 0; m < num_proj_RY; ++m) {
                const double* proj_RY_m = proj_RY + one_size * m;
                const double* dp_dx_m = buf_dp_dx + one_size * m;
                const double* dp_dy_m = buf_dp_dy + one_size * m;
                const double* dp_dz_m = buf_dp_dz + one_size * m;

                for (int k = 0; k < num_proj_RY; ++k) {
                    //const int iu = m + num_proj_RY * k;
                    const int il = k + num_proj_RY * m;

                    const double* proj_RY_k = proj_RY + one_size * k;

                    double sum = 0;
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_z = 0;
                    for (int i = 0; i < one_size; ++i) {
                        sum += proj_RY_k[i] * proj_RY_m[i];
                        sum_x += proj_RY_k[i] * dp_dx_m[i];
                        sum_y += proj_RY_k[i] * dp_dy_m[i];
                        sum_z += proj_RY_k[i] * dp_dz_m[i];
                    }
                    sum *= m_dx * m_dy * m_dz;
                    sum_x *= m_dx * m_dy * m_dz;
                    sum_y *= m_dx * m_dy * m_dz;
                    sum_z *= m_dx * m_dy * m_dz;

                    Mpp[il] += sum;
                    Mp_dp_dx[il] += sum_x;
                    Mp_dp_dy[il] += sum_y;
                    Mp_dp_dz[il] += sum_z;

                }
            }



#if CORRECTION_ANTI_SYMMETRIC_P_GRAD_P==1
            //反対称化する(正確には符号を除いた対称性が崩れているので補正する)/////////////////
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = m + 1; k < num_proj_RY; ++k) {
                    const int il = k + num_proj_RY * m;
                    const int ir = m + num_proj_RY * k;
                    if ((nonlocal_kspace.proj_odd_flags[m] & ODD_YLM_X) == 0) {
                        Mp_dp_dx[ir] = -Mp_dp_dx[il];
                    } else {
                        Mp_dp_dx[il] = -Mp_dp_dx[ir];
                    }
                    if ((nonlocal_kspace.proj_odd_flags[m] & ODD_YLM_Y) == 0) {
                        Mp_dp_dy[ir] = -Mp_dp_dy[il];
                    } else {
                        Mp_dp_dy[il] = -Mp_dp_dy[ir];
                    }
                    if ((nonlocal_kspace.proj_odd_flags[m] & ODD_YLM_Z) == 0) {
                        Mp_dp_dz[ir] = -Mp_dp_dz[il];
                    } else {
                        Mp_dp_dz[il] = -Mp_dp_dz[ir];
                    }
                }
            }
#elif CORRECTION_ANTI_SYMMETRIC_P_GRAD_P==2
            //反対称化する(正確には符号を除いた対称性が崩れているので補正する)/////////////////
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = m + 1; k < num_proj_RY; ++k) {
                    const int il = k + num_proj_RY * m;
                    const int ir = m + num_proj_RY * k;
                    if ((nonlocal_kspace.proj_odd_flags[m] & ODD_YLM_X)) {
                        Mp_dp_dx[ir] = -Mp_dp_dx[il];
                    } else {
                        Mp_dp_dx[il] = -Mp_dp_dx[ir];
                    }
                    if ((nonlocal_kspace.proj_odd_flags[m] & ODD_YLM_Y)) {
                        Mp_dp_dy[ir] = -Mp_dp_dy[il];
                    } else {
                        Mp_dp_dy[il] = -Mp_dp_dy[ir];
                    }
                    if ((nonlocal_kspace.proj_odd_flags[m] & ODD_YLM_Z)) {
                        Mp_dp_dz[ir] = -Mp_dp_dz[il];
                    } else {
                        Mp_dp_dz[il] = -Mp_dp_dz[ir];
                    }
                }
            }
#endif
#ifdef _DEBUG_2
            std::string filename("matrix-pp-diag2_Z" + std::to_string(Z) + ".txt");
            OutputMatrix(&Mpp[0], num_proj_RY, num_proj_RY, filename.c_str());
            std::string filename1("matrix-p-dpdx-diag2_Z" + std::to_string(Z) + ".txt");
            OutputMatrix(&Mp_dp_dx[0], num_proj_RY, num_proj_RY, filename1.c_str());
            std::string filename2("matrix-p-dpdy-diag2_Z" + std::to_string(Z) + ".txt");
            OutputMatrix(&Mp_dp_dy[0], num_proj_RY, num_proj_RY, filename2.c_str());
            std::string filename3("matrix-p-dpdz-diag2_Z" + std::to_string(Z) + ".txt");
            OutputMatrix(&Mp_dp_dz[0], num_proj_RY, num_proj_RY, filename3.c_str());
#endif
        }

        gy::AlignedFree(buffer);


    }

    void mCreateSymplecticProjectorMatrix_YDXmatrix(int Z) {



        auto it = m_nonlocal_symplectic.find(Z);
        if (it == m_nonlocal_symplectic.end()) {
            m_nonlocal_symplectic.emplace(std::make_pair(Z, NonlocalSymplectic{}));
            it = m_nonlocal_symplectic.find(Z);
        }
        auto& nonlocal_symplectic = it->second;

        auto& nonlocal_kspace = m_nonlocal_kspace[Z];

        const int num_proj_RY = nonlocal_kspace.num_proj_RY;
        auto& subgrid_actual = nonlocal_kspace.subgrid;


        //matrix <p|p>を対角化する://
        //固有値全てを求めれば対角化の射影行列になっているはず//
        {
            auto& Mpp = nonlocal_symplectic.matrix_pp;
            auto& Y = nonlocal_symplectic.Y;
            auto& diagD = nonlocal_symplectic.diagD;
            Y = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            diagD = gy::make_unique_aligned<double[]>(num_proj_RY);
            lapack_DSYEVD(&Mpp[0], &diagD[0], &Y[0], num_proj_RY);

#ifdef _DEBUG_2
            //ここからテスト1//////////////////////////////
            {
                std::string filename("matrix-Y-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&Y[0], num_proj_RY, num_proj_RY, filename.c_str());
            }
#endif
            
            //対角化できているかテスト//
            auto MY = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            auto D = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            blas_DGEMM_n(num_proj_RY, num_proj_RY, num_proj_RY, &Mpp[0], &Y[0], &MY[0], num_proj_RY, 1.0, 0.0);
            blas_DGEMM_t(num_proj_RY, num_proj_RY, num_proj_RY, &Y[0], &MY[0], &D[0], num_proj_RY, 1.0, 0.0);
            
#ifdef _DEBUG_2
            gy::Synchronize();
            {
                std::string filename2("matrix-D-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&D[0], num_proj_RY, num_proj_RY, filename2.c_str());
            }
#endif

            auto YY = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            blas_DGEMM_t(num_proj_RY, num_proj_RY, num_proj_RY, &Y[0], &Y[0], &YY[0], num_proj_RY, 1.0, 0.0);
            gy::Synchronize();
#ifdef _DEBUG_2
            {
                std::string filename3("matrix-YY-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&YY[0], num_proj_RY, num_proj_RY, filename3.c_str());
            }
#endif

            //対角化の結果, 対角成分と固有値が一致しているか確認//
            for (int i = 0; i < num_proj_RY; ++i) {
                if (fabs(D[i + num_proj_RY * i] - diagD[i]) > 1.0e-10) {
                    printf("WARNING: D[diagonal:%d], lambda = %.10f, %.10f\n", i, D[i + num_proj_RY * i], diagD[i]);
                }
            }
            //対角成分が正か確認//
            for (int i = 0; i < num_proj_RY; ++i) {
                if (diagD[i] <= 0.0) {
                    printf("WARNING: negative D[diagonal:%d] & lambda, %.10f, %.10f\n", i, D[i + num_proj_RY * i], diagD[i]);
                }
            }
            //////////////////////////////ここまでテスト1//

            //matrix B -> Bbar の算出//
            //MBK pseudo potentialの場合はBが対角行列(projector_energy)
            const auto& projector_energy = nonlocal_kspace.projector_energy_RY;

            //YDh = Y * D^{1/2}//
            auto YDh = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = 0; k < num_proj_RY; ++k) {
                    int i = k + num_proj_RY * m;
                    YDh[i] = Y[i] * sqrt(diagD[m]);
                }
            }
            //BYDh = B[diag] * YDh;
            auto BYDh = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = 0; k < num_proj_RY; ++k) {
                    int i = k + num_proj_RY * m;
                    BYDh[i] = projector_energy[k] * YDh[i];
                }
            }
            //Bbar = (YDh)^t B (YDh) = (YDh)^t BYDh
            auto Bbar = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            blas_DGEMM_t(num_proj_RY, num_proj_RY, num_proj_RY, &YDh[0], &BYDh[0], &Bbar[0], num_proj_RY, 1.0, 0.0);
#ifdef _DEBUG_2
            gy::Synchronize();
            {
                std::string filename("matrix-Bbar-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&Bbar[0], num_proj_RY, num_proj_RY, filename.c_str());
            }
#endif
            //Bbar の対角化//
            auto& X = nonlocal_symplectic.X;
            auto& diagLambda = nonlocal_symplectic.diagLambda;
            X = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            diagLambda = gy::make_unique_aligned<double[]>(num_proj_RY);
            lapack_DSYEVD(&Bbar[0], &diagLambda[0], &X[0], num_proj_RY);

            //ここからテスト2//////////////////////////////
#ifdef _DEBUG_2
            {
                std::string filename("matrix-X-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&X[0], num_proj_RY, num_proj_RY, filename.c_str());
            }
            {
                std::string filename("lambda-X-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&diagLambda[0], 1, num_proj_RY, filename.c_str());
            }
#endif

            //対角化できているかテスト//
            auto BbarX = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            auto XtBbarX = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            blas_DGEMM_n(num_proj_RY, num_proj_RY, num_proj_RY, &Bbar[0], &X[0], &BbarX[0], num_proj_RY, 1.0, 0.0);
            blas_DGEMM_t(num_proj_RY, num_proj_RY, num_proj_RY, &X[0], &BbarX[0], &XtBbarX[0], num_proj_RY, 1.0, 0.0);
#ifdef _DEBUG_2
            gy::Synchronize();
            {
                std::string filename2("matrix-XtBbarX-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&XtBbarX[0], num_proj_RY, num_proj_RY, filename2.c_str());
            }
#endif

            auto XX = gy::make_unique_aligned<double[]>(num_proj_RY* num_proj_RY);
            blas_DGEMM_t(num_proj_RY, num_proj_RY, num_proj_RY, &X[0], &X[0], &XX[0], num_proj_RY, 1.0, 0.0);
            gy::Synchronize();
#ifdef _DEBUG_2
            {
                std::string filename3("matrix-XX-diag2_Z" + std::to_string(Z) + ".txt");
                OutputMatrix(&XX[0], num_proj_RY, num_proj_RY, filename3.c_str());
            }
#endif
            //対角化の結果, 対角成分と固有値が一致しているか確認//
            for (int i = 0; i < num_proj_RY; ++i) {
                if (fabs(XtBbarX[i + num_proj_RY * i] - diagLambda[i]) > 1.0e-10) {
                    printf("WARNING: XtBbarX[diagonal:%d], lambda = %.10f, %.10f\n", i, XtBbarX[i + num_proj_RY * i], diagLambda[i]);
                }
            }
            
            //////////////////////////////ここまでテスト2//

            // よく使う(Y D^{-1/2} X)の作成//
            auto& YDmh = YDh;//recycle the buffer//
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = 0; k < num_proj_RY; ++k) {
                    int i = k + num_proj_RY * m;
                    YDmh[i] = Y[i] / sqrt(diagD[m]);
                }
            }
            auto& YDmhX = nonlocal_symplectic.YDmhX;
            YDmhX = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
            blas_DGEMM_n(num_proj_RY, num_proj_RY, num_proj_RY, &YDmh[0], &X[0], &YDmhX[0], num_proj_RY, 1.0, 0.0);            
            gy::Synchronize();

            //ここからテスト3//////////////////////////////
            // YDmhX * Lambda * (YDmhX)^t = B_{mk}
            // となっているか確認
            // ただし、ここでは MBK pseudo-potential、すなわち、//
            // B_{mk}が対角行列である(対角成分はprojector_energy)ことを前提とした確認//
            auto YDmhXL = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = 0; k < num_proj_RY; ++k) {
                    int i = k + num_proj_RY * m;
                    YDmhXL[i] = YDmhX[i] * diagLambda[m];
                }
            }
            auto reB = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
            blas_DGEMM_bt(num_proj_RY, num_proj_RY, num_proj_RY, &YDmhXL[0], &YDmhX[0], &reB[0], num_proj_RY, 1.0, 0.0);
            gy::Synchronize();

            //Bを再度求めた結果、対角成分がが一致しているか確認//
            for (int i = 0; i < num_proj_RY; ++i) {
                if (fabs(reB[i + num_proj_RY * i] - projector_energy[i]) > 1.0e-8) {
                    printf("WARNING: reB[diagonal:%d], projector_energy = %.10f, %.10f\n", i, reB[i + num_proj_RY * i], projector_energy[i]);
                }
            }

            //////////////////////////////ここまでテスト3//

        }


    }

    
    void mSetSymplecticLiouvilian(double dt) {
        if (dt == m_dt) return;

        m_dt = dt;

        for (auto&& a : m_nonlocal_symplectic) {
            auto& nonlocal_symplectic = a.second;
            const int num_proj_RY = nonlocal_symplectic.num_proj_RY;
            auto& YDmhX = nonlocal_symplectic.YDmhX;
            const auto& diagLambda = nonlocal_symplectic.diagLambda;
            auto& YDmhX_expL_sub_one_XtDmhYt_re = nonlocal_symplectic.YDmhX_expL_sub_one_XtDmhYt_re;
            auto& YDmhX_expL_sub_one_XtDmhYt_im = nonlocal_symplectic.YDmhX_expL_sub_one_XtDmhYt_im;
            YDmhX_expL_sub_one_XtDmhYt_re = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
            YDmhX_expL_sub_one_XtDmhYt_im = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
            //
            
            //real part
            auto YDmhX_expL_sub_one = gy::make_unique_aligned<double[]>(num_proj_RY * num_proj_RY);
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = 0; k < num_proj_RY; ++k) {
                    int i = k + num_proj_RY * m;
                    double phase = -dt * diagLambda[m];
                    YDmhX_expL_sub_one[i] = YDmhX[i] * (cos(phase)-1.0);
                }
            }
            blas_DGEMM_bt(num_proj_RY, num_proj_RY, num_proj_RY, &YDmhX_expL_sub_one[0], &YDmhX[0], &YDmhX_expL_sub_one_XtDmhYt_re[0], num_proj_RY, 1.0, 0.0);
            gy::Synchronize();

            //imaginaly part            
            for (int m = 0; m < num_proj_RY; ++m) {
                for (int k = 0; k < num_proj_RY; ++k) {
                    int i = k + num_proj_RY * m;
                    double phase = -dt * diagLambda[m];
                    YDmhX_expL_sub_one[i] = YDmhX[i] * sin(phase);
                }
            }
            blas_DGEMM_bt(num_proj_RY, num_proj_RY, num_proj_RY, &YDmhX_expL_sub_one[0], &YDmhX[0], &YDmhX_expL_sub_one_XtDmhYt_im[0], num_proj_RY, 1.0, 0.0);
            gy::Synchronize();

        }
    }
    

public:
    //with grad(psi)//
    double SymplecticIntegration_bundle(double dt, double* forces, SoAComplex* l_psi_next, const SoAComplex* l_psi_and_grad_psi,
        const double* occupancies, int num_bundle,
        const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei,
        int id_spin_kpoint, double gx, double gy, double gz) {

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];;

        watch_pp.Restart();
        
        mSetSymplecticLiouvilian(dt);

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const int local_size = l_grid.Size3D();

        const double dV = m_dx * m_dy * m_dz;


        const int num_all_nonlocal = m_num_all_nonlocal;


        auto l_inner = gy::make_unique_aligned<double[]>(num_all_nonlocal * num_bundle * 2 * 4);
        //sum_innerはMPI_Iallreduceのためにホストメモリである必要がある//
        auto sum_inner = std::make_unique<double[]>(num_all_nonlocal * num_bundle * 2 * 4);


        auto& info_range = m_num_list_nonlocal;


        //working buffer used for local(cut) |psi> and the operated H|psi>. 
        auto cut_Hp_0 = gy::make_unique_aligned<double[]>(m_max_grid_size * 2 * num_bundle);
        BlockCutInfo2* block_info = gy::AlignedAlloc<BlockCutInfo2>(8);

        //calculate < p_{ n,j } | psi_i >
        mInnerNonlocalPsi_v4_bundle(l_psi_and_grad_psi, 0, num_bundle * 4, l_grid, nuclei, num_nuclei, &l_inner[0], &sum_inner[0], id_spin_kpoint, info_range, dV, &cut_Hp_0[0], block_info);
        

        double l_ene = 0.0;
        auto l_forces = gy::make_unique_aligned<double[]>(num_nuclei * 3 * 2);
        double* sum_forces = &l_forces[0] + num_nuclei * 3;
        memset(&l_forces[0], 0, sizeof(double) * num_nuclei * 3*2);


        for (int ni = 0; ni < num_nuclei; ++ni) {

            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;
            
            const int index = info_range[ni] * num_bundle * 2 * 4; //2 means Complex, and 4 means psi and its derivatives//


            const auto& blocks = m_nonlocal_blocks[ni];            
            const int64_t grid_size = blocks.grid_sizes;


            auto& nonlocal_symplectic = m_nonlocal_symplectic[nuclei[ni].Z];
            const int width = nonlocal_symplectic.num_proj_RY;
            auto& M_re = nonlocal_symplectic.YDmhX_expL_sub_one_XtDmhYt_re;
            auto& M_im = nonlocal_symplectic.YDmhX_expL_sub_one_XtDmhYt_im;
            auto M_p_psi = gy::make_unique_aligned<double[]>(width * num_bundle * 2);
            double* M_p_psi_re = &M_p_psi[0];
            double* M_p_psi_im = &M_p_psi[width * num_bundle];

            int stride_row = width * 8;
            auto p_psi_re = gy::make_unique_aligned<double[]>(width * num_bundle * 2);
            double* p_psi_im = &p_psi_re[width * num_bundle];
            for (int n = 0; n < num_bundle; ++n) {
                for (int w = 0; w < width; ++w) {
                    p_psi_re[w + width * n] = sum_inner[index + width * (8 * n) + w];
                    p_psi_im[w + width * n] = sum_inner[index + width * (8 * n + 1) + w];
                }
            }
            /*
            {
                {
                    std::string filename2("p-psi_n" + std::to_string(ni) + "_mpi" + std::to_string(proc_id) + ".txt");
                    OutputMatrix(&p_psi_im[0], num_bundle*2, width, filename2.c_str());
                }
            }*/

            // M_p_psi_re = M_re * p_psi_re - M_im * p_psi_im;
            blas_DGEMM_n(width, num_bundle, width, &M_re[0], &p_psi_re[0], &M_p_psi_re[0], width, 1.0, 0.0);
            blas_DGEMM_n(width, num_bundle, width, &M_im[0], &p_psi_im[0], &M_p_psi_re[0], width, -1.0, 1.0);

            // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;
            blas_DGEMM_n(width, num_bundle, width, &M_im[0], &p_psi_re[0], &M_p_psi_im[0], width, 1.0, 0.0);
            blas_DGEMM_n(width, num_bundle, width, &M_re[0], &p_psi_im[0], &M_p_psi_im[0], width, 1.0, 1.0);


            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;
            
            blas_DGEMM_n(grid_size, num_bundle, width, proj_p, &M_p_psi_re[0], &cut_Hp_0[0], grid_size, 1.0, 0.0);
            blas_DGEMM_n(grid_size, num_bundle, width, proj_p, &M_p_psi_im[0], &cut_Hp_0[0] + grid_size * num_bundle, grid_size, 1.0, 0.0);
            gy::Synchronize();

            if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
                for (int n = 0; n < num_bundle; ++n) {
                    double* cut_Hp_l_re = &cut_Hp_0[0] + grid_size * n;
                    double* cut_Hp_l_im = &cut_Hp_0[0] + grid_size * (num_bundle + n);

                    const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                    const double* expikx_im = expikx_re + grid_size;
                    for (int64_t i = 0; i < grid_size; ++i) {
                        const double p_re = cut_Hp_l_re[i];
                        const double p_im = cut_Hp_l_im[i];

                        cut_Hp_l_re[i] = expikx_re[i] * p_re + expikx_im[i] * p_im;
                        cut_Hp_l_im[i] = expikx_re[i] * p_im - expikx_im[i] * p_re;

                    }
                }
            }

            watch_pp.Record(6);


            for (int n = 0; n < num_bundle; ++n) {
                AddSubgridByRanges(l_grid, l_psi_next[n].re, blocks.range_blocks, &cut_Hp_0[0] + grid_size * n);
                AddSubgridByRanges(l_grid, l_psi_next[n].im, blocks.range_blocks, &cut_Hp_0[0] + grid_size * (num_bundle + n));
            }
            watch_pp.Record(7);


            if (IsRoot(mycomm)) { //required this conddition to sum only for host//


                //<psi|dp_dR_alpha> = - <psi|dp_dx> = <dpsi_dx|p> = <exp(kx) * dphi_dx|p> + ik* <exp(kx) * phi|p> 
                for (int n = 0; n < num_bundle; ++n) {
                    for (int w = 0; w < width; ++w) {

                        double& inner_re = sum_inner[index + width * (8 * n) + w];
                        double& inner_im = sum_inner[index + width * (8 * n + 1) + w];


                        double& diff_x_re = sum_inner[index + width * (8 * n + 2) + w];
                        double& diff_x_im = sum_inner[index + width * (8 * n + 3) + w];

                        double& diff_y_re = sum_inner[index + width * (8 * n + 4) + w];
                        double& diff_y_im = sum_inner[index + width * (8 * n + 5) + w];


                        double& diff_z_re = sum_inner[index + width * (8 * n + 6) + w];
                        double& diff_z_im = sum_inner[index + width * (8 * n + 7) + w];

                        diff_x_im += gx * inner_re;
                        diff_x_re += -gx * inner_im;
                        diff_y_im += gy * inner_re;
                        diff_y_re += -gy * inner_im;
                        diff_z_im += gz * inner_re;
                        diff_z_re += -gz * inner_im;

                        int i = w + width * n;
                        const double coef_for_cc = 2.0 * occupancies[n]; //to be c.c.// 
                        l_forces[ni * 3 + 0] += coef_for_cc * (diff_x_re * M_p_psi_im[i] - diff_x_im * M_p_psi_re[i]);
                        l_forces[ni * 3 + 1] += coef_for_cc * (diff_y_re * M_p_psi_im[i] - diff_y_im * M_p_psi_re[i]);
                        l_forces[ni * 3 + 2] += coef_for_cc * (diff_z_re * M_p_psi_im[i] - diff_z_im * M_p_psi_re[i]);

                    }


                }

#if 1                
                constexpr double coef_b = -1.0;
                {
                    auto& p_dp_dx = nonlocal_symplectic.matrix_p_dp_dx;
                    // tmp = <p|dp/dx> * (M <p|psi>);
                    auto tmp_re = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto tmp_im = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto F2 = gy::make_unique_aligned<double[]>(num_bundle * num_bundle);

                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dx[0], &M_p_psi_re[0], &tmp_re[0], width, 1.0, 0.0);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dx[0], &M_p_psi_im[0], &tmp_im[0], width, 1.0, 0.0);
                    // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;                
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_re[0], &tmp_im[0], &F2[0], num_bundle, 1.0, 0.0);
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_im[0], &tmp_re[0], &F2[0], num_bundle, -1.0, 1.0);
                    gy::Synchronize();
                    for (int n = 0; n < num_bundle; ++n) {
                        l_forces[ni * 3 + 0] += coef_b*F2[n + num_bundle * n] * occupancies[n];
                    }
                }

                {
                    auto& p_dp_dy = nonlocal_symplectic.matrix_p_dp_dy;
                    auto tmp_re = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto tmp_im = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto F2 = gy::make_unique_aligned<double[]>(num_bundle * num_bundle);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dy[0], &M_p_psi_re[0], &tmp_re[0], width, 1.0, 0.0);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dy[0], &M_p_psi_im[0], &tmp_im[0], width, 1.0, 0.0);
                    // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_re[0], &tmp_im[0], &F2[0], num_bundle, 1.0, 0.0);
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_im[0], &tmp_re[0], &F2[0], num_bundle, -1.0, 1.0);
                    gy::Synchronize();
                    for (int n = 0; n < num_bundle; ++n) {
                        l_forces[ni * 3 + 1] += coef_b * F2[n + num_bundle * n] * occupancies[n];
                    }
                }

                {
                    auto& p_dp_dz = nonlocal_symplectic.matrix_p_dp_dz;
                    auto tmp_re = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto tmp_im = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto F2 = gy::make_unique_aligned<double[]>(num_bundle * num_bundle);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dz[0], &M_p_psi_re[0], &tmp_re[0], width, 1.0, 0.0);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dz[0], &M_p_psi_im[0], &tmp_im[0], width, 1.0, 0.0);
                    // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_re[0], &tmp_im[0], &F2[0], num_bundle, 1.0, 0.0);
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_im[0], &tmp_re[0], &F2[0], num_bundle, -1.0, 1.0);
                    gy::Synchronize();
                    for (int n = 0; n < num_bundle; ++n) {
                        l_forces[ni * 3 + 2] += coef_b * F2[n + num_bundle * n] * occupancies[n];
                    }
                }
#endif
            }
        }



        

        watch_pp.Record(9);


        //double ene = 0.0;
        //MPI_Reduce(&l_ene, &ene, 1, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        MPI_Reduce(&l_forces[0], sum_forces, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        if (IsRoot(l_grid.mpi_comm)) {
            for (int ni = 0; ni < num_nuclei; ++ni) {
                forces[ni * 3 + 0] += sum_forces[ni * 3 + 0];
                forces[ni * 3 + 1] += sum_forces[ni * 3 + 1];
                forces[ni * 3 + 2] += sum_forces[ni * 3 + 2];

                printf("Fnl = %.10f, %.10f, %.10f\n", forces[ni * 3 + 0], forces[ni * 3 + 1], forces[ni * 3 + 2]);
            }
        }

        return 0.0;
    }

    /*
    *wrapper of SymplecticIntegration_bundle_ids
    * non-overlapの原子リストごとに計算する//
    *with grad(psi)
    *
    */
    double SymplecticIntegration_bundle2(double dt, double* forces, SoAComplex* l_psi_next, const SoAComplex* l_psi_and_grad_psi,
        const double* occupancies, int num_bundle,
        const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei, bool forward_or_backward,
        int id_spin_kpoint, double gx, double gy, double gz) {

        if (forward_or_backward > 0) {
            int head = 0;
            for (const auto& count : m_sorted_count_list) {

                SymplecticIntegration_bundle_ids(dt, forces, l_psi_next, l_psi_and_grad_psi, occupancies, num_bundle,
                    l_grid, nuclei, count, &m_sorted_proc_ids[head], id_spin_kpoint, gx, gy, gz);

                head += count;
            }
        } else {
            int head = num_nuclei;
            for (auto it = m_sorted_count_list.rbegin(); it != m_sorted_count_list.rend();++it) {
                int count = *it;
                head -= count;
                SymplecticIntegration_bundle_ids(dt, forces, l_psi_next, l_psi_and_grad_psi, occupancies, num_bundle,
                    l_grid, nuclei, count, &m_sorted_proc_ids[head], id_spin_kpoint, gx, gy, gz);
            }
        }

        return 0.0;
    }

    //with grad(psi)//
    double SymplecticIntegration_bundle_ids(double dt, double* forces, SoAComplex* l_psi_next, const SoAComplex* l_psi_and_grad_psi,
        const double* occupancies, int num_bundle,
        const GridRangeMPI& l_grid, const Nucleus* nuclei, int num_nuclei, const int* nucl_ids,
        int id_spin_kpoint, double gx, double gy, double gz) {

        const bool is_gamma = m_is_gamma_point_list[id_spin_kpoint];;

        watch_pp.Restart();

        mSetSymplecticLiouvilian(dt);

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const int local_size = l_grid.Size3D();

        const double dV = m_dx * m_dy * m_dz;


        const int num_all_nonlocal = m_num_all_nonlocal;


        double* l_inner = gy::AlignedAlloc<double>(num_all_nonlocal * num_bundle * 2 * 4);
        //sum_innerはMPI_Iallreduceのためにホストメモリである必要がある//
        auto sum_inner = std::make_unique<double[]>(num_all_nonlocal * num_bundle * 2 * 4);


        auto& info_range = m_num_list_nonlocal;
        auto cut_psi_0 = gy::AlignedAlloc<double>(m_max_grid_size * 2 * num_bundle);
        BlockCutInfo2* block_info = gy::AlignedAlloc<BlockCutInfo2>(8);
        watch_pp.Record(0);
#ifdef FORCE_DIFF_PROJ
        //use grad(p), where p is projector//
        mInnerNonlocalPsi_v5_bundle_ids(l_psi_and_grad_psi, 0, num_bundle, l_grid, nuclei, num_nuclei, nucl_ids, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_psi_0[0], block_info);
#else
        //use grad(psi), where psi is orbital//
        mInnerNonlocalPsi_v5_bundle_ids(l_psi_and_grad_psi, 0, num_bundle * 4, l_grid, nuclei, num_nuclei, nucl_ids, l_inner, sum_inner.get(), id_spin_kpoint, info_range, dV, &cut_psi_0[0], block_info);
#endif

        double l_ene = 0.0;
        double* l_forces = gy::AlignedAlloc<double>(num_nuclei * 3 * 4);
        double* sum_forces = l_forces + num_nuclei * 3;
        double* l_forces2 = l_forces + num_nuclei * 3 * 2;
        double* sum_forces2 = l_forces + num_nuclei * 3 * 3;
        gy::ZeroClear(l_forces, num_nuclei * 3 * 4);

        double* cut_Hp_0 = cut_psi_0;

        for (int nn = 0; nn < num_nuclei; ++nn) {
            int ni = nucl_ids[nn];

            auto mycomm = m_comm4atoms.GetComm(ni);
            if (mycomm == MPI_COMM_NULL) continue;

            const int index = info_range[ni] * num_bundle * 2 * 4; //2 means Complex, and 4 means psi and its derivatives//


            const auto& blocks = m_nonlocal_blocks[ni];
            const int64_t grid_size = blocks.grid_sizes;


            auto& nonlocal_symplectic = m_nonlocal_symplectic[nuclei[ni].Z];
            const int width = nonlocal_symplectic.num_proj_RY;
            auto& M_re = nonlocal_symplectic.YDmhX_expL_sub_one_XtDmhYt_re;
            auto& M_im = nonlocal_symplectic.YDmhX_expL_sub_one_XtDmhYt_im;
            auto M_p_psi = gy::make_unique_aligned<double[]>(width * num_bundle * 2);
            double* M_p_psi_re = &M_p_psi[0];
            double* M_p_psi_im = &M_p_psi[width * num_bundle];

            int stride_row = width * 8;
            auto p_psi_re = gy::make_unique_aligned<double[]>(width * num_bundle * 2);
            double* p_psi_im = &p_psi_re[width * num_bundle];
            for (int n = 0; n < num_bundle; ++n) {
                for (int w = 0; w < width; ++w) {
#ifdef FORCE_DIFF_PROJ
                    p_psi_re[w + width * n] = sum_inner[index + width * (8 * n) + w];
                    p_psi_im[w + width * n] = sum_inner[index + width * (8 * n + 4) + w];
#else
                    p_psi_re[w + width * n] = sum_inner[index + width * (8 * n) + w];
                    p_psi_im[w + width * n] = sum_inner[index + width * (8 * n + 1) + w];
#endif
                }
            }
            /*
            {
                {
                    std::string filename2("p-psi_n" + std::to_string(ni) + "_mpi" + std::to_string(proc_id) + ".txt");
                    OutputMatrix(&p_psi_im[0], num_bundle*2, width, filename2.c_str());
                }
            }*/

            // M_p_psi_re = M_re * p_psi_re - M_im * p_psi_im;
            blas_DGEMM_n(width, num_bundle, width, &M_re[0], &p_psi_re[0], &M_p_psi_re[0], width, 1.0, 0.0);
            blas_DGEMM_n(width, num_bundle, width, &M_im[0], &p_psi_im[0], &M_p_psi_re[0], width, -1.0, 1.0);

            // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;
            blas_DGEMM_n(width, num_bundle, width, &M_im[0], &p_psi_re[0], &M_p_psi_im[0], width, 1.0, 0.0);
            blas_DGEMM_n(width, num_bundle, width, &M_re[0], &p_psi_im[0], &M_p_psi_im[0], width, 1.0, 1.0);


            auto* proj_p = m_nonlocal_projectors[ni].projector_RY;

            blas_DGEMM_n(grid_size, num_bundle, width, proj_p, &M_p_psi_re[0], cut_Hp_0, grid_size, 1.0, 0.0);
            blas_DGEMM_n(grid_size, num_bundle, width, proj_p, &M_p_psi_im[0], cut_Hp_0 + grid_size * num_bundle, grid_size, 1.0, 0.0);
            gy::Synchronize();

            if (!is_gamma) {//k点サンプルのBlock定理によるずらし//
                for (int n = 0; n < num_bundle; ++n) {
                    double* cut_Hp_l_re = cut_Hp_0 + grid_size * n;
                    double* cut_Hp_l_im = cut_Hp_0 + grid_size * (num_bundle + n);

                    const double* expikx_re = m_nonlocal_projectors[ni].bloch_expikx[id_spin_kpoint];
                    const double* expikx_im = expikx_re + grid_size;
                    for (int64_t i = 0; i < grid_size; ++i) {
                        const double p_re = cut_Hp_l_re[i];
                        const double p_im = cut_Hp_l_im[i];

                        cut_Hp_l_re[i] = expikx_re[i] * p_re + expikx_im[i] * p_im;
                        cut_Hp_l_im[i] = expikx_re[i] * p_im - expikx_im[i] * p_re;

                    }
                }
            }

            watch_pp.Record(6);


            for (int n = 0; n < num_bundle; ++n) {
                AddSubgridByRanges(l_grid, l_psi_next[n].re, blocks.range_blocks, cut_Hp_0 + grid_size * n);
                AddSubgridByRanges(l_grid, l_psi_next[n].im, blocks.range_blocks, cut_Hp_0 + grid_size * (num_bundle + n));
            }
            watch_pp.Record(7);


            if (IsRoot(mycomm)) { //required this conddition to sum only for host//

#ifdef FORCE_DIFF_PROJ
                //<psi|dp_dR_alpha> = - <psi|dp_dx> = -conj(<dp_dx|psi>)
                for (int n = 0; n < num_bundle; ++n) {
                    for (int w = 0; w < width; ++w) {

                        OneComplex inner{ sum_inner[index + width * (8 * n) + w], sum_inner[index + width * (8 * n + 4) + w] };
                        

                        OneComplex psi_dpdRx{- sum_inner[index + width * (8 * n + 1) + w], sum_inner[index + width * (8 * n + 5) + w] };
                        OneComplex psi_dpdRy{- sum_inner[index + width * (8 * n + 2) + w], sum_inner[index + width * (8 * n + 6) + w] };
                        OneComplex psi_dpdRz{- sum_inner[index + width * (8 * n + 3) + w], sum_inner[index + width * (8 * n + 7) + w] };

                        int i = w + width * n;
                        const double coef_for_cc = 2.0 * occupancies[n]; //to be c.c.// 
                        l_forces[nn * 3 + 0] += coef_for_cc * (psi_dpdRx.r * M_p_psi_im[i] + psi_dpdRx.i * M_p_psi_re[i]);
                        l_forces[nn * 3 + 1] += coef_for_cc * (psi_dpdRy.r * M_p_psi_im[i] + psi_dpdRy.i * M_p_psi_re[i]);
                        l_forces[nn * 3 + 2] += coef_for_cc * (psi_dpdRz.r * M_p_psi_im[i] + psi_dpdRz.i * M_p_psi_re[i]);

                    }
                }




#else
                //<psi|dp_dR_alpha> = - <psi|dp_dx> = <dpsi_dx|p> = <exp(kx) * dphi_dx|p> + ik* <exp(kx) * phi|p> 
                for (int n = 0; n < num_bundle; ++n) {
                    for (int w = 0; w < width; ++w) {

                        double& inner_re = sum_inner[index + width * (8 * n) + w];
                        double& inner_im = sum_inner[index + width * (8 * n + 1) + w];

                        OneComplex psi_dpdRx{ sum_inner[index + width * (8 * n + 2) + w], -sum_inner[index + width * (8 * n + 3) + w] };
                        OneComplex psi_dpdRy{ sum_inner[index + width * (8 * n + 4) + w], -sum_inner[index + width * (8 * n + 5) + w] };
                        OneComplex psi_dpdRz{ sum_inner[index + width * (8 * n + 6) + w], -sum_inner[index + width * (8 * n + 7) + w] };

                        psi_dpdRx.i += gx * inner_re;
                        psi_dpdRx.r += -gx * inner_im;
                        psi_dpdRy.i += gy * inner_re;
                        psi_dpdRy.r += -gy * inner_im;
                        psi_dpdRz.i += gz * inner_re;
                        psi_dpdRz.r += -gz * inner_im;
                        
                        double& diff_x_re = sum_inner[index + width * (8 * n + 2) + w];
                        double& diff_x_im = sum_inner[index + width * (8 * n + 3) + w];

                        double& diff_y_re = sum_inner[index + width * (8 * n + 4) + w];
                        double& diff_y_im = sum_inner[index + width * (8 * n + 5) + w];

                        double& diff_z_re = sum_inner[index + width * (8 * n + 6) + w];
                        double& diff_z_im = sum_inner[index + width * (8 * n + 7) + w];
                        
                        diff_x_im += gx * inner_re;
                        diff_x_re += -gx * inner_im;
                        diff_y_im += gy * inner_re;
                        diff_y_re += -gy * inner_im;
                        diff_z_im += gz * inner_re;
                        diff_z_re += -gz * inner_im;
                        

                        int i = w + width * n;
                        const double coef_for_cc = 2.0 * occupancies[n]; //to be c.c.// 
                        
                        l_forces[nn * 3 + 0] += coef_for_cc * (psi_dpdRx.r * M_p_psi_im[i] + psi_dpdRx.i * M_p_psi_re[i]);
                        l_forces[nn * 3 + 1] += coef_for_cc * (psi_dpdRy.r * M_p_psi_im[i] + psi_dpdRy.i * M_p_psi_re[i]);
                        l_forces[nn * 3 + 2] += coef_for_cc * (psi_dpdRz.r * M_p_psi_im[i] + psi_dpdRz.i * M_p_psi_re[i]);
                        
 /*                       
                        l_forces[nn * 3 + 0] += coef_for_cc * (diff_x_re * M_p_psi_im[i] - diff_x_im * M_p_psi_re[i]);
                        l_forces[nn * 3 + 1] += coef_for_cc * (diff_y_re * M_p_psi_im[i] - diff_y_im * M_p_psi_re[i]);
                        l_forces[nn * 3 + 2] += coef_for_cc * (diff_z_re * M_p_psi_im[i] - diff_z_im * M_p_psi_re[i]);
*/                        
                    }
                }

#endif //FORCE_DIFF_PROJ
          
                constexpr double coef_b = -1.0;
                {
                    auto& p_dp_dx = nonlocal_symplectic.matrix_p_dp_dx;
                    // tmp = <p|dp/dx> * (M <p|psi>);
                    auto tmp_re = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto tmp_im = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto F2 = gy::make_unique_aligned<double[]>(num_bundle * num_bundle);

                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dx[0], &M_p_psi_re[0], &tmp_re[0], width, 1.0, 0.0);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dx[0], &M_p_psi_im[0], &tmp_im[0], width, 1.0, 0.0);
                    // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;                
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_re[0], &tmp_im[0], &F2[0], num_bundle, 1.0, 0.0);
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_im[0], &tmp_re[0], &F2[0], num_bundle, -1.0, 1.0);
                    gy::Synchronize();
                    for (int n = 0; n < num_bundle; ++n) {
                        l_forces2[nn * 3 + 0] += coef_b * F2[n + num_bundle * n] * occupancies[n];
                    }
                }

                {
                    auto& p_dp_dy = nonlocal_symplectic.matrix_p_dp_dy;
                    auto tmp_re = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto tmp_im = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto F2 = gy::make_unique_aligned<double[]>(num_bundle * num_bundle);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dy[0], &M_p_psi_re[0], &tmp_re[0], width, 1.0, 0.0);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dy[0], &M_p_psi_im[0], &tmp_im[0], width, 1.0, 0.0);
                    // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_re[0], &tmp_im[0], &F2[0], num_bundle, 1.0, 0.0);
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_im[0], &tmp_re[0], &F2[0], num_bundle, -1.0, 1.0);
                    gy::Synchronize();
                    for (int n = 0; n < num_bundle; ++n) {
                        l_forces2[nn * 3 + 1] += coef_b * F2[n + num_bundle * n] * occupancies[n];
                    }
                }

                {
                    auto& p_dp_dz = nonlocal_symplectic.matrix_p_dp_dz;
                    auto tmp_re = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto tmp_im = gy::make_unique_aligned<double[]>(width * num_bundle);
                    auto F2 = gy::make_unique_aligned<double[]>(num_bundle * num_bundle);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dz[0], &M_p_psi_re[0], &tmp_re[0], width, 1.0, 0.0);
                    blas_DGEMM_n(width, num_bundle, width, &p_dp_dz[0], &M_p_psi_im[0], &tmp_im[0], width, 1.0, 0.0);
                    // M_p_psi_im = M_im * p_psi_re + M_re * p_psi_im;
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_re[0], &tmp_im[0], &F2[0], num_bundle, 1.0, 0.0);
                    blas_DGEMM_t(num_bundle, num_bundle, width, &M_p_psi_im[0], &tmp_re[0], &F2[0], num_bundle, -1.0, 1.0);
                    gy::Synchronize();
                    for (int n = 0; n < num_bundle; ++n) {
                        l_forces2[nn * 3 + 2] += coef_b * F2[n + num_bundle * n] * occupancies[n];
                    }
                }

            }
        }



        gy::AlignedFree(l_inner);
        gy::AlignedFree(cut_psi_0);

        watch_pp.Record(9);

#if 1
        //double ene = 0.0;
        //MPI_Reduce(&l_ene, &ene, 1, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        //MPI_Reduce(l_forces, sum_forces, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        //MPI_Reduce(l_forces2, sum_forces2, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        //if (IsRoot(l_grid.mpi_comm)) {
        for (int nn = 0; nn < num_nuclei; ++nn) {
            int ni = nucl_ids[nn];
            forces[ni * 3 + 0] += l_forces[nn * 3 + 0] + l_forces2[nn * 3 + 0];
            forces[ni * 3 + 1] += l_forces[nn * 3 + 1] + l_forces2[nn * 3 + 1];
            forces[ni * 3 + 2] += l_forces[nn * 3 + 2] + l_forces2[nn * 3 + 2];
        }
        
#else
        //double ene = 0.0;
        //MPI_Reduce(&l_ene, &ene, 1, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        MPI_Reduce(l_forces, sum_forces, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        MPI_Reduce(l_forces2, sum_forces2, num_nuclei * 3, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
        if (IsRoot(l_grid.mpi_comm)) {
            for (int nn = 0; nn < num_nuclei; ++nn) {
                int ni = nucl_ids[nn];
                forces[ni * 3 + 0] += sum_forces[nn * 3 + 0] + sum_forces2[nn * 3 + 0];
                forces[ni * 3 + 1] += sum_forces[nn * 3 + 1] + sum_forces2[nn * 3 + 1];
                forces[ni * 3 + 2] += sum_forces[nn * 3 + 2] + sum_forces2[nn * 3 + 2];
#ifdef _DEBUG
                //printf("Fnl = %.10f, %.10f, %.10f\n", forces[ni * 3 + 0], forces[ni * 3 + 1], forces[ni * 3 + 2]);
                printf("Fnl1 = %.10f, %.10f, %.10f\n", sum_forces[nn * 3 + 0], sum_forces[nn * 3 + 1], sum_forces[nn * 3 + 2]);
                printf("Fnl2 = %.10f, %.10f, %.10f\n", sum_forces2[nn * 3 + 0], sum_forces2[nn * 3 + 1], sum_forces2[nn * 3 + 2]);
#endif
            }
        }
#endif
        gy::AlignedFree(l_forces);
        return 0.0;
    }




private:
#ifdef TIME_PP_MPI
    void mPrintTime() {
        //if (watch_pp.GetTime(4) > watch_pp.GetTime(5)) 


        printf("PP calculation times====\n");
        watch_pp.Print("mem-alloc      ", 0);
        watch_pp.Print("other          ", 1);
        watch_pp.Print("Cut_psi(bundle)", 3);
        watch_pp.Print("Cut_psi(force) ", 11);
        watch_pp.Print("Cut_psi(symple)", 13);
        watch_pp.Print("Inner(bundle)  ", 10);
        watch_pp.Print("Inner(force)   ", 12);
        watch_pp.Print("Inner(symple)  ", 4);
        watch_pp.Print("Iallreduce     ", 5);
        watch_pp.Print("pre-proj       ", 2);
        watch_pp.Print("Projection     ", 6);
        watch_pp.Print("Paste          ", 7);
        watch_pp.Print("Energy(post)   ", 8);
        watch_pp.Print("ForceNL(post)  ", 9);

        watch_pp.Print("CountNLProj    ", 14);
        watch_pp.Print("UpdatePosLocal ", 15);
        watch_pp.Print("mSetSubspaceBlk", 16);
        watch_pp.Print("mSetNLProjInfo ", 17);
        watch_pp.Print("comm4atoms     ", 18);
        watch_pp.Print("mCreatePrjRYSvF", 19);
#ifdef NONLOCAL_SvF_PARALLEL
        watch_pp.Print("--barrier      ", 20);
        watch_pp.Print("--pre          ", 21);
        watch_pp.Print("--SvF          ", 22);
        watch_pp.Print("--paste        ", 24);
        watch_pp.Print("--Allgath/toall", 23);
#endif
    }


    void mPrintTimeMax() {

        
#ifdef NONLOCAL_SvF_PARALLEL
        double max_tms[5];
        double tms[5];
        for (int i = 0; i < 5; ++i) {
            tms[i] = watch_pp.GetTime(20 + i);
        }
        //暫定処理MPI_COMM_WORLD//
        MPI_Reduce(tms, max_tms, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (is_root) {

            printf("(max)--barrier      : %f [s]\n", max_tms[0]);
            printf("(max)--pre          : %f [s]\n", max_tms[1]);
            printf("(max)--SvF          : %f [s]\n", max_tms[2]);
            printf("(max)--paste        : %f [s]\n", max_tms[4]);
            printf("(max)--Allgath/toall: % f[s]\n", max_tms[3]);
        }
#endif

    }

#endif

public:
    void BenchFlopsNonlocal(double* flops_list) {
#ifdef TIME_PP_MPI        
        flops_list[0] = (m_bench_flop_nonlocal == 0) ? 0.0 : (double)m_bench_flop_nonlocal / watch_pp.GetTime(10);
        flops_list[1] = (m_bench_flop_nonlocal_f == 0) ? 0.0 : (double)m_bench_flop_nonlocal_f / watch_pp.GetTime(12);
#else
        
#endif
    }



};



#endif  //USE_MPI
