#pragma once
#include <map>
#include "vps_loader.h"
#include "wave_function.h"
#include "nucleus.h"
//#include "SubspaceField.h"
#include "RealSphericalHarmonics.h"
#include "vecmath.h"
#include "ellipsoidal_integral.h"

//#define COMMON_CUTOFF_L     //slower than org.

inline double pow2(double x) { return x * x; };
inline double fold(double x, double box_w) { return (x *2.0> box_w) ? x - box_w : (x*2.0 < -box_w) ? x + box_w : x; };

/*
* Pseudo Potentialをsubspace(範囲の限定された)のグリッドデータで保持する
* subspaceの中心は原子座標であり、このクラスも原子毎に保持する
* 同じ原子番号であっても位置が異なればこのクラスのデータは異なるため
*/
/*
struct PP_nonlocal_each_atom {
	std::vector<SubspaceField> projector;
	std::vector<SubspaceField> Ylm;
};
*/

template<class YLM>
void SetYlmAroundAny(YLM& Y, double value_org, double* buf, int ix_begin, int iy_begin, int iz_begin,
		int subsize_x, int subsize_y, int subsize_z, 
		double R0_x, double R0_y, double R0_z,
		double dx, double dy, double dz	)
{

	
	constexpr double rr_min = 1.0e-14;

	for (int iz = 0; iz < subsize_z; ++iz) {
		const double z = dz * (double)(iz + iz_begin) - R0_z;
		const double rz2 = z * z;
		for (int iy = 0; iy < subsize_y; ++iy) {
			const double y = dy * (double)(iy + iy_begin) - R0_y;
			const double ry2 = y * y;
			for (int ix = 0; ix < subsize_x; ++ix) {
				const double x = dx * (double)(ix + ix_begin) - R0_x;
				const double rx2 = x * x;
				const size_t i = (size_t)ix + (size_t)subsize_x * ((size_t)iy + ((size_t)subsize_y * (size_t)iz));

				const double rr = rx2 + ry2 + rz2;
				if (rr_min > rr) {
					buf[i] = value_org;
				} else {
					const double r = sqrt(rr);
					buf[i] = Y(x / r, y / r, z / r);
				}
			}
		}
	}
}

template<int L, int M>
void SetYlmAround(double* buf, int ix_begin, int iy_begin, int iz_begin,
	int subsize_x, int subsize_y, int subsize_z,
	double R0_x, double R0_y, double R0_z,
	double dx, double dy, double dz)
{
	const double Y00 = 1.0 / sqrt(4.0 * M_PI);
	Ylm<L, M> Y;
	SetYlmAroundAny(Y, (L == 0) ? Y00 : 0.0, buf, ix_begin, iy_begin, iz_begin,
		subsize_x, subsize_y, subsize_z,
		R0_x, R0_y, R0_z, dx, dy, dz);
}



/*
* PPのベースクラス
* 実質的にはCoreCoreEnergyCorrection()関数だけが機能としては生きている
*/
class PseudoPotIntegrator {
protected:
    GridInfo m_grid;
    std::map<int, PseudoPot_MBK*> m_pp_list;

    //std::vector<PP_nonlocal_each_atom> m_pp_by_atom;

    double m_dx;
    double m_dy;
    double m_dz;

public:

    virtual ~PseudoPotIntegrator() {
        //printf("destruct PseudoPotIntegrator\n"); fflush(stdout);
        for (auto&& a : m_pp_list) {
            a.second->Release();
        }
    }

    /*
    * [usage] 派生クラスからは呼ばれない
    */
    void InitializeGrid(const GridInfo& grid_, double dx, double dy, double dz) {
        m_grid = grid_;
        m_dx = dx;
        m_dy = dy;
        m_dz = dz;
    }


    /*
    * [usage] 派生クラスから呼ばれる. 
    */
    PseudoPot_MBK* mFindPseudoPot(int Z)
    {
        auto it = m_pp_list.find(Z);
        if (it == m_pp_list.end()) {
            return nullptr;
        }
        return it->second;
    }

    double MaxCutoffInAllElements() {
        double max_cutoff = 0.0;
        for (auto&& a : m_pp_list) {
            double c = a.second->MaxCutoffLength();
            if (max_cutoff < c) max_cutoff = c;
        }
        return max_cutoff;
    }

    /*
    * 原子番号Zの原子のVlocal(動径関数)において、距離rの値を取得
    * 
    * [usage] publicに外部から呼ばれている
    */
    double GetCutoffVlocal(int Z) {
        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        return pp->cutoff_vlocal;
    }

    /*
    * [deplicated] より精度の高いエネルギー計算では必要ない. 初期実装との比較用に残していある.
    * [usage] 派生クラスでは呼ばれていない
    * 原子番号Zの原子のVlocal(動径関数)において、距離rの値を取得
    */
    double GetVlocalRadial(int Z, double r) {
        const PseudoPot_MBK* pp = mFindPseudoPot(Z);

        const double r_min = pp->radius[0];
        //const double r_max = pp->radius[pp->num_radial_grids - 2];
        const double r_max = pp->cutoff_vlocal;
        if (r_min > r) {
            return pp->V_local[0];
        } else if (r_max <= r) {
            return -(pp->valence_electron) / r;
        } else {
            return RadialGrid2::GetValue(r, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
        }
    }

    /*
    * [usage] 派生クラスから呼ばれない
    */
    double NumValenceElectron(int Z) {
        const PseudoPot_MBK* pp = mFindPseudoPot(Z);
        return pp->valence_electron;
    }


    /*
    * [usage] 派生クラスから呼ばれない
    * 原子核間がvlocalのカットオフ長の和よりも短い距離になった場合の補正
    * 原子核間の2体相互作用をvlocalとcore電子密度の相互作用として求める.
    * 本体のqumasun_baseクラスでは、これを差っ引いた後、Tohmas-Fermi原子核間ポテンシャル(R to 0でE to inftyになるもの, 将来的にReGZ)を加える.
    */
    double CoreCoreEnergyCorrection(double R, int iZ_a, int iZ_b, double* force) {

        const PseudoPot_MBK* pp_a = mFindPseudoPot(iZ_a);
        const PseudoPot_MBK* pp_b = mFindPseudoPot(iZ_b);
        const double cutoff_a = pp_a->cutoff_vlocal;
        const double cutoff_b = pp_b->cutoff_vlocal;
        const double Q_a = pp_a->valence_electron;
        const double Q_b = pp_b->valence_electron;
        const double Z_a = (double)iZ_a;
        const double Z_b = (double)iZ_b;


        struct ENN_SELF {
            double E_Vlocal_rho = 0.0;
            double Force_Vlocal_rho = 0.0;

            ENN_SELF& operator+=(const ENN_SELF& b) {
                E_Vlocal_rho += b.E_Vlocal_rho;
                Force_Vlocal_rho += b.Force_Vlocal_rho;
                return *this;
            }

            ENN_SELF operator*(const double b) const {
                ENN_SELF r;
                r.E_Vlocal_rho = E_Vlocal_rho * b;
                r.Force_Vlocal_rho = Force_Vlocal_rho * b;
                return r;
            }
        };


        if (pp_b->has_nucl_charge) {
            //cut off / 4.0 is better, empirically.
            ENN_SELF Enn = ReGZ::IntegrateEllipsoidal<ENN_SELF, 96, 64>(R, cutoff_a / 4.0, cutoff_b / 4.0,
                //ENN_SELF Enn = ReGZ::IntegrateEllipsoidal<ENN_SELF, 96, 64>(R, cutoff_a, cutoff_b,
                //double Enn = ReGZ::IntegrateEllipsoidal<96, 64>(R, 1.0, 1.0,
                [&R, &pp_a, &pp_b, &cutoff_a, &cutoff_b](double r_a, double r_b) {
                    if (r_b < cutoff_b) {
                        double V = RadialGrid2::GetValue(r_a, pp_a->V_local, pp_a->num_radial_grids, pp_a->xi_min, pp_a->xi_delta);
                        double dV_dr = RadialGrid2::GetDifferential1(r_a, pp_a->V_local, pp_a->num_radial_grids, pp_a->xi_min, pp_a->xi_delta);
                        double rho = RadialGrid2::GetValue(r_b, pp_b->nucl_charge, pp_b->num_radial_grids, pp_b->xi_min, pp_b->xi_delta);
                        
                        double cos_theta_a = (R * R + r_a * r_a - r_b * r_b) / (2.0 * R * r_a);
                        return ENN_SELF{ V * rho,  -dV_dr * rho * cos_theta_a };
                    } else {
                        return ENN_SELF{ 0.0, 0.0 };
                    }

                });
            
            *force = Enn.Force_Vlocal_rho;
            return Enn.E_Vlocal_rho;

        } else {
            //cut off / 4.0 is better, empirically.
            ENN_SELF Enn = ReGZ::IntegrateEllipsoidal<ENN_SELF, 96, 64>(R, cutoff_a / 4.0, cutoff_b / 4.0,
                //ENN_SELF Enn = ReGZ::IntegrateEllipsoidal<ENN_SELF, 96, 64>(R, cutoff_a, cutoff_b,
                //double Enn = ReGZ::IntegrateEllipsoidal<96, 64>(R, 1.0, 1.0,
                [&R, &pp_a, &pp_b, &cutoff_a, &cutoff_b](double r_a, double r_b) {
                    if (r_b < cutoff_b) {
                        double V = RadialGrid2::GetValue(r_a, pp_a->V_local, pp_a->num_radial_grids, pp_a->xi_min, pp_a->xi_delta);
                        double dV_dr = RadialGrid2::GetDifferential1(r_a, pp_a->V_local, pp_a->num_radial_grids, pp_a->xi_min, pp_a->xi_delta);
                        double rho = (1.0 / (2.0 * M_PI)) * RadialGrid2::GetLaplacian(r_b, pp_b->V_local, 0, pp_b->num_radial_grids, pp_b->xi_min, pp_b->xi_delta);
                        
                        double cos_theta_a = (R * R + r_a * r_a - r_b * r_b) / (2.0 * R * r_a);
                        return ENN_SELF{ V * rho,  -dV_dr * rho * cos_theta_a };
                    } else {
                        return ENN_SELF{ 0.0, 0.0 };
                    }

                });

            *force = Enn.Force_Vlocal_rho;
            return Enn.E_Vlocal_rho;
        }
        
    }
};

