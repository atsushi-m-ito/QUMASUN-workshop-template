/************************************************************
原子配置に対して左右反転対称性やXYミラー対称性があるか確認する

ただし、前提条件として、以下を仮定する
(1)simulation boxは周期境界とする
(2)Simulation boxの左下の点を原点(0,0,0)とする

*************************************************************/

#pragma once
#include <cstdint>
#include <cmath>
#include <memory>
#include "nucleus.h"


namespace QUMASUN {
    namespace KPOINT_SYMMETRY {
        static constexpr uint32_t NONE = 0;
        static constexpr uint32_t NEGAPOSI_X = 0x1;
        static constexpr uint32_t NEGAPOSI_Y = 0x2;
        static constexpr uint32_t NEGAPOSI_Z = 0x4;
        static constexpr uint32_t MIRROR_XY = 0x8;
        static constexpr uint32_t MIRROR_YZ = 0x10;
        static constexpr uint32_t MIRROR_ZX = 0x20;
        static constexpr uint32_t FULL_XYZ = 0x3F;

        static constexpr uint32_t NEGAPOSI_BOTH_XY = 0x100;
        static constexpr uint32_t NEGAPOSI_BOTH_YZ = 0x200;
        static constexpr uint32_t NEGAPOSI_BOTH_ZX = 0x400;
        static constexpr uint32_t NEGAPOSI_ALL_XYZ = 0x800;
        static constexpr uint32_t AUTO = 0x80000000;
    };

    class SymmetryChecker {
    private:
        double m_threshold_distance = 1.0e-4;
    public:
        uint32_t Check(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            //auto mirror_exist = std::make_unique<int[]>(num_nuclei);
            uint32_t flags = KPOINT_SYMMETRY::NONE;
            if (CheckMirrorXY(nuclei, num_nuclei, boxaxis)) {
                flags |= KPOINT_SYMMETRY::MIRROR_XY;
            }
            if (CheckMirrorYZ(nuclei, num_nuclei, boxaxis)) {
                flags |= KPOINT_SYMMETRY::MIRROR_YZ;
            }
            if (CheckMirrorZX(nuclei, num_nuclei, boxaxis)) {
                flags |= KPOINT_SYMMETRY::MIRROR_ZX;
            }
            if (CheckInversionX(nuclei, num_nuclei, boxaxis)) {
                flags |= KPOINT_SYMMETRY::NEGAPOSI_X;
            }
            if (CheckInversionY(nuclei, num_nuclei, boxaxis)) {
                flags |= KPOINT_SYMMETRY::NEGAPOSI_Y;
            }
            if (CheckInversionZ(nuclei, num_nuclei, boxaxis)) {
                flags |= KPOINT_SYMMETRY::NEGAPOSI_Z;
            }

            if (((flags & KPOINT_SYMMETRY::NEGAPOSI_X) == 0) && ((flags & KPOINT_SYMMETRY::NEGAPOSI_Y) == 0)) {
                if (CheckInversionBothXY(nuclei, num_nuclei, boxaxis)) {
                    flags |= KPOINT_SYMMETRY::NEGAPOSI_BOTH_XY;
                }
            }
            if (((flags & KPOINT_SYMMETRY::NEGAPOSI_Y) == 0) && ((flags & KPOINT_SYMMETRY::NEGAPOSI_Z) == 0)) {
                if (CheckInversionBothYZ(nuclei, num_nuclei, boxaxis)) {
                    flags |= KPOINT_SYMMETRY::NEGAPOSI_BOTH_YZ;
                }
            }
            if (((flags & KPOINT_SYMMETRY::NEGAPOSI_Z) == 0) && ((flags & KPOINT_SYMMETRY::NEGAPOSI_X) == 0)) {
                if (CheckInversionBothZX(nuclei, num_nuclei, boxaxis)) {
                    flags |= KPOINT_SYMMETRY::NEGAPOSI_BOTH_ZX;
                }
            }
            if (((flags & KPOINT_SYMMETRY::NEGAPOSI_X) == 0) && ((flags & KPOINT_SYMMETRY::NEGAPOSI_Y) == 0) && ((flags & KPOINT_SYMMETRY::NEGAPOSI_Z) == 0)) {
                if (CheckInversionAllXYZ(nuclei, num_nuclei, boxaxis)) {
                    flags |= KPOINT_SYMMETRY::NEGAPOSI_ALL_XYZ;
                }
            }
            return flags;
        }

    private:

        static inline double FoldingDistance(double dx, double box_w) {
            double x = dx / box_w;
            x = x - floor(x);
            x = std::min(x, 1.0 - x);
            return x * box_w;
        }

        //指定した座標の粒子と同じ位置の粒子を探す//
        bool Hit(Nucleus& target, const Nucleus* nuclei, int num_nuclei, const double* boxaxis)const  {
            const double d2_threshold = m_threshold_distance * m_threshold_distance;

            int num_mirror = 0;
            for (int i = 0; i < num_nuclei; ++i) {
                if (target.Z != nuclei[i].Z) continue;
                const double dx = FoldingDistance(target.Rx - nuclei[i].Rx, boxaxis[0]);
                const double dy = FoldingDistance(target.Ry - nuclei[i].Ry, boxaxis[4]);
                const double dz = FoldingDistance(target.Rz - nuclei[i].Rz, boxaxis[8]);

                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < d2_threshold) {
                    return true;
                }
            }
            return false;
        }

        template <class FUNC>
        bool HitAll(const Nucleus* nuclei, int num_nuclei, const double* boxaxis, FUNC convert) const {
            int hit_count = 0;
            for (int i = 0; i < num_nuclei; ++i) {
                Nucleus target = convert(nuclei[i]);//  座標変換//
                if (Hit(target, nuclei, num_nuclei, boxaxis)) {
                    ++hit_count;
                }
            }

            return (hit_count == num_nuclei);

        }

        bool CheckMirrorXY(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = a.Ry;
                    b.Ry = a.Rx;
                    b.Rz = a.Rz;
                    return b;
                });
        }

        bool CheckMirrorYZ(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = a.Rx;
                    b.Ry = a.Rz;
                    b.Rz = a.Ry;
                    return b;
                });
        }

        bool CheckMirrorZX(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = a.Rz;
                    b.Ry = a.Ry;
                    b.Rz = a.Rx;
                    return b;
                });
        }


        bool CheckInversionX(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = -a.Rx + boxaxis[0];
                    b.Ry = a.Ry;
                    b.Rz = a.Rz;
                    return b;
                });
        }
        bool CheckInversionY(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = a.Rx;
                    b.Ry = -a.Ry + boxaxis[4];
                    b.Rz = a.Rz;
                    return b;
                });
        }
        bool CheckInversionZ(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = a.Rx;
                    b.Ry = a.Ry;
                    b.Rz = -a.Rz + boxaxis[8];
                    return b;
                });
        }

        //x,yを同時反転させた場合に対称//
        bool CheckInversionBothXY(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = -a.Rx + boxaxis[0];
                    b.Ry = -a.Ry + boxaxis[4];
                    b.Rz = a.Rz;
                    return b;
                });
        }
        //y,zを同時反転させた場合に対称//
        bool CheckInversionBothYZ(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = a.Rx;
                    b.Ry = -a.Ry + boxaxis[4];
                    b.Rz = -a.Rz + boxaxis[8];
                    return b;
                });
        }

        //z,xを同時反転させた場合に対称//
        bool CheckInversionBothZX(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = -a.Rx + boxaxis[0];
                    b.Ry = a.Ry;
                    b.Rz = -a.Rz + boxaxis[8];
                    return b;
                });
        }

        //x,y,zを同時反転させた場合に対称//
        bool CheckInversionAllXYZ(const Nucleus* nuclei, int num_nuclei, const double* boxaxis) const {
            return HitAll(nuclei, num_nuclei, boxaxis,
                [&](const Nucleus& a) {
                    Nucleus b;
                    b.Z = a.Z;
                    b.Rx = -a.Rx + boxaxis[0];
                    b.Ry = -a.Ry + boxaxis[4];
                    b.Rz = -a.Rz + boxaxis[8];
                    return b;
                });
        }
    };
}
