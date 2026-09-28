#pragma once

#include "Vxc.h"



class LDA_PW92 {
public:
    static constexpr double SMALL_LIMIT = 1.0e-14;

    static double EnergyDensityPotential_C(double rho, double rho_diff, double* p_v_c_up, double* p_v_c_down) {
        if (rho < SMALL_LIMIT) {
            *p_v_c_up = *p_v_c_down = 0.0;
            return 0.0;
        }
        const double zeta = std::clamp(rho_diff / rho, -1.0, 1.0);
        const double rs = std::cbrt(3.0 / (4.0 * M_PI * rho));
        constexpr double c_2_4_3_sub_2 = 0.51984209978974632953442121455646;

        constexpr double A[3]{ 0.031091, 0.015545, 0.016887 };
        constexpr double alpha[3]{ 0.21370, 0.20548, 0.11125 };
        constexpr double beta1[3]{ 7.5957, 14.1189, 10.357 };
        constexpr double beta2[3]{ 3.5876, 6.1977, 3.6231 };
        constexpr double beta3[3]{ 1.6382, 3.3662, 0.88026 };
        constexpr double beta4[3]{ 0.49294, 0.62517, 0.49671 };

        auto get_e_c = [](double* p_de_c_drs, double rs, double zeta,
            double A, double alpha, double beta1, double beta2, double beta3, double beta4) {
                const double sqrt_rs = std::sqrt(rs);
                double mother = 2.0 * A * (beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs);
                double ln_term = std::log(1.0 + 1.0 / mother);
                double e_c = -2.0 * A * (1.0 + alpha * rs) * ln_term;

                double dmother_drs = 2.0 * A * (beta1 / (2.0 * sqrt_rs) + beta2 + (3.0 / 2.0) * beta3 * sqrt_rs + 2.0 * beta4 * rs);
                double dln_drs = -dmother_drs / ((mother + 1.0) * mother);

                *p_de_c_drs = -2.0 * A * (alpha * ln_term + (1.0 + alpha * rs) * dln_drs);                

                return e_c;
            };

        double de_c_drs_0, de_c_drs_1, da_c_drs;
        const double e_c_0 = get_e_c(&de_c_drs_0, rs, zeta, A[0], alpha[0], beta1[0], beta2[0], beta3[0], beta4[0]);
        const double e_c_1 = get_e_c(&de_c_drs_1, rs, zeta, A[1], alpha[1], beta1[1], beta2[1], beta3[1], beta4[1]);
        const double a_c = -get_e_c(&da_c_drs, rs, zeta, A[2], alpha[2], beta1[2], beta2[2], beta3[2], beta4[2]);
        da_c_drs = -da_c_drs;

        double cbrt_one_add_zeta = std::cbrt(1.0 + zeta);
        double cbrt_one_sub_zeta = std::cbrt(1.0 - zeta);

        double f_zeta = ((1.0 + zeta) * cbrt_one_add_zeta + (1.0 - zeta) * cbrt_one_sub_zeta - 2.0) / c_2_4_3_sub_2;
        constexpr double d2f_dz2 = (8.0 / 9.0) / c_2_4_3_sub_2; //=1.7099209341613653

        double zeta3 = zeta * zeta * zeta;
        double zeta4 = zeta * zeta * zeta * zeta;
        double e_c = e_c_0 + a_c * f_zeta / d2f_dz2 * (1.0 - zeta4) + (e_c_1 - e_c_0) * f_zeta * zeta4;

        double de_c_drs = de_c_drs_0 + da_c_drs * f_zeta / d2f_dz2 * (1.0 - zeta4) + (de_c_drs_1 - de_c_drs_0) * f_zeta * zeta4;

        double df_dzeta = ((4.0 / 3.0) / c_2_4_3_sub_2) * (cbrt_one_add_zeta - cbrt_one_sub_zeta);
        //double de_c_dzeta = a_c * df_dzeta / d2f_dz2 * (1.0 - zeta4) + (de_c_drs_1 - de_c_drs_0) * df_dzeta * zeta4
        //    + (-da_c_drs / d2f_dz2 + (de_c_drs_1 - de_c_drs_0)) * f_zeta * zeta3;
        double de_c_dzeta = (a_c / d2f_dz2) * (df_dzeta * (1.0 - zeta4) - 4.0 * f_zeta * zeta3)
            + (e_c_1 - e_c_0) * (df_dzeta * zeta4 + 4.0 * f_zeta * zeta3);
                

        double dE_c_drho = e_c - (1.0 / 3.0) * rs * de_c_drs;
        *p_v_c_up = dE_c_drho - (zeta - 1.0) * de_c_dzeta ;
        *p_v_c_down = dE_c_drho - (zeta + 1.0) * de_c_dzeta ;
        

        return e_c;
    }

};


class GGA_PBE {
public:
    //for exchange//
    static constexpr double kappa = 0.804;
    static constexpr double mu = 0.21951;

    //for correlation//
    static constexpr double beta = 0.066725;
    static constexpr double gamma = 0.031091;

    //parameters//
    //static const double C = mu / (std::pow(48.0 * M_PI * M_PI, 2.0 / 3.0) * kappa) ;
    static constexpr double c = mu / (60.770664964607961830508850127991 * kappa);

    // static const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
    static constexpr double K_lda = 3.0 * 0.62035049089940001666800681204778;

    static constexpr double SMALL_LIMIT = 1.0e-8;

    /*
    * rho_upとrho_downでそれぞれ呼ぶこと
    * e_x(rho)を返す
    */
    static double EnergyDensity_X(double rho_up, double sq_grad_rho_up) {
        if (rho_up < SMALL_LIMIT) return 0.0;

        //const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
        //const double rho_4_3 = std::pow(rho_up, 4.0 / 3.0);
        const double rho_1_3 = std::cbrt(rho_up);
        const double t = sq_grad_rho_up / ((rho_up * rho_1_3) * (rho_up * rho_1_3));

        const double f = 1.0 + kappa - kappa / (1.0 + c * t);
        //const double f = 1.0;

        return -(1.0 / 2.0) * K_lda * rho_1_3 * f;
    }


    /*
    * rho_up: electron density
    * sq_grad_rho_up; (\nabla \rho)^2
    * laplace_rho: \Delta \rho
    * dif_rho_vmv = \sum_{a,b} (\nabla_a \rho) (\nabla_b \rho) (\nabla_a \nabla_b \rho)
    */

    static double Potential_X(double rho_up, double sq_grad_rho_up, double laplace_rho, double dif_rho_vmv) {
        if (rho_up < SMALL_LIMIT) return 0.0;

        //const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
        const double n = 4.0 / 3.0;
        const double rho_n_1 = std::pow(rho_up, n - 1.0);
        const double rho_n = rho_n_1 * rho_up;
        const double m = 8.0 / 3.0;
        //const double t = sq_grad_rho_up / std::pow(rho_up, m);
        const double t = sq_grad_rho_up / (rho_n * rho_n);
        /*
        const double f = 1.0;
        const double df_dt = 0.0;
        const double d2f_dt2 = 0.0;
        */
        const double f = 1.0 + kappa - kappa / (1.0 + c * t);
        const double df_dt = kappa * c / ((1.0 + c * t) * (1.0 + c * t));
        const double d2f_dt2 = -2.0 * kappa * c * c / ((1.0 + c * t) * (1.0 + c * t) * (1.0 + c * t));


        return -(1.0 / 2.0) * K_lda * (
            n * rho_n_1 * f
            //+ (m-2.0*n) * rho_n_1 * df_dt(t) * t    this is zero because m = 2*n
            - 2.0 / rho_n * df_dt * laplace_rho
            + 2.0 * m * rho_n_1 * d2f_dt2 * t * t
            - 4.0 * d2f_dt2 * dif_rho_vmv / (rho_n * rho_n * rho_n) //2m-n = 3n
            );
    }

#if 1

    static double Potential_X_test1(double rho_up, double sq_grad_rho_up, double laplace_rho, double dif_rho_vmv, double* p_v_x_up) {
        if (rho_up < SMALL_LIMIT) {
            *p_v_x_up = 0.0;
            return 0.0;
        }


        //const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
        const double n = 4.0 / 3.0;
        const double rho_n_1 = std::pow(rho_up, n - 1.0);
        const double rho_n = rho_n_1 * rho_up;
        //const double m = 8.0 / 3.0;
        const double m = 8.0 / 3.0;   
        const double rho_m = std::pow(rho_up, m);
        const double rho_m_1 = std::pow(rho_up, m-1.0);
        //const double t = sq_grad_rho_up / std::pow(rho_up, m);
        //const double t = sq_grad_rho_up / (rho_n * rho_n);
        const double s = sq_grad_rho_up ;
        
        const double f = 1.0 + kappa - kappa *(rho_m / (rho_m + c * s));
        const double df_drho = - m * kappa * c * (rho_m_1 * s / ((rho_m + c * s) * (rho_m + c * s)));
        const double df_ds = kappa * c * (rho_m / ((rho_m + c * s) * (rho_m + c * s)));
        const double d2f_drho_ds = m * kappa * c * (rho_m_1 * (c * s - rho_m) / ((rho_m + c * s) * (rho_m + c * s)* (rho_m + c * s)));
        const double d2f_ds2 = -2.0 * kappa * c * c * ( rho_m/ ((rho_m + c * s) * (rho_m + c * s) * (rho_m + c * s)));

        /*
        *p_v_x_up = -(1.0 / 2.0) * K_lda * (
            n * rho_n_1 * f
            + rho_n * df_drho
            - 2.0 * n * rho_n_1 * df_ds * q
            - 2.0 * rho_n * df_ds * laplace_rho
            - 2.0 * rho_n * d2f_drho_ds  * q
            - 4.0 * rho_n * d2f_ds2 * dif_rho_vmv
            );
            */

        double corre = kappa * rho_n_1 * c * s * (n * (c * s) * (c * s) - 3.0 * m * (c * s) * rho_m + (m - n) * rho_m * rho_m) / ((rho_m + c * s) * (rho_m + c * s) * (rho_m + c * s))
            - 2.0 * rho_n * df_ds * laplace_rho
            - 4.0 * rho_n * d2f_ds2 * dif_rho_vmv;
        *p_v_x_up = -(1.0 / 2.0) * K_lda * (
            n * rho_n_1
            + corre
            );

#ifdef _DEBUG
        if (fabs(n * rho_n_1) * 10.0 < fabs(corre)) {
            printf("WARN: %f, %f\n", n * rho_n_1, corre);
        }
#endif

        return -(1.0 / 2.0) * K_lda * rho_n_1 * f;

    }


#else
    static double Potential_X_test1(double rho_up, double sq_grad_rho_up, double laplace_rho, double dif_rho_vmv, double* p_v_x_up) {
        if (rho_up < SMALL_LIMIT) {
            *p_v_x_up = 0.0;
            return 0.0;
        }


        //const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
        const double n = 4.0 / 3.0;
        const double rho_n_1 = std::pow(rho_up, n - 1.0);
        const double rho_n = rho_n_1 * rho_up;
        //const double m = 8.0 / 3.0;
        const double m = 8.0 / 3.0;   //m=<6.0では上手く収束, m==8/3で収束しない//
        const double rho_m = std::pow(rho_up, m);
        //const double t = sq_grad_rho_up / std::pow(rho_up, m);
        //const double t = sq_grad_rho_up / (rho_n * rho_n);
        const double t = sq_grad_rho_up / rho_m;
        /*
        const double f = 1.0;
        const double df_dt = 0.0;
        const double d2f_dt2 = 0.0;
        */
        const double f = 1.0 + kappa - kappa / (1.0 + C * t);
        const double df_dt = kappa * C / ((1.0 + C * t) * (1.0 + C * t));
        const double d2f_dt2 = -2.0 * kappa * C * C / ((1.0 + C * t) * (1.0 + C * t) * (1.0 + C * t));


        double corre = +n * rho_n_1 * (f - 1.0)
            + (m - 2.0 * n) * rho_n_1 * df_dt * t    //this is zero because m = 2*n
            - 2.0 * rho_n * df_dt * laplace_rho / rho_m
            + 2.0 * m * rho_n_1 * d2f_dt2 * t * t
            - 4.0 * rho_n * d2f_dt2 * dif_rho_vmv / (rho_m * rho_m)     //2m-n = 3n
            ;

        *p_v_x_up = -(1.0 / 2.0) * K_lda * (
            n * rho_n_1
            + corre
            );

        if (fabs(n * rho_n_1) * 10.0 < fabs(corre)) {
            printf("WARN: %f, %f\n", n * rho_n_1, corre);
        }

        return -(1.0 / 2.0) * K_lda * rho_n_1 * f;

    }

#endif


    /*
    * 引数にはrho = rho_up + rho_downの右辺のうちどちらか一方だけを代入する
    * 
    * 
    */
    static double Potential_X_test2(double rho_up, double sq_grad_rho_up, double* p_dE_drho, double* p_dE_ds) {
        if (rho_up < SMALL_LIMIT) {
            *p_dE_drho = 0.0;
            *p_dE_ds = 0.0;
            return 0.0;
        }


        //const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
        const double n = 4.0 / 3.0;
        const double rho_n_1 = std::pow(rho_up, n - 1.0);
        const double rho_n = rho_n_1 * rho_up;
        //const double m = 8.0 / 3.0;
        const double m = 8.0 / 3.0;
        const double rho_m = std::pow(rho_up, m);
        const double rho_m_1 = std::pow(rho_up, m - 1.0);
        //const double t = sq_grad_rho_up / std::pow(rho_up, m);
        //const double t = sq_grad_rho_up / (rho_n * rho_n);
        const double s = sq_grad_rho_up;

        const double f = 1.0 + kappa - kappa * (rho_m / (rho_m + c * s));
        const double df_drho = -m * kappa * c * (rho_m_1 * s / ((rho_m + c * s) * (rho_m + c * s)));
        const double df_ds = kappa * c * (rho_m / ((rho_m + c * s) * (rho_m + c * s)));
        const double d2f_drho_ds = m * kappa * c * (rho_m_1 * (c * s - rho_m) / ((rho_m + c * s) * (rho_m + c * s) * (rho_m + c * s)));
        const double d2f_ds2 = -2.0 * kappa * c * c * (rho_m / ((rho_m + c * s) * (rho_m + c * s) * (rho_m + c * s)));

        
        /*double corre = kappa * rho_n_1 * C * q * (n * (C * q) * (C * q) - 3.0 * m * (C * q) * rho_m + (m - n) * rho_m * rho_m) / ((rho_m + C * q) * (rho_m + C * q) * (rho_m + C * q))
            - 2.0 * rho_n * df_ds * laplace_rho
            - 4.0 * rho_n * d2f_ds2 * dif_rho_vmv;
            */
        *p_dE_drho = -(1.0 / 2.0) * K_lda * (
            n * rho_n_1 * f
            + rho_n*df_drho
            );

        *p_dE_ds = K_lda * ( rho_n * df_ds );  //部分積分の効果を加味して符号は逆にして返す
/*
#ifdef _DEBUG
        if (fabs(n * rho_n_1) * 10.0 < fabs(corre)) {
            printf("WARN: %f, %f\n", n * rho_n_1, corre);
        }
#endif
*/
        return -(1.0 / 2.0) * K_lda * rho_n_1 * f;

    }


    /*
    * 引数rho_sigmaにはrho = rho_up + rho_downの右辺のうちどちらか一方だけを代入する
    *
    * vx = \delta E_x[rho,|\nabra rho|]/delta rho
    *    = v^LDA_x(rho) F(q) + rho epsilon^LDA_x(rho) dF/dq - \nabra * [2 P_sigma \nabra rho]
    * そこで、第一項と第二項をvx_by_rhoに返し、
    * 第三項のP_sigmaの部分を別途返す
    * ただし、
    * P_sigma = rho epsilon^LDA_x(rho) dF/dq dq/d((\nabra rho)^2)) 
    * 
    *
    */
    static double Potential_X_test3(double rho_sigma, double sq_grad_rho_sigma, double* vx_by_rho, double* P_sigma) {
        if (rho_sigma < SMALL_LIMIT) {
            *vx_by_rho = 0.0;
            *P_sigma = 0.0;
            return 0.0;
        }


        //const double K_lda = 3.0 * std::cbrt(3.0 / (4.0 * M_PI));
        //const double n = 4.0 / 3.0;
        const double rho_1_3 = std::cbrt(rho_sigma);
        const double rho_4_3 = rho_1_3 * rho_sigma;
        static const double cbrt_6_pi = std::cbrt(6.0 / M_PI);
        static const double c_6pi2_2_3 = (6.0 * M_PI) / cbrt_6_pi;
        const double q_mother = (4.0 * c_6pi2_2_3 * rho_4_3 * rho_4_3);
        const double q = sq_grad_rho_sigma / q_mother;
        const double F = 1.0 + kappa - kappa / (1.0 + q * mu / kappa);
        const double dF_dq = mu / ((1.0 + q * mu / kappa) * (1.0 + q * mu / kappa));
        const double dq_drho = -(8.0/3.0) * q / rho_sigma;
        const double dq_ddivrho = 1.0 / q_mother;

        const double rho_E_x = -(3.0 / 4.0) * cbrt_6_pi * rho_4_3 * F;
        

        *vx_by_rho = -cbrt_6_pi * (rho_1_3 * F + (3.0 / 4.0) * rho_4_3 * dF_dq * dq_drho);

        *P_sigma = -(3.0 / 4.0) * cbrt_6_pi * rho_4_3 * dF_dq * dq_ddivrho;



        return rho_E_x;

    }


    static double EnergyDensityPotential_C(double rho, double rho_diff, double sq_grad_rho, double laplace_rho, double dif_rho_vnv, double grad_rho_grad_rho_up, double grad_rho_grad_rho_down, double* p_v_c_up, double* p_v_c_down) {
        if (rho < SMALL_LIMIT) {
            *p_v_c_up = 0.0;
            *p_v_c_down = 0.0;
            return 0.0;
        }

        double v_c_up_lda, v_c_down_lda;
        const double e_c_lda = LDA_PW92::EnergyDensityPotential_C(rho, rho_diff, &v_c_up_lda, &v_c_down_lda);

        const double zeta = rho_diff / rho;
        const double cbrt_one_add_zeta = std::cbrt(1.0 + zeta);
        const double cbrt_one_sub_zeta = std::cbrt(1.0 - zeta);

        const double phi = 0.5 * (cbrt_one_add_zeta * cbrt_one_add_zeta + cbrt_one_sub_zeta * cbrt_one_sub_zeta);
        const double phi3 = phi * phi * phi;
        const double exp_e_c = std::exp(-e_c_lda / (gamma * phi3));
        const double A = (beta/gamma) / (exp_e_c - 1.0);
        const double c = std::cbrt(M_PI / 3.0) / 16.0;
        const double cbrt_rho = std::cbrt(rho);
        const double t2 = c * sq_grad_rho / (rho * rho * cbrt_rho * phi*phi);
        const double At2 = A * t2;
        const double h2 = 1.0 + At2 + At2 * At2;
        const double hln = std::log(1.0 + (beta / gamma) * t2 * (1.0 + At2) / h2);
        const double H = gamma * phi3 * hln;

        const double h1 = h2 + (beta / gamma) * t2 * (1.0 + At2);
        const double dh2_dA = t2 * (1.0 + 2.0 * At2);
        const double dh2_dt2 = A * (1.0 + 2.0 * At2);
        const double dh1_dA = dh2_dA + (beta / gamma) * t2 * t2;
        const double dh1_dt2 = dh2_dt2 + (beta / gamma) * (1.0 + 2.0 * At2);

        const double dh_dA = dh1_dA / h1 - dh2_dA / h2;
        const double dh_dt2 = dh1_dt2 / h1 - dh2_dt2 / h2;
        const double d2h_dt22 = (2.0*A*(A+(beta/gamma))) / h1
            - (dh1_dt2 / h1)*(dh1_dt2 / h1)
            - (2.0*A*A) / h2
            + (dh2_dt2 / h2) * (dh2_dt2 / h2);
        
        const double d2h_dAdt2 = (1.0 + 4.0*At2 + 2.0*(beta/gamma)*t2)/h1
            - (dh1_dt2 / h1) * (dh1_dA / h1)
            - (1.0+4.0*At2)/ h2
            + (dh2_dt2 / h2) * (dh2_dA / h2);
        
        const double dA_dphi = -3.0*(beta / gamma) / ((exp_e_c - 1.0) * (exp_e_c - 1.0)) * exp_e_c * e_c_lda / (gamma * phi3*phi);
        const double dphi_dzeta = (1.0/3.0)* (1.0/ cbrt_one_add_zeta - 1.0/ cbrt_one_sub_zeta); //but, derivative by e_c_lda is fixed.
        
        const double dA_de_c_lda = (beta / gamma) / ((exp_e_c - 1.0) * (exp_e_c - 1.0)) * exp_e_c / (gamma * phi3);
        const double de_c_lda_drho_up = (v_c_up_lda - e_c_lda) * rho;
        const double de_c_lda_drho_down = (v_c_down_lda - e_c_lda) * rho;

        
        double grad_rho_grad_phi = -(2.0/3.0) * sq_grad_rho * phi / rho
            + (2.0/3.0) / rho * (grad_rho_grad_rho_up/cbrt_one_add_zeta + grad_rho_grad_rho_down / cbrt_one_sub_zeta);
        double grad_rho_grad_t2 = -(7.0/3.0)*t2*sq_grad_rho/rho
            + 2.0*c/(rho*rho*cbrt_rho*phi*phi)* dif_rho_vnv
            - 2.0*t2 / rho * grad_rho_grad_phi;
        double grad_rho_grad_A = dA_de_c_lda * (de_c_lda_drho_up * grad_rho_grad_rho_up + de_c_lda_drho_down * grad_rho_grad_rho_down)
            + dA_dphi * grad_rho_grad_phi;

        double dE_c_drho = H - (7.0/3.0) * gamma * phi3 * dh_dt2 * t2;
        double dH_dzeta = 3.0 * gamma * phi * phi * hln * dphi_dzeta
            + gamma * phi3 * dh_dA * dA_dphi * dphi_dzeta;

        //grad_rho_grad_parts is: - grad { 2C \gamma \rho^(-4/3) \phi^3 (dh/d(t^2)) grad{rho} }
        double grad_rho_grad_parts = (8.0 / 3.0) * c * gamma / (rho * rho * cbrt_rho) * phi3 * dh_dt2 * sq_grad_rho
            - 6.0 * c * gamma * phi * phi / (rho * cbrt_rho) * dh_dt2 * grad_rho_grad_phi
            - 2.0 * c * gamma * phi3 / (rho * cbrt_rho) * d2h_dt22 * grad_rho_grad_t2
            - 2.0 * c * gamma * phi3 / (rho * cbrt_rho) * d2h_dAdt2 * grad_rho_grad_A;

        *p_v_c_up = v_c_up_lda + dE_c_drho -(zeta - 1.0) * dH_dzeta
            + gamma * rho * phi3* dh_dA * dA_de_c_lda * de_c_lda_drho_up
            + grad_rho_grad_parts;
        *p_v_c_down = v_c_down_lda + dE_c_drho - (zeta + 1.0) * dH_dzeta
            + gamma * rho * phi3 * dh_dA * dA_de_c_lda * de_c_lda_drho_down
            + grad_rho_grad_parts;
#if 0//def _DEBUG
        if (fabs(H) > fabs(e_c_lda) * 10.0) {
            printf("fabs(H) > fabs(e_c_lda): %g, %g\n", fabs(H) , fabs(e_c_lda));
        }
        if (fabs(*p_v_c_up) > fabs(v_c_up_lda) * 10.0) {
            printf("fabs(v_c_up) > fabs(v_c_up_lda): %g, %g\n", fabs(*p_v_c_up), fabs(v_c_up_lda));
        }
        if (fabs(*p_v_c_down) > fabs(v_c_down_lda) * 10.0) {
            printf("fabs(v_c_down) > fabs(v_c_down_lda): %g, %g\n", fabs(*p_v_c_down), fabs(v_c_down_lda));
        }
#endif
        return e_c_lda + H;
    }

    static double EnergyDensityPotential_C_test2(double rho, double rho_diff, double sq_grad_rho, double* p_dE_drho_up, double* p_dE_drho_down, double* p_dE_ds) {
        if (rho < SMALL_LIMIT) {
            *p_dE_drho_up = 0.0;
            *p_dE_drho_down = 0.0;
            *p_dE_ds = 0.0;
            return 0.0;
        }

        double v_c_up_lda, v_c_down_lda;
        const double e_c_lda = LDA_PW92::EnergyDensityPotential_C(rho, rho_diff, &v_c_up_lda, &v_c_down_lda);

        const double zeta = rho_diff / rho;
        const double cbrt_one_add_zeta = std::cbrt(1.0 + zeta);
        const double cbrt_one_sub_zeta = std::cbrt(1.0 - zeta);

        const double phi = 0.5 * (cbrt_one_add_zeta * cbrt_one_add_zeta + cbrt_one_sub_zeta * cbrt_one_sub_zeta);
        const double phi3 = phi * phi * phi;
        const double exp_e_c = std::exp(-e_c_lda / (gamma * phi3));

        const double alpha = (exp_e_c - 1.0);
        const double c = (beta / gamma) * std::cbrt(M_PI / 3.0) / 16.0;
        const double cbrt_rho = std::cbrt(rho);
        const double s = sq_grad_rho;
        
        const double cs = c * s;
        const double rho_7_3 = rho * rho * std::cbrt(rho);
        const double B = alpha * rho_7_3 * phi * phi;
        const double h2 = cs * (B + cs);
        const double one_e_h2 = B * B + exp_e_c * h2;
        const double H = std::log(one_e_h2 / (B * B + h2));
        const double dH_de = h2 / one_e_h2;
        
        const double dH_dB = (2.0 * B + exp_e_c * cs) / one_e_h2 - (2.0 * B + cs) / (B * B + h2);
        //const double dH_dB_per_alpha = - (B * Y * (B + 2.0*Y)) / (one_e_h2 * (B * B + h2));
        //const double dH_dB = alpha * dH_dB_per_alpha;


        double dE_c_drho = gamma * phi3 * H
            + (7.0 / 3.0) * gamma * phi3 * dH_dB * B;

        double dE_c_dphi_per_rho = 3.0 * gamma * phi * phi * H
            + 2.0 * gamma * phi * phi * dH_dB * B
            + 3.0 * e_c_lda * exp_e_c / (phi) * (dH_dB * (rho_7_3 * phi * phi) + dH_de);
            

        //double dphi_dzeta = (cbrt_one_add_zeta - cbrt_one_sub_zeta) / 3.0;
        const double dphi_dzeta = (1.0 / cbrt_one_add_zeta - 1.0 / cbrt_one_sub_zeta) / 3.0; //but, derivative by e_c_lda is fixed.

        double rho_dphi_drho_up = -dphi_dzeta * (zeta - 1.0);
        double rho_dphi_drho_down = -dphi_dzeta * (zeta + 1.0);


        double dE_c_depsilon_per_rho = - exp_e_c * (dH_dB * (rho_7_3 * phi * phi) + dH_de);
        const double rho_de_c_lda_drho_up = (v_c_up_lda - e_c_lda) ;
        const double rho_de_c_lda_drho_down = (v_c_down_lda - e_c_lda) ;


        double dH_ds = c *(B + 2.0 * cs) * (exp_e_c / one_e_h2 - 1.0 / (B * B + h2));
        double dE_ds = -2.0 * rho * gamma * phi3 * dH_ds;
        //double dE_ds = -2.0 * rho * gamma * phi3 * alpha * B*B*(B+2.0*Y)*C / (one_e_h2 * (B * B + h2));
        

        *p_dE_drho_up = v_c_up_lda + dE_c_drho + dE_c_dphi_per_rho * rho_dphi_drho_up + dE_c_depsilon_per_rho * rho_de_c_lda_drho_up;
        *p_dE_drho_down = v_c_down_lda + dE_c_drho + dE_c_dphi_per_rho * rho_dphi_drho_down + dE_c_depsilon_per_rho * rho_de_c_lda_drho_down;
        *p_dE_ds = dE_ds;

        return e_c_lda + gamma * phi3 * H;

    }

    /*
    * P = rho * dH/dq 
    * where q = |nabra rho|^2
    */
    static double EnergyDensityPotential_C_test3(double rho, double rho_diff, double sq_grad_rho, double* vc_up_by_rho, double* vc_down_by_rho, double* P) {
        if (rho < SMALL_LIMIT) {
            *vc_up_by_rho = 0.0;
            *vc_down_by_rho = 0.0;
            *P = 0.0;
            return 0.0;
        }

        double v_c_up_lda, v_c_down_lda;
        const double e_c_lda = LDA_PW92::EnergyDensityPotential_C(rho, rho_diff, &v_c_up_lda, &v_c_down_lda);

        const double rho_up = (rho + rho_diff) / 2.0;
        const double rho_down = (rho - rho_diff) / 2.0;

        const double zeta = std::clamp(rho_diff / rho, -1.0 +1.0e-10, 1.0 - 1.0e-10);
        const double cbrt_one_add_zeta = std::cbrt(1.0 + zeta);
        const double cbrt_one_sub_zeta = std::cbrt(1.0 - zeta);

        const double phi = 0.5 * (cbrt_one_add_zeta * cbrt_one_add_zeta + cbrt_one_sub_zeta * cbrt_one_sub_zeta);
        const double phi3 = phi * phi * phi;
        const double exp_e_c = std::exp(-e_c_lda / (gamma * phi3));

        const double alpha = (exp_e_c - 1.0);
        const double C = (beta / gamma) * std::cbrt(M_PI / 3.0) / 16.0;
        const double cbrt_rho = std::cbrt(rho);
        const double q = sq_grad_rho;

        const double Y = C * q;
        const double rho_7_3 = rho * rho * std::cbrt(rho);
        const double B = alpha * rho_7_3 * phi * phi;
        const double B_eY = std::max(B + exp_e_c * Y, 1e-30);
        const double B_Y = std::max(B + Y, 1e-30);
        double logBY = std::log(B_eY / B_Y);
        const double H = gamma * phi3 * logBY;

        //here, E^H_c = \int rho H dx
        // vH_up = \delta E^H_c / \delta \rho_up

        const double dphi_drho_up = (rho_up < SMALL_LIMIT) ? 0.0 : (1.0 / cbrt_one_add_zeta - 1.0 / cbrt_one_sub_zeta) / 3.0 * (1.0 - zeta) / rho;
        const double dphi_drho_down = (rho_down < SMALL_LIMIT) ? 0.0 : -(1.0 / cbrt_one_add_zeta - 1.0 / cbrt_one_sub_zeta) / 3.0 * (1.0 + zeta) / rho;
        const double dexp_drho_up = (rho_up < SMALL_LIMIT) ? 0.0 : -(exp_e_c / (gamma * phi3)) * (v_c_up_lda - e_c_lda) / rho
            + 3.0 * exp_e_c * e_c_lda / (gamma * phi3*phi) * dphi_drho_up;
        const double dexp_drho_down = (rho_down < SMALL_LIMIT) ? 0.0  : -(exp_e_c / (gamma * phi3)) * (v_c_down_lda - e_c_lda) / rho
            + 3.0 * exp_e_c * e_c_lda / (gamma * phi3 * phi) * dphi_drho_down;
        const double dB_drho_up = (7.0 / 3.0) * alpha * cbrt_rho * rho * phi * phi
            + 2.0 * alpha * rho_7_3 * phi * dphi_drho_up
            + rho_7_3 * phi * phi * dexp_drho_up;
        const double dB_drho_down = (7.0 / 3.0) * alpha * cbrt_rho * rho * phi * phi
            + 2.0 * alpha * rho_7_3 * phi * dphi_drho_down
            + rho_7_3 * phi * phi * dexp_drho_down;

        const double vH_c_up = H + 3.0 * rho * gamma * phi * phi * logBY * dphi_drho_up
            + rho * gamma * phi3 * ((-alpha * Y/ (B_eY * B_Y)) * dB_drho_up
                + Y / B_eY * dexp_drho_up);
        const double vH_c_down = H + 3.0 * rho * gamma * phi * phi * logBY * dphi_drho_down
            + rho * gamma * phi3 * ((-alpha * Y / (B_eY * B_Y)) * dB_drho_down
                + Y / B_eY * dexp_drho_down);

        //here, q = (nabra rho)^2
        //const double dH_dq = rho * gamma * phi3 * (exp_e_c / B_eY - 1.0 / B_Y) * C;
        const double dH_dq = rho * gamma * phi3 * (alpha * B / (B_eY * B_Y)) * C;
        
        *vc_up_by_rho = v_c_up_lda + vH_c_up;
        *vc_down_by_rho = v_c_down_lda + vH_c_down;
        *P = dH_dq;

        return e_c_lda + H;

    }

};

