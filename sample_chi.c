#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <globes/globes.h>   /* GLoBES library */

#include <gsl/gsl_math.h>    /* GNU Scientific library (required for root finding) */
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_spline.h>

 double degree    = M_PI/180;


#include <stdarg.h>
 double min(int n, ...) {
  /*
    n是參數個數，後面才是參數本身 
  */
  int i;
  double min_num = 10000000;
  double input;
  va_list vl;
  va_start(vl,n);
  for ( i = 0 ; i < n ; i++ ){
    input = va_arg(vl,double);
    if( input < min_num )
      min_num = input;
  } // for
  va_end(vl);
  
  return min_num;
} // min()

/***************************************************************************
 *                            M A I N   P R O G R A M                      *
 ***************************************************************************/
   
  /* 定義global fit參數(Normal Ordering, NuFIT 5.0, 2020) */
  double theta12_N = 33.44; 
  double theta13_N = 8.57;
  double theta23_N = 49;
  double sdm_g_N = 7.42;
  double ldm_g_N = 2.514;
 /* 1 sigma range (Normal Ordering, NuFIT 5.0, 2020) */
  double delta_12_N =(35.86-31.27)/3;
  double delta_13_N =( 8.97- 8.20)/3;
  double delta_23_N =(51.80-39.60)/3;
  double delta_sdm_N=( 8.04- 6.82)/3;
  double delta_ldm_N=(2.598-2.431)/3;

  /* 定義global fit參數(Inverse Ordering, NuFIT 5.0, 2020) */
  double theta12_I = 33.45; 
  double theta13_I = 8.61;
  double theta23_I = 49.3;
  double sdm_g_I = 7.42;
  double ldm_g_I = -2.497;
 /* 1 sigma range (Inverse Ordering, NuFIT 5.0, 2020) */
  double delta_12_I =(35.87-31.27)/3;
  double delta_13_I =( 8.98- 8.24)/3;
  double delta_23_I =(52.00-39.90)/3;
  double delta_sdm_I=( 8.04- 6.82)/3;
  double delta_ldm_I=(2.583-2.412)/3;

  

//  /* 計算chi2 without projection */ //參數:{系統誤差on_off:[GLB_ON,GLB_OFF], 選定實驗EXP:[0,1,GLB_ALL], theta23:[40,50], true value的deltacp}
// double chi2 (int on_off, int EXP ,double theta23, double deltacp)
// {
//     glb_params test_values = glbAllocParams(); 
//     glb_params true_values = glbAllocParams(); 
  
//   /* 定義true value */  
//     glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23*degree, deltacp*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
//     glbSetDensityParams(true_values,1.0,GLB_ALL);

//   /* 計算test value中delta_cp = 0的chi_0 */
//     glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23*degree,  0*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
//     glbSetDensityParams(test_values,1.0,GLB_ALL); 
//     glbSetOscillationParameters(test_values);
//     glbSetRates();
//   double chi2_0;
//   glbSwitchSystematics(EXP,GLB_ALL,on_off);
//   chi2_0=glbChiSys(true_values,EXP,GLB_ALL);

//   /* 計算test value中delta_cp = 180的chi2_pi */
//     glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23*degree,  180*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
//     glbSetDensityParams(test_values,1.0,GLB_ALL); 
//     glbSetOscillationParameters(test_values);
//     glbSetRates();
//   double chi2_pi;
//   glbSwitchSystematics(EXP,GLB_ALL,on_off);
//   chi2_pi=glbChiSys(true_values,EXP,GLB_ALL);

//   /* 取最小的chi2 */
//   if(chi2_0 < chi2_pi) {return chi2_0;}
//   else {return chi2_pi;}
// }



// /* 計算chi2 with projection onto deltacp */ 
// //參數:{系統誤差on_off:[GLB_ON,GLB_OFF], 選定實驗EXP:[0,1,GLB_ALL], theta23:[40,50], true value的deltacp}
// double chi2_proj (int on_off, int EXP ,double theta23, double deltacp)
// {
//     glb_params test_values = glbAllocParams();
//     glb_params true_values = glbAllocParams(); 
//     glb_params input_errors = glbAllocParams(); 
  
//   /* 定義true value */  
//     glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23*degree, deltacp*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
//     glbSetDensityParams(true_values,1.0,GLB_ALL);

//   // /* 設定Prior ON */   
//   //   glbDefineParams(input_errors,delta_12_N*degree,delta_13_N*degree,delta_23_N*degree,
//   //                   0,1e-5*delta_sdm_N,1e-3*delta_ldm_N);
//   //   glbSetDensityParams(input_errors,1,GLB_ALL);
//   //   glbSetInputErrors(input_errors);
//   //   glbSetCentralValues(true_values); 

//   /* 設定Prior OFF */   
//     glbDefineParams(input_errors,0,0,0,0,0,0);
//     glbSetDensityParams(input_errors,0,GLB_ALL);
//     glbSetInputErrors(input_errors);
//     glbSetCentralValues(true_values); 
 
//   /* 設定Projection */   
//     glb_projection projection_cp = glbAllocProjection();
//     //GLB_FIXED/GLB_FREE                theta12    theta13  theta23    deltacp     m21        m31
//     glbDefineProjection(projection_cp, GLB_FIXED, GLB_FREE, GLB_FREE, GLB_FIXED, GLB_FIXED, GLB_FREE);//deltacp不動，其他可變
//     glbSetDensityProjectionFlag(projection_cp,GLB_FIXED,GLB_ALL);//matter density不變
//     glbSetProjection(projection_cp);
  
//   /* 計算test value中delta_cp = 0的chi_0 */
//     glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23*degree,  0*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
//     glbSetDensityParams(test_values,1.0,GLB_ALL); 
//     glbSetOscillationParameters(test_values);
//     glbSetRates();
//   double chi2_0;
//   glbSwitchSystematics(EXP,GLB_ALL,on_off);
//   chi2_0 = glbChiNP(true_values,NULL,EXP);

//   /* 計算test value中delta_cp = 180的chi2_pi */
//     glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23*degree,  180*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
//     glbSetDensityParams(test_values,1.0,GLB_ALL); 
//     glbSetOscillationParameters(test_values);
//     glbSetRates();
//   double chi2_pi;
//   glbSwitchSystematics(EXP,GLB_ALL,on_off);
//   chi2_pi = glbChiNP(true_values,NULL,EXP);

//   /* 取最小的chi2 */
//   if(chi2_0 < chi2_pi) {return chi2_0;}
//   else {return chi2_pi;}
// }


/* 計算chi2 with projection onto deltacp, 4 conditions for test value (0,NO) (pi,NO) (0,IO) (pi,IO) */ 
//參數:{系統誤差on_off:[GLB_ON,GLB_OFF], 選定實驗EXP:[0,1,GLB_ALL], theta23:[40,50], true value的deltacp}
double chi2_proj (int on_off, int EXP ,double theta23, double deltacp)
{
    glb_params test_values = glbAllocParams(); 
    glb_params true_values = glbAllocParams(); 
    glb_params input_errors = glbAllocParams(); 
  
  /* 定義true value (依照NO) */ 
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23*degree, deltacp*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);

  // /* 設定Prior ON */   
  //   glbDefineParams(input_errors,delta_12_N*degree,delta_13_N*degree,delta_23_N*degree,
  //                   0,1e-5*delta_sdm_N,1e-3*delta_ldm_N);
  //   glbSetDensityParams(input_errors,1,GLB_ALL);
  //   glbSetInputErrors(input_errors);
  //   glbSetCentralValues(true_values); 

  /* 設定Prior OFF */   
    glbDefineParams(input_errors,0,0,0,0,0,0);
    glbSetDensityParams(input_errors,0,GLB_ALL);
    glbSetInputErrors(input_errors);
    glbSetCentralValues(true_values); 
 
  /* 設定Projection */   
    glb_projection projection_cp = glbAllocProjection();
    //GLB_FIXED/GLB_FREE                theta12    theta13  theta23    deltacp     m21        m31
    glbDefineProjection(projection_cp, GLB_FIXED, GLB_FREE, GLB_FREE, GLB_FIXED, GLB_FIXED, GLB_FREE);//deltacp theta12 m21 不動，其他可變
    glbSetDensityProjectionFlag(projection_cp,GLB_FIXED,GLB_ALL);//matter density不變
    glbSetProjection(projection_cp);
  
  /* 1. 計算test value中 (delta_cp = 0, NO) 的chi_0_N */
    glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23*degree,  0*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_0_N;
  glbSwitchSystematics(EXP,GLB_ALL,on_off);
  chi2_0_N = glbChiNP(true_values,NULL,EXP);

  /* 2. 計算test value中 (delta_cp = 180, NO) 的chi2_pi_N */
    glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23*degree,  180*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_pi_N;
  glbSwitchSystematics(EXP,GLB_ALL,on_off);
  chi2_pi_N = glbChiNP(true_values,NULL,EXP);

  /* 3. 計算test value中 (delta_cp = 0, IO) 的chi_0_I */
    glbDefineParams(test_values, theta12_I*degree, theta13_I*degree, theta23*degree,  0*degree , 1e-5*sdm_g_I, 1e-3*ldm_g_I);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_0_I;
  glbSwitchSystematics(EXP,GLB_ALL,on_off);
  chi2_0_I = glbChiNP(true_values,NULL,EXP);

  /* 4. 計算test value中 (delta_cp = 180, IO) 的chi2_pi_I */
    glbDefineParams(test_values, theta12_I*degree, theta13_I*degree, theta23*degree,  180*degree , 1e-5*sdm_g_I, 1e-3*ldm_g_I);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_pi_I;
  glbSwitchSystematics(EXP,GLB_ALL,on_off);
  chi2_pi_I = glbChiNP(true_values,NULL,EXP);

  /* 取最小的chi2 */
// printf("%g %g %g %g\n",chi2_0_N, chi2_pi_N, chi2_0_I, chi2_pi_I);
return min(4, chi2_0_N, chi2_pi_N, chi2_0_I, chi2_pi_I);

}





int main(int argc, char *argv[])
{ 
    glbInit(argv[0]); 

    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

    FILE* OUT =  fopen("sample_chi_test.dat","w");//建立輸出檔案
    

  double step = 3;
  double deltacp;
  for (deltacp = -180; deltacp <= 180; deltacp = deltacp + step )
  { printf("%g \n",deltacp); //看進度

  // /* 不考慮系統誤差(GLB_OFF) */
  // double a0 = chi2(GLB_OFF, 0,       40, deltacp);   // 0代表DUNE, 1代表T2HK
  // double b0 = chi2(GLB_OFF, 0,       50, deltacp);
  // double c0 = chi2(GLB_OFF, 1,       40, deltacp);
  // double d0 = chi2(GLB_OFF, 1,       50, deltacp);
  // double e0 = chi2(GLB_OFF, GLB_ALL, 40, deltacp);
  // double f0 = chi2(GLB_OFF, GLB_ALL, 50, deltacp);
  
  // /* 考慮系統誤差(GLB_ON) */  
  // double a1 = chi2(GLB_ON, 0,       40, deltacp);
  // double b1 = chi2(GLB_ON, 0,       50, deltacp); 
  // double c1 = chi2(GLB_ON, 1,       40, deltacp);
  // double d1 = chi2(GLB_ON, 1,       50, deltacp);
  // double e1 = chi2(GLB_ON, GLB_ALL, 40, deltacp);
  // double f1 = chi2(GLB_ON, GLB_ALL, 50, deltacp);

 // fprintf(OUT,"%g %g %g %g %g %g %g %g %g %g %g %g %g \n",deltacp,a0,b0,c0,d0,e0,f0,a1,b1,c1,d1,e1,f1);

////////////// 4.1_run_1 ////////////////
   /* 不考慮系統誤差(GLB_OFF) */
  double a0 = chi2_proj(GLB_OFF, 0,       theta23_N, deltacp);   // 0代表DUNE, 1代表T2HK
  double b0 = chi2_proj(GLB_OFF, 1,       theta23_N, deltacp);
  double c0 = chi2_proj(GLB_OFF, GLB_ALL, theta23_N, deltacp);
  
  /* 考慮系統誤差(GLB_ON) */  
  double a1 = chi2_proj(GLB_ON, 0,       theta23_N, deltacp);  
  double b1 = chi2_proj(GLB_ON, 1,       theta23_N, deltacp);
  double c1 = chi2_proj(GLB_ON, GLB_ALL, theta23_N, deltacp);
  fprintf(OUT,"%g %g %g %g %g %g %g \n",deltacp,a0,b0,c0,a1,b1,c1);
//////////////// End /////////////////////

  }
  return 0;  
}

