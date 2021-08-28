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

 /* 定義global fit參數(Normal Ordering, NuFIT 5.0, 2020) */
  double theta12_N = 33.44; 
  double theta13_N = 8.57;
  double theta23_N = 49;
  double sdm_N = 7.42;
  double ldm_N = 2.514;

  /* 定義global fit參數(Inverse Ordering, NuFIT 5.0, 2020) */
  double theta12_I = 33.45; 
  double theta13_I = 8.61;
  double theta23_I = 49.3;
  double sdm_I = 7.42;
  double ldm_I = -2.497;


/* 計算chi2 with projection onto deltacp, 4 conditions for test value (0,NO) (pi,NO) (0,IO) (pi,IO) */ 
//參數:{系統誤差on_off:[GLB_ON,GLB_OFF], 選定實驗EXP:[0,1,GLB_ALL], true value的deltacp}
double delta_deltacp_1sigma (int on_off, int EXP ,double deltacp_true)
{
    glb_params test_values = glbAllocParams(); 
    glb_params true_values = glbAllocParams(); 
    glb_params input_errors = glbAllocParams();
  
  /* 定義true value (依照NO) */ 
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree, deltacp_true*degree ,1e-5*sdm_N,1e-3*ldm_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
  /* 定義test value (依照NO) */ 
    glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23_N*degree,  deltacp_true*degree , 1e-5*sdm_N, 1e-3*ldm_N); 
    glbSetDensityParams(test_values,1.0,GLB_ALL);

  /* 設定true value 為oscillation parameters */ 
    glbSetOscillationParameters(true_values);
    glbSetRates();

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
    glbDefineProjection(projection_cp, GLB_FIXED, GLB_FIXED, GLB_FIXED, GLB_FIXED, GLB_FIXED, GLB_FIXED);//deltacp theta12 m21 不動，其他可變
    glbSetDensityProjectionFlag(projection_cp,GLB_FIXED,GLB_ALL);//matter density不變
    glbSetProjection(projection_cp);

  /* 開關系統誤差 */  
    glbSwitchSystematics(EXP,GLB_ALL,on_off);

  /* 1. 計算 deltacp 的 Upper Bound */
    double chi_f = 0.0001 ; 
    double chi_i ; 
    double up_bound ;
    double gradiant;
    double deltacp_test;
    deltacp_test = deltacp_true + 1;
    while(chi_f < 1){
      glbSetOscParams(test_values,deltacp_test*degree,GLB_DELTA_CP);
      chi_i = chi_f;
      chi_f = glbChiNP(test_values,NULL,EXP);
      gradiant = (chi_f-chi_i)/chi_i;
      deltacp_test = deltacp_test + tanh(gradiant)+ 0.1;
      // printf("%g %g %g \n",chi_i,tanh(gradiant),chi_f); //看進度 
  }
    up_bound = deltacp_test ;

  /* 2. 計算 deltacp 的 Lower Bound */
    chi_f = 0.0001 ; 
    deltacp_test = deltacp_true + 1;
    double low_bound ;
    while(chi_f < 1){
      glbSetOscParams(test_values,deltacp_test*degree,GLB_DELTA_CP);
      chi_i = chi_f;
      chi_f = glbChiNP(test_values,NULL,EXP);
      gradiant = (chi_f-chi_i)/chi_i;
      deltacp_test = deltacp_test - tanh(gradiant)- 0.1;
  }
    low_bound = deltacp_test;

  /* 回傳 delta_deltacp*/
  return (up_bound-low_bound)/2;

}



int main(int argc, char *argv[])
{ 
  /* Initialize libglobes */
  glbInit(argv[0]);

  FILE* OUT =  fopen("4-2_A.dat","w");//建立輸出檔案

  /* Initialize experiment NFstandard.glb */
  glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
  glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

  /* Iteration over all values to be computed */
  double step = 3;
  double deltacp;
  for (deltacp = -180; deltacp <= 180; deltacp = deltacp + step )
  { printf("%g \n",deltacp); //看進度

////////////// 4.2_run_1 ////////////////
   /* 不考慮系統誤差(GLB_OFF) */
  double a0 = delta_deltacp_1sigma(GLB_OFF, 0,       deltacp);   // 0代表DUNE, 1代表T2HK
  double b0 = delta_deltacp_1sigma(GLB_OFF, 1,       deltacp);
  double c0 = delta_deltacp_1sigma(GLB_OFF, GLB_ALL, deltacp);
  
  /* 考慮系統誤差(GLB_ON) */  
  double a1 = delta_deltacp_1sigma(GLB_ON, 0,        deltacp);  
  double b1 = delta_deltacp_1sigma(GLB_ON, 1,        deltacp);
  double c1 = delta_deltacp_1sigma(GLB_ON, GLB_ALL,  deltacp);
  

  fprintf(OUT,"%g %g %g %g %g %g %g \n",deltacp,a0,b0,c0,a1,b1,c1);

  }
  return 0;  
}
