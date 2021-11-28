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
  double sdm_g_N = 7.42;
  double ldm_g_N = 2.514;

  /* 定義global fit參數(Inverse Ordering, NuFIT 5.0, 2020) */
  double theta12_I = 33.45; 
  double theta13_I = 8.61;
  double theta23_I = 49.3;
  double sdm_g_I = 7.42;
  double ldm_g_I = -2.497;

  /* 定義chi2 Function */ 
double chi2 (int EXP ,double theta23,double deltacp)
{
    glb_params test_values_NO = glbAllocParams(); 
    glb_params test_values_IO = glbAllocParams(); 
    glb_params true_values = glbAllocParams(); 
    glb_params input_errors = glbAllocParams(); 
  
  /* 定義true value (依照NO) */ 
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree, 195*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
  /* 定義test value (依照NO) */ 
    glbDefineParams(test_values_NO, theta12_N*degree, theta13_N*degree, theta23*degree,  deltacp*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N); 
    glbSetDensityParams(test_values_NO,1.0,GLB_ALL);
    /* 定義test value (依照IO) */ 
    glbDefineParams(test_values_IO, theta12_I*degree, theta13_I*degree, theta23*degree,  deltacp*degree , 1e-5*sdm_g_I, 1e-3*ldm_g_I); 
    glbSetDensityParams(test_values_IO,1.0,GLB_ALL);

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
    glb_projection projection = glbAllocProjection();
    //GLB_FIXED/GLB_FREE             theta12    theta13  theta23    deltacp     m21        m31
    glbDefineProjection(projection, GLB_FIXED, GLB_FREE, GLB_FIXED, GLB_FIXED, GLB_FIXED, GLB_FREE);//deltacp theta12 theta23 m21 不動，其他可變
    glbSetDensityProjectionFlag(projection,GLB_FIXED,GLB_ALL);//matter density不變
    glbSetProjection(projection);

  /* 關閉系統誤差 */  
    glbSwitchSystematics(EXP,GLB_ALL,GLB_OFF);

  /* 計算chi2 */ 
    double chi_NO = glbChiNP(test_values_NO,NULL,EXP);
    double chi_IO = glbChiNP(test_values_IO,NULL,EXP);
  /* 取最小的chi2 */
    if(chi_NO < chi_IO) {return chi_NO;}
    else {return chi_IO;}
}

int main(int argc, char *argv[])
{ 
  /* Initialize libglobes */
  glbInit(argv[0]);

  /* 建立輸出檔案 */
  FILE* OUT_DUNE =    fopen("one_sigma_contour_DUNE.dat","w");
  FILE* OUT_T2HK =    fopen("one_sigma_contour_T2HK.dat","w");
  FILE* OUT_DUT2 =  fopen("one_sigma_contour_DUT2.dat","w");

  /* Initialize experiment NFstandard.glb */
  glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
  glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

  /* Iteration over all values to be computed */
  double x,y; // x為theta23, y為deltacp   
  double chi2_output;  

  // 推薦的theta23/deltacp設置範圍(1 sigma)
  /* DUNE x=48.5;x<=49.5;x=x+0.05  y=188;y<=205;y=y+1 */
  /* T2HK x=48.5;x<=49.5;x=x+0.05  y=188;y<=204;y=y+1 */
  /* DUNE+T2HK x=48.5;x<=49.5;x=x+0.05  y=190;y<=200;y=y+1 */

  /* DUNE */
  printf("DUNE \n"); //看進度
  for(x=48.2;x<=49.5;x=x+0.05) // 推薦的theta23/deltacp設置範圍(1 sigma)
  for(y=178;y<=215;y=y+1)
  { 
    chi2_output = chi2(0, x ,y);
    printf("%g %g %g\n",x,y,chi2_output); //看進度

    if(chi2_output < 2.3 ) { 
    fprintf(OUT_DUNE,"%g %g %g \n",x,y,chi2_output);
    }
  }

    /* T2HK */
  printf("T2HK \n"); //看進度
  for(x=48.5;x<=49.5;x=x+0.05)
  for(y=180;y<=210;y=y+1)
  { 
    chi2_output = chi2(1, x ,y);
    printf("%g %g %g\n",x,y,chi2_output); //看進度

    if(chi2_output < 2.3 ) { 
    fprintf(OUT_T2HK,"%g %g %g \n",x,y,chi2_output);
    }
  }

    /* DUNE+T2HK */
  printf("DUNE+T2HK \n"); //看進度
  for(x=48.5;x<=49.5;x=x+0.05)
  for(y=185;y<=210;y=y+1)
  { 
    chi2_output = chi2(-1, x ,y);
    printf("%g %g %g\n",x,y,chi2_output); //看進度

    if(chi2_output < 2.3 ) { 
    fprintf(OUT_DUT2,"%g %g %g \n",x,y,chi2_output);
    }
  }
   

  return 0;  
}
