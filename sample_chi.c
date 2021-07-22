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
/***************************************************************************
 *                            M A I N   P R O G R A M                      *
 ***************************************************************************/
   
  /* 定義global fit參數 */
  double theta12_g = 33.72; 
  double theta13_g = 8.46;
  double sdm_g = 7.49;
  double ldm_g = 2.526;
 

 /* 計算chi2 */ //參數:{系統誤差on_off:[GLB_ON,GLB_OFF], 選定實驗EXP:[0,1,GLB_ALL], theta23:[40,50], true value的deltacp}
double chi2 (int on_off, int EXP ,double theta23, double deltacp)
{
    glb_params test_values = glbAllocParams(); //給test_values一個記憶體位置
    glb_params true_values = glbAllocParams(); //給true_values一個記憶體位置
  
  /* 定義true value */  
    glbDefineParams(true_values,theta12_g*degree,theta13_g*degree,theta23*degree, deltacp*degree ,1e-5*sdm_g,1e-3*ldm_g);
    glbSetDensityParams(true_values,1.0,GLB_ALL);

  /* 計算test value中delta_cp = 0的chi_0 */
    glbDefineParams(test_values, theta12_g*degree, theta13_g*degree, theta23*degree,  0*degree , 1e-5*sdm_g, 1e-3*ldm_g);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_0;
  glbSwitchSystematics(EXP,GLB_ALL,on_off);
  chi2_0=glbChiSys(true_values,EXP,GLB_ALL);

  /* 計算test value中delta_cp = 180的chi2_pi */
    glbDefineParams(test_values, theta12_g*degree, theta13_g*degree, theta23*degree,  180*degree , 1e-5*sdm_g, 1e-3*ldm_g);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_pi;
  glbSwitchSystematics(EXP,GLB_ALL,on_off);
  chi2_pi=glbChiSys(true_values,EXP,GLB_ALL);

  /* 取最小的chi2 */
  if(chi2_0 < chi2_pi) {return chi2_0;}
  else {return chi2_pi;}
}


int main(int argc, char *argv[])
{ 
    glbInit(argv[0]); 

    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

    FILE* OUT =  fopen("sample_chi.dat","w");//建立輸出檔案
    

  double step = 3;
  double deltacp;
  for (deltacp = -180; deltacp <= 180; deltacp = deltacp + step )
  {
  /* 不考慮系統誤差(GLB_OFF) */
  double a0 = chi2(GLB_OFF, 0,       40, deltacp);   // 0代表DUNE, 1代表T2HK
  double b0 = chi2(GLB_OFF, 0,       50, deltacp);
  double c0 = chi2(GLB_OFF, 1,       40, deltacp);
  double d0 = chi2(GLB_OFF, 1,       50, deltacp);
  double e0 = chi2(GLB_OFF, GLB_ALL, 40, deltacp);
  double f0 = chi2(GLB_OFF, GLB_ALL, 50, deltacp);
  
  /* 考慮系統誤差(GLB_ON) */  
  double a1 = chi2(GLB_ON, 0,       40, deltacp);
  double b1 = chi2(GLB_ON, 0,       50, deltacp);
  double c1 = chi2(GLB_ON, 1,       40, deltacp);
  double d1 = chi2(GLB_ON, 1,       50, deltacp);
  double e1 = chi2(GLB_ON, GLB_ALL, 40, deltacp);
  double f1 = chi2(GLB_ON, GLB_ALL, 50, deltacp);

  fprintf(OUT,"%g %g %g %g %g %g %g %g %g %g %g %g %g \n",deltacp,a0,b0,c0,d0,e0,f0,a1,b1,c1,d1,e1,f1);
  printf("%g \n",deltacp);  
  }


  return 0;  
}

