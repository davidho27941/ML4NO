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

  /* 定義global fit參數 */
  double theta12_g = 33.72; 
  double theta13_g = 8.46;
  double theta23_g = 40;
  double sdm_g = 7.49;
  double ldm_g = 2.526;
  double deltacp;


int main(int argc, char *argv[])
{ 
  glbInit(argv[0]); 

    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

    FILE* OUT =  fopen("test0726.dat","w");//建立輸出檔案
 
    glb_params test_values = glbAllocParams(); 
    glb_params true_values = glbAllocParams(); 
    glb_params input_errors = glbAllocParams(); 
    glb_params minimum = glbAllocParams(); 

for (deltacp = 0; deltacp <= 90; deltacp = deltacp + 5 )
  {
  /* 定義true value */  
    glbDefineParams(true_values,theta12_g*degree,theta13_g*degree,theta23_g*degree, deltacp*degree ,1e-5*sdm_g,1e-3*ldm_g);
    glbSetDensityParams(true_values,1.0,GLB_ALL);

  /* 設定Projection */   
    glbDefineParams(input_errors,0,0,0,0,0,0);
    glbSetDensityParams(input_errors,0,GLB_ALL);
    glbSetInputErrors(input_errors);
    glbSetCentralValues(true_values); 
    
    glb_projection projection_cp = glbAllocProjection();
    glbDefineProjection(projection_cp, GLB_FIXED, GLB_FIXED, GLB_FREE, GLB_FIXED, GLB_FIXED, GLB_FIXED);//deltacp不動，其他可變
    glbSetDensityProjectionFlag(projection_cp,GLB_FIXED,GLB_ALL);//matter density可變
    glbSetProjection(projection_cp);

/* 計算test value中delta_cp = 0的chi_0 */
    glbDefineParams(test_values, theta12_g*degree, theta13_g*degree, theta23_g*degree,  0*degree , 1e-5*sdm_g, 1e-3*ldm_g);  
    glbSetDensityParams(test_values,1.0,GLB_ALL); 
    glbSetOscillationParameters(test_values);
    glbSetRates();
  double chi2_0;
  glbSwitchSystematics(GLB_ALL,GLB_ALL,GLB_OFF);
  chi2_0 = glbChiNP(true_values,minimum,GLB_ALL);
  printf("%g %g \n",deltacp,chi2_0);
  glbPrintParams(stdout,minimum);
  //fprintf(OUT,"%g %g %g  \n",deltacp,chi2_0);
  }
  return 0;  
}
