/* Generate single NuFit Spectrum*/

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

  /* 定義global fit參數(Normal Ordering, NuFIT 5.0, 2020) */
    double theta12_N = 33.44; 
    double theta13_N = 8.57;
    double theta23_N = 49;
    double sdm_N = 7.42;
    double ldm_N = 2.514;
    double deltacp_N = 195;
  /* 3 sigma range (Normal Ordering, NuFIT 5.0, 2020) */
    // double delta_12_N =(35.86-31.27);
    // double delta_13_N =( 8.97- 8.20);
    // double delta_23_N =(51.80-39.60);
    // double delta_sdm_N=( 8.04- 6.82);
    // double delta_ldm_N=(2.598-2.431);

  /* 定義global fit參數(Inverse Ordering, NuFIT 5.0, 2020) */
    double theta12_I = 33.45; 
    double theta13_I = 8.61;
    double theta23_I = 49.3;
    double sdm_I = 7.42;
    double ldm_I = -2.497;
    double deltacp_I = 286;
  /* 3 sigma range (Inverse Ordering, NuFIT 5.0, 2020) */
    // double delta_12_I =(35.87-31.27);
    // double delta_13_I =( 8.98- 8.24);
    // double delta_23_I =(52.00-39.90);
    // double delta_sdm_I=( 8.04- 6.82);
    // double delta_ldm_I=(2.583-2.412);
//


int LABEL_OT(double x)
{
if(x>45) {return 1;}
else if (x==45) {return 0;}
else {return -1;}
}

int LABEL_CP(double x)
{
if((x==0)||(x==180)) {return 0;}
else {return 1;}
}

int LABEL_MO(double x)
{
if(x>0) {return 1;}
else {return -1;}
}



int main(int argc, char *argv[])
{ 

    glbInit(argv[0]);                /* Initialize GLoBES */
  //  int TOTALsample= atoi(argv[1]); //choosed by the user
    
    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glb_params true_values_N = glbAllocParams();
    glb_params true_values_I = glbAllocParams();

  /* Set standard oscillation parameters (cf. hep-ph/0405172v5) */

    FILE* OUT =   fopen("sample_NuFit.dat","w");//建立輸出檔案

    int ew_low, ew_high;
    int i,exp,channel;
    int channel_max;

  /* Normal Ordering */
  int OCTANT_N= LABEL_OT(theta23_N);
  int CPV_N    =LABEL_CP(deltacp_N);
  int MO_N     =LABEL_MO(ldm_N    );

  glbDefineParams(true_values_N,theta12_N*degree,theta13_N*degree,theta23_N*degree,deltacp_N*degree,1e-5*sdm_N,1e-3*ldm_N);
  glbSetDensityParams(true_values_N,1.0,GLB_ALL);
  glbSetOscillationParameters(true_values_N);
  glbSetRates();
    
  exp=0;channel=0;
  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  double *true_rates_N = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates_N[i]);}
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_N,theta13_N,theta23_N,deltacp_N,sdm_N,ldm_N,OCTANT_N,CPV_N,MO_N);
  

  /* Inverse Ordering */
  int OCTANT_I= LABEL_OT(theta23_I);
  int CPV_I    =LABEL_CP(deltacp_I);
  int MO_I    =LABEL_MO(ldm_I    );

  glbDefineParams(true_values_I,theta12_I*degree,theta13_I*degree,theta23_I*degree,deltacp_I*degree,1e-5*sdm_I,1e-3*ldm_I);
  glbSetDensityParams(true_values_I,1.0,GLB_ALL);
  glbSetOscillationParameters(true_values_I);
  glbSetRates();
    
  exp=0;channel=0;
  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  double *true_rates_I = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates_I[i]);}
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_I,theta13_I,theta23_I,deltacp_I,sdm_I,ldm_I,OCTANT_I,CPV_I,MO_I);
  

  /* Clean up */
  glbFreeParams(true_values_N);
  glbFreeParams(true_values_I);

  return 0;  
}