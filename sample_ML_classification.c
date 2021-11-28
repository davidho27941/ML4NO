/* Generate NuFit Spectrum for classification with different deltacp*/

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

    glbInit(argv[0]);                /* Initialize GLoBES and define chi^2 functions */
  //  int TOTALsample= atoi(argv[1]); //choosed by the user

    
    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);
    
    FILE* OUT =   fopen("sample_ML_classification.dat","w");//建立輸出檔案

    double j ;
    double deltacp;
    double *true_rates;
    int OCTANT, CPV, MO;
    int ew_low, ew_high;
    int i,exp,channel, channel_max;
    glb_params true_values = glbAllocParams();
 /*
-180   ~  -177 0.2 
-177   ~  -5     1
-5      ~   5      0. 2
5       ~   177   1
177   ~   180  0.2
*/

 /*-180   ~  -177 , 0.2 */

  for(j = -1800; j < -1770; j = j + 2){
    deltacp = j/10;
    printf("%g \n",deltacp);
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree,deltacp*degree,1e-5*sdm_N,1e-3*ldm_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
    glbSetOscillationParameters(true_values);
    glbSetRates();
    OCTANT= LABEL_OT(theta23_N);
    CPV   =LABEL_CP(deltacp);
    MO   =LABEL_MO(ldm_N    );

  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  true_rates = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates[i]);
  }
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_N,theta13_N,theta23_N,deltacp,sdm_N,ldm_N,OCTANT,CPV,MO);
  }

 /*-177   ~  -5 , 1 */

  for(j = -1770; j < -50; j = j + 10){
    deltacp = j/10;
    printf("%g \n",deltacp);
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree,deltacp*degree,1e-5*sdm_N,1e-3*ldm_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
    glbSetOscillationParameters(true_values);
    glbSetRates();
    OCTANT= LABEL_OT(theta23_N);
    CPV   =LABEL_CP(deltacp);
    MO   =LABEL_MO(ldm_N    );

  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  true_rates = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates[i]);
  }
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_N,theta13_N,theta23_N,deltacp,sdm_N,ldm_N,OCTANT,CPV,MO);
  }

 /*-5   ~  5 , 0.2 */

  for(j = -50; j < 50; j = j + 2){
    deltacp = j/10;
    printf("%g \n",deltacp);
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree,deltacp*degree,1e-5*sdm_N,1e-3*ldm_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
    glbSetOscillationParameters(true_values);
    glbSetRates();
    OCTANT= LABEL_OT(theta23_N);
    CPV   =LABEL_CP(deltacp);
    MO   =LABEL_MO(ldm_N    );

  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  true_rates = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates[i]);
  }
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_N,theta13_N,theta23_N,deltacp,sdm_N,ldm_N,OCTANT,CPV,MO);
  }

   /*5  ~  177 , 1 */

  for(j = 50; j < 1770; j = j + 10){
    deltacp = j/10;
    printf("%g \n",deltacp);
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree,deltacp*degree,1e-5*sdm_N,1e-3*ldm_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
    glbSetOscillationParameters(true_values);
    glbSetRates();
    OCTANT= LABEL_OT(theta23_N);
    CPV   =LABEL_CP(deltacp);
    MO   =LABEL_MO(ldm_N    );

  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  true_rates = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates[i]);
  }
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_N,theta13_N,theta23_N,deltacp,sdm_N,ldm_N,OCTANT,CPV,MO);
  }

 /*177   ~  180 , 0.2 */

  for(j = 1770; j < 1802; j = j + 2){
    deltacp = j/10;
    printf("%g \n",deltacp);
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23_N*degree,deltacp*degree,1e-5*sdm_N,1e-3*ldm_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
    glbSetOscillationParameters(true_values);
    glbSetRates();
    OCTANT= LABEL_OT(theta23_N);
    CPV   =LABEL_CP(deltacp);
    MO   =LABEL_MO(ldm_N    );

  for(exp=0; exp<2; exp++){
  channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  true_rates = glbGetRuleRatePtr(exp, channel);
  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates[i]);
  }
  }
  }
  fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_N,theta13_N,theta23_N,deltacp,sdm_N,ldm_N,OCTANT,CPV,MO);
  }


  return 0;  
}