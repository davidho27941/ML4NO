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

float keithRandom() {
    // Random number function based on the GNU Scientific Library
    // Returns a random float between 0 and 1, exclusive; e.g., (0,1)
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    struct timeval tv; // Seed generation based on time
    gettimeofday(&tv,0);
    unsigned long mySeed = tv.tv_sec + tv.tv_usec;
    T = gsl_rng_default; // Generator setup
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, mySeed);
    double u = gsl_rng_uniform(r); // Generate it!
    gsl_rng_free (r);
    return (float)u;
}


double randn (double mu, double sigma) /*mu is the central value, sigma is the width of gaussian distribution. */
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }
    
    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (double) X1);
}

double TCRandom(double mu, double sigma) {
    // Random number function based on the GNU Scientific Library
    // Returns a random float between 0 and 1, exclusive; e.g., (0,1)
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    struct timeval tv; // Seed generation based on time
    gettimeofday(&tv,0);
    unsigned long mySeed = tv.tv_sec + tv.tv_usec;
    T = gsl_rng_default; // Generator setup
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, mySeed);
    double u = mu + gsl_ran_gaussian(r, sigma); // Generate it!
//    printf ("%g %g\n",mu,sigma);
    gsl_rng_free (r);
    
    return u;
}


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
    glb_params true_values = glbAllocParams();

  /* Set standard oscillation parameters (cf. hep-ph/0405172v5) */

    FILE* BIN =   fopen("bin_setup.dat","w");//建立輸出檔案
    FILE* OUT =   fopen("sample.dat","w");//建立輸出檔案
    
  double theta12_c = 33.44; 
  // double theta13_c = 8.57;
  // double theta23_c = 45;
  double sdm_c = 7.42;
  // double ldm_c = 2.514;

    // glb_params test_values = glbAllocParams();    
    // glbDefineParams(test_values,theta12_c*degree,theta13_c*degree,theta23_c*degree,195,1e-5*sdm_c,1e-3*ldm_c);  
    // glbSetDensityParams(test_values,1.0,GLB_ALL); 

  double delta_12 =(35.86-31.27)/2;
  double delta_13 =( 8.97- 8.20)/2;
  double delta_23 =(51.80-39.60)/2;
  double delta_sdm=( 8.04- 6.82)/2;
  double delta_ldm=(2.598-2.431)/2;
  
  int location;
  int TOTALsample=1000000 ;//choosed by the user 

    int ew_low, ew_high;
    int i,exp,channel;
    
    
    for(exp=0; exp<2; exp++){
        double *bin_c_energy  = glbGetBinCentersListPtr(exp);
        double *bin_size      = glbGetBinSizeListPtr(exp);
        int channel_max= glbGetNumberOfRules(exp);
        for(channel=0;channel<channel_max;channel++){
    glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
    fprintf(BIN,"#central energy %i %i [GeV] \n",exp,channel);
    for (int i=ew_low; i <= ew_high; i++)
    { fprintf(BIN,"%g ",bin_c_energy[i]);} 
    fprintf(BIN,"\n");
    fprintf(BIN,"#bin size %i %i [GeV]\n",exp,channel);
    for (int i=ew_low; i <= ew_high; i++)
    { fprintf(BIN,"%g ",bin_size[i]);}
    fprintf(BIN,"\n");
        }}
    
  for (location=0;location<TOTALsample;location++){
  
  // double theta12 = TCRandom (theta12_c, delta_12 ); 
  // double theta13 = TCRandom (theta13_c, delta_13 );
  // double theta23 = TCRandom (theta23_c, delta_23 );
  // double sdm     = TCRandom (    sdm_c, delta_sdm);  //eV^2
  // double ldm     = TCRandom (    ldm_c, delta_ldm);  //eV^2
  
  // double theta12 = (35.86+31.27)/2 + (keithRandom()-0.5)*(35.86-31.27);
  double theta13 = ( 8.97+ 8.20)/2 + (keithRandom()-0.5)*( 8.97- 8.20);
  double theta23 = (45 + 45)/2 + (keithRandom()-0.5)*(51.80-39.60);
  // double sdm = ( 8.04 + 6.82)/2 + (keithRandom()-0.5)*( 8.04- 6.82);
  double ldm = (2.598 + 2.431)/2 + (keithRandom()-0.5)*(2.598-2.431);

  double sign_ldm= 1;
  if(keithRandom()<0.5) {sign_ldm= -1;} 
  ldm = sign_ldm* ldm;

  // if(keithRandom()<0.33) {theta23 = 45;} 

  // double deltacp = keithRandom()*360.0*2.0-180.0;
  // if (deltacp<0){deltacp=0;}
  // else if (deltacp>360){deltacp=180;}

  double deltacp = keithRandom()*360.0;
  
  int OCTANT= LABEL_OT(theta23);
  int CPV    =LABEL_CP(deltacp);
  int MO     =LABEL_MO(ldm    );

  glbDefineParams(true_values,theta12_c*degree,theta13*degree,theta23*degree,deltacp*degree,1e-5*sdm_c,1e-3*ldm);
  glbSetDensityParams(true_values,1.0,GLB_ALL);
  
  glbSetOscillationParameters(true_values);
  glbSetRates();

    
    
  exp=0;channel=0;
  for(exp=0; exp<2; exp++){
  int channel_max= glbGetNumberOfRules(exp);
  for(channel=0;channel<channel_max;channel++){
      
  glbGetEnergyWindowBins(exp, channel, &ew_low, &ew_high);
  
  double *true_rates = glbGetRuleRatePtr(exp, channel);
int count = 0;

  for (i=ew_low; i <= ew_high; i++){fprintf(OUT,"%g ",true_rates[i]);
  count += 1;
  }
    // printf("count = %d \n",count);
  //printf("%i %i %i \n",exp,channel,ew_high-ew_low);
  }
  }

    // double chi2;
    // chi2=glbChiSys(test_values,GLB_ALL,GLB_ALL);
    fprintf(OUT,"  %g %g %g %g %g %g %i %i %i\n",theta12_c,theta13,theta23,deltacp,sdm_c,ldm,OCTANT,CPV,MO);
    printf("%i \n",location);  
  }
  /* Clean up */
  glbFreeParams(true_values);

  return 0;  
}