/* include */
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

/* Define Poisson True Value Spectrum */
double ve_dune_poisson[66];
double vebar_dune_poisson[66];
double vu_dune_poisson[66];
double vubar_dune_poisson[66];

double ve_t2hk_poisson[8];
double vebar_t2hk_poisson[12];
double vu_t2hk_poisson[8];
double vubar_t2hk_poisson[12];

#include <stdarg.h>
 double min(int n, ...) 
 {
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
//
  
int random_poisson(double mu) 
{
    const gsl_rng_type * T;
    gsl_rng * r;
    int test;
    int i;
    gsl_rng_env_setup();
    struct timeval tv; // Seed generation based on time
    gettimeofday(&tv,0);
    unsigned long mySeed = tv.tv_sec + tv.tv_usec;
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, mySeed);
    unsigned int k = gsl_ran_poisson (r, mu);
    gsl_rng_free (r);
    return k;
}

inline double poisson_likelihood(double true_rate, double fit_rate)
{
    double res;
    if (true_rate==0) true_rate=true_rate+0.001;
    if (fit_rate==0) fit_rate=fit_rate+0.001;
    res = fit_rate - true_rate;
    if (fit_rate <= 0.0)
        { 
         res = 1e100;
        }
    else if (true_rate > 0)
        {
        res += true_rate * log(true_rate/fit_rate);
        }
    return 2.0 * res;
}

int do_poisson_fluctuation(glb_params test_values)
{
  glbSetOscillationParameters(test_values);
    glbSetRates();
      double *ve_dune      = glbGetRuleRatePtr(0, 0);
      double *vebar_dune   = glbGetRuleRatePtr(0, 1);
      double *vu_dune      = glbGetRuleRatePtr(0, 2);
      double *vubar_dune   = glbGetRuleRatePtr(0, 3);    
      double *ve_t2hk      = glbGetRuleRatePtr(1, 0);
      double *vebar_t2hk   = glbGetRuleRatePtr(1, 1);    
      double *vu_t2hk      = glbGetRuleRatePtr(1, 2);
      double *vubar_t2hk   = glbGetRuleRatePtr(1, 3);
    
    int ew_low, ew_high, i;
  for (int exp=0; exp <= 1; exp++){
      int rule_max= glbGetNumberOfRules(exp);
      for (int rule = 0; rule < rule_max; rule++){
      glbGetEnergyWindowBins(exp, rule, &ew_low, &ew_high);

    //以下開始判斷是哪個Spectrum
    if (exp == 0){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        ve_dune_poisson[i] = random_poisson(ve_dune[i]);}
      }
      if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        vebar_dune_poisson[i] = random_poisson(vebar_dune[i]);}
      }
      if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        vu_dune_poisson[i] = random_poisson(vu_dune[i]);}
      }
      if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        vubar_dune_poisson[i] = random_poisson(vubar_dune[i]);}
      }
    }

    if (exp == 1){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        ve_t2hk_poisson[i] = random_poisson(ve_t2hk[i]);}
      }
      if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        vebar_t2hk_poisson[i] = random_poisson(vebar_t2hk[i]);}
      }
      if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        vu_t2hk_poisson[i] = random_poisson(vu_t2hk[i]);}
      }
      if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        vubar_t2hk_poisson[i] = random_poisson(vubar_t2hk[i]);}
      }
    }
      }
  } 
  return 0;
}

double chi2_poisson(int exp, int rule, int np, double *x, double *errors, void* user_data)
{
    double *signal_fit_rate = glbGetSignalFitRatePtr(exp, rule);
    double *bg_fit_rate     = glbGetBGFitRatePtr(exp, rule);
    double fit_rate;
    double chi2 = 0.0;
    int i;
    int ew_low, ew_high;
    glbGetEnergyWindowBins(exp, rule, &ew_low, &ew_high);
    fit_rate = signal_fit_rate[i] + bg_fit_rate[i];
    
    if (exp == 0){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(ve_dune_poisson[i], fit_rate);}
      }
      if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(vebar_dune_poisson[i], fit_rate);}
      }
      if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(vu_dune_poisson[i], fit_rate);}
      }
      if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(vubar_dune_poisson[i], fit_rate);}
      }
    }

    if (exp == 1){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(ve_t2hk_poisson[i], fit_rate);}
      }
      if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(vebar_t2hk_poisson[i], fit_rate);}
      }
      if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(vu_t2hk_poisson[i], fit_rate);}
      }
      if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        chi2 += poisson_likelihood(vubar_t2hk_poisson[i], fit_rate);}
      }
    }
 
    return chi2;
}


//參數:{CP : [0, 1] , true value的deltacp}
double delta_chi2 (double CP , double deltacp) //根據CP的假設，生成Poisson Sample，計算其test statistic
{
    glb_params test_values_cpc = glbAllocParams(); 
    glb_params test_values_cpv = glbAllocParams(); 
    glb_params input_errors = glbAllocParams(); 

    /* 定義test_values_cpc */ 
      glbDefineParams(test_values_cpc,theta12_N*degree,theta13_N*degree,theta23_N*degree, 0*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
      glbSetDensityParams(test_values_cpc,1.0,GLB_ALL);

    /* 定義test_values_cpv */                                                         //deltacp為Input的值
      glbDefineParams(test_values_cpv,theta12_N*degree,theta13_N*degree,theta23_N*degree, deltacp*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
      glbSetDensityParams(test_values_cpv,1.0,GLB_ALL);

    /* 設定Projection */   
      glb_projection projection_cp = glbAllocProjection();
      //GLB_FIXED/GLB_FREE                theta12    theta13  theta23    deltacp     m21        m31
      glbDefineProjection(projection_cp, GLB_FIXED, GLB_FREE, GLB_FREE, GLB_FIXED, GLB_FIXED, GLB_FREE);//deltacp theta12 m21 不動，其他可變
      glbSetDensityProjectionFlag(projection_cp,GLB_FIXED,GLB_ALL);//matter density不變
      glbSetProjection(projection_cp);
   
    /* 關閉系統誤差 */   
      glbSwitchSystematics(GLB_ALL,GLB_ALL,GLB_OFF);

  if (CP == 0){ //CPC

    /* 根據CPC的假設，生成Poisson True Spectrum */   
      do_poisson_fluctuation(test_values_cpc);
     
    /* 設定Prior OFF */   
      glbDefineParams(input_errors,0,0,0,0,0,0);
      glbSetDensityParams(input_errors,0,GLB_ALL);
      glbSetInputErrors(input_errors);
      
    /* 計算Chi square under cpc */ 
      glbSetOscillationParameters(test_values_cpc);
      glbSetRates();
      glbDefineChiFunction(&chi2_poisson,0,"chi2_poisson",NULL);
      glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
      glbSetCentralValues(test_values_cpc); 
      double a = glbChiNP(test_values_cpc,NULL,GLB_ALL);
    
    /* 計算Chi square under cpv */ 
      glbSetOscillationParameters(test_values_cpv);
      glbSetRates();
      glbDefineChiFunction(&chi2_poisson,0,"chi2_poisson",NULL);
      glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
      glbSetCentralValues(test_values_cpv); 
      double b = glbChiNP(test_values_cpv,NULL,GLB_ALL);
    
    /* 輸出Delta Chi square */ 
    // printf("a = %g, b = %g  \n",a,b);
      return a - b ;
  }

  if (CP == 1){ //CPV

    /* 根據CPV的假設，生成Poisson True Spectrum */   
      do_poisson_fluctuation(test_values_cpv);
     
    /* 設定Prior OFF */   
      glbDefineParams(input_errors,0,0,0,0,0,0);
      glbSetDensityParams(input_errors,0,GLB_ALL);
      glbSetInputErrors(input_errors);
      
    /* 計算Chi square under cpc */ 
      glbSetOscillationParameters(test_values_cpc);
      glbSetRates();
      glbDefineChiFunction(&chi2_poisson,0,"chi2_poisson",NULL);
      glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
      glbSetCentralValues(test_values_cpc); 
      double a = glbChiNP(test_values_cpc,NULL,GLB_ALL);
    
    /* 計算Chi square under cpv */ 
      glbSetOscillationParameters(test_values_cpv);
      glbSetRates();
      glbDefineChiFunction(&chi2_poisson,0,"chi2_poisson",NULL);
      glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
      glbSetCentralValues(test_values_cpv); 
      double b = glbChiNP(test_values_cpv,NULL,GLB_ALL);
    
    /* 輸出Delta Chi square */ 
      // printf("a = %g, b = %g  \n",a,b);
      return a - b ;
  }
}


int main(int argc, char *argv[])
{ 

    glbInit(argv[0]); 

    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

    FILE* OUT =  fopen("two_delta_chi2_distribution.dat","w");//建立輸出檔案

int TOTALsample = 10000;
  for (int count = 0; count < TOTALsample; count++)
  { printf("%d \n",count); //看進度
    double q0 = delta_chi2(0 , 90);
    double q1 = delta_chi2(1 , 90);
    fprintf(OUT," %g %g \n ",q0 ,q1 );
  
  }
  
  return 0;  
}