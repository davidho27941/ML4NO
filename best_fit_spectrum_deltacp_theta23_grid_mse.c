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
  double ve_dune_poisson[70];    //66
  double vebar_dune_poisson[70]; //66
  double vu_dune_poisson[70];    //66
  double vubar_dune_poisson[70]; //66

  double ve_t2hk_poisson[8];      //8
  double vu_t2hk_poisson[12];     //12
  double vebar_t2hk_poisson[8];   //8
  double vubar_t2hk_poisson[12];  //12

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

 /* 定義3 sigma range 的Prior (For NO) */
double prior_3sigma_NO(const glb_params in, void* user_data)
{
  glb_params central_values = glbAllocParams();
  glb_params input_errors = glbAllocParams();
  glb_projection p = glbAllocProjection();
  glbGetCentralValues(central_values);
  glbGetInputErrors(input_errors);
  glbGetProjection(p);
  int i;
  double pv = 0.0;
  double fit_theta12 ; 
  double fit_theta13 ;
  double fit_theta23 ;
  double fit_deltacp ;
  double fit_ldm ;
  double fit_sdm ;

  /* 取得參數目前Fit Value */
  fit_theta12 = glbGetOscParams(in,0);
  fit_theta13 = glbGetOscParams(in,1);
  fit_theta23 = glbGetOscParams(in,2);
  fit_deltacp = glbGetOscParams(in,3);
  fit_sdm     = glbGetOscParams(in,4);
  fit_ldm     = glbGetOscParams(in,5);

  /* 判斷參數是否要引入Prior */
  if(glbGetProjectionFlag(p,0)==GLB_FREE){
    if(fit_theta12  > 35.86 *degree || fit_theta12 < 31.27 *degree){
      pv += 1e20;
    }
  }
  if(glbGetProjectionFlag(p,1)==GLB_FREE){
    if(fit_theta13  > 8.97 *degree || fit_theta13 < 8.20 *degree){
      pv += 1e20;
    }
  }
  if(glbGetProjectionFlag(p,2)==GLB_FREE){
    if(fit_theta23 > 51.80 *degree  || fit_theta23 < 39.60 *degree ){
      pv += 1e20;
    }
  }
  // if(glbGetProjectionFlag(p,3)==GLB_FREE){
  //   if(fit_deltacp == 0 *degree || fit_deltacp == 180 *degree || fit_deltacp < 0 *degree|| fit_deltacp > 360 *degree){
  //     pv += 1e20;
  //   }
  // }
  if(glbGetProjectionFlag(p,4)==GLB_FREE){
    if(fit_sdm > 8.04 *1e-5  || fit_sdm  < 6.82 *1e-5){
      pv += 1e20;
    }
  }
  if(glbGetProjectionFlag(p,5)==GLB_FREE){
    if(fit_ldm  > 2.598 *1e-3 || fit_ldm  < 2.431 *1e-3){
      pv += 1e20;
    }
  }

  glbFreeParams(central_values);
  glbFreeParams(input_errors);
  glbFreeProjection(p);
  // printf("pv = %g \n",pv);
  return pv;
}

 /* 定義3 sigma range 的Prior (For IO) */
double prior_3sigma_IO(const glb_params in, void* user_data)
{
  glb_params central_values = glbAllocParams();
  glb_params input_errors = glbAllocParams();
  glb_projection p = glbAllocProjection();
  glbGetCentralValues(central_values);
  glbGetInputErrors(input_errors);
  glbGetProjection(p);
  int i;
  double pv = 0.0;
  double fit_theta12 ; 
  double fit_theta13 ;
  double fit_theta23 ;
  double fit_deltacp ;
  double fit_ldm ;
  double fit_sdm ;

  /* 取得參數目前Fit Value */
  fit_theta12 = glbGetOscParams(in,0);
  fit_theta13 = glbGetOscParams(in,1);
  fit_theta23 = glbGetOscParams(in,2);
  fit_deltacp = glbGetOscParams(in,3);
  fit_sdm     = glbGetOscParams(in,4);
  fit_ldm     = glbGetOscParams(in,5);

  /* 判斷參數是否要引入Prior */
  if(glbGetProjectionFlag(p,0)==GLB_FREE){
    if(fit_theta12  > 35.87 *degree || fit_theta12 < 31.27 *degree){
      pv += 1e20;
    }
  }
  else if(glbGetProjectionFlag(p,1)==GLB_FREE){
    if(fit_theta13  > 8.98 *degree || fit_theta13 < 8.24 *degree){
      pv += 1e20;
    }
  }
  else if(glbGetProjectionFlag(p,2)==GLB_FREE){
    if(fit_theta23 > 52.00 *degree  || fit_theta23 < 39.90 *degree ){
      pv += 1e20;
    }
  }
  // else if(glbGetProjectionFlag(p,3)==GLB_FREE){
  //   if(fit_deltacp == 0 *degree || fit_deltacp == 180 *degree ){
  //     pv += 1e20;
  //   }
  // }
  else if(glbGetProjectionFlag(p,4)==GLB_FREE){
    if(fit_sdm > 8.04 *1e-5  || fit_sdm  < 6.82 *1e-5){
      pv += 1e20;
    }
  }
  else if(glbGetProjectionFlag(p,5)==GLB_FREE){
    if(fit_ldm  < -2.583 *1e-3 || fit_ldm  > -2.412 *1e-3){
      pv += 1e20;
    }
  }

  glbFreeParams(central_values);
  glbFreeParams(input_errors);
  glbFreeProjection(p);
  // printf("pv = %g \n",pv);
  return pv;
}

/* Poisson亂數生成器 */
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

/* 定義Poisson Likelihood Function */
inline double poisson_likelihood(double true_rate, double fit_rate)
{
    double res = 0 ;
    if (true_rate==0) true_rate=true_rate+1e-9;
    if (fit_rate==0) fit_rate=fit_rate+1e-9;
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
    //return 1;
}

/* 對True Value Spectrum 做Poisson Fluctuation */
int do_poisson_fluctuation(glb_params test_values)
{
  glbSetOscillationParameters(test_values);
  glbSetRates();
      double *ve_dune      = glbGetRuleRatePtr(0, 0);
      double *vebar_dune   = glbGetRuleRatePtr(0, 1);
      double *vu_dune      = glbGetRuleRatePtr(0, 2);
      double *vubar_dune   = glbGetRuleRatePtr(0, 3);    
      double *ve_t2hk      = glbGetRuleRatePtr(1, 0);
      double *vu_t2hk      = glbGetRuleRatePtr(1, 1);    
      double *vebar_t2hk   = glbGetRuleRatePtr(1, 2);
      double *vubar_t2hk   = glbGetRuleRatePtr(1, 3);
    
      int ew_low, ew_high, i;
  
  // /* 光譜測試 */
  // glbGetEnergyWindowBins(0, 0, &ew_low, &ew_high);
  // for (i=ew_low; i <= ew_high; i++) {
  //   printf("%g ",ve_dune[i]);}
  //   printf("\n");
  //   printf("\n");
  
  // for (i=0; i < 66; i++) {
  //   printf("%g ",ve_dune[i]);}
  //   printf("\n");
  for (int exp=0; exp <= 1; exp++){
      int rule_max= glbGetNumberOfRules(exp);
      for (int rule = 0; rule < rule_max; rule++){
        glbGetEnergyWindowBins(exp, rule, &ew_low, &ew_high);
      // printf("ew_low = %d, ew_high = %d ",ew_low,ew_high);
    //以下開始判斷是哪個Spectrum
    if (exp == 0){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        ve_dune_poisson[i] = random_poisson(ve_dune[i]);

        // printf("%d ",i);
        }
      }
      else if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        vebar_dune_poisson[i] = random_poisson(vebar_dune[i]);

        // printf("%d ",i);
        }
      }

      else if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        vu_dune_poisson[i] = random_poisson(vu_dune[i]);

        // printf("%d ",i);
        }
      }
      else if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        vubar_dune_poisson[i] = random_poisson(vubar_dune[i]);

        // printf("%d ",i);
        }
      }
    }

    else if (exp == 1){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        ve_t2hk_poisson[i] = random_poisson(ve_t2hk[i]);

        // printf("%d ",i);
        }
      }
      else if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        vu_t2hk_poisson[i] = random_poisson(vu_t2hk[i]);

        // printf("%d ",i);
        }
      }
      else if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        vebar_t2hk_poisson[i] = random_poisson(vebar_t2hk[i]);

        // printf("%d ",i);
        }
      }
      else if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        vubar_t2hk_poisson[i] = random_poisson(vubar_t2hk[i]);

        // printf("%d ",i);
        }
      }
    }
      }
  } 
 
  return 0;
}

/* 定義 Chi Square */
double chi2_poisson(int exp, int rule, int np, double *x, double *errors, void* user_data)
{
    double *signal_fit_rate = glbGetSignalFitRatePtr(exp, rule);
    double *bg_fit_rate     = glbGetBGFitRatePtr(exp, rule);
    double fit_rate;
    double chi2 = 0.0;
    int i;
    int ew_low, ew_high;
    glbGetEnergyWindowBins(exp, rule, &ew_low, &ew_high);

    
    if (exp == 0){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];
        chi2 += poisson_likelihood(ve_dune_poisson[i], fit_rate);}
      }
      else if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(vebar_dune_poisson[i], fit_rate);}
      }
      else if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(vu_dune_poisson[i], fit_rate);}
      }
      else if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(vubar_dune_poisson[i], fit_rate);}
      }
    }

    else if (exp == 1){
      if (rule == 0){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(ve_t2hk_poisson[i], fit_rate);}
      }
      else if (rule == 1){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(vu_t2hk_poisson[i], fit_rate);}
      }
      else if (rule == 2){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(vebar_t2hk_poisson[i], fit_rate);}
      }
      else if (rule == 3){
        for (i=ew_low; i <= ew_high; i++) {
        fit_rate = signal_fit_rate[i] + bg_fit_rate[i];          
        chi2 += poisson_likelihood(vubar_t2hk_poisson[i], fit_rate);}
      }
    }
    return chi2;
}
/***************************************************************************
 *                            M A I N   P R O G R A M                      *
 ***************************************************************************/
   
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


/* 計算chi2 with projection onto deltacp, 4 conditions for test value (0,NO) (pi,NO) (0,IO) (pi,IO) */ 
//參數:{系統誤差on_off:[GLB_ON,GLB_OFF], 選定實驗EXP:[0,1,GLB_ALL], true value的deltacp}
double chi2_proj (int on_off, int EXP , int MO , double deltacp, double theta23, double deltacp_ini, double theta23_ini, double fit_values[10])
{
    glb_params test_values = glbAllocParams(); 
    glb_params true_values = glbAllocParams(); 
    glb_params input_errors = glbAllocParams(); 

    glb_params fit_values_N = glbAllocParams(); 
    glb_params fit_values_I = glbAllocParams(); 
    // clock_t t1, t2;

  if(MO == 1){
  /* 定義true value (依照NO) */ 
    glbDefineParams(true_values,theta12_N*degree,theta13_N*degree,theta23*degree, deltacp*degree ,1e-5*sdm_g_N,1e-3*ldm_g_N);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
  }

  if(MO == -1){
  /* 定義true value (依照IO) */ 
    glbDefineParams(true_values,theta12_I*degree,theta13_I*degree,theta23*degree, deltacp*degree ,1e-5*sdm_g_I,1e-3*ldm_g_I);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
  }
 
  /* 對True Value Spectrum 做Poisson Fluctuation */
    do_poisson_fluctuation(true_values);

  /* 設定Projection */   
    glb_projection projection_cp_fixed = glbAllocProjection();
    //GLB_FIXED/GLB_FREE                      theta12    theta13  theta23    deltacp     m21        m31
    glbDefineProjection(projection_cp_fixed, GLB_FIXED, GLB_FREE, GLB_FREE, GLB_FREE, GLB_FIXED, GLB_FREE);//deltacp theta12 m21 不動，其他可變
    glbSetDensityProjectionFlag(projection_cp_fixed,GLB_FIXED,GLB_ALL);//matter density不變
    glbSetProjection(projection_cp_fixed);

  /* 開關系統誤差 */ 
    glbSwitchSystematics(EXP,GLB_ALL,on_off);  

  /* 定義 Chi Square */
    glbDefineChiFunction(&chi2_poisson,0,"chi2_poisson",NULL);
    glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);

////////////////////////////////////////////////////////////////////////////////////////////////////////

  double chi2_N, chi2_I;
  double chi2 = 1e10;
      
    /* 1. 計算test value中 (delta_cp = NO) 的chi_N */
      glbDefineParams(test_values, theta12_N*degree, theta13_N*degree, theta23_ini*degree,  deltacp_ini*degree , 1e-5*sdm_g_N, 1e-3*ldm_g_N);  
      glbSetDensityParams(test_values,1.0,GLB_ALL); 
      /* 設定Prior (3 sigma range, Normal Ordering)*/
      glbRegisterPriorFunction(prior_3sigma_NO,NULL,NULL,NULL);
      glbDefineParams(input_errors,0,0,0,0,0,0);
      glbSetDensityParams(input_errors,0,GLB_ALL);
      glbSetInputErrors(input_errors);
      glbSetCentralValues(test_values); 

      chi2_N = glbChiNP(test_values,fit_values_N,EXP);

    /* 2. 計算test value中 (delta_cp = IO) 的chi_I */
      glbDefineParams(test_values, theta12_I*degree, theta13_I*degree, theta23_ini*degree,  deltacp_ini*degree , 1e-5*sdm_g_I, 1e-3*ldm_g_I);  
      glbSetDensityParams(test_values,1.0,GLB_ALL); 
    /* 設定Prior (3 sigma range, Inverse Ordering)*/
      glbRegisterPriorFunction(prior_3sigma_IO,NULL,NULL,NULL);
      glbDefineParams(input_errors,0,0,0,0,0,0);
      glbSetDensityParams(input_errors,0,GLB_ALL);
      glbSetInputErrors(input_errors);
      glbSetCentralValues(test_values); 

      chi2_I = glbChiNP(test_values,fit_values_I,EXP);
    
    double chi2_tmp = min(2, chi2_N, chi2_I);
    //printf("chi2_tmp =  %f \n", chi2_tmp);
      if(chi2_tmp < chi2){
        chi2 = chi2_tmp;

          if(chi2 == chi2_N){
            for(int i = 0; i<=5; i++){
              fit_values[i] = glbGetOscParams(fit_values_N, i);
            }
          }
          else if(chi2_tmp == chi2_I){
            for(int i = 0; i<=5; i++){
              fit_values[i] = glbGetOscParams(fit_values_I, i);
            }
          }
      }
      
return chi2;
}



int main(int argc, char *argv[])
{ 
  // clock_t t_start, t_end;
  // t_start = clock();
    glbInit(argv[0]); 

    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);



  /* 設定輸出檔案位置 方式1*/ 
    // int len = strlen("./sample_grid/best_fit_spectrum_deltacp_theta23_grid_mse_ver") + strlen(argv[3]) + strlen(".dat") + 1;
    // char path[len];
    // strcpy(path,"./sample_grid/best_fit_spectrum_deltacp_theta23_grid_mse_ver");
    // strcat(path, argv[3]);
    // strcat(path, ".dat");
    // printf("File Path : %s\n",path);
    // FILE* OUT =  fopen(path,"w");//建立輸出檔案

  /* 設定輸出檔案位置 方式2*/   
  FILE* OUT =  fopen("best_fit_spectrum_deltacp_theta23_grid_mse.dat","w");

  double fit_values[10];
  double chi_square;
  //printf("Event #%i \n",location);  

  /* Random True Value of deltacp & theta23 */
  double deltacp = keithRandom()*360.0;
  double theta23 = (45 + 45)/2 + (keithRandom()-0.5)*(50-40);
  double theta23_initial;
  double deltacp_initial;
  /* Calculate Chi2 in different Grids*/
  clock_t t1, t2;
  double t;
  t1 = clock();
  for (int i = 4000; i <= 5000; i += 25 ){
    theta23_initial = i/100.000;
    for (int j =0; j <= 360; j += 9){
    deltacp_initial = j;
    printf("theta23_initial = %g ,deltacp_initial = %g \n",theta23_initial,deltacp_initial);

      chi_square = chi2_proj(GLB_OFF, atof(argv[1]), atof(argv[2]),  deltacp, theta23, deltacp_initial, theta23_initial, fit_values);   // 0代表DUNE, 1代表T2HK
      
      /* Relocate fit_deltacp to 0-2pi range */
      if(fit_values[3] > 2*M_PI){
        int run; 
        run = fit_values[3]/2/M_PI; 
        fit_values[3] = fit_values[3]-run*2*M_PI;
        }
      else if(fit_values[3] < 0){
        int run; 
        run = -fit_values[3]/2/M_PI; 
        run = run+1; 
        fit_values[3] = fit_values[3]+run*2*M_PI;
        }
      
      if (atof(argv[2]) == 1){
        fprintf(OUT,"%g %g %g %g %g %g   %g %g %g   %g %g %g %g %g %g \n ",theta12_N*degree,theta13_N*degree,theta23*degree, deltacp ,1e-5*sdm_g_N,1e-3*ldm_g_N,
        theta23_initial,deltacp_initial,chi_square,fit_values[0],fit_values[1],fit_values[2],fit_values[3]/degree,fit_values[4],fit_values[5]);
      }
      if (atof(argv[2]) == -1){
        fprintf(OUT,"%g %g %g %g %g %g   %g %g %g   %g %g %g %g %g %g \n ",theta12_I*degree,theta13_I*degree,theta23*degree, deltacp ,1e-5*sdm_g_I,1e-3*ldm_g_I,
        theta23_initial,deltacp_initial,chi_square,fit_values[0],fit_values[1],fit_values[2],fit_values[3]/degree,fit_values[4],fit_values[5]);
      }
    }
  }
  t2 = clock();
  t = (double)(t2-t1)/CLOCKS_PER_SEC;
  fprintf(OUT,"總運行時間 : %g sec",t);
  return 0;  
}

/* 
使用方式：./best_fit_spectrum_deltacp_theta23_grid_mse [EXP = 0,1,-1] [MO = 1,-1] [Version: 1,2,3......]
Output dat 格式 : 
[theta12_true, theta13_true, theta23_true(random), deltacp_true(random), sdm_true, sdm_true,
theta23_initial, deltacp_initial,chi2,
theta12_fit, theta13_fit, theta23_fit, deltacp_fit, sdm_fit, sdm_fit]

for i in range(25):
    print("nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 " +str(i+1)+ " > 1207.log &")

nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 1 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 2 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 3 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 4 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 5 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 6 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 7 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 8 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 9 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 10 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 11 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 12 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 13 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 14 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 15 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 16 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 17 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 18 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 19 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 20 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 21 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 22 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 23 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 24 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 25 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 26 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 27 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 28 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 29 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 30 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 31 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 32 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 33 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 34 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 35 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 36 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 37 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 38 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 39 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 40 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 41 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 42 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 43 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 44 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 45 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 46 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 47 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 48 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 49 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 50 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 51 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 52 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 53 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 54 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 55 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 56 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 57 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 58 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 59 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 60 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 61 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 62 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 63 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 64 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 65 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 66 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 67 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 68 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 69 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 70 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 71 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 72 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 73 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 74 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 75 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 76 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 77 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 78 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 79 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 80 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 81 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 82 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 83 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 84 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 85 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 86 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 87 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 88 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 89 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 90 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 91 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 92 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 93 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 94 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 95 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 96 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 97 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 98 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 99 > 1207.log &
nohup ./best_fit_spectrum_deltacp_theta23_grid_mse -1 1 100 > 1207.log &
*/