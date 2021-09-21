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
#include "hdf5.h"
#include <stdarg.h>

#define degree 0.0174
double min (int n, ...) {
	/*
	n是參數個數，後面才是參數本身 
	*/
	int 	i;
	double 	min_num = 1e20;
	double 	input;
	va_list vl;
	va_start(vl,n);
	for ( i = 0 ; i < n ; i++ ) {
		input = va_arg(vl,double);
		min_num = min_num > input ? input : min_num;
	} 
	va_end(vl);
	return min_num;
 } 

/***************************************************************************
 *                            M A I N   P R O G R A M                      *
 ***************************************************************************/

/* 定義3 sigma range 的Prior (For NO) */
double prior_3sigma_NO(const glb_params in, void* user_data) {
	glb_params 		central_values 	= 	glbAllocParams();
	glb_params 		input_errors 	= 	glbAllocParams();
	glb_projection 	p 				= 	glbAllocProjection();

	glbGetCentralValues(central_values);
	glbGetInputErrors(input_errors);
	glbGetProjection(p);
	int i;
	double 	pv = 0.0,
		  	fit_theta12,
		   	fit_theta13,
		   	fit_theta23,
		   	fit_deltacp,
		   	fit_ldm,
		   	fit_sdm;

    /* 取得參數目前Fit Value */
	fit_theta12 = glbGetOscParams(in,0);
	fit_theta13 = glbGetOscParams(in,1);
	fit_theta23 = glbGetOscParams(in,2);
	fit_deltacp = glbGetOscParams(in,3);
	fit_sdm     = glbGetOscParams(in,4);
	fit_ldm     = glbGetOscParams(in,5);

    /* 判斷參數是否要引入Prior */
	if (glbGetProjectionFlag(p,0)==GLB_FREE){
		if (fit_theta12  > 35.86 *degree || fit_theta12 < 31.27 *degree){
		    pv += 1e20;
		}
	}
	if (glbGetProjectionFlag(p,1)==GLB_FREE){
		if (fit_theta13  > 8.97 *degree || fit_theta13 < 8.20 *degree){
		    pv += 1e20;
		}
	}
	if (glbGetProjectionFlag(p,2)==GLB_FREE){
		if (fit_theta23 > 51.80 *degree  || fit_theta23 < 39.60 *degree ){
		    pv += 1e20;
		}
	}
	if (glbGetProjectionFlag(p,3)==GLB_FREE){
		if (fit_deltacp == 0 *degree || fit_deltacp == 180 *degree || fit_deltacp < 0 *degree|| fit_deltacp > 360 *degree){
		    pv += 1e20;
		}
	}
	if (glbGetProjectionFlag(p,4)==GLB_FREE){
		if (fit_sdm > 8.04 *1e-5  || fit_sdm  < 6.82 *1e-5){
		    pv += 1e20;
		}
	}
	if (glbGetProjectionFlag(p,5)==GLB_FREE){
		if (fit_ldm  > 2.598 *1e-3 || fit_ldm  < 2.431 *1e-3){
		    pv += 1e20;
		}
	}

	glbFreeParams(central_values);
	glbFreeParams(input_errors);
	glbFreeProjection(p);
	return pv;
}

 /* 定義3 sigma range 的Prior (For IO) */
double prior_3sigma_IO(const glb_params in, void* user_data)
{
	glb_params 		central_values 	= 	glbAllocParams();
	glb_params 		input_errors 	= 	glbAllocParams();
	glb_projection 	p 				= 	glbAllocProjection();
	
	glbGetCentralValues(central_values);
	glbGetInputErrors(input_errors);
	glbGetProjection(p);
	int i;
	double 	pv = 0.0,
		   	fit_theta12,
		   	fit_theta13,
		   	fit_theta23,
		   	fit_deltacp,
		   	fit_ldm,
		   	fit_sdm;

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
	if(glbGetProjectionFlag(p,1)==GLB_FREE){
		if(fit_theta13  > 8.98 *degree || fit_theta13 < 8.24 *degree){
		    pv += 1e20;
		}
	}
	if(glbGetProjectionFlag(p,2)==GLB_FREE){
		if(fit_theta23 > 52.00 *degree  || fit_theta23 < 39.90 *degree ){
		    pv += 1e20;
		}
	}
	if(glbGetProjectionFlag(p,3)==GLB_FREE){
		if(fit_deltacp == 0 *degree || fit_deltacp == 180 *degree ){
		    pv += 1e20;
		}
	}
	if(glbGetProjectionFlag(p,4)==GLB_FREE){
		if(fit_sdm > 8.04 *1e-5  || fit_sdm  < 6.82 *1e-5){
		    pv += 1e20;
		}
	}
	if(glbGetProjectionFlag(p,5)==GLB_FREE){
		if(fit_ldm  < -2.583 *1e-3 || fit_ldm  > -2.412 *1e-3){
		    pv += 1e20;
		}
	}

	glbFreeParams(central_values);
	glbFreeParams(input_errors);
	glbFreeProjection(p);
	return pv;
}

/* Poisson亂數生成器 */
int random_poisson(double mu) {
	const 	gsl_rng_type * T;
	gsl_rng * r;
	int 	test;
	int 	i;
	gsl_rng_env_setup();
	struct 	timeval tv; // Seed generation based on time
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
inline double poisson_likelihood(double true_rate, double fit_rate) {
	double res = 0 ;
	true_rate = true_rate == 0 ? (true_rate+1e-9) : true_rate;
	fit_rate = fit_rate == 0 ? (fit_rate+1e-9) : fit_rate;

	res = fit_rate - true_rate;
	if (fit_rate <= 0.0) { 
		res = 1e100;
	} else if (true_rate > 0) {
		res += true_rate * log(true_rate/fit_rate);
	}
		
	return 2.0 * res;
}

/* 對True Value Spectrum 做Poisson Fluctuation */
void do_poisson_fluctuation(glb_params test_values, int mode_expr, double* dataset, int* dset_info) {
	
	glbSetOscillationParameters(test_values);
	glbSetRates();

	int		ew_low, 
			ew_high, 
			i, 
			ladder = 0;
	int    	count_dpf = 0;
	int 	num_channel,
			num_bins, 
			cumu_bins;
	if (mode_expr == 0){
		double *ve_dune      = glbGetRuleRatePtr(0, 0);
		double *vebar_dune   = glbGetRuleRatePtr(0, 1);
		double *vu_dune      = glbGetRuleRatePtr(0, 2);
		double *vubar_dune   = glbGetRuleRatePtr(0, 3);    
		
		num_channel = 4;
		cumu_bins	= 0;
		for (int channel = 0; channel < num_channel; channel++){
			glbGetEnergyWindowBins(mode_expr, channel, &ew_low, &ew_high);
			num_bins = ew_hight - ew_low + 1;
			count_dpf = 0;
			for (i=ew_low; i <= ew_high; i++) { 
				*(dataset + cumu_bins + count_dpf) = random_poisson(ve_dune[i]);		
				count_dpf += 1;
			}
			cumu_bins += num_bins;
			*(dset_info + channel) = num_bins;
		}
	} else if (mode_expr == 1){ 
		double *ve_t2hk      = glbGetRuleRatePtr(1, 0);
		double *vu_t2hk      = glbGetRuleRatePtr(1, 1);    
		double *vebar_t2hk   = glbGetRuleRatePtr(1, 2);
		double *vubar_t2hk   = glbGetRuleRatePtr(1, 3);	
		
		num_channel = 4;
		cumu_bins	= 0;
		for (int channel = 0; channel < num_channel; channel++){
			glbGetEnergyWindowBins(mode_expr, channel, &ew_low, &ew_high);
			num_bins = ew_hight - ew_low + 1;
			count_dpf = 0;
			for (i=ew_low; i <= ew_high; i++) { 
				*(dataset + cumu_bins + count_dpf) = random_poisson(ve_dune[i]);		
				count_dpf += 1;
			}
			cumu_bins += num_bins;
			*(dset_info + channel) = num_bins;
		}
	} else if (mode_expr == -1) {
		double *ve_dune      = glbGetRuleRatePtr(0, 0);
		double *vebar_dune   = glbGetRuleRatePtr(0, 1);
		double *vu_dune      = glbGetRuleRatePtr(0, 2);
		double *vubar_dune   = glbGetRuleRatePtr(0, 3);    
		double *ve_t2hk      = glbGetRuleRatePtr(1, 0);
		double *vu_t2hk      = glbGetRuleRatePtr(1, 1);    
		double *vebar_t2hk   = glbGetRuleRatePtr(1, 2);
		double *vubar_t2hk   = glbGetRuleRatePtr(1, 3);
		
		num_expr	= 2;
		num_channel = 4;
		cumu_bins	= 0;
		int expr;
		for ( expr = 0; expr < num_expr; expr++) {
			for (int channel = 0; channel < num_channel; channel++){
				glbGetEnergyWindowBins(expr, channel, &ew_low, &ew_high);
				num_bins = ew_hight - ew_low + 1;
				count_dpf = 0;
				for (i=ew_low; i <= ew_high; i++) { 
					*(dataset + cumu_bins + count_dpf) = random_poisson(ve_dune[i]);		
					count_dpf += 1;
				}
				cumu_bins += num_bins;
				*(dset_info + channel) = num_bins;
			}
		}
	} else {
		printf("Please inpuit a correct experiment protocol.");
	}
}

/* 定義 Chi Square */
double chi2_poisson(int exp, int rule, int np, double *x, double *errors, void* user_data) {
	double 	*signal_fit_rate = glbGetSignalFitRatePtr(exp, rule);
	double 	*bg_fit_rate     = glbGetBGFitRatePtr(exp, rule);
	double 	fit_rate;
	double 	chi2 = 0.0;
	int 	i;
	int 	ew_low, 
			ew_high;
	glbGetEnergyWindowBins(exp, rule, &ew_low, &ew_high);
	int 	sum_y = 0, 
			index, 
			pos, 
			count_cp;
	int 	data_info_dune[4] = { 70, 70, 70, 70}; /* {(y_axis_length)} */
	int 	data_info_t2hk[4] = { 8, 12, 8, 12};
	int 	data_info_all[8] = { 70, 70, 70, 70, 8, 12, 8, 12};
	
	switch (exp) {
		case 0:
			for ( pos = 0; pos < rule; pos ++){
				sum_y += data_info_dune[pos];
			}
			
			for (i=ew_low; i <= ew_high; i++) {
				fit_rate = signal_fit_rate[i] + bg_fit_rate[i];
				chi2 += poisson_likelihood( *( (double*) user_data + sum_y + i), fit_rate);
			}
			break;
		case 1:
			for ( pos = 0; pos < rule; pos ++){
				sum_y += data_info_t2hk[pos];
			}
			
			for (i=ew_low; i <= ew_high; i++) {
				fit_rate = signal_fit_rate[i] + bg_fit_rate[i];
				chi2 += poisson_likelihood( *( (double*) user_data + sum_y + i), fit_rate);
			}
			break;
	}
    return chi2;
}


/* 定義 Test Statistic (Delta Chi-Square) */
//參數:{CP : [0, 180, 1], MO : [1, -1] , CPV Hypothesis的deltacp}  
// 0 : CPC at deltacp=0; 180 : CPC at deltacp=180; 1 : CPV at deltacp
//選定實驗EXP:[0,1,GLB_ALL]

double delta_chi2 (int CP, int MO, double deltacp, int EXP) {
	double  a = 0,
			b = 0; 
	double  chi_0_NO, 
			chi_0_IO, 
			chi_pi_NO, 
			chi_pi_IO, 
			chi_cpv_NO, 
			chi_cpv_IO;

	int data_info_dune[6] = {4, 280, 70, 70, 70, 70}; /* {(y_axis_length)} */
	int data_info_t2hk[6] = {4, 40, 8, 12, 8, 12};
	int data_info_all[10] = {8, 320, 70, 70, 70, 70, 8, 12, 8, 12};
	double *darray;
	int sum_y  = 0,
		ladder = 0,
		index_sum, 
		i_idx, j_idx;
	int LENGTH = EXP != -1 ? 4 : 8;
	int dset_y_info[LENGTH];

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

	switch (EXP){
		case 0:
			for ( index_sum = 2; index_sum < 6; index_sum++ ){
				sum_y += data_info_dune[index_sum];
			}
			darray = malloc( data_info_dune[0] * sum_y * sizeof(double));
			break;
		case 1:
			for ( index_sum = 2; index_sum < 6; index_sum++ ){
				sum_y += data_info_t2hk[index_sum];
			}
			darray = malloc( data_info_t2hk[0] * sum_y * sizeof(double));
			break;
		case -1:
			for ( index_sum = 2; index_sum < 10; index_sum++ ){
				sum_y += data_info_all[index_sum];
			}
			darray = malloc( data_info_all[0] * sum_y * sizeof(double));
			break;
	}

	//根據CP、MO的假設，生成Poisson Sample，計算其test statistic
	/* 定義glb_params */
	glb_params test_values_cpc_0_NO  = glbAllocParams(); 
	glb_params test_values_cpc_0_IO  = glbAllocParams(); 
	glb_params test_values_cpc_pi_NO = glbAllocParams(); 
	glb_params test_values_cpc_pi_IO = glbAllocParams(); 
	glb_params test_values_cpv_NO    = glbAllocParams(); 
	glb_params test_values_cpv_IO    = glbAllocParams();     
	glb_params input_errors = glbAllocParams(); 
	glb_params minimum = glbAllocParams(); //////////

	/* 定義test_values_cpc_0_NO */ 
	glbDefineParams(test_values_cpc_0_NO, theta12_N*degree, theta13_N*degree, theta23_N*degree, 0*degree, 1e-5*sdm_N, 1e-3*ldm_N);
	glbSetDensityParams(test_values_cpc_0_NO,1.0,GLB_ALL);

	/* 定義test_values_cpc_0_IO */ 
	glbDefineParams(test_values_cpc_0_IO, theta12_I*degree, theta13_I*degree, theta23_I*degree, 0*degree, 1e-5*sdm_I, 1e-3*ldm_I);
	glbSetDensityParams(test_values_cpc_0_IO,1.0,GLB_ALL);

	/* 定義test_values_cpc_pi_NO */ 
	glbDefineParams(test_values_cpc_pi_NO, theta12_N*degree, theta13_N*degree, theta23_N*degree, 180*degree, 1e-5*sdm_N, 1e-3*ldm_N);
	glbSetDensityParams(test_values_cpc_pi_NO,1.0,GLB_ALL);      

	/* 定義test_values_cpc_pi_IO */ 
	glbDefineParams(test_values_cpc_pi_IO, theta12_I*degree, theta13_I*degree, theta23_I*degree, 180*degree, 1e-5*sdm_I, 1e-3*ldm_I);
	glbSetDensityParams(test_values_cpc_pi_IO,1.0,GLB_ALL);

	/* 定義test_values_cpv_NO */                                                         //deltacp為Input的值
	glbDefineParams(test_values_cpv_NO, theta12_N*degree, theta13_N*degree, theta23_N*degree, deltacp*degree, 1e-5*sdm_N, 1e-3*ldm_N);
	glbSetDensityParams(test_values_cpv_NO,1.0,GLB_ALL);

	/* 定義test_values_cpv_IO */                                                         //deltacp為Input的值
	glbDefineParams(test_values_cpv_IO, theta12_I*degree, theta13_I*degree, theta23_I*degree, deltacp*degree, 1e-5*sdm_I, 1e-3*ldm_I);
	glbSetDensityParams(test_values_cpv_IO,1.0,GLB_ALL);

	/* 設定Projection */   
	glb_projection projection_cp_fixed = glbAllocProjection();
	glb_projection projection_cp_free  = glbAllocProjection();

	//GLB_FIXED/GLB_FREE                      theta12    theta13  theta23    deltacp     m21        m31
	glbDefineProjection(projection_cp_fixed, GLB_FIXED, GLB_FREE, GLB_FREE, GLB_FIXED, GLB_FIXED, GLB_FREE);//deltacp theta12 m21 不動，其他可變
	glbSetDensityProjectionFlag(projection_cp_fixed,GLB_FIXED,GLB_ALL);//matter density不變

	//GLB_FIXED/GLB_FREE                      theta12    theta13  theta23    deltacp     m21        m31
	glbDefineProjection(projection_cp_free,  GLB_FIXED, GLB_FREE, GLB_FREE, GLB_FREE, GLB_FIXED, GLB_FREE);// theta12 m21 不動，其他可變
	glbSetDensityProjectionFlag(projection_cp_free,GLB_FIXED,GLB_ALL);//matter density不變


	/* 關閉系統誤差 */   
	glbSwitchSystematics(GLB_ALL,GLB_ALL,GLB_OFF);

	/* 設定Input_errors */  
	glbDefineParams(input_errors,0,0,0,0,0,0);
	glbSetDensityParams(input_errors,0,GLB_ALL);
	glbSetInputErrors(input_errors);
    
	/* 根據MO,CP的假設，生成 Poisson Spectrum */ 
	switch (MO) {
		case 1:	
	  		if (CP == 0){ //CPC at deltacp=0
				printf("生成deltacp = 0, NO 的Poisson Spectrum \n");
				/* 根據CPC_0_NO的假設，生成Poisson True Spectrum */   
				do_poisson_fluctuation(test_values_cpc_0_NO, EXP, darray, dset_y_info);
			}
	  		if (CP == 180){ //CPC at deltacp=180
				printf("生成deltacp = 180, NO 的Poisson Spectrum \n");
				/* 根據CPC_0_IO的假設，生成Poisson True Spectrum */   
		  		do_poisson_fluctuation(test_values_cpc_pi_NO, EXP, darray, dset_y_info);
			}
	  		if (CP == 1){ //CPV
				printf("生成deltacp = input value, NO 的Poisson Spectrum \n");
				/* 根據CPV_NO的假設，生成Poisson True Spectrum */   
				do_poisson_fluctuation(test_values_cpv_NO, EXP, darray, dset_y_info);
			}
			break;
		case -1:	
	  		if (CP == 0){ //CPC at deltacp=0
				printf("生成deltacp = 0, IO 的Poisson Spectrum \n");
				/* 根據CPC_0_NO的假設，生成Poisson True Spectrum */   
		  		do_poisson_fluctuation(test_values_cpc_0_IO, EXP, darray, dset_y_info);
			}
	  		if (CP == 180){ //CPC at deltacp=180
				printf("生成deltacp = 180, IO 的Poisson Spectrum \n");
				/* 根據CPC_0_IO的假設，生成Poisson True Spectrum */   
		  		do_poisson_fluctuation(test_values_cpc_pi_IO, EXP, darray, dset_y_info);
			}
	  		if (CP == 1){ //CPV
				printf("生成deltacp = input value, IO 的Poisson Spectrum \n");
				/* 根據CPV_IO的假設，生成Poisson True Spectrum */   
		  		do_poisson_fluctuation(test_values_cpv_IO, EXP, darray, dset_y_info);
			}
			break;
	}
 
	/* 計算CPC Hypothesis (4種情況)*/

	/* 設定Prior (3 sigma range, Normal Ordering)*/
	glbRegisterPriorFunction(prior_3sigma_NO,NULL,NULL,NULL);

	/* 計算Chi square under cpc_0_NO */ 
	glbSetProjection(projection_cp_fixed); //設定Projection deltacp_Fixed
	glbSetOscillationParameters(test_values_cpc_0_NO);
	glbSetRates();
	glbDefineChiFunction(&chi2_poisson, 0, "chi2_poisson", darray);
	glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
	glbSetCentralValues(test_values_cpc_0_NO); 
	chi_0_NO = glbChiNP(test_values_cpc_0_NO, minimum ,EXP);

	/* 計算Chi square under cpc_pi_NO */ 
	glbSetProjection(projection_cp_fixed); //設定Projection deltacp_Fixed
	glbSetOscillationParameters(test_values_cpc_pi_NO);
	glbSetRates();
	glbDefineChiFunction(&chi2_poisson, 0, "chi2_poisson", darray);
	glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
	glbSetCentralValues(test_values_cpc_pi_NO); 
	chi_pi_NO = glbChiNP(test_values_cpc_pi_NO, minimum ,EXP);

	/* 設定Prior (3 sigma range, Inverse Ordering)*/
	glbRegisterPriorFunction(prior_3sigma_IO,NULL,NULL,NULL);

	/* 計算Chi square under cpc_0_IO */ 
	glbSetProjection(projection_cp_fixed); //設定Projection deltacp_Fixed
	glbSetOscillationParameters(test_values_cpc_0_IO);
	glbSetRates();
	glbDefineChiFunction(&chi2_poisson, 0, "chi2_poisson", darray);
	glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
	glbSetCentralValues(test_values_cpc_0_IO); 
	chi_0_IO = glbChiNP(test_values_cpc_0_IO, minimum ,EXP);

	/* 計算Chi square under cpc_pi_IO */ 
	glbSetProjection(projection_cp_fixed); //設定Projection deltacp_Fixed
	glbSetOscillationParameters(test_values_cpc_pi_IO);
	glbSetRates();
	glbDefineChiFunction(&chi2_poisson, 0, "chi2_poisson", darray);
	glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
	glbSetCentralValues(test_values_cpc_pi_IO); 
	chi_pi_IO = glbChiNP(test_values_cpc_pi_IO, minimum ,EXP);

	/* 取 chi_0_NO , chi_pi_NO, chi_0_IO , chi_pi_IO  四者之最小值 */   
	a = min(4, chi_0_NO, chi_pi_NO, chi_0_IO, chi_pi_IO);
          
	/* 計算CPV Hypothesis (2種情況)*/      

	/* 設定Prior (3 sigma range, Normal Ordering)*/
	glbRegisterPriorFunction(prior_3sigma_NO, NULL, NULL, NULL);

	/* 設定Projection (deltacp_Free)*/  
	glbSetProjection(projection_cp_free);

	/* 計算Chi square under cpv_NO */ 
	glbSetOscillationParameters(test_values_cpv_NO);
	glbSetRates();
	glbDefineChiFunction(&chi2_poisson, 0, "chi2_poisson", darray);
	glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
	glbSetCentralValues(test_values_cpv_NO); 
	chi_cpv_NO = glbChiNP(test_values_cpv_NO, minimum, EXP);
		                                                    

	/* 設定Prior (3 sigma range, Inverse Ordering)*/
	glbRegisterPriorFunction(prior_3sigma_IO, NULL, NULL, NULL);

	/* 設定Projection (deltacp_Free)*/  
	glbSetProjection(projection_cp_free);
    
	/* 計算Chi square under cpv_IO */ 
	glbSetOscillationParameters(test_values_cpv_IO);
	glbSetRates();
	glbDefineChiFunction(&chi2_poisson, 0, "chi2_poisson", darray);
	glbSetChiFunction(GLB_ALL, GLB_ALL, GLB_OFF, "chi2_poisson", NULL);
	glbSetCentralValues(test_values_cpv_IO); 
	chi_cpv_IO = glbChiNP(test_values_cpv_IO, minimum, EXP);

	/* 取 chi_cpv_NO , chi_cpv_IO  兩者之最小值 */   
	b = min(2, chi_cpv_NO, chi_cpv_IO);

	/* 輸出Delta Chi square */ 
	printf("a = %g, b = %g  \n",a,b);
	printf("a - b = %g  \n",a-b);
	free(darray);
	return a-b;
}


int main(int argc, char *argv[]) { 

	char   filename[32];
	strcpy(filename, argv[1]);
	int    TOTALsample = atof(argv[2]);
	double angle       = atof(argv[3]);
	int    expr        = atof(argv[4]);

	/*
     * Declare the variables for hdf5.
     */
	hid_t		file_id, 
				space_id, 
				dset_id,
				group_id,
				sub_group_id;
	herr_t		status;
	hsize_t		dims_q0[2] = {TOTALsample, 4}, 
				dims_q1[1] = {TOTALsample};
	

	glbInit(argv[0]);
	glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
	glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);

	printf("File name: %s\n",filename);

	double 	q0,  q1;
	double 	Q0[TOTALsample][4], 
			Q1[TOTALsample];
	for (int num_simulatiom = 0; num_simulatiom < TOTALsample; num_simulatiom++) { 
		q0 = delta_chi2(0 , 1 , angle, expr); 
		Q0[num_simulatiom][0] = q0;

		q0 = delta_chi2(180, 1 , angle, expr);
		Q0[num_simulatiom][1] = q0; 

		q0 = delta_chi2(0, -1 , angle, expr);
		Q0[num_simulatiom][2] = q0;

		q0 = delta_chi2(180, -1 , angle, expr);
		Q0[num_simulatiom][3] = q0;

		q1 = delta_chi2(1 , 1, angle, expr);
		Q1[num_simulatiom] = q1;
	}

	file_id   	= 	H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	space_id  	=  	H5Screate_simple(2, dims_q0, NULL);
	dset_id    	=  	H5Dcreate(file_id, "q0", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status     	=  	H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Q0);
	status    	=  	H5Dclose(dset_id);
	status    	=  	H5Sclose(space_id);
	
	space_id  	=  	H5Screate_simple(1, dims_q1, NULL);
	dset_id    	=  	H5Dcreate(file_id, "q1", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status     	=  	H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Q1);
	status    	=  	H5Dclose(dset_id);
	status    	=  	H5Sclose(space_id);
	status     	=  	H5Fclose(file_id);
	return 0;  
}

