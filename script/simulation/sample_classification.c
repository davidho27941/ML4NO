/* 生成sample for Regression */
/* theta12, m21 fixed */
/* theta23, theta13, m31 flat distribution in 3 sigma range */
/* deltacp flat distribution in 0~360 */

/* 使用方式(產100K組) :  ./sample_regression <filename.h5> <num> */
/* Caution: `num` should not greater than 1000. */

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

#define degree M_PI/180

/***************************************************************************
 *                            M A I N   P R O G R A M                      *
 ***************************************************************************/

float keithRandom() {
	// Random number function based on the GNU Scientific Library
	// Returns a random float between 0 and 1, exclusive; e.g., (0,1)
	const 	gsl_rng_type * T;
	gsl_rng * r;
	gsl_rng_env_setup();
	struct 	timeval tv; // Seed generation based on time
	gettimeofday(&tv,0);
	unsigned long mySeed = tv.tv_sec + tv.tv_usec;
	T = gsl_rng_default; // Generator setup
	r = gsl_rng_alloc (T);
	gsl_rng_set(r, mySeed);
	double u = gsl_rng_uniform(r); // Generate it!
	gsl_rng_free (r);
	return (float)u;
}


double randn (double mu, double sigma) {
	/*mu is the central value, sigma is the width of gaussian distribution. */
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;
    
	if (call == 1) {
		call = !call;
		return (mu + sigma * (double) X2);
	}
    
	do{
		U1 = -1 + ((double) rand () / RAND_MAX) * 2;
		U2 = -1 + ((double) rand () / RAND_MAX) * 2;
		W = pow (U1, 2) + pow (U2, 2);
	} while (W >= 1 || W == 0);
    
	mult = sqrt ((-2 * log (W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double) X1);
}

double TCRandom(double mu, double sigma) {
	// Random number function based on the GNU Scientific Library
	// Returns a random float between 0 and 1, exclusive; e.g., (0,1)
	const 	gsl_rng_type * T;
	gsl_rng * r;
	gsl_rng_env_setup();
	struct 	timeval tv; // Seed generation based on time
	gettimeofday(&tv,0);
	unsigned long mySeed = tv.tv_sec + tv.tv_usec;
	T = gsl_rng_default; // Generator setup
	r = gsl_rng_alloc (T);
	gsl_rng_set(r, mySeed);
	double u = mu + gsl_ran_gaussian(r, sigma); // Generate it!
	gsl_rng_free (r);

	return u;
}

int case_judger (int mode, double value) {
	/*
	 * mode 1: Octant;
	 * mode 2: CPV;
	 * mode 3: MO;
	 */
	int result;
	switch (mode){
		case 1:
			if ((value == 0) || (value==180)){
				result = 0;
			} else {
				result = 1;
			}
		break;
	case 2:
		if (value > 45) {
		    result = 1;
		} else if (value == 45) {
		    result = 0;
		} else {
		    result = -1;
		}
		break;
	case 3:
		result = value > 0 ? 1 : -1;
		break;
	}
	return result;
}

int main(int argc, char *argv[]) { 

	/* Initialize GLoBES and define chi^2 functions */
	glbInit(argv[0]);                

	/*
	 * Read the prefix of file name and add suffix to two child string.
	 */
	char 	filename[32],
	 	 	Group_name[32],
		 	channel_info_dset_name[] = "/Spectrum/expr_0/channel_0\0",
		 	size_info_dset_name[] = "/Spectrum/expr_0/channel_0/Bin_ize\0",
		 	energy_info_dset_name[] = "/Spectrum/expr_0/channel_0/Bin_energy\0",
		 	expr_info_dset_name[] = "/Spectrum/expr_0\0";
	strcpy(filename, argv[1]);

	int  	TOTALsample = atof(argv[2]);

    /*
     * Declare the variables for hdf5.
     */
    hid_t	file_id, 
			space_id, 
			dset_id,
			group_id,
			sub_group_id,
			sub2_group_id,
			sub3_group_id;
    herr_t 	status;
    hsize_t	dims[2];

	/*
	 * Initialize experiment parameters.
	 */
	glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
	glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);
	glb_params true_values = glbAllocParams();

	/* Set standard oscillation parameters (cf. hep-ph/0405172v5) */
	double theta12_c = 33.44; 
	// double theta13_c = 8.57;
	// double theta23_c = 45;
	double sdm_c = 7.42;
	// double ldm_c = 2.514;


	double delta_12  = (35.86 - 31.27)/2;
	double delta_13  = (8.97 - 8.20)/2;
	double delta_23  = (51.80 - 39.60)/2;
	double delta_sdm = (8.04 - 6.82)/2;
	double delta_ldm = (2.598 - 2.431)/2;

	int    num_simulatiom;

	int    ew_low, ew_high;
	int    i, j, num_expriment, channel;
    
	file_id   = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	group_id = H5Gcreate(file_id, "/Spectrum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	for ( num_expriment = 0;  num_expriment < 2;  num_expriment++){
		double *bin_c_energy  =   glbGetBinCentersListPtr( num_expriment );
		double *bin_size      =   glbGetBinSizeListPtr( num_expriment );
		int    channel_max    =   glbGetNumberOfRules( num_expriment );
		
		expr_info_dset_name[15] = num_expriment+'0';
		sub_group_id = H5Gcreate(file_id, expr_info_dset_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		for(channel=0;channel<channel_max;channel++){
			glbGetEnergyWindowBins( num_expriment, channel, &ew_low, &ew_high);
			int energy_window = ew_high - ew_low;
			hsize_t dims_s[1] = {energy_window};

			double dset_channel_eng[energy_window];
			double dset_channel_size[energy_window];

			j = 0;
			for ( i = ew_low; i <= ew_high; i++) { 
				dset_channel_eng[j]  = bin_c_energy[i];
				dset_channel_size[j] = bin_size[i];
				j += 1;
			}
		
			channel_info_dset_name[15]  =  num_expriment+'0';
			channel_info_dset_name[25]  =  channel+'0';
			energy_info_dset_name[15]   =  num_expriment+'0';
			energy_info_dset_name[25]   =  channel+'0';
	 	    size_info_dset_name[15]   =  num_expriment+'0';
			size_info_dset_name[25]   =  channel+'0';

			sub2_group_id =  H5Gcreate(file_id, channel_info_dset_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			space_id  =  H5Screate_simple(1, dims_s, NULL);

			dset_id       =  H5Dcreate(sub2_group_id, energy_info_dset_name, H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT,
						            H5P_DEFAULT);
			status        =  H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_channel_eng);
			status        =  H5Dclose(dset_id);
			dset_id       =  H5Dcreate(sub2_group_id, size_info_dset_name, H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT,
						            H5P_DEFAULT);
			status        =  H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_channel_size);
			status        =  H5Dclose(dset_id);
			status        =  H5Sclose(space_id);
			status        =  H5Gclose(sub2_group_id);
		}
		status        =  H5Gclose(sub_group_id);
	}
	status        =  H5Gclose(group_id);
	 
	double     dataset[TOTALsample][9], 
			   TRUE_RATE[TOTALsample][2][4][8],
			   TRUE_RATE_PRIME[2][4][8][TOTALsample];
	for (num_simulatiom=0;num_simulatiom<TOTALsample;num_simulatiom++){

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

		double sign_ldm = 1;
		if ( keithRandom() < 0.5) sign_ldm = -1;
		ldm = sign_ldm * ldm;

		// if(keithRandom()<0.33) {theta23 = 45;} 

		double deltacp = keithRandom()*360.0*2.0-180.0;
		if (deltacp<0){
			deltacp=0;
		}	else if (deltacp>360){
			deltacp=180;
		}

		//double deltacp = keithRandom() * 360.0;

		int octant = case_judger(1, theta23);
		int cpv    = case_judger(2, deltacp);
		int mo     = case_judger(3, ldm);

		glbDefineParams(true_values,theta12_c*degree,theta13*degree,theta23*degree,deltacp*degree,1e-5*sdm_c,1e-3*ldm);
		glbSetDensityParams(true_values,1.0,GLB_ALL);

		glbSetOscillationParameters(true_values);
		glbSetRates();
	   
		num_expriment = 0;
		channel = 0; 

		for ( num_expriment = 0; num_expriment < 2; num_expriment++){
			int channel_max = glbGetNumberOfRules( num_expriment );		
			for ( channel = 0; channel < channel_max ; channel++){
				glbGetEnergyWindowBins( num_expriment, channel, &ew_low, &ew_high);
				double *true_rates = glbGetRuleRatePtr( num_expriment, channel );
				int count = 0;
				for (i=ew_low; i <= ew_high; i++){
					TRUE_RATE[num_simulatiom][num_expriment][channel][count] = true_rates[i];
					count += 1;
				}
			}
		}
	 	double tmp_array[9] = {theta12_c, theta13, theta23, deltacp, sdm_c, ldm, octant, cpv, mo};
		int  count = 0;
		for ( count = 0; count < 9; count++){
			dataset[num_simulatiom][count] = tmp_array[count];
		}
	}
	int k, l;
	for ( i = 0; i < TOTALsample; i++){
		for ( j = 0; j < 2; j++){
			for ( k = 0; k < 4; k++){
				for ( l = 0; l < 8; l++){
			   		TRUE_RATE_PRIME[j][k][l][i] = TRUE_RATE[i][j][k][l];
			   	}
			}
		}
	}

	char 	channel_spec_true_name[] = "/Parameter/true/expr_0/channel_0\0",
		 	expr_spec_true_name[]    = "/Parameter/true/expr_0\0", 
		 	expr_spec_sim_name[]     = "/Parameter/simulation\0";
    
	hsize_t dims_s[2]    =  {TOTALsample, 8};
	group_id     		 =  H5Gcreate(file_id, "/Parameter", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	sub_group_id 		 =  H5Gcreate(file_id, "/Parameter/true/", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	for ( i = 0; i < 2; i ++){
		expr_spec_true_name[21] = i+'0';
		sub2_group_id =  H5Gcreate(file_id, expr_spec_true_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		for ( j = 0; j < 4; j++){
			channel_spec_true_name[21] = i+'0';
			channel_spec_true_name[31] = j+'0';
			space_id      =  H5Screate_simple(2, dims_s, NULL);
			dset_id       =  H5Dcreate(sub2_group_id, channel_spec_true_name, H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT,
				                H5P_DEFAULT);
			status        =  H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, TRUE_RATE_PRIME[i][j]);
			status        =  H5Dclose(dset_id);
			status        =  H5Sclose(space_id);
		}
		status        =  H5Gclose(sub2_group_id);
	}
    status        =  H5Gclose(sub_group_id);
    
	hsize_t dims_d[2]    = 	{TOTALsample, 9};
	sub_group_id  		 =  H5Gcreate(file_id, "/Parameter/simulation/", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	space_id      		 =  H5Screate_simple(2, dims_d, NULL);
	dset_id       		 =  H5Dcreate(sub_group_id, "/Parameter/simulation/dataset", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT,
		                        H5P_DEFAULT);
	status        		 =  H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
	status        		 =  H5Dclose(dset_id);
	status        		 =  H5Sclose(space_id);
	status        		 =  H5Gclose(sub_group_id); 
	status        		 =  H5Gclose(group_id);

	/* Clean up */
	glbFreeParams(true_values);

	status        =  H5Fclose(file_id);
	return 0;  
}
