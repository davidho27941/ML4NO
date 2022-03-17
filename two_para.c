/*Delta chi^2 value againt epsilon_emu and epsilon_etau
 without the restriction of mutau reflection symmetry*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include<float.h>
#include<complex.h>
#include<gsl/gsl_complex.h>
#include<gsl/gsl_complex_math.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_blas.h>
#include <globes/globes.h>   /* GLoBES library */
#include  "snu.h"

double const degree= M_PI/180;


double my_prior(const glb_params in, void* user_data)//Mark
{
    glb_params central_values = glbAllocParams();
    glb_params input_errors = glbAllocParams();
    glb_projection p = glbAllocProjection();
    glbGetCentralValues(central_values);
    glbGetInputErrors(input_errors);
    glbGetProjection(p);
    int i;
    double pv = 0.0;
    double fitvalue,centralvalue,inputerror;

    /*
     theta12     0    FIXED
     theta13    1  
     theta23:  2
     deltacp   3
     sdm  4           FIXED
     ldm 5

     emu    25
     etau:  27
     mumu   29
     */

    /*impose the absolute value of off-diagonal terms to be positive */

if(glbGetProjectionFlag(p,25)==GLB_FREE){if(fabs(glbGetOscParams(in,25))>1) {glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p); return 1e10;}}
if(glbGetProjectionFlag(p,27)==GLB_FREE){if(fabs(glbGetOscParams(in,27))>1) {glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p); return 1e10;}}
if(glbGetProjectionFlag(p,29)==GLB_FREE){if(fabs(glbGetOscParams(in,30))>1) {glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p); return 1e10;}}


    //restriction on theta13 theta23
if(glbGetProjectionFlag(p,1)==GLB_FREE){if((glbGetOscParams(in,1)<8.2*degree)||(glbGetOscParams(in,1)>8.97*degree)){glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p); return 1e10;}}    
if(glbGetProjectionFlag(p,2)==GLB_FREE){if((glbGetOscParams(in,2)<38.9*degree)||(glbGetOscParams(in,2)>51.1*degree)){glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p);return 1e10;} }   
    //prior for DM31 for both hierarchy
if(glbGetProjectionFlag(p,5)==GLB_FREE){if((fabs(glbGetOscParams(in,5))<2.431e-3)||(fabs(glbGetOscParams(in,5))>2.598e-3)){glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p);return 1e10;}}    


 glbFreeParams(central_values); glbFreeParams(input_errors); glbFreeProjection(p);
    
    return 0;
}



typedef struct ParaNode{
  int key;
  char para_name[10];
  float lower_limit_true;
  float upper_limit_true;
  float step_true;
  float lower_limit_test;
  float upper_limit_test;
  float step_test;
}ParaNode;

ParaNode paras[7];

void SetParaNode( int i, int key, char para_name[], float lower_limit_true, float upper_limit_true, float step_true,
                     float lower_limit_test, float upper_limit_test, float step_test ) {
  paras[i].key = key;
  strcpy(paras[i].para_name,para_name);
  paras[i].lower_limit_true = lower_limit_true;
  paras[i].upper_limit_true = upper_limit_true;
  paras[i].step_true = step_true;
  paras[i].lower_limit_test = lower_limit_test;
  paras[i].upper_limit_test = upper_limit_test;
  paras[i].step_test = step_test;
}

void ParaInit() {
  SetParaNode( 0, 1, "theta13" , 8.2 *degree, 8.97 *degree, 0.0077 *degree*10, 0 , 0 , 0 );
  SetParaNode( 1, 2, "theta23", 35 *degree, 55.1 *degree,  0.2 *degree*10, 40 *degree, 50.1 *degree,  5 *degree);
  SetParaNode( 2, 3, "deltacp",   0 *degree , 360*degree,  3.6 *degree*10,   0 *degree , 360*degree, 90 *degree);
  SetParaNode( 3, 5, "ldm" ,   0,    0,    0, -2.514e-3 , 2.515e-3, 2.514e-3*2);
  SetParaNode( 4, 25, "emu" , -1, 1+ 1e-5, 0.02*10 , 0, 1, 1);
  SetParaNode( 5, 27, "etau", -1, 1+ 1e-5, 0.02*10 , 0, 1, 1);
  SetParaNode( 6, 29, "mumu", -1, 1+ 1e-5, 0.02*10 , 0, 1, 1);
} 

// void ParaInit() {
//   SetParaNode( 0, 1, "theta13" , 8.2 *degree, 8.97 *degree, 0.0077 *degree, 0 , 0 , 0 );
//   SetParaNode( 1, 2, "theta23", 35 *degree, 55.1 *degree,  0.2 *degree, 40 *degree, 50.1 *degree,  5 *degree);
//   SetParaNode( 2, 3, "deltacp",   0 *degree , 360*degree,  3.6 *degree,   0 *degree , 360*degree, 90 *degree);
//   SetParaNode( 3, 5, "ldm" ,   0,    0,    0, -2.514e-3 , 2.515e-3, 2.514e-3*2);
//   SetParaNode( 4, 25, "emu" , -1, 1+ 1e-5, 0.02 , 0, 1, 1);
//   SetParaNode( 5, 27, "etau", -1, 1+ 1e-5, 0.02 , 0, 1, 1);
//   SetParaNode( 6, 29, "mumu", -1, 1+ 1e-5, 0.02 , 0, 1, 1);
// } 

ParaNode findPara( int key ) {
  for ( int i = 0 ; i < 7 ; i++ ) {
    if ( paras[i].key == key )
      return paras[i];
  } // for
  
  printf( "輸入錯誤的key值\n" );
  return paras[0];
}

ParaNode findNotTwoPara( int key1, int key2, int testNum ) {
  for ( int i = 0 ; i < 7 ; i++ ) {
    if ( paras[i].key != key1 && paras[i].key != key2 ) {
      testNum--;
      if ( testNum == 0 )
        return paras[i];
    }
  } // for
  
  printf( "無剩餘的參數\n" );
  return paras[0];
}



int main(int argc, char *argv[])
{
  ParaInit();     
  glbInit(argv[0]);             

  //printf( "./output/chi_%s_%s.dat",findPara(atoi(argv[1])).para_name,findPara(atoi(argv[2])).para_name );
    int para1 = atoi(argv[1]);
    int para2 = atoi(argv[2]);
    int version = atof(argv[3]);

    char output[256];
    sprintf(output,"./output/ver%i_chi_%s_%s.dat",version ,findPara(para1).para_name,findPara(para2).para_name);
    FILE* OUT=fopen(output, "w");

    /* Initialize libglobes */
    glbInit(argv[0]);
    glbInitExperiment("./DUNE2021/DUNE_GLoBES.glb",&glb_experiment_list[0],&glb_num_of_exps);
//    glbInitExperiment("./HK_globes/HK_combined_coarse.glb",&glb_experiment_list[0],&glb_num_of_exps);
    
    snu_init_probability_engine_3();
    glbRegisterProbabilityEngine(6*9-3,      /* Number of parameters,   6*SQR(n_flavors) - n_flavors */
                                 &snu_probability_matrix,
                                 &snu_set_oscillation_parameters,
                                 &snu_get_oscillation_parameters,
                                 NULL);
    
    /* Define true oscillation parameters */ //Mark
    double theta12, theta13, theta23;
    double sdm       = 0.0;
    double ldm       = 0.0;
    double deltacp   = 0.0;;
    double ee        = 0.0;
    double mumu      = 0.0;
    double tautau    = 0.0;
    double emu       = 0.0;
    double etau      = 0.0;
    double mutau     = 0.0;
    double phi_emu   = 0.0*M_PI;
    double phi_etau = 0.0*M_PI;
    double phi_mutau = 0.0*M_PI;
    
    
        FILE* fp_chans_para = fopen("./parameters/parameters","r");
        int num_chans=0;    int ret;        
        if (fp_chans_para == NULL) {
            printf ("Cannot open file parameters/parameters \n");
            exit(0);
        }
        char chbuf[1000];
        num_chans=0;
        while (!feof(fp_chans_para)) {
            fgets(chbuf,1000,fp_chans_para);
            if (chbuf != NULL) {/*0:MC OFF*/
                ret = sscanf(chbuf,"%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",\
                &theta12,&theta13,&theta23,&deltacp,&sdm,&ldm,&ee,&mumu,&tautau,&emu,&etau,&mutau,&phi_emu,&phi_etau,&phi_mutau);
                num_chans++;
            }
        }
        fclose(fp_chans_para);
    
    printf("input values %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",\
    theta12,theta13,theta23,deltacp,sdm,ldm,ee,mumu,tautau,emu,etau,mutau,phi_emu,phi_etau,phi_mutau);

    fprintf(OUT,"input values %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",\
    theta12,theta13,theta23,deltacp,sdm,ldm,ee,mumu,tautau,emu,etau,mutau,phi_emu,phi_etau,phi_mutau);

    
    glb_params true_values = glbAllocParams();
    for(int i=0; i < 51; i++)
    {
        glbSetOscParams(true_values, 0.0, i);
    }
    glbDefineParams(true_values,theta12*degree,theta13*degree,theta23*degree,deltacp*degree,sdm*1e-5,ldm*1e-3);
    glbSetDensityParams(true_values,1.0,GLB_ALL);
    //############ NSI Parameter ############//
    /*
     ee     24
     emu    25
     etau:  27
     mumu   29
     mutau  30
     tautau 32
     */
    glbSetOscParams(true_values, ee,     24);  // eps_ee
    glbSetOscParams(true_values, emu,    25);  // eps_mue magnitude
    glbSetOscParams(true_values, phi_emu,  26);  // eps_mue phase
    glbSetOscParams(true_values, etau,   27);  // eps_etau
    glbSetOscParams(true_values, phi_etau,  28);  // eps_etau phase
    glbSetOscParams(true_values, mumu,   29);  // eps_mumu
    glbSetOscParams(true_values, mutau,  30);  // eps_mutau
    glbSetOscParams(true_values, phi_mutau,  31);  // eps_mutau phase
    glbSetOscParams(true_values, tautau, 32);  // eps_tautau
    glbSetOscillationParameters(true_values);
    glbSetRates();
    
    /* Define "test" oscillation parameter vector */
    glb_params test_values = glbAllocParams();

    
    /*initialise the prior*/
    glb_params input_errors = glbAllocParams();
    glb_params centers = glbAllocParams();
    for(int i=0; i < 51; i++)
    {
        glbSetOscParams(input_errors, 0.0, i);
        glbSetOscParams(centers, 0.0, i);
    }
    glbSetDensityParams(centers, 1.0, GLB_ALL);
    glbSetDensityParams(input_errors, 0.05, GLB_ALL);
    glbSetInputErrors(input_errors);
    glbSetCentralValues(centers);
    glbRegisterPriorFunction(my_prior,NULL,NULL,NULL);

    /*projection set up*/
    glb_projection projection = glbAllocProjection();
    glbSetDensityProjectionFlag(projection, GLB_FREE, GLB_ALL);
    for (int i=0; i<51; i++)  glbSetProjectionFlag(projection, GLB_FIXED,i);

    /*
     theta12     0    FIXED
     theta13    1  
     theta23:  2
     deltacp   3
     sdm  4           FIXED
     ldm 5


     emu    25
     etau:  27
     mumu   29
     */


    glbSetProjectionFlag(projection, GLB_FREE,1);
    glbSetProjectionFlag(projection, GLB_FREE,2);
    glbSetProjectionFlag(projection, GLB_FREE,3);
    glbSetProjectionFlag(projection, GLB_FREE,5);

    glbSetProjectionFlag(projection, GLB_FREE,25); 
    glbSetProjectionFlag(projection, GLB_FREE,27);
    glbSetProjectionFlag(projection, GLB_FREE,29);  

    glbSetProjectionFlag(projection, GLB_FIXED,para1);
    glbSetProjectionFlag(projection, GLB_FIXED,para2);
    glbSetProjection(projection);
    
    double x_test,y_test;

        glbCopyParams(true_values,test_values);


void Many_test( int para1, int para2, double *min ) {
  double res;
  if ( para1 == para2 ) {
    printf( "Error : para1 == para2" );
  }
  else if ( para1 == 1 || para2 == 1 ) {
    ParaNode test1 = findNotTwoPara(para1,para2,1);
    ParaNode test2 = findNotTwoPara(para1,para2,2);
    ParaNode test3 = findNotTwoPara(para1,para2,3);
    ParaNode test4 = findNotTwoPara(para1,para2,4);
    ParaNode test5 = findNotTwoPara(para1,para2,5);
    for ( double d1 = test1.lower_limit_test ; d1 < test1.upper_limit_test ; d1 += test1.step_test ){
      glbSetOscParams(test_values, d1, test1.key);
      for ( double d2 = test2.lower_limit_test ; d2 < test2.upper_limit_test ; d2+= test2.step_test ){
        glbSetOscParams(test_values, d2, test2.key);
        for ( double d3 = test3.lower_limit_test ; d3 < test3.upper_limit_test ; d3+= test3.step_test ) {
          glbSetOscParams(test_values, d3, test3.key);
          for ( double d4 = test4.lower_limit_test ; d4 < test4.upper_limit_test ; d4+= test4.step_test ) {
            glbSetOscParams(test_values, d4, test4.key);
            for ( double d5 = test5.lower_limit_test ; d5 < test5.upper_limit_test ; d5+= test5.step_test ) {
              glbSetOscParams(test_values, d5, test5.key);
              res=glbChiNP(test_values,NULL,GLB_ALL);
              if(res<*min)
                *min=res;
            }
          }
        }
      }
    }
  }
else {
    ParaNode test1 = findNotTwoPara(para1,para2,2);
    ParaNode test2 = findNotTwoPara(para1,para2,3);
    ParaNode test3 = findNotTwoPara(para1,para2,4);
    ParaNode test4 = findNotTwoPara(para1,para2,5);
    for ( double d1 = test1.lower_limit_test ; d1 < test1.upper_limit_test ; d1 += test1.step_test ){
      glbSetOscParams(test_values, d1, test1.key);
      for ( double d2 = test2.lower_limit_test ; d2 < test2.upper_limit_test ; d2+= test2.step_test ){
        glbSetOscParams(test_values, d2, test2.key);
        for ( double d3 = test3.lower_limit_test ; d3 < test3.upper_limit_test ; d3+= test3.step_test ) {
          glbSetOscParams(test_values, d3, test3.key);
          for ( double d4 = test4.lower_limit_test ; d4 < test4.upper_limit_test ; d4+= test4.step_test ) {
            glbSetOscParams(test_values, d4, test4.key);
            res=glbChiNP(test_values,NULL,GLB_ALL);
            if(res<*min)
              *min=res;
          }
        }
      }
    }
  }
}

     for(x_test = findPara(para1).lower_limit_true; x_test <= findPara(para1).upper_limit_true; x_test += findPara(para1).step_true){
        glbSetOscParams(test_values, x_test, para1);                        
     for(y_test = findPara(para2).lower_limit_true; y_test <= findPara(para2).upper_limit_true; y_test += findPara(para2).step_true){
        glbSetOscParams(test_values, y_test, para2);
    
        double min=1e20;
        Many_test(para1, para2, &min);

        fprintf(stdout, "%g %g %g\n",x_test,y_test, min);
        fprintf(OUT, "%g %g %g\n",x_test,y_test, min);
      
      }  }

    
    fclose(OUT);
    
    /* Destroy parameter and projection vector(s) */
    glbFreeParams(true_values);
    glbFreeParams(test_values);
    glbFreeParams(input_errors);
    
    return 0;
}

/*
nohup ./two_para 1 2 2 > 0313.log &
nohup ./two_para 1 3 2 > 0313.log &
nohup ./two_para 1 25 2 > 0313.log &
nohup ./two_para 1 27 2 > 0313.log &
nohup ./two_para 1 29 2 > 0313.log &
nohup ./two_para 2 3 2 > 0313.log &
nohup ./two_para 2 25 2 > 0313.log &
nohup ./two_para 2 27 2 > 0313.log &
nohup ./two_para 2 29 2 > 0313.log &
nohup ./two_para 3 25 2 > 0313.log &
nohup ./two_para 3 27 2 > 0313.log &
nohup ./two_para 3 29 2 > 0313.log &
nohup ./two_para 25 27 2 > 0313.log &
nohup ./two_para 25 29 2 > 0313.log &
nohup ./two_para 27 29 2 > 0313.log &
*/