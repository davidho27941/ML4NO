#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TH2.h>
#include <TH1.h>

void mig_matrix()
{
  ofstream outputFile;
  outputFile.open("enumu_reco_NC.dat");

  TFile *f = new TFile("enumu_reco_NC.root");
  f->ls();
  TH2F * enu_reco = (TH2F*)f->Get("enu_reco");

  Int_t binx,biny,max_binx,max_biny;
  const int max_binx = enu_reco->GetNbinsX();
  const int max_biny = enu_reco->GetNbinsY();
  Double_t norm_fact[26]; 
  Double_t matrix[26][26];

  for(binx=1;binx<max_binx+1;binx++)
    {
      norm_fact[binx]=0;
      for(biny=1;biny<max_biny+1;biny++)
	{
	  norm_fact[binx] = norm_fact[binx] + enu_reco->GetBinContent(binx,biny);
	}
      cout << norm_fact[binx] << endl;
    };
 
  for(binx=1;binx<max_binx+1;binx++)
    {
      for(biny=1;biny<max_biny+1;biny++)
	{
	  if(norm_fact[binx] !=0)
	    {
	      matrix[binx-1][biny-1]=(enu_reco->GetBinContent(binx,biny))/norm_fact[binx];
	    }
	  else
	    {
	      matrix[binx-1][biny-1] = 0;
	    }
	}
    };
  
  for(biny=0;biny<max_biny;biny++)
    {    
      outputFile <<"{0, 26 ";
      for(binx=0;binx<max_binx;binx++)
	{
	  outputFile << ", " << matrix[binx][biny];
	}
      outputFile << "}:" << endl;
    }
  
}


