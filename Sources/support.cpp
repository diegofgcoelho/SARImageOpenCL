/*
 * support.h
 *
 *  Created on: Oct 6, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  This file contains some supporting function for measuring time.
 */

#include "support.h"

timespec diff_time(timespec init, timespec end){
	/*
	 * Input:
	 * init and end are timespec structures that represents the starting and ending time of the measure
	 * that we are interested in computing the difference
	 * Output:
	 * out_time is a timespec structure representing the time elapsed from the sart to end
	 * Description:
	 * this function computes the difference between start and end time, which means the time elapsed between the beginning
	 * of the measurement and the end
	 */

	//Output variable
	timespec out_time;

	if((end.tv_nsec-init.tv_nsec) < 0){
		out_time.tv_sec = end.tv_sec-init.tv_sec-1;
		out_time.tv_nsec = 1e9 + end.tv_nsec - init.tv_nsec;
	} else {
		out_time.tv_sec = end.tv_sec - init.tv_sec;
		out_time.tv_nsec = end.tv_nsec - init.tv_nsec;
	}

	return out_time;
}

double get_millisecs(timespec time){
	/*
	 * Input:
	 * time is a timespec structure
	 * Output:
	 * out_time is a double representing the time in milliseconds
	 * Description:
	 * this function converts the time in the timespec structure time to milliseconds measures
	 */

	double out_time = static_cast<double>(time.tv_sec)*1000+static_cast<double>(time.tv_nsec)/1000000;
	return out_time;
}

void read_r_data(std::string filename, mcmatrix** matrices, long unsigned int * n_matrices){
	/*
	 * Input:
	 * filename representing the rdata file containing the matrices
	 * matrices, which is a pointer to the array of mcmatrix objects
	 * n_matrices, which represents the number of matrices that were loaded from the rdata file
	 * Output:
	 * matrices and n_matrices are also output arguments
	 * Description:
	 * This function opens the file named filename, which is assumed to be a rdata file and read the
	 * Hermitian matrices (from PolSAR imagery). Those matrices are returned in the array pointed by matrices
	 * and the total number of matrices is stored in n_matrices. The memory for the array matrices is allocated inside
	 * this function. The calling function must deallocate the memory. In this particular function, we assume that the variable
	 * holding the data to be read is named 'ImgMP'. It can be made variable, but we are not interested on it for right now.
	 */

	//Each of these matrices represents a band in the rdata format
	//Rcpp::NumericMatrix X1, X2, X3, X4r, X4i, X5r, X5i, X6r, X6i;


	const int argc = 1;
	const char* argv[1];
	argv[1] = "tempRInstance";
	RInside Rinstance(argc, argv);
	std::string Rcommands = "cat('Loading the data file.'); load('"+filename+"'); "
			"X1 <- Re(ImgMP[,,1]); X2 <- Re(ImgMP[,,2]); X3 <- Re(ImgMP[,,3]);"
			"X4r <- Re(ImgMP[,,4]); X4i <- Im(ImgMP[,,4]); X5r <- Re(ImgMP[,,5]); X5i <- Im(ImgMP[,,5]);"
			"X6r <- Re(ImgMP[,,6]); X6i <- Im(ImgMP[,,6]);"
			"cat(' File successfully read.');";

	//Execute R commands
	Rinstance.parseEval(Rcommands);

	Rcommands = "X1";
	//Execute R commands and get results
	Rcpp::NumericMatrix X1 = Rinstance.parseEval(Rcommands);

	Rcommands = "X2";
	//Execute R commands and get results
	Rcpp::NumericMatrix X2 = Rinstance.parseEval(Rcommands);

	Rcommands = "X3";
	//Execute R commands and get results
	Rcpp::NumericMatrix X3 = Rinstance.parseEval(Rcommands);

	Rcommands = "X4r";
	//Execute R commands and get results
	Rcpp::NumericMatrix X4r = Rinstance.parseEval(Rcommands);
	Rcommands = "X4i";
	//Execute R commands and get results
	Rcpp::NumericMatrix X4i = Rinstance.parseEval(Rcommands);

	Rcommands = "X5r";
	//Execute R commands and get results
	Rcpp::NumericMatrix X5r = Rinstance.parseEval(Rcommands);
	Rcommands = "X5i";
	//Execute R commands and get results
	Rcpp::NumericMatrix X5i = Rinstance.parseEval(Rcommands);


	Rcommands = "X6r";
	//Execute R commands and get results
	Rcpp::NumericMatrix X6r = Rinstance.parseEval(Rcommands);
	Rcommands = "X6i";
	//Execute R commands and get results
	Rcpp::NumericMatrix X6i = Rinstance.parseEval(Rcommands);

	//The matrices will be organized in a row-wise manner
	const unsigned int nncols = X1.ncol();
	const unsigned int nnrows = X1.nrow();
	//Allocate memory for the mcmatrix array
	*matrices = new mcmatrix[nncols*nnrows];
	if(*matrices == NULL){
		std::cout << "Bad allocation. Terminating the program." << std::endl;
		exit(-1);
	} else{
		std::cout << "Array of matrix created. nrow = " << nnrows << " and nncol = " << nncols << std::endl;
	}
	//Filling matrices
	for(unsigned int i = 0; i < nnrows; i++){
		for(unsigned int j = 0; j < nncols; j++){
			//std::cout << "X1("<<i<<","<<j<<") = " << X1(i,j) << std::endl;
			//(*matrices)[i].a.a = 0;
			(*matrices)[i*nncols+j].a.a = X1(i,j); (*matrices)[i*nncols+j].a.b = 0.0;
			(*matrices)[i*nncols+j].d.a = X2(i,j); (*matrices)[i*nncols+j].d.b = 0.0;
			(*matrices)[i*nncols+j].f.a = X3(i,j); (*matrices)[i*nncols+j].f.b = 0.0;
			(*matrices)[i*nncols+j].b.a = X4r(i,j); (*matrices)[i*nncols+j].b.b = X4i(i,j);
			(*matrices)[i*nncols+j].c.a = X5r(i,j); (*matrices)[i*nncols+j].c.b = X5i(i,j);
			(*matrices)[i*nncols+j].e.a = X6r(i,j); (*matrices)[i*nncols+j].e.b = X6i(i,j);
		}
	}

	std::cout << "Matrix assignment complete" << std::endl;

	//Passing the number of matrices
	*n_matrices = static_cast<long unsigned int>(nnrows*nncols);

}
