/*
 * support.h
 *
 *  Created on: Oct 6, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  This file contains some supporting function for measuring time.
 */

#ifndef SUPPORT_H_
#define SUPPORT_H_

#include <ctime>
#include "../OpenCL/numerical_t.h"
#include <Rcpp/include/Rcpp.h>
#include <RInside/include/RInside.h>

timespec diff_time(timespec init, timespec end);
double get_millisecs(timespec time);
void read_r_data(std::string filename, mcmatrix** matrices, long unsigned int * n_matrices);

#endif /* SUPPORT_H_ */
