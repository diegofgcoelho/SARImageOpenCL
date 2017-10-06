/*
 * numerical_t.h
 *
 *  Created on: Sep 28, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  	This file implements some useful functions described in numeric_t.h.
 *
 *  	Because OpenCL is follows C99, everything in this file follows this standard.
 */

#ifndef NUMERICAL_S_H_
#define NUMERICAL_S_H_

#include "../OpenCL/numerical_t.h"

//mreal functions
mreal mrrand();

//mcomplex functions
mcomplex mcrand();

void printmc(mcomplex a);

void setmcm(mcomplex* to, mcomplex* from);

void setmcr(mcomplex* to, mreal* from);

#endif /* NUMERICAL_S_H_ */
