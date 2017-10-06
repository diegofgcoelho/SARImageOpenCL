/*
 * numerical_t.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  	This file implements some useful functions described in numeric_t.h.
 *
 */

#include "numerical_s.h"
#include "../OpenCL/numerical_t.h"
#include <cstdlib>
#include <cstdio>

//mreal functions
mreal mrrand(){
	return static_cast<double>(rand())/RAND_MAX;
}

mcomplex mcrand(){
	/*
	 * This function generate random mcomplex number
	 */
	mcomplex c;
	c.a = mrrand();
	c.b = mrrand();
	return c;
}

void printmc(mcomplex a){
	printf("%f+j%f", a.a, a.b);
}

void setmcm(mcomplex* to, mcomplex* from){
	/*
	 * This function sets a mcomplex from another mcomplex
	 */
	to->a = from->a;
	to->b = from->b;
}
void setmcr(mcomplex* to, mreal* from){
	/*
	 * This function sets a mcomplex from another mreal
	 */
	to->a = *from;
	to->b = 0;
}
