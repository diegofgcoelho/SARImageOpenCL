/*
 * numerical_t.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  	This file implements some useful functions described in numeric_s.h.
 *
 */

#include "numerical_s.h"
#include "../OpenCL/numerical_t.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>

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

double norm(mcmatrix a){
	/*
	 * This function returns the Frobenius norm of the mcmatrix object,
	 * which represents an Hermitian matrix.
	 */

	double t = static_cast<double>((a.a.a)*(a.a.a)+(a.a.b)*(a.a.b));
	t += static_cast<double>((a.b.a)*(a.b.a)+(a.b.b)*(a.b.b));
	t += static_cast<double>((a.c.a)*(a.c.a)+(a.c.b)*(a.c.b));
	t += static_cast<double>((a.d.a)*(a.d.a)+(a.d.b)*(a.d.b));
	t += static_cast<double>((a.e.a)*(a.e.a)+(a.e.b)*(a.e.b));
	t += static_cast<double>((a.f.a)*(a.f.a)+(a.f.b)*(a.f.b));

	return sqrt(t);
}

double mcmatrix_norm(mcmatrix a, mcmatrix b){
	/*
	 * This function returns the Frobenius norm of the difference between mcmatrix objects,
	 * which represent an Hermitian matrices.
	 */

	mcmatrix c;
	c.a.a = a.a.a-b.a.a;
	c.a.b = a.a.b-b.a.b;

	c.b.a = a.b.a-b.b.a;
	c.b.b = a.b.b-b.b.b;

	c.c.a = a.c.a-b.c.a;
	c.c.b = a.c.b-b.c.b;

	c.d.a = a.d.a-b.d.a;
	c.d.b = a.d.b-b.d.b;

	c.e.a = a.e.a-b.e.a;
	c.e.b = a.e.b-b.e.b;

	c.f.a = a.f.a-b.f.a;
	c.f.b = a.f.b-b.f.b;

	return norm(c);
}
