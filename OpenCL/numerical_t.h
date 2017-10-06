/*
 * numerical_t.h
 *
 *  Created on: Sep 28, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  	This file describes the numerical types required for the matrix computation regarding
 *  	this project. In particular, we are interested in developing fast algorithms that are capable
 *  	of simultaneously compute the inverse and the determinant of a 3x3 symmetric matrix with
 *  	complex entries. This problem occurs in imagining in synthetic aperture radar (SAR) for detection
 *  	and separation between urban, rural, sea ... areas. The computation involved is intensive because
 *  	every pixel in the obtained images are 3x3 symmetric complex matrices. The hypothesis testing
 *  	required for separation between areas demand the computation of the inverse and determinant of these 3x3 matrices.
 *  	Computing them efficiently, not only reduces the required energy for performing the same task, but also
 *  	reduces the computation time.
 *
 *  	Because OpenCL is follows C99, everything in this file follows this standard.
 */

#ifndef NUMERICAL_T_H_
#define NUMERICAL_T_H_

/*
 * Defining our own numerical type. This is important because in case
 * of change from double to other formats, we just need to change the definition
 * of our numerical type. The word mreal stands for my real.
 */

typedef double mreal;

/*
 * This is our complex number representation based on the mreal type.
 * If mreal type is changed, the complex type is changed accordingly.
 * The word mcomplex stands for my complex.
 */

//mcomplex type
typedef struct{
	mreal a;
	mreal b;
} mcomplex;

//mcmatrix type
typedef struct {
	mcomplex a;
	mcomplex b;
	mcomplex c;
	mcomplex d;
	mcomplex e;
	mcomplex f;
} mcmatrix;

#endif /* NUMERICAL_T_H_ */
