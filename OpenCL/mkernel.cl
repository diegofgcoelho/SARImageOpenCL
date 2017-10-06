/*
 * mkernel.cl
 *
 *  Created on: Sep 28, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  	This file describes the numerical functions and kernels required for the matrix computation regarding
 *  	this project.
 *
 *  	Because OpenCL is follows C99, everything in this file follows this standard.
 */

#include "numerical_t.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

mcomplex addmc(mcomplex x, mcomplex y);
mcomplex submc(mcomplex x, mcomplex y);
mcomplex multmc(mcomplex x, mcomplex y);
mcomplex divmc(mcomplex x, mcomplex y);
mcomplex invmc(mcomplex x);
mcomplex conjmc(mcomplex x);
mreal magmc(mcomplex x);
mcomplex mulmcc(mcomplex x, mreal c);
mcomplex sqrtmc(mcomplex x);


//mcomplex functions
mcomplex addmc(mcomplex x, mcomplex y){
	/*
	 * This function returns x+y
	 */
	mcomplex c;
	c.a = x.a+y.a;
	c.b = x.b+y.b;
	return c;
}

mcomplex submc(mcomplex x, mcomplex y){
	/*
	 * This function returns x-y
	 */
	mcomplex c;
	c.a = x.a-y.a;
	c.b = x.b-y.b;
	return c;
}

mcomplex multmc(mcomplex x, mcomplex y){
	/*
	 * This function returns x*y
	 */
	mcomplex c;
	c.a = x.a*y.a-x.b*y.b;
	c.b = x.a*y.b+y.a*x.b;
	return c;
}

mcomplex divmc(mcomplex x, mcomplex y){
	/*
	 * This function returns x/y
	 */
	mcomplex c = mulmcc(multmc(x,conjmc(y)), 1/pow(magmc(y),2.0));
	return c;
}

mcomplex invmc(mcomplex x){
	/*
	 * This function returns 1/x
	 */
	mcomplex c = mulmcc(conjmc(x), 1/pow(magmc(x),2));
	return c;
}

mcomplex conjmc(mcomplex x){
	/*
	 * This function returns conjmc(x)
	 */
	mcomplex c;
	c.a = x.a;
	c.b = -x.b;
	return c;
}

mreal magmc(mcomplex x){
	/*
	 * This function returns |x|
	 */
	mreal c = (mreal) sqrt(x.a*x.a+x.b*x.b);
	return c;
}

mcomplex mulmcc(mcomplex x, mreal c){
	/*
	 * This function returns c*x
	 */
	mcomplex r;
	r.a = c*x.a;
	r.b = c*x.b;
	return r;
}

mcomplex sqrtmc(mcomplex x){
	/*
	 * This function returns sqrtmc(x).
	 * In this function, we use the Euler
	 * formula to compute the principal
	 * square root.
	 */
	mreal xmag = magmc(x);
	mreal xsin = x.b/xmag;
	mreal xcos = x.a/xmag;
	mreal angle = asin(xsin);
	//Correcting the angle value
	if(xsin > 0.0){
		if(xcos < 0.0){
			angle = M_PI-angle;
		}
	} else {
		if(xcos < 0.0){
			angle = M_PI-angle;
		}
	}

	mreal hangle = angle/2;

	mcomplex c;
	c.a = xmag*cos(hangle);
	c.b = xmag*sin(hangle);
	return c;
}

//Actual kernels

__kernel void inv_det_lu(__global mcmatrix * in, __global mcmatrix * out, __global mcomplex * det){\
	size_t id = get_global_id(0);
	
	//We know that the elements in the diagonal are real
	mcomplex l00; l00.a = sqrt(in[id].a.a); l00.b = 0;
	mreal v0 = 1/l00.a;
	mcomplex l10 = mulmcc(conjmc(in[id].b), v0);
	mcomplex l20 = mulmcc(conjmc(in[id].c), v0);
	//We know that the elements in the diagonal are real
	mcomplex l11; l11.a = in[id].d.a-pow(magmc(l10), 2); l11.b = 0;
	l11.a = sqrt(l11.a);
	mreal v1 = 1/l11.a;
	mcomplex l21 = mulmcc(submc(conjmc(in[id].e), multmc(conjmc(l10), l20)), v1);
	//We know that the elements in the diagonal are real
	mcomplex l22; l22.a = in[id].f.a-pow(magmc(l20), 2)-pow(magmc(l21),2); l22.b = 0;
	l22.a = sqrt(l22.a);
	
	//Determinant of input matrix
	mreal tdet = pow(l00.a*l11.a*l22.a, 2);
	det[id].a = tdet; det[id].b = 0;
	mreal idet = 1/tdet;

	mreal t00 = l11.a*l22.a;
	mcomplex t01 = mulmcc(l10, l22.a); t01.a = -t01.a; t01.b = -t01.b;
	mcomplex t02 = submc(multmc(l10,l21), multmc(l20,l11));
	mreal t11 = l00.a*l22.a;
	mcomplex t12 = multmc(l00,l21); t12.a = -t12.a; t12.b = -t12.b;
	mreal t22 = l00.a*l11.a;

	//Assigning the inverse
	out[id].a.a = (pow(t00,2)+pow(magmc(t01),2)+pow(magmc(t02),2))*idet; out[id].a.b = 0.0;
	out[id].b = conjmc(mulmcc(addmc(mulmcc(t01,t11), multmc(t02,conjmc(t12))), idet));
	out[id].c = conjmc(mulmcc(t02, t22*idet));
	out[id].d.a = (pow(t11,2)+pow(magmc(t12), 2))*idet; out[id].d.b = 0.0;
	out[id].e = conjmc(mulmcc(mulmcc(t12,t22), idet));
	out[id].f.a = pow(t22,2)*idet; out[id].f.b = 0.0;
}

__kernel void inv_det_fast(__global mcmatrix * in, __global mcmatrix * out, __global mcomplex * det){\
	size_t id = get_global_id(0);
	
	mcomplex A = multmc(in[id].d, in[id].f);
	A.a = A.a-pow(magmc(in[id].e), 2);
	mcomplex B = mulmcc(submc(multmc(conjmc(in[id].b), in[id].f), multmc(in[id].e, conjmc(in[id].c))), -1.0);
	mcomplex C = submc(multmc(conjmc(in[id].b), conjmc(in[id].e)), multmc(in[id].d, conjmc(in[id].c)));
	mcomplex D = multmc(in[id].a, in[id].f); D.a = D.a - pow(magmc(in[id].c), 2);
	mcomplex E = mulmcc(submc(multmc(in[id].a, conjmc(in[id].e)), multmc(in[id].b, conjmc(in[id].c))), -1.0);
	mcomplex F = multmc(in[id].a, in[id].d); F.a = F.a - pow(magmc(in[id].b), 2);

	//Determinant of input matrix
	det[id] = addmc(multmc(in[id].a, A), addmc(multmc(in[id].b, B), multmc(in[id].c, C)));

	mcomplex invD = invmc(det[id]);

	//Assigning the inverse
	out[id].a = conjmc(multmc(invD, A));
	out[id].b = conjmc(multmc(invD, B));
	out[id].c = conjmc(multmc(invD, C));
	out[id].d = conjmc(multmc(invD, D));
	out[id].e = conjmc(multmc(invD, E));
	out[id].f = conjmc(multmc(invD, F));
}