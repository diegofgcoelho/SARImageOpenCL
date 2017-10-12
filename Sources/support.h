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

timespec diff_time(timespec init, timespec end);
double get_millisecs(timespec time);

#endif /* SUPPORT_H_ */
