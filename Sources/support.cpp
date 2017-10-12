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
	 * that we are insterested in computing the difference
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
	 * out_time is a double representins the time in milliseconds
	 * Description:
	 * this function converts the time in the timespec structure time to milliseconds measures
	 */

	double out_time = static_cast<double>(time.tv_sec)*1000+static_cast<double>(time.tv_nsec)/1000000;
	return out_time;
}
