#!/bin/bash
 
# Copyright 2015, 2016, 2017 Ingo Steinwart
#
# This file is part of liquidSVM.
#
# liquidSVM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# liquidSVM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.


 


##############################################################
#
# This script defines hyper-parameter grid, a hyper-parameter
# search strategy as well as a partition strategy according to
# the values of $GRID_CHOICE, $ADAPTIVITY_CONTROL, and 
# $PARTITION_CHOICE. Negative values of  $GRID_CHOICE create 
# grids in which all grid points are manually chosen, and 
# positive values create grids with geometrically spaced grid 
# points.
#
##############################################################





##############################################################
# 
# Grid Section
#
##############################################################


if [[ "$GRID_CHOICE" = -2 ]] || ! [[ -z "$C_VALUES" ]]
then
# 	Here we create a grid with gamma and C values, where
# 	the C values are the classical term in front of the 
# 	empirical error term. You can replace the lists by any 
# 	other space separated lists, but you should not change
# 	GRID_OPTIONS= ... line.

	: ${C_VALUES:="0.01 0.1 1 10 100 1000 10000"}
	: ${GAMMAS:="10.0 5.0 2.0 1.0 0.5 0.25 0.1 0.05"}
	
	GRID_OPTIONS="-g [ "$GAMMAS" ] -l [ "$C_VALUES" ] 1"
elif [[ "$GRID_CHOICE" = -1 ]] || ! [[ -z "$GAMMAS" ]] || ! [[ -z "$LAMBDAS" ]]
then
# 	Here we create a grid with gamma and lambda values, where
# 	the lambda values are the classical regularization parameter
# 	in front of the norm term. You can replace the lists by any 
# 	other space separated lists, but you should not change
# 	GRID_OPTIONS= ... line.

	: ${LAMBDAS:="1.0 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001"}
	: ${GAMMAS:="10.0 5.0 2.0 1.0 0.5 0.25 0.1 0.05"}
	
	GRID_OPTIONS="-g [ "$GAMMAS" ] -l [ "$LAMBDAS" ] 0"
else
# 	In the next three code blocks, geometrically spaced grids of 
# 	different sizes are created. Note the min and max values are 
# 	scaled according the the number of samples, the dimensionality
# 	of the data sets, the number of folds used, and the estimated 
# 	diameter of the data set. You can freely change the min and 
# 	max values as well as the step values.

	if [[ "$GRID_CHOICE" = 0 ]]
	then
# 	This gives a 10-by-10 grid, which is the default grid
	
		: ${MIN_LAMBDA:=0.001}
		: ${MAX_LAMBDA:=0.01}
		: ${MIN_GAMMA:=0.2}
		: ${MAX_GAMMA:=5.0}

		: ${LAMBDA_STEPS:=10}
		: ${GAMMA_STEPS:=10}
	elif [[ "$GRID_CHOICE" = 1 ]]
	then
# 	This is a 15-by-15 grid 

		: ${MIN_LAMBDA:=0.0001}
		: ${MAX_LAMBDA:=0.01}
		: ${MIN_GAMMA:=0.1}
		: ${MAX_GAMMA:=10.0}

		: ${LAMBDA_STEPS:=15}
		: ${GAMMA_STEPS:=15}
	elif [[ "$GRID_CHOICE" = 2 ]]
	then
# 	And finally, this gives a 20-by-20 grid

		: ${MIN_LAMBDA:=0.00001}
		: ${MAX_LAMBDA:=0.01}
		: ${MIN_GAMMA:=0.05}
		: ${MAX_GAMMA:=20.0}

		: ${LAMBDA_STEPS:=20}
		: ${GAMMA_STEPS:=20}
	fi
	
# 	The next lines combines all set variables

	GRID_OPTIONS="-g "$GAMMA_STEPS" "$MIN_GAMMA" "$MAX_GAMMA" -l "$LAMBDA_STEPS" "$MIN_LAMBDA" "$MAX_LAMBDA
fi


# 	Next an adaptive grid search is activated. The higher the values
# 	of MAX_LAMBDA_INCREASES and MAX_NUMBER_OF_WORSE_GAMMAS are set
# 	the more conservative the search strategy is. The values can be 
# 	freely modified.

if [[ "$ADAPTIVITY_CONTROL" = 1 ]]
then
	ADAPTIVE_SEARCH=1
	MAX_LAMBDA_INCREASES=4
	MAX_NUMBER_OF_WORSE_GAMMAS=4
elif [[ "$ADAPTIVITY_CONTROL" = 2 ]]
then
	ADAPTIVE_SEARCH=1
	MAX_LAMBDA_INCREASES=3
	MAX_NUMBER_OF_WORSE_GAMMAS=3
elif [[ ! -z "$ADAPTIVITY_CONTROL" ]]
then
  ADAPTIVE_SEARCH="$ADAPTIVITY_CONTROL"
else
	ADAPTIVE_SEARCH=0
fi


# Finally, the defined grid is combined with the parameters set by the chosen s
# grid search trategy.

GRID_OPTIONS=$GRID_OPTIONS" -a "$ADAPTIVE_SEARCH" "$MAX_LAMBDA_INCREASES" "$MAX_NUMBER_OF_WORSE_GAMMAS



##############################################################
# 
# Partition Section
#
##############################################################


# The details of the following settings can be obtained by typing
# svm-train -P

if ! [[ -z "$PARTITION_CHOICE" ]]
then
	if [[ "$PARTITION_CHOICE" = 1 ]]
	then
# This gives a partition into random chunks of size 2000

		VORONOI="1 2000"
	elif [[ "$PARTITION_CHOICE" = 2 ]]
	then
# This gives a partition into 10 random chunks

		VORONOI="2 10"
	elif [[ "$PARTITION_CHOICE" = 3 ]]
	then
# This gives a Voronoi partition into cells with radius 
# not larger than 1.0. For its creation a subsample containing
# at most 50.000 samples is used. 

		VORONOI="3 1.0 50000"
	elif [[ "$PARTITION_CHOICE" = 4 ]]
	then
# This gives a Voronoi partition into cells with at most 2000 
# samples (approximately). For its creation a subsample containing
# at most 50.000 samples is used. A shrinking heuristic is used 
# to reduce the number of cells.

		VORONOI="4 2000 1 50000"
	elif [[ "$PARTITION_CHOICE" = 5 ]]
	then
# This gives a overlapping regions with at most 2000 samples
# (approximately). For its creation a subsample containing
# at most 50.000 samples is used. A stopping heuristic is used 
# to stop the creation of regions if 0.5 * 2000 samples have
# not been assigned to a region, yet. 

		VORONOI="5 2000 0.5 50000 1"
	elif [[ "$PARTITION_CHOICE" = 6 ]]
	then
# Experimental so far ...
	
		VORONOI="6 2000 1 50000 2.0 20 4"
	else
	  VORONOI="$PARTITION_CHOICE"
	fi
fi



