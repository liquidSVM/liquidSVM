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


 


# Determine where I am and where the data might be.

SML_SCRIPTS_DIR=`dirname $0`
source $SML_SCRIPTS_DIR/auxiliary/get-directories.sh


# Set default values

source $SML_SCRIPTS_DIR/set-default-values.sh
: ${NPL_CLASS:=1}
: ${NPL_CONSTRAINT:=0.05}


# Here, you define, which weighted classification problems will be considered.
# The meaning of the values can be found by typing svm-train -w
# The choice is usually a bit tricky. Good luck ...

: ${WEIGHT_STEPS:=10}
: ${MIN_WEIGHT:=0.001}
: ${MAX_WEIGHT:=0.5}
: ${GEO_WEIGHTS:=1}



# Display help message

if [[ -z "$1" ]]
then
	source $SML_SCRIPTS_DIR/print-default-values.sh "npl-svm.sh" "[<class>] [<constraint>]" "1"
	echo class:"                 "$NPL_CLASS
	echo constraint:"            "$NPL_CONSTRAINT
	echo
	source $SML_SCRIPTS_DIR/print-meaning-of-parameters.sh "1"
	
	echo
	echo "<class>                     The class, the <constraint> is enforced on."
	echo
	echo "<constraint>                The constraint on the false alarm rate. The script"
	echo "                            actually considers a couple of values around the"
	echo "                            value of <constraint> to give the user an informed"
	echo "                            choice."
	echo 
	exit           
fi


# Copy command line arguments into variables 

source $SML_SCRIPTS_DIR/read-standard-arguments.sh $1 $2 $3 $4 $4 $5 $5

PARTITION_CHOICE=0
ADAPTIVE_SEARCH=0

if ! [[ -z $6 ]]
then
	NPL_CLASS=$6
fi

if ! [[ -z $7 ]]
then
	NPL_CONSTRAINT=$7
fi
NPL_CONSTRAINT_BASE=$NPL_CONSTRAINT

# Set some extra options 

FOLDS=$STRATIFIED" "$NUM_FOLDS
SVM_TYPE=$SVM_HINGE_2D
LOSS_TYPE="0"
VOTE_TYPE=$VOTE_METHOD" 2 "$NPL_CLASS
CLIPPING=1

if [[ $NPL_CLASS = 1 ]]
then
	NPL_SWAP=1
else
	NPL_SWAP=0
fi


# Get file names and grid specifications  

source $SML_SCRIPTS_DIR/get-file-names.sh
source $SML_SCRIPTS_DIR/get-grid-and-partition.sh 



# Train and test the SVM
# 5 false alarm rates around the defined NPL_CONSTRAINT are considered.

source $SML_SCRIPTS_DIR/svm-train.sh
for CONSTRAINT_FACTOR in {3,4,6,9,12}
do
	NPL_CONSTRAINT=$(echo "scale=8; $CONSTRAINT_FACTOR * $NPL_CONSTRAINT_BASE / 6" | bc)
	source $SML_SCRIPTS_DIR/svm-select.sh
done
source $SML_SCRIPTS_DIR/svm-test.sh



