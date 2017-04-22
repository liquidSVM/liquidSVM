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


# Set default values. If you wish to perform default bootstrapping differently,
# change the BOOT_STRAP line. The first number specifies the number of resamples, 
# and the second number the size of each resample. 

source $SML_SCRIPTS_DIR/set-default-values.sh
SVM_TYPE=$SVM_HINGE_2D
BOOT_STRAP="5 500"


# Display help message

if [[ -z "$1" ]]
then
	source $SML_SCRIPTS_DIR/print-default-values.sh "bs-svm.sh" "[<svm-type>] \"[<ws-number> <ws-size>]\""
	echo "svm-type:              "$SVM_TYPE
	echo "ws-number & ws-size:   "$BOOT_STRAP
	echo
	source $SML_SCRIPTS_DIR/print-meaning-of-parameters.sh
	echo
	echo "<svm-type>                  This parameter determines the solver type used."
	echo "                            The numbering is that of 'svm-train -S' ."
	echo
	echo "[<ws-number> <ws-size>]     This parameter determines the number of resamples"
	echo "                            and the number of samples in each resample."
	echo 
	exit       
fi


# Copy command line arguments into variables

source $SML_SCRIPTS_DIR/read-standard-arguments.sh "${@:1:7}"

if ! [[ -z $8 ]]
then
	SVM_TYPE=$8
fi

if ! [[ -z $9 ]]
then
	BOOT_STRAP=$9
fi


# Set some extra variables according to the chosen solver type.

if [[ $SVM_TYPE = $KERNEL_RULE ]]
then
	VOTE_TYPE=$VOTE_METHOD" 0"
	LOSS_TYPE="0"
	FOLDS=$STRATIFIED" "$NUM_FOLDS
elif [[ $SVM_TYPE = $SVM_LS_2D ]]
then
	VOTE_TYPE=$VOTE_METHOD" 1"
	LOSS_TYPE=2
	CLIPPING=0
	FOLDS=$RANDOM_FOLDS" "$NUM_FOLDS
elif [[ $SVM_TYPE = $SVM_HINGE_2D ]]
then
	VOTE_TYPE=$VOTE_METHOD" 0"
	LOSS_TYPE="0"
	CLIPPING=1
	FOLDS=$STRATIFIED" "$NUM_FOLDS
elif [[ $SVM_TYPE = $SVM_QUANTILE ]]
then
	VOTE_TYPE=$VOTE_METHOD" 1"
	LOSS_TYPE="4"
	CLIPPING=0
	FOLDS=$RANDOM_FOLDS" "$NUM_FOLDS
elif [[ $SVM_TYPE = $SVM_EXPECTILE ]]
then
	VOTE_TYPE=$VOTE_METHOD" 1"
	LOSS_TYPE="3"
	CLIPPING=0
	FOLDS=$RANDOM_FOLDS" "$NUM_FOLDS
fi




# Get file names and grid specifications

source $SML_SCRIPTS_DIR/get-file-names.sh
source $SML_SCRIPTS_DIR/get-grid-and-partition.sh 



# Train and test the SVM

source $SML_SCRIPTS_DIR/svm-train.sh
source $SML_SCRIPTS_DIR/svm-select.sh
source $SML_SCRIPTS_DIR/svm-test.sh



