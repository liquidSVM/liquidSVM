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
MC_TYPE=0


# Display help message

if [[ -z "$1" ]]
then
	source $SML_SCRIPTS_DIR/print-default-values.sh "mc-svm.sh" "[<mc-type>]"
	echo mc-type:"               "$MC_TYPE
	echo
	source $SML_SCRIPTS_DIR/print-meaning-of-parameters.sh
	
	echo
	echo "<mc-type>                   This parameter determines the multiclass strategy:"
	echo "                              <mc-type> = 0  =>   AvA with hinge loss."
	echo "                              <mc-type> = 1  =>   OvA with least squares loss."
	echo "                              <mc-type> = 2  =>   OvA with hinge loss."
	echo "                              <mc-type> = 3  =>   AvA with least squares loss."
	echo
	exit         
fi


# Copy command line arguments into variables 

source $SML_SCRIPTS_DIR/read-standard-arguments.sh "${@:1:7}"

if ! [[ -z $8 ]]
then
	MC_TYPE=$8
fi


# Set some extra variables according to the chosen multiclass type

if [[ $MC_TYPE = 0 ]]
then
	WS_TYPE="1"
	SVM_TYPE=$SVM_HINGE_2D
	VOTE_TYPE=$VOTE_METHOD" 0"
	LOSS_TYPE="0"
elif [[ $MC_TYPE = 1 ]]
then
	WS_TYPE="2"
	SVM_TYPE=$SVM_LS_2D
	VOTE_TYPE=$VOTE_METHOD" 1"
	LOSS_TYPE="2"
elif [[ $MC_TYPE = 2 ]]
then
	WS_TYPE="2"
	SVM_TYPE=$SVM_HINGE_2D
	VOTE_TYPE=$VOTE_METHOD" 1"
	LOSS_TYPE="0"
elif [[ $MC_TYPE = 3 ]]
then
	WS_TYPE="1"
	SVM_TYPE=$SVM_LS_2D
	VOTE_TYPE=$VOTE_METHOD" 1"
	LOSS_TYPE="2"
fi


# Set some extra options for classification

FOLDS=$STRATIFIED" "$NUM_FOLDS
CLIPPING=1.0


# Get file names and grid specifications

source $SML_SCRIPTS_DIR/get-file-names.sh
source $SML_SCRIPTS_DIR/get-grid-and-partition.sh 


# Train and test the SVM

source $SML_SCRIPTS_DIR/svm-train.sh
source $SML_SCRIPTS_DIR/svm-select.sh
source $SML_SCRIPTS_DIR/svm-test.sh

