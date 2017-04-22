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


# Here, you define, which weighted classification problems will be considered.
# The meaning of the values can be found by typing svm-train -w
# The choice is usually a bit tricky. Good luck ...

: ${WEIGHT_STEPS:=9}
: ${MAX_WEIGHT:=0.9}
: ${MIN_WEIGHT:=0.1}
: ${GEO_WEIGHTS:=0}



# Display help message

if [[ -z "$1" ]]
then
	source $SML_SCRIPTS_DIR/print-default-values.sh "roc-svm.sh" ""
	source $SML_SCRIPTS_DIR/print-meaning-of-parameters.sh
	echo
	exit      
fi


# Copy command line arguments into variables used by svm.sh

source $SML_SCRIPTS_DIR/read-standard-arguments.sh "${@:1:7}"



# Set some extra options

FOLDS=$STRATIFIED" "$NUM_FOLDS
SVM_TYPE=$SVM_HINGE_2D
CLIPPING=1.0
VOTE_TYPE=$VOTE_METHOD" 0"
LOSS_TYPE="0"
DISPLAY_ROC_STYLE=1



# Get file names and grid specifications

source $SML_SCRIPTS_DIR/get-file-names.sh
source $SML_SCRIPTS_DIR/get-grid-and-partition.sh 



# Train and test the SVM

source $SML_SCRIPTS_DIR/svm-train.sh
for (( WEIGHT_NUMBER=1; WEIGHT_NUMBER<=$WEIGHT_STEPS; WEIGHT_NUMBER++ ))
do
	source $SML_SCRIPTS_DIR/svm-select.sh
done
source $SML_SCRIPTS_DIR/svm-test.sh



