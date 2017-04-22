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


 


#############################################################################
#
# This script runs all svm-test with the help of some variables set elsewhere
#
#############################################################################


# Make sure only options are used for which the corresponding variable is set

# Global options

source $SML_SCRIPTS_DIR/global-svm-options.sh


#----------------------------- Options for svm-test ------------------------------------------------------

if ! [[ -z "$LOSS_TYPE" ]]
then
	LOSS_TYPE_OPT="-L "$LOSS_TYPE
fi

if ! [[ -z "$DISPLAY_ROC_STYLE" ]]
then
	DISPLAY_ROC_STYLE_OPT="-o "$DISPLAY_ROC_STYLE
fi



if ! [[ -z "$VOTE_TYPE" ]]
then
	VOTE_TYPE_OPT="-v "$VOTE_TYPE
fi


echo
echo -------------------------- svm-test --------------------------------------------

$SML_BIN_DIR/svm-test $VOTE_TYPE_OPT   $LOSS_TYPE_OPT  $DISPLAY_ROC_STYLE_OPT    $SVM_GLOBAL_OPTS  $SVM_TEST_FILENAMES

