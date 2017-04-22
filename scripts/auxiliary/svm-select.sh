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


 


###############################################################################
#
# This script runs all svm-select with the help of some variables set elsewhere
#
###############################################################################


# Make sure only options are used for which the corresponding variable is set

# Global options

source $SML_SCRIPTS_DIR/global-svm-options.sh


#----------------------------- Options for svm-select ------------------------------------------------------

if ! [[ -z $RETRAIN_METHOD ]]
then
	RETRAIN_METHOD_OPT="-R "$RETRAIN_METHOD
fi

if ! [[ -z "$NPL_CLASS" ]]
then
	NPL_SELECT_OPT=$(echo -N $NPL_CLASS $NPL_CONSTRAINT)
fi

if ! [[ -z "$WEIGHT_NUMBER" ]]
then
	WEIGHT_SELECT_OPT=$(echo -W $WEIGHT_NUMBER)
fi




echo
echo -------------------------- svm-select ------------------------------------------

$SML_BIN_DIR/svm-select $RETRAIN_METHOD_OPT  $NPL_SELECT_OPT  $WEIGHT_SELECT_OPT  $DISPLAY_OPT   $SVM_SELECT_FILENAMES


