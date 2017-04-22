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


 


##################################################################################################################
#
# This script generates various file names for svm-train, svm-select, and svm-test based on the base file name $1.
#
##################################################################################################################


if [[ -z $SML_DIR ]]
then
	if [[ "$SML_SCRIPTS_DIR" == "."* ]] 
	then
		SML_DIR=$SML_SCRIPTS_DIR/..
	else
		SML_DIR=`dirname "$SML_SCRIPTS_DIR"`
	fi
fi
SML_BIN_DIR=$SML_DIR/bin
SML_SCRIPTS_DIR=$SML_DIR/scripts/auxiliary


source $SML_SCRIPTS_DIR/get-package-name.sh
echo "$(pwd)"|grep -q $SML_DIR && IN_SML="yes"


# If one is in one of the package folders, then use the result folder of the package.
# Otherwise, the SML_RESULT_PATH will be searched.

if ! [[ -z $IN_SML ]]
then
	SML_RESULT_DIR=$SML_DIR/results
fi

: ${SML_DATA_PATH:=.:data:../data:"$SML_DIR/data:$HOME/$PACKAGE_NAME""data:$HOME/$PACKAGE_NAME-data:$HOME/$PACKAGE_NAME""_data"}
: ${SML_RESULT_PATH:=results:../results:"$HOME/$PACKAGE_NAME""results:$HOME/$PACKAGE_NAME-results:$HOME/$PACKAGE_NAME""_results:."}







