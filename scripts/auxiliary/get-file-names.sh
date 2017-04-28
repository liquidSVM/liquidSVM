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
# This script generates various file names for svm-train, svm-select, and svm-test based on $BASE_FILENAME.
# In addition $SOL_TRAIN_FILENAME is set and some log files are deleted according to the values of 
# $CREATE_SOL_TRAIN and $DELETE_OLD_LOGS_ETC.
#
##################################################################################################################



# Iterate over all candidates of the search path if the variable SML_DATA_DIR has not been set
# by the user, and take the first one, which containes the data.

if [[ -z $SML_DATA_DIR ]]
then
	# We need to change IFS to colon so we can iterate over locations in data path
	OLD_IFS=$IFS
	IFS=:
	for DIR in $SML_DATA_PATH; do
		if [[ -f "$DIR/$BASE_FILENAME.train.csv" ]] || [[ -f "$DIR/$BASE_FILENAME.train.uci" ]] || [[ -f "$DIR/$BASE_FILENAME.train.lsv" ]] || [[ -f "$DIR/$BASE_FILENAME.nla" ]]
		then
			SML_DATA_DIR="$DIR"
			break
		fi
	done
	#   Reverse definition
	IFS=$OLD_IFS
fi
echo
echo Reading data files from folder: $SML_DATA_DIR 



# Do the same thing for the result path

if [[ -z $SML_RESULT_DIR ]]
then
	OLD_IFS=$IFS
	IFS=:
	for DIR in $SML_RESULT_PATH; do
		if [[ -d "$DIR" ]]
		then
			SML_RESULT_DIR="$DIR"
			break
		fi
	done
	IFS=$OLD_IFS
fi
echo Writing result files to folder: $SML_RESULT_DIR 


# Determine file format of the training and test set.

TRAIN_VAL_TEST=TRUE
if [[ -f "$SML_DATA_DIR/$BASE_FILENAME.train.csv" ]]
then
	EXT=csv
elif [[ -f "$SML_DATA_DIR/$BASE_FILENAME.train.uci" ]]
then
	EXT=uci
elif [[ -f "$SML_DATA_DIR/$BASE_FILENAME.train.lsv" ]]
then
	EXT=lsv
elif [[ -f "$SML_DATA_DIR/$BASE_FILENAME.train.nla" ]] && [[ $ALLOW_NLA = TRUE ]]
then
	EXT=nla
elif [[ -f "$SML_DATA_DIR/$BASE_FILENAME.nla" ]] && [[ $ALLOW_NLA = TRUE ]]
then
	TRAIN_VAL_TEST=FALSE
	EXT=nla
	ALLOWED_EXT="nla"
fi


# Define training and test file names

if [[ $TRAIN_VAL_TEST = TRUE ]]
then
	TRAIN_FILENAME="$SML_DATA_DIR/$BASE_FILENAME.train.$EXT"
	TEST_FILENAME="$SML_DATA_DIR/$BASE_FILENAME.test.$EXT"
	VAL_FILENAME="$SML_DATA_DIR/$BASE_FILENAME.val.$EXT"
	ALLOWED_EXT="[csv | uci | lsv "
	if [[ $ALLOW_NLA = TRUE ]]
	then
		ALLOWED_EXT=$ALLOWED_EXT"| nla "
	fi
	ALLOWED_EXT=$ALLOWED_EXT"]"
	ALLOWED_TRAIN_EXT=".train."$ALLOWED_EXT
	ALLOWED_TEST_EXT=".test."$ALLOWED_EXT
else
	TRAIN_FILENAME="$SML_DATA_DIR/$BASE_FILENAME.$EXT"
fi


# Check whether files exist

if ! [[ -f "$TRAIN_FILENAME" ]]
then
	echo
	echo "SCRIPT ERROR: Could not determine extension of training file since" 
	echo
	if ! [[ $ALLOW_NLA = TRUE ]]
	then 
		echo "   "$BASE_FILENAME$ALLOWED_TRAIN_EXT
	else
		echo "   "$BASE_FILENAME$ALLOWED_TRAIN_EXT"   or   "$BASE_FILENAME".nla"
	fi
	echo 
	echo could not be found.
	echo
	exit
fi

if ! [[ -f "$TEST_FILENAME" ]] && [[ $TRAIN_VAL_TEST = TRUE ]]
then
	echo
	echo "SCRIPT ERROR: Could not determine extension of test file since" 
	echo
	echo "   "$BASE_FILENAME$ALLOWED_TEST_EXT
	echo 
	echo could not be found.
	echo
	exit
fi


# Define auxiliary filenames

LOG_SUMMARY_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.summary.log"

LOG_TRAIN_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.train.log"
AUX_TRAIN_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.train.aux"
if ! [[ -z $CREATE_SOL_TRAIN ]] && [[ $CREATE_SOL_TRAIN="yes" ]]
then
	SOL_TRAIN_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.train.sol"
fi

LOG_SELECT_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.select.log"
SOL_SELECT_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.sol"

LOG_TEST_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.test.log"
RES_TEST_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.result.csv"
FULL_RES_TEST_FILENAME="$SML_RESULT_DIR/$BASE_FILENAME.full_result.csv"

SVM_TRAIN_FILENAMES=$TRAIN_FILENAME" "$LOG_TRAIN_FILENAME" "$LOG_SUMMARY_FILENAME" "$SOL_TRAIN_FILENAME
SVM_SELECT_FILENAMES=$TRAIN_FILENAME" "$LOG_TRAIN_FILENAME" "$LOG_SELECT_FILENAME" "$SOL_SELECT_FILENAME" "$LOG_SUMMARY_FILENAME
SVM_TEST_FILENAMES=$TRAIN_FILENAME" "$SOL_SELECT_FILENAME" "$TEST_FILENAME" "$LOG_TEST_FILENAME" "$RES_TEST_FILENAME" "$LOG_SUMMARY_FILENAME




# Clean up from previous training runs

if ! [[ -z $CREATE_SOL_TRAIN ]] && [[ $CREATE_SOL_TRAIN == "yes" ]]
then
	rm -f "$SOL_TRAIN_FILENAME"
fi


if [[ -z $DELETE_OLD_LOGS_ETC ]] || [[ $DELETE_OLD_LOGS_ETC == "yes" ]]
then
	rm -f "$LOG_TRAIN_FILENAME"
	rm -f "$AUX_TRAIN_FILENAME"

	rm -f "$LOG_SELECT_FILENAME"
	rm -f "$SOL_SELECT_FILENAME"

	rm -f "$LOG_TEST_FILENAME"
	rm -f "$RES_TEST_FILENAME"
	rm -f "$FULL_RES_TEST_FILENAME"
fi
