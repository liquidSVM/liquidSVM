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


COMMAND=$1



if ! [[ -z $COMMAND ]]
then
	echo 
	echo 'Being in the ./scripts folder, the command called for this example is:'
	echo
	echo $COMMAND


	echo
	echo
	read -p "To start this command, press any key." -n1 -s

	echo
	./$COMMAND
fi

echo
read -p "Press any key to continue ... " -n1 -s

