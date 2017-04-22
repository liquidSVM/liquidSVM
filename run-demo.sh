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


 


# Make sure everything is prepared for the demo

find . -name \*.sh | xargs chmod a+x 

if ! [[ -f ./bin/svm-train ]]
then
	make svm-train
fi

if ! [[ -f ./bin/svm-select ]]
then
	make svm-select
fi


if ! [[ -f ./bin/svm-test ]]
then
	make svm-test
fi




cd ./scripts




printf "\033c"
echo
echo 'The package contains scripts for the following learning scenarios:'
echo '- (weighted) binary classification'
echo '- multiclass classification (both AvA and OvA)'
echo '- Neyman-Pearson-type classification'
echo '- Least squares regression'
echo '- Quantile regression'
echo '- Expectile regression'
echo 


./auxiliary/run-command-and-wait.sh ''


printf "\033c"
echo
echo 'The first run is on the binary classification data set banana-bc. During'
echo 'training, a five fold cross validation is performed on a 10x10 grid of'
echo 'hyper-parameters (regularization parameter and width of the Gaussian kernel).'
echo  'Then, the best hyper-parameter pair is determined for each fold by svm-select.'
echo 'Finally, svm-test creates a weighted average of the corresponding five decision'
echo 'functions and evaluates them on a test set.'
echo 'Of course, the cross-validation, as well as the behavior of svm-select and'
echo 'svm-test can be modifi ed in various ways ...'

./auxiliary/run-command-and-wait.sh './mc-svm.sh banana-bc'





printf "\033c"
echo
echo 'The next run is on the multiclass data set banana-mc. Without any parameters'
echo 'the All-versus-All strategy is performed. A five-fold cross validation on a'
echo '10 x 0 grid is performed for each pairing. The reported error of the first'
echo 'task is the overall-multiclass classification error, the remaining errors are'
echo 'the binary classification errors for the different pairings.' 
echo 'The error reported for task 0 will be the overall test error.'


./auxiliary/run-command-and-wait.sh './mc-svm.sh banana-mc'



printf "\033c"
echo
echo "Now let us use the One-versus-All strategy with the same setup. Unfortunately"
echo "this may take a litte longer."
echo 'The error reported for task 0 will be the overall test error.'


./auxiliary/run-command-and-wait.sh './mc-svm.sh banana-mc 1 0 0 0 0 1 1'


printf "\033c"
echo
echo "Now let us suppose we want to do binary classification but need to put a"
echo "constraint on one type of error. This can be achieved by either estimating"
echo "the conditional probability of positive samples using the least-squares-"
echo "solver or by working a couple of weighted binary classification problems."
echo "For the latter approach, which is usually a bit more robust, the package"
echo "has its own script, which in its default mode offers five different false"
echo "alarm rates around 0.05."


./auxiliary/run-command-and-wait.sh './npl-svm.sh banana-bc'



printf "\033c"
echo
echo "Alternatively, one can explore the ROC curve, by either a least square"
echo "approach or again by a couple of different weighted binary classification"
echo "problems. The latter approach has its own script again."



./auxiliary/run-command-and-wait.sh './roc-svm.sh banana-bc'



printf "\033c"
echo
echo "Now let us turn to regression. The first script in this category does least"
echo "squares regression."


./auxiliary/run-command-and-wait.sh './ls-svm.sh reg-1d'


printf "\033c"
echo
echo "Quantile regression is also contained in the package. The next example estimates"
echo "the tau-quantiles for tau = 0.05, 0.1, 0.5, 0.9, 0.95."


./auxiliary/run-command-and-wait.sh './qt-svm.sh reg-1d'




printf "\033c"
echo
echo "Now, let's have a look at the interface of the scripts we called so far. By typing"
echo "  <script-name>"
echo "you get a description of the interface. To illustrate this for the classification"
read -p "script mc-svm.sh, press any key." -n1 -s

./mc-svm.sh

read -p "Press any key to continue ..." -n1 -s

printf "\033c"
echo
echo "You may have noticed that the scripts call three different programs for training,"
echo "parameter selection, and testing. Their names are:"
echo "  svm-train"
echo "  svm-select"
echo "  svm-test"
echo "Again, these programs have an online help function that can be reached by typing"
echo "  <command-name>"
echo "if you are in the ./bin folder or have added this folder to your PATH-variable."
read -p "To illustrate this for svm-train, press any key." -n1 -s
echo

../bin/svm-train

read -p "Press any key to continue ..." -n1 -s

printf "\033c"
echo
echo "Ok, this only gives you an overview. To get a detailed help for a specific option"
echo "you need to type"
echo "  <command-name> -<option>"
read -p "To illustrate this for svm-train -P, press any key." -n1 -s
echo

../bin/svm-train -P

read -p "Press any key to continue ..." -n1 -s


printf "\033c"
echo
echo "This option has actually been chosen on purpose, as it makes it possible to deal"
echo "with large data sets. To illustrate this download the larger covtype datasets from" 
echo "http://www.isa.uni-stuttgart.de/software/ into the ./data folder and unzip them"
echo "there. Then switch to the ./scripts folder and type, for example,"
echo "  ./mc-svm.sh covtype.35000 1 0 4"
echo "Of course, 35.000 samples is not really big so you can try the full covtype data"
echo "set, which contains about 523.000 samples and can be found on the web page, too."  
echo "Finally, if you have a machine with 16GB RAM, you can compare the training time"
echo "to the training without splitting by typing"
echo "  ./mc-svm.sh covtype.35000"
echo
echo "Have fun ..."
echo

cd ..
