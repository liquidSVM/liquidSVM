# Copyright 2015-2017 Philipp Thomann
# 
# This file is part of liquidSVM.
# 
#  liquidSVM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# liquidSVM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.
# 

require(liquidSVM)

context("liquidSVM-bugs")

orig <- options(liquidSVM.warn.suboptimal=FALSE, threads=1)[[1]]

test_that("trees bug",{
  set.seed(123)
  
  ### seems to be fixed!
  expect_true(!is.null(s_trees <- svm(Height ~ ., trees)))
})

options(liquidSVM.warn.suboptimal=orig)

