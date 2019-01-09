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

context("liquidSVM-liquidData")

COVERAGE <- TRUE

test_that("print liquidData",{
  tt <- liquidData('banana-bc')
  expect_output(print(tt))
  tt <- liquidData('reg-1d')
  expect_output(print(tt))
})

test_that("list liquidData",{
  datas <- liquidData(loc=system.file('data',package='liquidSVM'))
  expect_true(all(c("banana-bc", "banana-mc", "reg-1d") %in% datas))
})

test_that("list liquidData http",{
  skip("Web Server www.isa.uni-stuttgart.de was redesigned and data sets are not available anymore.")
  thenames <- c("banana-bc", "banana-mc", "reg-1d", 'covtype.1000','gisette')
  mockHtml <- paste0('<a href="',thenames,'.train.csv">')
  with_mock(
    readLines = function(f) return(mockHtml),
    datas <- liquidData(loc='http://www.isa.uni-stuttgart.de/liquidData')
  )
  expect_true(all(thenames %in% datas))
})

test_that("liquidData sampling",{
  tt <- liquidData('reg-1d')
  expect_warning(tt2 <- sample.liquidData(tt, trainSize=1e6))
  expect_equal(nrow(tt2$train), nrow(tt$train))
  tt3 <- sample.liquidData(tt, trainSize=Inf)
  expect_equal(nrow(tt3$train), nrow(tt$train))
})

test_that("write liquidData",{
  skip_on_cran()
  
  testit <- function(name, dir, type, ext=type, tt=liquidData(name)){
    trainName <- paste0(dir,'/',name,'.train.',ext)
    testName <- paste0(dir,'/',name,'.test.',ext)
    unlink(trainName)
    unlink(testName)
    
    write.liquidData(tt, loc=dir, type=type)
    
    if(type != "GURLS"){
      expect_true(file.exists(trainName))
      expect_true(file.exists(testName))
      
      if(type == 'csv' && missing(tt)){
        tt2 <- liquidData(name, loc=dir)
        expect_equal(tt2$name, tt$name)
        expect_equivalent(tt2$train, tt$train)
        expect_equivalent(tt2$test, tt$test)
      }
      
      unlink(trainName)
      unlink(testName)
    }else{
      dirName <- paste0(dir,'/',name)
      expect_true(dir.exists(dirName))
      unlink(dirName, recursive=TRUE)
    }
  }
  testit('reg-1d', '.', 'csv')
  testit('iris', '.', 'csv', tt=ttsplit(iris))
  testit('banana-bc', '.', 'csv')
  testit('banana-bc', '.', 'lsv')
  if(COVERAGE){
    testit('banana-bc', '.', 'nla')
    testit('banana-bc', '.', 'uci')
    testit('banana-bc', '.', 'GURLS')
    testit('banana-bc', '.', 'm-svm', 'txt')
  }
  
})


