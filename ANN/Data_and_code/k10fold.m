clc;
clear all;
load x.csv
load y.csv
load kcross.csv
indices = crossvalind('Kfold',kcross,10);
 % forming 10 cross validation.    
kfoldnumber = 1;
xxtrain = zeros(135,4); %train input
yytrain = zeros(135,3); %train output
xxtest = zeros(15,4); %test input
yytest = zeros(15,3); %test output
testindex = 1;
trainindex = 1;

for index=1:length(indices) %iteration for all K-folds
    if indices(index,1) == kfoldnumber
      xxtest(testindex,:) = x(index,1:4);
      yytest(testindex,:) = y(index,1:3);
      testindex = testindex + 1;
      continue;
    end
    xxtrain(trainindex,:) = x(index,1:4);
    yytrain(trainindex,:) = y(index,1:3);
    trainindex = trainindex + 1;
    
end
