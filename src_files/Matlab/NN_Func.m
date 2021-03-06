% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created Thu Feb 19 14:09:38 CET 2015
%
% This script assumes these variables are defined:
%
%   features - input data.
%   targets - target data.

load features_func
load targets_func

display('NN_func');

[x, PS] = mapstd(features_func');
t = targets_func';

load test_features_func.csv
load test_targets_func.csv
in = mapstd('apply', test_features_func', PS);
out = test_targets_func';
percentErrors = 0;
testErrors = 0;

for i = 1:30          
% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
% net = patternnet(hiddenSizes, trainFcn)


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = percentErrors + sum(tind ~= yind)/numel(tind);
performance = perform(net,t,y);

pred = net(in);
out_ind = vec2ind(out);
pred_ind = vec2ind(pred);
testErrors = testErrors + sum(out_ind ~= pred_ind)/numel(out_ind);
end

precentErrors = percentErrors/30;
testErrors = testErrors/30;
display(percentErrors);
display(testErrors);

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)
%figure, ploterrhist(e)


% load test_features_func.csv
% load test_targets_func.csv
% in = mapstd('apply', test_features_func', PS);
% out = test_targets_func';
% pred = net(in);
% plotconfusion(out, pred)


