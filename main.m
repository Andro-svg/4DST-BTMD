 close all;
clear all;
clc
clear
%% parameter setting
format long
addpath('utils/');
addpath('metric_utils\');
addpath('tensor_toolbox\');

frame = 30;
patchSize=60;
lambdaL=1.188;
mu = 0.00044;

readPath = '.\dataset';
savePath = '.\result';  

if ~exist(savePath)
    mkdir(savePath);
end
tuneopts.temporal_step = frame;
tuneopts.patchSize = patchSize;
tuneopts.lambdaL = lambdaL;
tuneopts.mu=mu; 

target_detection(char(readPath), savePath, tuneopts);                  
