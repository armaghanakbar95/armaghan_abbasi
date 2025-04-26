clc;
clear;
close all;

% Load ground truth data
load('export_ground_truth.mat'); 

% Initialize ground truth structure
nFrames = numel(gTruth.LabelData.Car);  % Number of frames
gtData = cell(nFrames, 1);  % Cell array to store corrected ground truth

% Correct the ground truth data
for n = 1:nFrames
    gt = gTruth.LabelData.Car{n};
    if isa(gt, 'struct')
        gt = gTruth.LabelData.Car{n}.Position;  % Extract position if it's a struct
    end
    if isempty(gt)
    gt = zeros(0, 4);  % If empty, set as an empty matrix
    end
    gtData{n} = gt;  % Store the ground truth for the frame
end



% Main
videoFile = 'traffic.mj2';  
vidObj = VideoReader(videoFile);


%Call each Function which will return Precision, recall and F1 array we

%% Threshold Detector and Calculating Mean
[T_Precision, T_Recall, T_F1] = Threshold_Detection(vidObj, gtData);
Threshold_Precision = mean(T_Precision);
Threshold_Recall = mean(T_Recall);
Threshold_F1 = mean(T_F1);

disp("Threshold Precision: "+Threshold_Precision)
disp("Threshold Recall: "+ Threshold_Recall)
disp("Threshold F1 Value: "+ Threshold_F1)

close all;

%% YOLO Detector and Calculating Mean
[Y_Precision, Y_Recall, Y_F1 ]= Yolo4_detection(vidObj, gtData);
Yolo4_detection_Precision = mean(Y_Precision);
Yolo4_detection_Recall = mean(Y_Recall);
Yolo4_detection_F1 = mean(Y_F1);

disp("Yolo4_detection Precision: " + Yolo4_detection_Precision)
disp("Yolo4_detection Recall: " + Yolo4_detection_Recall)
disp("Yolo4_detection F1 Value: " + Yolo4_detection_F1)

close all;

%% Gaussian_mixture Detector and Calculating Mean
[G_Precision, G_Recall, G_F1 ]= Gaussian_mixture(vidObj, gtData);
Gaussian_mixture_Precision = mean(G_Precision);
Gaussian_mixture_Recall = mean(G_Recall);
Gaussian_mixture_F1 = mean(G_F1);

disp("Gaussian_mixture Precision: " + Gaussian_mixture_Precision)
disp("Gaussian_mixture Recall: " + Gaussian_mixture_Recall)
disp("Gaussian_mixture F1 Value: " + Gaussian_mixture_F1)

close all;

%% ACF Detector and Calculating Mean
[C_Precision, C_Recall, C_F1 ]= other_detector(vidObj, gtData);

ACF_Precision = mean(C_Precision);
ACF_Recall = mean(C_Recall);
ACF_F1 = mean(C_F1);

disp("ACF Precision: "+ACF_Precision)
disp("ACF Recall: "+ ACF_Recall)
disp("ACF F1 Value: "+ ACF_F1)
close all;

%% Creating Bar figures for comparison
detectors = {'Threshold', 'Gaussian Mixture','YOLOv4'};

precision = [Threshold_Precision, Gaussian_mixture_Precision, Yolo4_detection_Precision];
recall = [Threshold_Recall, Gaussian_mixture_Recall, Yolo4_detection_Recall];
f1_value  = [Threshold_F1, Gaussian_mixture_F1, Yolo4_detection_F1];

data = [precision; recall; f1_value];

figure;
hold on;
bar(data', 'grouped');
set(gca, 'XTickLabel', detectors);

% Add labels and title
xlabel('Detection Methods');
ylabel('Scores');
title('Comparison of Detection Methods');
legend({'Precision', 'Recall', 'F1 Value'}, 'Location', 'Best');