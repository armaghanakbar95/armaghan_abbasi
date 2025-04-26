function [G_Precision, G_Recall, G_F1] = Gaussian_mixture(vidObj, gtData)
detector = vision.ForegroundDetector(NumTrainingFrames=5,InitialVariance=30*30);

blob = vision.BlobAnalysis(...
       CentroidOutputPort=false,...
       AreaOutputPort=false, ...
       BoundingBoxOutputPort=true, ...
       MinimumBlobAreaSource="Property",...
       MinimumBlobArea=250);

nFrames = vidObj.NumFrames;
threshold = 0.5;
Array_precision = zeros(nFrames, 1);
Array_recall = zeros(nFrames, 1);
Array_f1 = zeros(nFrames, 1);

hFig = figure;
set(hFig, 'Name', 'Gaussian_mixture', 'NumberTitle', 'off');

for n = 1:vidObj.NumFrames
    frame = read(vidObj, n);  % Read the nth frame
    
    % Get ground truth for the current frame
    gt = gtData{n};

    fgMask=detector(frame);
    bboxes = blob(fgMask);
    labels='Car';
    detectedImg = insertObjectAnnotation(frame,"Rectangle",bboxes,labels);
    imshow(detectedImg)
    % claculate the values of Precision and Recall by using built-in
    % function For Each Frame

    [Gprecision,Grecall] = bboxPrecisionRecall(bboxes,gt, threshold);

    %Calculating Values of F1 For Each Frame
    Gf1 = 2 * (Gprecision * Grecall) / (Gprecision + Grecall);
    Gf1(isnan(Gf1))=0;

    % Saving values By each Frame
    Array_precision(n) = Gprecision;
    Array_recall(n) = Grecall;
    Array_f1(n) = Gf1;
end
    %return Arrays
    G_Precision = Array_precision;
    G_Recall = Array_recall;
    G_F1 = Array_f1;
end

