function [Y_Precision, Y_Recall, Y_F1 ]= Yolo4_detection(vidObj, gtData)
    % Load pre-trained YOLOv4 model
    detector = yolov4ObjectDetector('tiny-yolov4-coco');
    

    nFrames = vidObj.NumFrames;
    Array_precision = zeros(nFrames, 1);
    Array_recall = zeros(nFrames, 1);
    Array_f1 = zeros(nFrames, 1);

    hFig = figure;
    set(hFig, 'Name', 'Yolo4_detection', 'NumberTitle', 'off');

    for n = 1:vidObj.NumFrames
        frame = read(vidObj, n);  % Read the nth frame
    
        % Get ground truth for the current frame
        gt = gtData{n};
    
        % Perform detection on the input frame
        [bboxes,scores,labels] = detect(detector,frame,Threshold=0.1);
        labels='Car';
        detectedImg = insertObjectAnnotation(frame,"Rectangle",bboxes,labels);
        imshow(detectedImg)

        threshold = 0.5;
        [Yprecision,Yrecall] = bboxPrecisionRecall(bboxes,gt, threshold);

        %Calculating Values of F1 For Each Frame
        Yf1 = 2 * (Yprecision * Yrecall) / (Yprecision + Yrecall);

        % Saving values By each Frame
        Array_precision(n) = Yprecision;
        Array_recall(n) = Yrecall;
        Array_f1(n) = Yf1;
    end
    %return Arrays
    Y_Precision = Array_precision;
    Y_Recall = Array_recall;
    Y_F1 = Array_f1;
end