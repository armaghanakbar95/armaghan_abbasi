function [T_Precision, T_Recall, T_F1] = Threshold_Detection(vidObj, gtData)
    nFrames = vidObj.NumFrames;
    threshold = 0.5;
    % Initialize arrays to save values for each frame
    Array_precision = zeros(nFrames, 1);
    Array_recall = zeros(nFrames, 1);
    Array_f1 = zeros(nFrames, 1);
    nhood = 5;
    SE = strel('square',nhood);
    hFig = figure;
    set(hFig, 'Name', 'Threshold Detection', 'NumberTitle', 'off');
    for n = 1:vidObj.NumFrames
        frame = read(vidObj, n);  % Read the nth frame

        % Get ground truth for the current frame
        gt = gtData{n};

        grayFrame = rgb2gray(frame); %converting RGB to Gray Image

        %%Pixels > 170 → 1 (foreground/possible vehicle)
        %%Pixels ≤ 170 → 0 (background)

        binaryFrame = grayFrame >170; % 
        %Applying Morphological Operations
        binaryFrame = imclose(binaryFrame,SE);
        binaryFrame = imopen(binaryFrame,SE);
       
        binaryFrame = imdilate(binaryFrame,SE);
       
        %imshow(binaryFrame);
        
       
        CC = bwconncomp(binaryFrame);  % Find connected components
        
        %Initialize an array to store bounding boxes for each frame
        bboxes = [];
        
        % calculate its bounding box
        for i = 1:CC.NumObjects
            % Get the pixel indices of the current connected component
            pixelIndices = CC.PixelIdxList{i};
            
            % Convert linear indices to row and column indices
            [rows, cols] = ind2sub(size(binaryFrame), pixelIndices);
            
            %bounding box of the current Frame
            minRow = min(rows);      % Minimum row top
            maxRow = max(rows);      % Maximum row bottom
            minCol = min(cols);      % Minimum column left side
            maxCol = max(cols);      % Maximum column right side
            
            % Store the bounding box (minCol, minRow, width, height)
            width = maxCol - minCol + 1;
            height = maxRow - minRow + 1;
            bboxes = [bboxes; minCol, minRow, width, height];
            size(bboxes);
        end
        
        % Visualize the bounding boxes on the frame
        labels = 'Car';
        detectedImg = frame;  % Start with the original frame
        
        for i = 1:size(bboxes, 1)
            bbox = bboxes(i, :);  % Get the current bounding box
            detectedImg = insertObjectAnnotation(detectedImg, 'Rectangle', bbox, labels);  % Draw the bounding box on the frame
        end

        % Display the result
        imshow(detectedImg);
        % Calculating Precision and recall By using built in function
        if isempty(bboxes)
            Tprecision = 0;
            Trecall = 0;
        else
            [Tprecision,Trecall] = bboxPrecisionRecall(bboxes,gt, threshold);
        end

        %Calculating Values of F1 For Each Frame

        if Tprecision==0 && Trecall == 0
            Tf1 = 0;
        else 
            Tf1 = 2 * (Tprecision * Trecall) / (Tprecision + Trecall);
        end

        % Saving values By each Frame
        Array_precision(n) = Tprecision;
        Array_recall(n) = Trecall;
        Array_f1(n) = Tf1;
   
    end
    %return Arrays
    T_Precision = Array_precision;
    T_Recall = Array_recall;
    T_F1 = Array_f1;
end
