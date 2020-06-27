% Loading and splitting data in train, validation and test

maskds = imageDatastore('Mask_Dataset',"IncludeSubfolders",true,"LabelSource",'foldernames')
[maskTrain, maskVal, maskTest] = splitEachLabel(maskds,0.7,0.1,0.2,'randomized');

% Load AlexNet & Data augmentation for AlexNet 

net = alexnet
augmenter = imageDataAugmenter( ...
    'RandRotation',[0 360], ...
    'RandScale',[0.5 1], ...
    'RandXShear',[0.5 1], ...
    'RandYShear', [1 1.5])

[maskTest_aug] = augmentedImageDatastore([227 227 3],maskTest,'ColorPreprocessing','gray2rgb')  
[maskVal_aug] = augmentedImageDatastore([227 227 3],maskVal,'ColorPreprocessing','gray2rgb')
[maskTrain_aug] = augmentedImageDatastore([227 227 3],maskTrain,'DataAugmentation',augmenter','ColorPreprocessing','gray2rgb')

% Extracting Layers & modifying them for transfer learning

ly = net.Layers
ly(23) = fullyConnectedLayer(2)
ly(25) = classificationLayer

% Training options

options = trainingOptions('adam','MaxEpochs',40,'ValidationData',maskVal_aug,'ValidationFrequency',1,'ValidationPatience',15,"Plots","training-progress",'InitialLearnRate',0.0001)
% Training modified network with options  and saving it

[masknet, info] = trainNetwork(maskTrain_aug,ly,options)
save masknet_alexnet_1

% classifying test images 

% Uncomment to load after training 
% load gearnet_001_adam_default.mat gearnet

%classifying test images with trained network 

[preds, scores] = classify(masknet,maskTest_aug)

% Training accuracy 

truetest = maskTest.Labels;
nnz(preds == truetest)/numel(preds)
plot(info.TrainingLoss)

% confusion matrix

confusionchart(truetest,preds)

                           

