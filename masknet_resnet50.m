% Loading and splitting data in train, validation and test

maskds = imageDatastore('Mask_Dataset',"IncludeSubfolders",true,"LabelSource",'foldernames')
[maskTrain, maskVal, maskTest] = splitEachLabel(maskds,0.7,0.1,0.2,'randomized');

%Data augmentation 

augmenter = imageDataAugmenter( ...
    'RandRotation',[0 360], ...
    'RandScale',[0.5 1], ...
    'RandXShear',[0.5 1], ...
    'RandYShear', [1 1.5])

[maskTest_aug] = augmentedImageDatastore([224 224 3],maskTest,'DataAugmentation',augmenter','ColorPreprocessing','gray2rgb')  
[maskVal_aug] = augmentedImageDatastore([224 224 3],maskVal,'ColorPreprocessing','gray2rgb')
[maskTrain_aug] = augmentedImageDatastore([224 224 3],maskTrain,'DataAugmentation',augmenter','ColorPreprocessing','gray2rgb')


% Load DAGnet 
dagnet = resnet50
ly = dagnet.Layers;

% Get layer graph of dagnet

lgraph = layerGraph(dagnet)

%Create a new fully connected layer and classification layer

newfc = fullyConnectedLayer(2,'Name','fc')
newcl = classificationLayer("Name",'gear')

%Replace fc layer and classification layer in lgraph

lgraph = replaceLayer(lgraph,'fc1000',newfc)
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newcl)

% set training options, train resnet and save 

options = trainingOptions('rmsprop','MiniBatchSize',20,'MaxEpochs',5,'ValidationData',maskVal_aug,'ValidationFrequency',1,'ValidationPatience',7,"Plots","training-progress",'InitialLearnRate',0.0001)
[masknet_dag,info] = trainNetwork(maskTrain_aug,lgraph,options);
save masknet_dag_rmsprop

%classifying test images 

% Uncomment to load after training 
% load masknetnet_dag_rmsprop.mat masknet_dag

% classifying test images with trained network 

[preds, scores] = classify(masknet_dag,maskTest_aug)

% Training accuracy 

truetest = maskTest.Labels;
nnz(preds == truetest)/numel(preds)
plot(info.TrainingLoss)

% confusion matrix

confusionchart(truetest,preds)

test_img = imread(fullfile('new_test/test_img_1.jpg'));
imshow(test_img);
test_imds = imageDatastore('new_test','FileExtensions',{'.jpg','.tif'})
img = readimage(test_imds, 1);
aug_test_imds = augmentedImageDatastore([224 224 3],test_imds,'DataAugmentation',augmenter','ColorPreprocessing','gray2rgb')  

[preds_new, scores_new] = classify(masknet_dag,aug_test_imds)

