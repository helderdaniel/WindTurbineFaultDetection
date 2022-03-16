%Use a non function script to avoid passing by
%copy large variables

%config

%assumes features and labels are the input variable names
%labels are strings

trainSize = 0.9;

labelsint  = uint8(str2num(labels));
labelsuni  = unique(labelsint);
numClasses = numel(labelsuni);

samples = uint32(size(labels)(1) / numClasses);
splitPoint = uint32(samples*trainSize);

features_training = [];
labels_training   = [];
features_test     = [];
labels_test       = [];

for i = 0:numClasses-1
    i
    
    %get class
    st = i*samples + 1;
    
    %shuffle each class
    p = randperm (samples)-1;
    X = features (st+p, :);
    Y = labelsint(st+p, :);
    
    %add shuffled class to train and test sets
    features_training = [features_training; X(1:splitPoint, :)];
    labels_training   = [labels_training;   Y(1:splitPoint, :)];
    features_test     = [features_test;     X(splitPoint+1:samples, :)];
    labels_test       = [labels_test;       Y(splitPoint+1:samples, :)];
endfor

%save('-binary', '-zip', 't.mat', 'features_training', 'labels_training', 'features_test', 'labels_test' )
