imageSize = [480 640 3];
numClasses = 5;
network = "resnet18";
net = deeplabv3plus(imageSize,numClasses,network, ...
             DownsamplingFactor=16);

analyzeNetwork(net)