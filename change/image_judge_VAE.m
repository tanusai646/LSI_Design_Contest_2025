% OX (Circle or Cross) Judgement Program (Variational Auto Encoder) 
% LSI Design Contest 2025
%
% OX_judge_VAE
%
% Requrired : OX_judge_VAE.m, Neuralnetwork_VAE.m, Neuralnetwork_forward_VAE.m
%
% see also OX_judge_VAE, Neuralnetwork_VAE, Neuralnetwork_forward_VAE.

clear;
close all;
clc;

rng(2025);        % Set random seed.

TestCount = 64;

image_name = "input.bmp";

[CorrectData, LabelData, TestData, count] = image_to_correct(image_name, TestCount);

figure(10);
imshow(reshape(TestData(:,1),8,8), 'InitialMagnification','fit');

for i = 1:TestCount
    TLabelData(:,i) = LabelData(:,i);
end

% Parameter setting
Layer1 = 64;                     % Number of input layer units
Layer2 = 2;                     % Number of hidden layer units
Layer3 = Layer1;                % Number of output layer units

L2func = 'ReLUfnc';             % Algorithm of hidden layer ('Sigmoid' or Default: 'ReLUfnc')
L3func = 'Sigmoid_BCE';         % Algorithm of output layer and error ('Sigmoid_MSE' or Default : 'Sigmoid_BCE' (Binary Cross Entropy))
         
%% 中間層と出力層の重みとバイアスの初期値
% Initialization values of weights (Hidden layer and Output layer)
% Initialization values of bias (Hidden layer and Output layer) 

w2_mean = rand(Layer2,Layer1);	% Hidden layer's weight matrix for supervisor data
w2_var = rand(Layer2,Layer1);	% Hidden layer's weight matrix for supervisor data
w3 = rand(Layer3,Layer2);       % Output layer's weight matrix for supervisor data

b2_mean = (-0.5)*ones(Layer2,1);
b2_var = (-0.5)*ones(Layer2,1);
b3 = (-0.5)*ones(Layer3,1);

% 学習率
% Learning rate
eta = 0.001;		%学習率が高すぎると更新した係数が大きくなりすぎてコストが減らなくなる	
                    %If the learning rate is too high, the updated coefficient becomes too large and the cost may not decrease

epoch = 5000;

%% Pre-learning Test
X = TestData;

[z2_mean,z2_var, a2_mean, a2_var,a2,z3,a3] = Neuralnetwork_forward_VAE(X,w2_mean,w2_var,w3,b2_mean,b2_var,b3);

fprintf('Initial Weight\n');
fprintf('w2_mean\n');   disp(w2_mean);
fprintf('w2_var\n');    disp(w2_var);
fprintf('b2_mean\n');   disp(b2_mean);
fprintf('b2_var\n');    disp(b2_var);
fprintf('w3\n');        disp(w3);
fprintf('b3\n');        disp(b3);

fprintf('Initial Weight Test\n');
fprintf('Test data X\n');   disp(X);
fprintf('z2_mean\n');   disp(z2_mean);
fprintf('a2_mean\n');   disp(a2_mean);
fprintf('z2_var\n');    disp(z2_var);
fprintf('a2_var\n');    disp(a2_var);
fprintf('a2\n');        disp(a2);
fprintf('z3\n');        disp(z3);
fprintf('a3\n');        disp(a3);


% 学習前のテストにおける中間層の分布のグラフ表示
% Test before learning and graph the distribution of hidden layer.
figure(1);
hold on;
for i = 1:TestCount
    plot(a2(1,i), a2(2,i),'or');
end

% 学習前を画像として表示
figure(13);
imshow(reshape(a3(:,1),8,8), 'InitialMagnification','fit');

%% VAE Learning
X = TestData;
t = TLabelData;

[w2_mean,w2_var,w3,b2_mean,b2_var,b3,w2_mean_t,w2_var_t,w3_t,b2_mean_t,b2_var_t,b3_t,E] = Neuralnetwork_VAE(X,t,w2_mean,w2_var,w3,b2_mean,b2_var,b3,eta,epoch,L2func,L3func);

%% Post-learning Test
X = TestData;

[z2_mean,z2_var, a2_mean, a2_var,a2,z3,a3] = Neuralnetwork_forward_VAE(X,w2_mean,w2_var,w3,b2_mean,b2_var,b3);

fprintf('Final Weight\n');
fprintf('w2_mean\n');   disp(w2_mean);
fprintf('b2_mean\n');   disp(b2_mean);
fprintf('w2_var\n');    disp(w2_var);
fprintf('b2_var\n');    disp(b2_var);
fprintf('w3\n');        disp(w3);
fprintf('b3\n');        disp(b3);

fprintf('Final Weight Test\n');
fprintf('Test data X\n');   disp(X);
fprintf('a2_mean\n');   disp(a2_mean);
fprintf('a2_var\n');    disp(a2_var);
fprintf('a2\n');        disp(a2);
fprintf('z3\n');        disp(z3);
fprintf('a3\n');        disp(a3);

% 学習後のテスト入力における中間層の分布のグラフ表示
% Test input after the study and display a graph of the distribution of hidden layer.
figure(2);
hold on;
for i = 1:TestCount
    plot(a2(1,i), a2(2,i),'or');
end


hold off;
xlabel('y_1 = a^2_1'); ylabel('y_2 = a^2_2');
title('Latent Variable (Final weights and bias)');
box('on');



% 学習後を画像として表示
figure(14);
imshow(reshape(a3(:,1),8,8), 'InitialMagnification','fit');

% 学習過程のグラフ表示（各エポックごとのの誤差関数の値）
% Graphically display the learning process (Error function values for each epoch)
figure(3);
plot(E);
xlabel('Epoch'); ylabel('Error');

% 学習後の潜在空間を使用して画像を出力
reput_image(a3, TestCount);
