% OX (Circle or Cross) Judgement Program (Variational Auto Encoder) 
% LSI Design Contest in Okinawa 2024
%
% OX_judge_VAE
%
% Requrired : OX_judge_VAE.m, Neuralnetwork_VAE.m,
% Neuralnetwork_forward_VAE.m, Neuralnetwork_generate_VAE.m
%
% see also OX_judge_VAE, Neuralnetwork_VAE, Neuralnetwork_forward_VAE, Neralnetwork_generate_VAE.
%
clear;
close all;

rng(2024);          % Set random seed.

MARU = [1 1 1;
        1 0 1;
        1 1 1];   % 〇 Circle (Maru in Japanese)
BATU = [1 0 1;
        0 1 0;
        1 0 1];   % × Cross (Batu in Japanese)


New64 = double(im2gray(imread('new_64.png')))/255.0;
figure(100); subplot(2,1,1);imshow(New64);
Mamou64 = double(im2gray(imread('mamou_64.png')))/255.0;
subplot(2,1,2);imshow(Mamou64);
%
imagex = 64; imagey=64;

% 横方向に順に並べた縦ベクトルを作るには、'をつけておく。
%　Teacher data of 〇　（新しいソールの教師データ）
TrainData(:,1) = reshape(New64', imagex*imagey,1);
LabelData(:,1) = TrainData(:,1);   % Correct answer when input is 〇（入力が〇の時の正解）

% Teacher data of ×（×の教師データ）
%TrainData(:,2) = reshape([1 0 1;0 1 0;1 0 1]', 9,1);
%LabelData(:,2) = TrainData(:,2);   % Correct answer when input is ×　（入力が×の時の正解）

% Create test data
% Maru and Maru + 1bit error
TestData(:,1) = reshape(New64', imagex*imagey,1);
TestData(:,2) = reshape(Mamou64', imagex*imagey,1);
% TestData(:,3) = reshape([1 0 1;1 0 1;1 1 1]', 9,1);
% TestData(:,4) = reshape([1 1 0;1 0 1;1 1 1]', 9,1);
% TestData(:,5) = reshape([1 1 1;0 0 1;1 1 1]', 9,1);
% TestData(:,6) = reshape([1 1 1;1 1 1;1 1 1]', 9,1);
% TestData(:,7) = reshape([1 1 1;1 0 0;1 1 1]', 9,1);
% % Batsu and Batsu + 1bit error
% TestData(:,8) = reshape([1 0 1;0 1 0;1 0 1]', 9,1);
% TestData(:,9) = reshape([0 0 1;0 1 0;1 0 1]', 9,1);
% TestData(:,10) = reshape([1 1 1;0 1 0;1 0 1]', 9,1);
% TestData(:,11) = reshape([1 0 0;0 1 0;1 0 1]', 9,1);
% TestData(:,12) = reshape([1 0 1;1 1 0;1 0 1]', 9,1);
% TestData(:,13) = reshape([1 0 1;0 0 0;1 0 1]', 9,1);
% TestData(:,14) = reshape([1 0 1;0 1 1;1 0 1]', 9,1);

for i =1:2
    TLabelData(:,i) = TestData(:,i);
end
% for i =8:14
%     TLabelData(:,i) = TestDat(:,2);
% end

% Parameter setting

Layer1 = imagex*imagey;                     % 入力層のユニット数
Layer2 = 128;                     % 中間層（隠れ層，AEの出力層）のユニット数
Layer3 = Layer1;                % 復元層（出力層）のユニット数

L2func = 'ReLUfnc';             % 中間層のアルゴリズム（'Sigmoid' or Default: 'ReLUfnc' 文字数は等しくないとエラーを起こす）
L3func = 'Sigmoid_BCE';         % 復元層のアルゴリズムと誤差（'Sigmoid_MSE' or Default : 'Sigmoid_BCE' (Binary Cross Entropy)）
         
%% 中間層と出力層の重みの初期値
% Initialization values of weights (Hidden layer and Output layer)

w12_mean = randn(Layer2,Layer1);	% 教師データ用の行列(中間層1の重み)	 hidden layer's weight matrix for supervisor data
w12_var = randn(Layer2,Layer1);	% 教師データ用の行列(中間層1の重み)	 hidden layer's weight matrix for supervisor data
w23 = randn(Layer3,Layer2);	% 教師データ用の行列(復元層２の重み)	hidden layer's weight matrix for supervisor data

% 中間層と出力層のバイアスの初期値
% Initialization values of bias (Hidden layer and Output layer) 
b2_mean = randn(Layer2,1);
b2_var = randn(Layer2,1);
b3 = randn(Layer3,1);

% Step size
eta = 0.0001;		%学習率が高すぎると更新した係数が大きくなりすぎてコストが減らなくなる	
				%Learning rate. If the learning rate is too high, the updated coefficient becomes too large and the cost may not decrease

epoch = 10000;  
%epoch = 1000000;  

%%
% 初期状態における a3 の出力
% Forward Process (Initial weight)
X = TestData;
t = TLabelData;

[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3);

fprintf('Initial Weight\n');
% fprintf('w12_mean\n');   disp(w12_mean);
% fprintf('w12_var\n');   disp(w12_var);
% fprintf('b2_mean\n');    disp(b2_mean);
% fprintf('b2_var');    disp(b2_var);
% fprintf('w23\n');   disp(w23);
% fprintf('b3\n');    disp(b3);

fprintf('Initial Weight Test\n');
% fprintf('Test data X \n');   disp(X);
% fprintf('z2_mean\n');    disp(z2_mean);
% fprintf('a2_mean\n');    disp(a2_mean);
% fprintf('z2_var\n');    disp(z2_var);
% fprintf('a2_var\n');    disp(a2_var);
% fprintf('z\n');    disp(z);
% fprintf('z3\n');    disp(z3);
% fprintf('a3\n');    disp(a3);

figure(101);
for j = 1:size(a3,2)
    subplot(size(a3,2),1,j);
    imshow(reshape(a3(:,j), [64 64])');
end

% 学習前の初期ウェイトと初期バイアスにおけるテストにおける中間層の分布のグラフ表示
% figure(1);
% hold on;
% for i=1:7
%     plot(z(1,i), z(2,i),'or');
% end
% for i=8:14
%     plot(z(1,i), z(2,i),'xk');
% end
% hold off;
% xlabel('y_1 = z_1'); ylabel('y_2 = z_2');
% title('Latent Variable (Initial weights and bias)');
% box('on');

%% AE の学習
tic;
X = TrainData;
t = LabelData;
[w12_mean,w12_var,w23,b2_mean,b2_var,b3,w12_mean_t,w12_var_t,w23_t,b2_mean_t,b2_var_t,b3_t,C] = Neuralnetwork_VAE(X,t,w12_mean,w12_var,w23,b2_mean,b2_var,b3,eta,epoch,L2func,L3func);
toc

%% 学習後のテスト
X = TestData;

[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3);

% 値の表示
fprintf('Final Weight\n');
% fprintf('w12_mean\n');   disp(w12_mean);
% fprintf('b2_mean\n');    disp(b2_mean);
% fprintf('w12_var\n');   disp(w12_var);
% fprintf('b2_var\n');    disp(b2_var);
% fprintf('w23\n');   disp(w23);
% fprintf('b3\n');    disp(b3);

fprintf('Final Weight Test\n');
% fprintf('Test data X \n');   disp(X);
% fprintf('a2_mean\n');    disp(a2_mean);
% fprintf('a2_var\n');    disp(a2_var);
% fprintf('z\n');    disp(z);
% fprintf('z3\n');    disp(z3);
% fprintf('a3\n');    disp(a3);


figure(102);
for j = 1:size(a3,2)
    subplot(size(a3,2),1,j);
    imshow(reshape(a3(:,j), [64 64])');
end


% 学習後のテスト入力における中間層の分布のグラフ表示
% figure(2);
% hold on;
% for i=1:7
%     plot(z(1,i), z(2,i),'or');
% end
% for i=8:14
%     plot(z(1,i), z(2,i),'xk');
% end
% hold off;
% xlabel('y_1 = z_1'); ylabel('y_2 = z_2');
% title('Latent Variable (Final weights and bias)');
% % xlim([0 ceil(max(a2(1,:))/10)*10]);
% % ylim([0 ceil(max(a2(2,:))/10)*10]);
% box('on');
% 
% figure(1);
% xlim([0 ceil(max(a2(1,:))/10)*10]);
% ylim([0 ceil(max(a2(2,:))/10)*10]);

% 学習過程のグラフ表示（各エポックごとのの誤差関数の値）
figure(3);
plot(C);
xlabel('Epoch'); ylabel('Error');

% z = [0 1 3 4 5 6 7 8];
% [z3,a3] = Neuralnetwork_generate_VAE(z, w23,b3)
% fprintf('Generation \n');
% fprintf('z\n');    disp(z);
% fprintf('z3\n');    disp(z3);
% fprintf('a3\n');    disp(a3);

