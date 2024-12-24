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

% 16×16の画像ブロックをVAEに通す

clear;
close all;

rng(2025);          % Set random seed.

%% 各種変数設定
% 入力画像数の設定
input_count = 1024;

% imageの画像の大きさ設定
imagex = 16; 
imagey=16;


% Step size
eta = 0.0001;   %学習率が高すぎると更新した係数が大きくなりすぎてコストが減らなくなる	
				%Learning rate. If the learning rate is too high, the updated coefficient becomes too large and the cost may not decreas
epoch = 80000;  %実行回数

% レイヤーの設定
Layer1 = imagex*imagey;                     % 入力層のユニット数
Layer2 = 32;                     % 中間層（隠れ層，AEの出力層）のユニット数
Layer3 = Layer1;                % 復元層（出力層）のユニット数

L2func = 'ReLUfnc';             % 中間層のアルゴリズム（'Sigmoid' or Default: 'ReLUfnc' 文字数は等しくないとエラーを起こす）
L3func = 'Sigmoid_BCE';         % 復元層のアルゴリズムと誤差（'Sigmoid_MSE' or Default : 'Sigmoid_BCE' (Binary Cross Entropy)）

%% 道路ブロックの情報取得
block_num = readtable("road_only_block2.xlsx");
block_num = table2array(block_num);
block_num_size = size(block_num,2);


%% 初期データの取得
for i = 1:block_num_size
    num = block_num(1,i);
    filename = sprintf("block_test%d.bmp", num);
    filepath = fullfile("Blocks_test", filename);
    New64(:,:,i) = double(im2gray(imread(filepath)))/255.0;
end

figure(100); 
%subplot(2,1,1);
%imshow(New64(:,:,1));
%subplot(2,1,2);imshow(New64(:,:,2));

for i = 1:2
    subplot(2,1,i);
    imshow(New64(:,:,i)); %サイズ変更
end



%% 教師データを縦ベクトルにし，LabelDataに格納
% 横方向に順に並べた縦ベクトルを作るには、'をつけておく。
%　Teacher data of 〇　（新しいソールの教師データ）
for i = 1:block_num_size
    TrainData(:,i) = reshape(New64(:,:,i)', imagex*imagey,1);
    LabelData(:,i) = TrainData(:,i); 
end

%% テストデータの入力
% テストデータとして教師データ及び，教師データの次のデータを入力
for i = 1:block_num_size
    TestData(:,i) = reshape(New64(:,:,i)', imagex*imagey,1);
    TLabelData(:,i) = TestData(:,i);
end

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

%%
% 初期状態における a3 の出力
% Forward Process (Initial weight)
X = TestData;
t = TLabelData;

[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork2_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3);

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
for j = 1:2
    subplot(2,1,j);
    imshow(reshape(a3(:,j), [16 16])');
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
[w12_mean,w12_var,w23,b2_mean,b2_var,b3,w12_mean_t,w12_var_t,w23_t,b2_mean_t,b2_var_t,b3_t,C] = Neuralnetwork2_VAE(X,t,w12_mean,w12_var,w23,b2_mean,b2_var,b3,eta,epoch,L2func,L3func);
toc

%% 学習後のテスト
X = TestData;

[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork2_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3);

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
for j = 1:2
    subplot(2,1,j);
    imshow(reshape(a3(:,j), [16 16])'); %サイズ変更
end


% 学習後のテスト入力における中間層の分布のグラフ表示
%figure(2);
%hold on;
%for i=1:input_count
%     plot3(z(1,i), z(2,i), z(3,i),'or');
%end
%xlabel('z1'); ylabel('z2'); zlabel('z3');
%grid on;

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

%% ブロック番号をエクセルに仮保存
writematrix(a3, "a3_output.xlsx");

