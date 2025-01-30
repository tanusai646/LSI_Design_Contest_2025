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

%% 初期データの取得
% 元画像の情報取得
% テストで使用する画像を挿入
New64_1(:,:,1) = double(im2gray(imread("block_test401.bmp")))/255.0;
%New64_1(:,:,1) = double(im2gray(imread("block1.bmp")))/255.0;

figure()
imshow(New64_1(:,:,1), "InitialMagnification", "fit"); %サイズ変更


%% テストデータを縦ベクトルにし，LabelDataに格納
% 横方向に順に並べた縦ベクトルを作るには、'をつけておく。
%　Teacher data of 〇　（新しいソールの教師データ）
TestData(:,1) = reshape(New64_1(:,:,1)', imagex*imagey,1);
LabelData(:,1) = TestData(:,1); 

load("vae_model.mat");

X = TestData;
[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork2_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3);


mse = sum((X-a3).^2) / (16*16);
psnr = 10 * log10(1^2 / mse);
figure(101);
imshow(reshape(a3(:,1), [16 16])', "InitialMagnification", "fit");
writematrix(New64_1, "X.csv");


