% tb_forward_VAE_929
clear;
close all;

hdlsetuptoolpath('ToolName', 'Xilinx Vivado', 'ToolPath', 'C:\Xilinx\Vivado\2022.2\bin\vivado.bat');

rng(2024);          % Set random seed.

% 横方向に順に並べた縦ベクトルを作るには、'をつけておく。
%　Teacher data of 〇　（〇の教師データ）
TrainData(:,1) = reshape([1 1 1;1 0 1;1 1 1]', 9,1);
LabelData(:,1) = TrainData(:,1);   % Correct answer when input is 〇（入力が〇の時の正解）

% Teacher data of ×（×の教師データ）
TrainData(:,2) = reshape([1 0 1;0 1 0;1 0 1]', 9,1);
LabelData(:,2) = TrainData(:,2);   % Correct answer when input is ×　（入力が×の時の正解）

 k = TrainData;		%教師データ用の行列(入力)		input matrix for supervisor data
% Labeled output
 t = LabelData;
 
 % Create test data
% Maru and Maru + 1bit error
TestData(:,1) = reshape([1 1 1;1 0 1;1 1 1]', 9,1);
TestData(:,2) = reshape([0 1 1;1 0 1;1 1 1]', 9,1);
TestData(:,3) = reshape([1 0 1;1 0 1;1 1 1]', 9,1);
TestData(:,4) = reshape([1 1 0;1 0 1;1 1 1]', 9,1);
TestData(:,5) = reshape([1 1 1;0 0 1;1 1 1]', 9,1);
TestData(:,6) = reshape([1 1 1;1 1 1;1 1 1]', 9,1);
TestData(:,7) = reshape([1 1 1;1 0 0;1 1 1]', 9,1);
% Batsu and Batsu + 1bit error
TestData(:,8) = reshape([1 0 1;0 1 0;1 0 1]', 9,1);
TestData(:,9) = reshape([0 0 1;0 1 0;1 0 1]', 9,1);
TestData(:,10) = reshape([1 1 1;0 1 0;1 0 1]', 9,1);
TestData(:,11) = reshape([1 0 0;0 1 0;1 0 1]', 9,1);
TestData(:,12) = reshape([1 0 1;1 1 0;1 0 1]', 9,1);
TestData(:,13) = reshape([1 0 1;0 0 0;1 0 1]', 9,1);
TestData(:,14) = reshape([1 0 1;0 1 1;1 0 1]', 9,1);

% Parameter setting
Layer1 = 9;                     % 入力層のユニット数
Layer2 = 2;                     % 中間層（隠れ層，AEの出力層）のユニット数
Layer3 = Layer1;                % 復元層（出力層）のユニット数

L2func = 'ReLUfnc';             % 中間層のアルゴリズム（'Sigmoid' or Default: 'ReLUfnc' 文字数は等しくないとエラーを起こす）
L3func = 'Sigmoid_BCE';         % 復元層のアルゴリズムと誤差（'Sigmoid_MSE' or Default : 'Sigmoid_BCE' (Binary Cross Entropy)）
         
%% 中間層と出力層の重みの初期値
% Initialization values of weights (Hidden layer and Output layer)

w12_mean = rand(Layer2,Layer1);	% 教師データ用の行列(中間層1の重み)	 hidden layer's weight matrix for supervisor data
w12_var = rand(Layer2,Layer1);	% 教師データ用の行列(中間層1の重み)	 hidden layer's weight matrix for supervisor data
w23 = rand(Layer3,Layer2);	% 教師データ用の行列(復元層２の重み)	hidden layer's weight matrix for supervisor data

% 中間層と出力層のバイアスの初期値
% Initialization values of bias (Hidden layer and Output layer) 
b2_mean = (-0.5)*ones(Layer2,1);
b2_var = (-0.5)*ones(Layer2,1);
b3 = (-0.5)*ones(Layer3,1);

% Step size
eta = 0.01;		%学習率が高すぎると更新した係数が大きくなりすぎてコストが減らなくなる	
				%Learning rate. If the learning rate is too high, the updated coefficient becomes too large and the cost may not decrease

epoch = 10000;  
%epoch = 1000000;  

%%
% 初期状態における a3 の出力
% Forward Process (Initial weight)
X = TestData;
t = LabelData;
eps = randn(Layer2,1);

[z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3,eps);

%%
% 初期状態における a3 の出力 (HW)
CWL=24;             CFL=16;
CWL_W2=32;      CFL_W2=24;
CWL_W3=32;      CFL_W3=24;

%% z2_mean の計算
% 入力の準備
for i=1:9
    assignin('base', sprintf('X_%d', i),  timeseries(X(i)));
end
% w2の準備（今回のw2はw12_mean）
w2_11 = timeseries(w12_mean(1,1));
w2_12 = timeseries(w12_mean(1,2));
w2_13 = timeseries(w12_mean(1,3));
w2_14 = timeseries(w12_mean(1,4));
w2_15 = timeseries(w12_mean(1,5));
w2_16 = timeseries(w12_mean(1,6));
w2_17 = timeseries(w12_mean(1,7));
w2_18 = timeseries(w12_mean(1,8));
w2_19 = timeseries(w12_mean(1,9));

for i=1:9
    assignin('base', sprintf('w2_2%d', i),  timeseries(w12_mean(2,i)));
end

% b2の準備（今回のb2はb2_mean）
b2_1 = timeseries(b2_mean(1));
b2_2 = timeseries(b2_mean(2));

% w23の準備
for i=1:9
    for j=1:2
        assignin('base', sprintf('w3_%d%d', i,j),  timeseries(w23(i,j)));
    end
end
% b3の準備
for i=1:9
    assignin('base', sprintf('b3_%d', i),  timeseries(b3(i)));
end
% zの準備
for i=1:2
    assignin('base', sprintf('z_%d', i),  timeseries(0));
end

% HWでのシミュレーション
sim('forward_VAE');
%%
z2_1=ans.z2_1; z2_2=ans.z2_2;
a2_1=ans.a2_1; a2_2=ans.a2_2;
z2_mean_hw = [double(z2_1.Data(10)); double(z2_2.Data(10))];
a2_mean_hw = z2_mean_hw;

%% z2_varの計算
% w2の準備（今回は，w2はw12_var）
for i=1:9
    assignin('base', sprintf('w2_1%d', i),  timeseries(w12_var(1,i)));
    assignin('base', sprintf('w2_2%d', i),  timeseries(w12_var(2,i)));
end
% b2の準備（今回のb2はb2_var）
b2_1 = timeseries(b2_var(1));
b2_2 = timeseries(b2_var(2));
% HWでのシミュレーション
sim('forward_VAE.slx');

z2_1=ans.z2_1; z2_2=ans.z2_2;
a2_1=ans.a2_1; a2_2=ans.a2_2;
z2_var_hw = [double(z2_1.Data(10)); double(z2_2.Data(10))];
% Softplus 関数（ソフトプラス関数）
% Sofplus関数の微分は，Sigmoid関数
a2_var_hw = log(1+exp(z2_var_hw));

%% zの計算
z_hw = a2_mean_hw + sqrt(a2_var_hw).*eps;

%% Generate Image
% w23の準備
for i=1:9
    for j=1:2
        assignin('base', sprintf('w3_%d%d', i,j),  timeseries(w23(i,j)));
    end
end
% b3の準備
for i=1:9
    assignin('base', sprintf('b3_%d', i),  timeseries(b3(i)));
end
% zの準備
for i=1:2
    assignin('base', sprintf('z_%d', i),  timeseries(z_hw(i)));
end
% HWでのシミュレーション
sim('forward_VAE.slx');

%%
z3_1=ans.z3_1; z3_2=ans.z3_2; z3_3=ans.z3_3; 
z3_4=ans.z3_4; z3_5=ans.z3_5; z3_6=ans.z3_6;
z3_7=ans.z3_7; z3_8=ans.z3_8; z3_9=ans.z3_9;
a3_1=ans.a3_1; a3_2=ans.a3_2; a3_3=ans.a3_3; 
a3_4=ans.a3_4; a3_5=ans.a3_5; a3_6=ans.a3_6;
a3_7=ans.a3_7; a3_8=ans.a3_8; a3_9=ans.a3_9;

fprintf('Initial Weight\n');
fprintf('w12_mean\n');   disp(w12_mean);
fprintf('w12_var\n');   disp(w12_var);
fprintf('b2_mean\n');    disp(b2_mean);
fprintf('b2_var\n');    disp(b2_var);
fprintf('w23\n');   disp(w23);
fprintf('b3\n');    disp(b3);

fprintf('Initial Weight Test\n');
fprintf('Test data X \n');   disp(X);
fprintf('a2_mean\n');    disp(a2_mean);
fprintf('a2_var\n');    disp(a2_var);
fprintf('a3\n');    disp(a3);
    
    % 結果の表示
    fprintf('HWでの計算結果の検証：SWはDoubleでの計算\n');
    fprintf('z2_mean (HW) = %9f, %9f\n', ...
        double(z2_mean_hw(1)), double(z2_mean_hw(2)) );
    fprintf('z2_mean (SW) = %9f, %9f\n', ...
        double(z2_mean(1)), double(z2_mean(2)) );
    fprintf('a2_mean (HW) = %9f, %9f\n', ...
        double(a2_mean_hw(1)), double(a2_mean_hw(2))  );
    fprintf('a2_mean (SW) = %9f, %9f\n', ...
        double(a2_mean(1)), double(a2_mean(2)) );
    fprintf('Diff    = %9f, %9f\n', ...
        double(a2_mean(1))-double(a2_mean_hw(1)), double(a2_mean(1))-double(a2_mean_hw(1)));
    fprintf('\n');
    fprintf('z2_var (HW) = %9f, %9f\n', ...
        double(z2_var_hw(1)), double(z2_var_hw(2)) );
    fprintf('z2_var (SW) = %9f, %9f\n', ...
        double(z2_var(1)), double(z2_var(2)) );
    fprintf('a2_var (HW) = %9f, %9f\n', ...
        double(a2_var_hw(1)), double(a2_var_hw(2))  );
    fprintf('a2_var (SW) = %9f, %9f\n', ...
        double(a2_var(1)), double(a2_var(2)) );
    fprintf('Diff    = %9f, %9f\n', ...
        double(a2_mean(1))-double(a2_mean_hw(1)), double(a2_mean(1))-double(a2_mean_hw(1)));
    fprintf('\n');
    fprintf('a3 (HW) = %9f, %9f, %9f, %9f, %9f, %9f, %9f, %9f, %9f\n',...
        double(a3_1.Data(10)), double(a3_2.Data(10)), double(a3_3.Data(10)),...
        double(a3_4.Data(10)), double(a3_5.Data(10)), double(a3_6.Data(10)),...
        double(a3_7.Data(10)), double(a3_8.Data(10)), double(a3_9.Data(10)));
    fprintf('a3 (SW) =  %9f, %9f, %9f, %9f, %9f, %9f, %9f, %9f, %9f\n', ...
        double(a3(1)), double(a3(2)), double(a3(3)), ...
        double(a3(4)), double(a3(5)), double(a3(6)), ...
        double(a3(7)), double(a3(8)), double(a3(9)));
    fprintf('Diff    =  %9f, %9f, %9f, %9f, %9f, %9f, %9f, %9f, %9f\n', ...
        double(a3(1))-double(a3_1.Data(10)),  double(a3(2))-double(a3_2.Data(10)),  double(a3(3))-double(a3_3.Data(10)), ...
        double(a3(4))-double(a3_4.Data(10)),  double(a3(5))-double(a3_5.Data(10)),  double(a3(6))-double(a3_6.Data(10)), ...
        double(a3(7))-double(a3_7.Data(10)),  double(a3(8))-double(a3_8.Data(10)),  double(a3(9))-double(a3_9.Data(10)));
