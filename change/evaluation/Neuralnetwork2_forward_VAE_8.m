% OX (Circle or Cross) Judgement Program (Variational Auto Encoder) 
% LSI Design Contest in Okinawa 2024
%
% Neuralnetwork_forward_VAE function
%  Forward process 
%
% Input:
%           X : Input data
%           w12_mean : Weight (Input Layer to Hidden Layer (Mean): Encoder)
%           w12_var : Weight (Input Layer to Hidden Layer (Variance): Encoder)
%           w23 : Weight (Hidden Layer to Output Layer: Decoder)
%           b2_mean : Bias (Hidden Layer, Mean)
%           b2_var : Bias (Hidden Layer, Variance)
%           b3 : Bias (Output Layer)
%   
% Output: 
%           z2_mean : Hidden Layer 
%           z2_var : Hidden Layer 
%           a2_mean : Hidden Layer (潜在変数: latent variable)
%           a2_var : Hidden Layer (潜在変数: latent variable)
%           z : 潜在変数: latent variable
%           z3 : Output Layer  
%           a3 ; Output Layer
%
% Requrired : OX_judge_VAE.m, Neuralnetwork_VAE.m,
% Neuralnetwork_forward_VAE.m, Neuralnetwork_generate_VAE.m
%
% see also OX_judge_VAE, Neuralnetwork_VAE, Neuralnetwork_forward_VAE, Neralnetwork_generate_VAE.
%
function [z2_mean,z2_var, a2_mean, a2_var,z,z3,a3, a3_fixed_d] = Neuralnetwork2_forward_VAE_8(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3)

z2_mean = w12_mean * X + b2_mean;                      % 中間層（隠れ層）の重み付き入力        input weight for hidden layer
z2_var = w12_var * X + b2_var;                      % 中間層（隠れ層）の重み付き入力

z2_mean = z2_mean/16;
z2_var = z2_var/16;
% input weight for hidden layer
% if(L2func == 'Sigmoid')                 % 中間層（隠れ層）の出力a2
%     a2 = 1./(1+exp(-z2));               % 中間層（隠れ層）の出力a2（活性化関数：シグモイド関数）
% elseif(L2func == 'ReLUfnc')
%     a2 = zeros(size(z2));               % 中間層（隠れ層）の出力a2（活性化関数：ReLU関数）
%     a2(find(z2>0)) = z2(find(z2>0));
% end

a2_mean = z2_mean;
% Softplus 関数（ソフトプラス関数）
% Sofplus関数の微分は，Sigmoid関数
a2_var = log(1+exp(z2_var));

eps = randn(size(z2_mean));
z = a2_mean + sqrt(a2_var).*eps;

z_fixed = fi(z, 1, 8, 3);
z_fixed_d = double(z_fixed);

% ノーマルの計算
z3 = w23 * z + b3;                     % 復元層（出力層）の重み付き入力        input weight for hidden layer
z3 = z3/16.0;
a3= 1.0001 ./(1+exp(-z3));                    % 復元層の出力                output for hidden layer

% 固定小数点数の計算
z3_fixed_d = w23 * z_fixed_d + b3;
z3_fixed_d = z3_fixed_d / 16.0;
a3_fixed_d = 1.0001 ./ (1 + exp(-z3_fixed_d));



%writematrix(z2_mean, "z2_mean.csv");
%writematrix(z2_var, "z2_var.csv");
%writematrix(z3, "z3.csv");
%writematrix(eps, "eps.csv");
end
