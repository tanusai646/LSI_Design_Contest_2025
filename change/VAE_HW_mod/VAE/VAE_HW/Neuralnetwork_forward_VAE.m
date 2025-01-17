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
%           a2_mean : Hidden Layer (���ݕϐ�: latent variable)
%           a2_var : Hidden Layer (���ݕϐ�: latent variable)
%           z : ���ݕϐ�: latent variable
%           z3 : Output Layer  
%           a3 ; Output Layer
%
% Requrired : OX_judge_VAE.m, Neuralnetwork_VAE.m,
% Neuralnetwork_forward_VAE.m, Neuralnetwork_generate_VAE.m
%
% see also OX_judge_VAE, Neuralnetwork_VAE, Neuralnetwork_forward_VAE, Neralnetwork_generate_VAE.
%
function [z2_mean,z2_var, a2_mean, a2_var,z,z3,a3] = Neuralnetwork_forward_VAE(X,w12_mean,w12_var,w23,b2_mean,b2_var,b3,eps)

z2_mean = w12_mean * X + b2_mean;                      % ���ԑw�i�B��w�j�̏d�ݕt������        input weight for hidden layer
z2_var = w12_var * X + b2_var;                      % ���ԑw�i�B��w�j�̏d�ݕt������        input weight for hidden layer
% if(L2func == 'Sigmoid')                 % ���ԑw�i�B��w�j�̏o��a2
%     a2 = 1./(1+exp(-z2));               % ���ԑw�i�B��w�j�̏o��a2�i�������֐��F�V�O���C�h�֐��j
% elseif(L2func == 'ReLUfnc')
%     a2 = zeros(size(z2));               % ���ԑw�i�B��w�j�̏o��a2�i�������֐��FReLU�֐��j
%     a2(find(z2>0)) = z2(find(z2>0));
% end

a2_mean = z2_mean;
% Softplus �֐��i�\�t�g�v���X�֐��j
% Sofplus�֐��̔����́CSigmoid�֐�
a2_var = log(1+exp(z2_var));

% eps = randn(size(z2_mean));
z = a2_mean + sqrt(a2_var).*eps;

z3 = w23 * z + b3;                     % �����w�i�o�͑w�j�̏d�ݕt������        input weight for hidden layer
a3= 1./(1+exp(-z3));                    % �����w�̏o��                output for hidden layer

end
