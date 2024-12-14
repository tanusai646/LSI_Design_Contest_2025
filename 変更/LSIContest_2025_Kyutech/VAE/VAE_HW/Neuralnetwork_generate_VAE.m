% OX (Circle or Cross) Judgement Program  (Variational Auto Encoder) 
% LSI Design Contest in Okinawa 2024
%
% Neuralnetwork_Generation_VAE function
%  Forward process 
%
% Input:
%           z : ���ݕϐ�: latent variable
%           w23 : Weight (Hidden Layer to Output Layer: Decoder)
%           b3 : Bias (Output Layer)
%   
% Output: 
%           z3 : Output Layer  
%           a3 ; Output Layer
%
% Requrired : OX_judge_VAE.m, Neuralnetwork_VAE.m,
% Neuralnetwork_forward_VAE.m, Neuralnetwork_generate_VAE.m
%
% see also OX_judge_VAE, Neuralnetwork_VAE, Neuralnetwork_forward_VAE, Neralnetwork_generate_VAE.
%
function [z3,a3] = Neuralnetwork_generate_VAE(z, w23,b3)

z3 = w23 * z + b3;                     % �����w�i�o�͑w�j�̏d�ݕt������        input weight for hidden layer
a3= 1./(1+exp(-z3));                    % �����w�̏o��                output for hidden layer

end