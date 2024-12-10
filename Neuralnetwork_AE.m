% OX (Circle or Cross) Judgement Program (Auto Encoder) 
% LSI Design Contest 2025
%
% Neuralnetwork_AE function
%  Learing process (backpropagation)
%
% Input:
%           X : Input data
%           t : Label
%           w2 : Initial weight (Input Layer to Hidden Layer: Encoder)
%           w3 : Initial weight (Hidden Layer to Output Layer: Decoder)
%           b2 : Initial bias (Hidden Layer)
%           b3 : Initial bias (Output Layer)
%           eta : Step size
%           epoch : The number of epochs
%           L2func : Activation function in Layer2 ('Sigmoid' or 'ReLUfnc')
%           L3func : Activation function in Layer3 ('Sigmoid_MSE' or 'Sigmoid_BCE' )
%
% Output:
%           w2 : Final weights (Input Layer to Hidden Layer: Encoder)
%           w3 : Final weights (Hidden Layer to Output Layer: Decoder)
%           b2 : Final bias (Hidden Layer)
%           b3 : Final bias (Output Layer)
%           w2_t : History of weights
%           w3_t : History of weights
%           b2_t : History of bias
%           b3_t : History of bias
%           C : Losss history
%
% Requrired : OX_judge_AE.m, Neuralnetwork_AE.m, Neuralnetwork_forward_AE.m
%
% see also OX_judge_AE, Neuralnetwork_AE, Neuralnetwork_forward_AE.
%
function [w2,w3,b2,b3,w2_t,w3_t,b2_t,b3_t,E] = Neuralnetwork_AE(X,t,w2,w3,b2,b3,eta,epoch,L2func,L3func)

    w2_t = zeros(size(w2,1),size(w2,2),epoch);
    w3_t = zeros(size(w3,1),size(w3,2),epoch);
    b2_t = zeros(size(b2,1),size(b2,2),epoch);
    b3_t = zeros(size(b3,1),size(b3,2),epoch);

    E = zeros(1,epoch);                             % Array for storing errors

    %% Forwardpropagation & Backpropagation
    for j = 1:epoch

        z2 = w2 * X + b2;                           % z2: Weighted sum of hidden layers

        if(L2func == 'Sigmoid')                     % a2: Output for hidden layer
            a2 = 1./(1+exp(-z2));                   % Sigmoid
        elseif(L2func == 'ReLUfnc')
            a2 = zeros(size(z2));                   % ReLU
            a2(find(z2>0)) = z2(find(z2>0));
        end

        z3 = w3 * a2 + b3;                          % z3: Weighted sum of hidden layers

        if(L3func == 'Sigmoid_MSE')                 % a3: Output for output layer

            a3 = 1./(1+exp(-z3));                   % Sigmoid

            Esum = sum(sum((t-a3).^2)/2,2);         % MSE

            E(1,j) = Esum/size(X,2);                % Stores error per piece of data

            dEda3 = (a3-t);
            dEdz3 = dEda3 .* a3 .* (1-a3);
            delta3 = dEdz3;

        elseif(L3func == 'Sigmoid_BCE')             % a3: Output for output layer

            a3 = 1./(1+exp(-z3));                   % Sigmoid

            Esum = -t.*log(a3) - (1-t).*log(1-a3);  % BCE
            Esum = sum(sum(Esum),2);                % Sum of errors for all data

            E(1,j) = Esum/size(X,2);                % Stores error per piece of data

            dEda3 = -t./a3 + (1-t)./(1-a3);
            dEdz3 = dEda3 .* a3 .* (1-a3);
            delta3 = dEdz3;
        end

        % �t�`�d�@�i�o�b�N�v���p�Q�[�V�����j
        dEdb3 = sum(delta3,2);
        dEdw3 = (a2 * (delta3).').';

        dEda2 =  w3.' * (delta3);

        if(L2func == 'Sigmoid')                     % a2: Ouput for hidden layer
            dEdz2 = dEda2 .* a2 .* (1-a2);
        elseif(L2func == 'ReLUfnc')
            da2dz2 = zeros(size(a2));
            da2dz2(a2 > 0) = 1;
            dEdz2 = dEda2 .* da2dz2;
        end

        delta2 = dEdz2;

        dEdb2 = sum(delta2,2);
        dEdw2 = (X * delta2.').';

        % Parameter Update
        b3 = b3 - eta * dEdb3;
        w3 = w3 - eta * dEdw3;

        b2 = b2 - eta * dEdb2;
        w2 = w2 - eta * dEdw2;

        w2_t(:,:,j) = w2;                           % w12_t = zeros(size(w12,1),size(w12,2),epoch);
        w3_t(:,:,j) = w3;
        b2_t(:,:,j) = b2;                           % b2_t = zeros(size(b2,1), size(b2,2),epoch);
        b3_t(:,:,j) = b3;
    end
end



