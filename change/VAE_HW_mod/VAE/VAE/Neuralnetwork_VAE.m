% OX (Circle or Cross) Judgement Program (Variational Auto Encoder) 
% LSI Design Contest in Okinawa 2024
%
% NNeuralnetwork_VAE function
%  Learing process (backpropagation)
%
% Input:
%           X : Input data
%           t : Label
%           w12 : Initial weight (Input Layer to Hidden Layer: Encoder)
%           w23 : Initial weight (Hidden Layer to Output Layer: Decoder)
%           b2 : Initial bias (Hidden Layer)
%           b3 : Initial bias (Output Layer)
%           eta : Step size
%           epoch : The number of epochs
%           L2func : Activation function in Layer2 ('Sigmoid' or 'ReLUfnc')
%           L3func : Activation function in Layer3 ('Sigmoid_MSE' or 'Sigmoid_BCE' )
%
% Output:
%           w12 : Final weights (Input Layer to Hidden Layer: Encoder)
%           w23 : Final weights (Hidden Layer to Output Layer: Decoder)
%           b2 : Final bias (Hidden Layer)
%           b3 : Final bias (Output Layer)
%           w12_t : History of weights
%           w23_t : History of weights
%           b2_t : History of bias
%           b3_t : History of bias
%           C : Losss history
%
% Requrired : OX_judge_VAE.m, Neuralnetwork_VAE.m,
% Neuralnetwork_forward_VAE.m, Neuralnetwork_generate_VAE.m
%
% see also OX_judge_VAE, Neuralnetwork_VAE, Neuralnetwork_forward_VAE, Neralnetwork_generate_VAE.
%
function [w12_mean,w12_var,w23,b2_mean,b2_var,b3,w12_mean_t,w12_var_t,w23_t,b2_mean_t,b2_var_t,b3_t,C] = Neuralnetwork_VAE(X,t,w12_mean,w12_var,w23,b2_mean,b2_var,b3,eta,epoch,L2func,L3func)

        %% �f�[�^�T�C�Y���傫���ꍇ�́C�ϐ��̗������Z�[�u���Ȃ�
%     w12_mean_t = zeros(size(w12_mean,1),size(w12_mean,2),epoch);
%     w12_var_t = zeros(size(w12_var,1),size(w12_var,2),epoch);
%     w23_t = zeros(size(w23,1),size(w23,2),epoch);
%     b2_mean_t = zeros(size(b2_mean,1), size(b2_mean,2),epoch);
%     b2_var_t = zeros(size(b2_var,1), size(b2_var,2),epoch);
%     b3_t = zeros(size(b3,1), size(b3,2),epoch);

    C = zeros(1,epoch);                         % �덷�i�[�p�z��

    %% Forwardpropagation & Backpropagation
    for j=1:epoch
        if rem(j,1000)== 0
            fprintf('%d/%d\n',j, epoch);
        else
        end
        z2_mean = w12_mean * X + b2_mean;          % ���ԑw�i�B��w�j�̏d�ݕt������        input weight for hidden layer
        z2_var = w12_var * X + b2_var;                      % ���ԑw�i�B��w�j�̏d�ݕt������        input weight for hidden layer

        z2_mean = z2_mean/64;
        z2_var = z2_var/64;
        
        %         if(L2func == 'Sigmoid')                     % ���ԑw�i�B��w�j�̏o��a2
        %             a2 = 1./(1+exp(-z2));                   % ���ԑw�i�B��w�j�̏o��a2�i�������֐��F�V�O���C�h�֐��j
        %         elseif(L2func == 'ReLUfnc')
        %             a2 = zeros(size(z2));                   % ���ԑw�i�B��w�j�̏o��a2�i�������֐��FReLU�֐��j
        %             a2(find(z2>0)) = z2(find(z2>0));
        %         end

        a2_mean = z2_mean;
        % Softplus �֐��i�\�t�g�v���X�֐��j
        % Sofplus�֐��̔����́CSigmoid�֐�
        a2_var = log(1+exp(z2_var));

        eps = randn(size(z2_mean));
        z = a2_mean + sqrt(a2_var).*eps;

        z3 = w23 * z + b3;                       % ���ԑw�i�B��w�j�̏d�ݕt���az3
        z3 = z3/64.0;
        a3 = 1.0001./(1+exp(-z3));                   % �����w�i�o�͑w�j�̏o��a3�i�������֐��F�V�O���C�h�֐��j
        %if��
        c = -t.*log(a3)-(1-t).*log(1-a3);                                     % �����i�N���X�j�G���g���s�[�덷�Z�o
        L = -1/2 *(1+log(a2_var) - a2_mean.^2 - a2_var);        % KL Divergence

        % figure(200); subplot(2,1,1);plot(c);axis([0 64*64 0 1]); subplot(2,1,2);plot(L);
        %c_z =

        ctot=sum(sum(c),2)+sum(sum(L),2);                        % �S�f�[�^�̌덷���a

        C(1,j) = ctot/size(X,2);                           % �f�[�^1������̌덷���i�[
        %%%%% �`�F�b�N�̂��߂̉摜�\��
        if rem(j,250)== 0
            figure(103);
            for k = 1:size(a3,2)
                subplot(size(a3,2)+3,1,k);
                imshow(reshape(a3(:,k), [64 64])');                   % ���t�摜����ꂽ�Ƃ��̏o�͉摜
            end
            subplot(size(a3,2)+3,1,size(a3,2)+1);
            plot(c); axis([0 64*64 0 2]);
            title(sprintf('%d/%d, Reconstruction Error (BCE)', j, epoch));
            subplot(size(a3,2)+3,1,size(a3,2)+2);
            plot(L); axis([0 128 0 100]);
            title(sprintf(' KL Divergence'));
            subplot(size(a3,2)+3,1,size(a3,2)+3);
            plot(C(1,:)); 
            title(sprintf('Error (Total)', j, epoch));
        else
        end        
        %%%%%

        dCda3 = -t./a3 + (1-t)./(1-a3);         % -p/q + 1-p/1-q (E = -p+logq - (1-p)*log(1-q) = -t_1*log a-3_1 -(1-t_1)*log (1-a-3_1) )
        dCdz3 = dCda3 .* a3 .* (1-a3);         % �o�͑w�̌덷 ��13�@(1)��
        delta3 = dCdz3;

    % �t�`�d�@�i�o�b�N�v���p�Q�[�V�����j

    dCdb3 = sum(delta3,2);                  % �o�C�A�Xb3�̌��z (3)��
    % �Hsum�̗��R�H:���`�d�ł̃o�C�A�X�̉��Z�́C�e�f�[�^�ɑ΂��čs����D
    % ���̂��߁C�t�`�d�ł͊e�f�[�^�̌��z�̒l��1�ɏW�񂷂�K�v������̂Řa���Ƃ�D

    dCdw23 = (z * (delta3).').';            % �d��w23�̌��z (2)��
    % �H.'�̈Ӗ��H:�����f�[�^������ N�̂Ƃ��Ca2��m�sN��̍s��CdCdz3��n�sN��̍s��
    % (m:�B��w�̃��j�b�g���Cn:�o�͑w�̃��j�b�g��)�ɂȂ�D���̎��C�o�͑w�̏d��w23��m�sn��ɂȂ邽�߁C
    % ���̌��z�ł���dCdw23��m�sn��ł���K�v������D

    dCda2_mean =  w23.' * (delta3)+a2_mean;
    size(w23);
    size(delta3);
    size(eps);
    size(a2_var);
    dCda2_var =  (w23.' * (delta3)).*eps./(2*sqrt(a2_var)) + 0.5*(1-1./a2_var);

    dCdz2_mean = dCda2_mean;
    dCdz2_var = (w23.' * (delta3)).*eps./(2*sqrt(a2_var)).*(1./(1+exp(z2_var)))+0.5*(1-1./a2_var).*(1./(1+exp(z2_var))); % �\�t�g�v���X�̔���

    delta2_mean = dCdz2_mean;
    delta2_var = dCdz2_var;

    dCdb2_mean = sum(delta2_mean,2);                  % �o�C�A�Xb2�̌��z (6)��
    dCdw12_mean = (X * delta2_mean.').';              % �d��w12�̌��z (5)��
    dCdb2_var = sum(delta2_var,2);                  % �o�C�A�Xb2�̌��z (6)��
    dCdw12_var = (X * delta2_var.').';              % �d��w12�̌��z (5)��

    % �p�����[�^�̍X�V
    b3 = b3 - eta * dCdb3;                  % �o�͑w�̃o�C�A�X (8)��
    w23 = w23 - eta * dCdw23;               % �o�͑w�̏d�� (7)��

    b2_mean = b2_mean - eta * dCdb2_mean;                  % �o�͑w�̃o�C�A�X (8)��
    w12_mean = w12_mean - eta * dCdw12_mean;               % �o�͑w�̏d�� (7)��
    b2_var = b2_var - eta * dCdb2_var;                  % �o�͑w�̃o�C�A�X (8)��
    w12_var = w12_var - eta * dCdw12_var;               % �o�͑w�̏d�� (7)��

   %% �T�C�Y���傫���ꍇ�́A�f�[�^�̕ۑ������Ȃ�
%     w12_mean_t(:,:,j) = w12_mean;         % w12_t = zeros(size(w12,1),size(w12,2),epoch);
%     w12_var_t(:,:,j) = w12_var;         % w12_t = zeros(size(w12,1),size(w12,2),epoch);
%     w23_t(:,:,j) = w23;
%     b2_mean_t(:,:,j) = b2_mean;           % b2_t = zeros(size(b2,1), size(b2,2),epoch);
%     b2_var_t(:,:,j) = b2_var;           % b2_t = zeros(size(b2,1), size(b2,2),epoch);
%     b3_t(:,:,j) = b3;
     w12_mean_t = w12_mean;         % w12_t = zeros(size(w12,1),size(w12,2),epoch);
     w12_var_t = w12_var;         % w12_t = zeros(size(w12,1),size(w12,2),epoch);
     w23_t = w23;
     b2_mean_t = b2_mean;           % b2_t = zeros(size(b2,1), size(b2,2),epoch);
     b2_var_t = b2_var;           % b2_t = zeros(size(b2,1), size(b2,2),epoch);
     b3_t = b3;
    end
end



