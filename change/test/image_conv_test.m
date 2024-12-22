%% テスト
%% 指定した場所のみ画像として出力できるかのテストプログラム
clear; close all;

test_img = rand(9,9);



h_img = zeros(9,9);

for i = 1:9
    for j = 1:9
        h_img(i,j) = 0;
        if i == 4
            h_img(i,j) = 1;
        end
    end
end

conv_img = test_img .* h_img;

figure();
subplot(1,2,1);
imshow(test_img,"InitialMagnification","fit");
subplot(1,2,2);
imshow(conv_img,"InitialMagnification","fit");
