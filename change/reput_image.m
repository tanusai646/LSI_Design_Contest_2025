function retput_image(a3, TestCount)
% data: 64xNの入力データ (Nは列数)

block_size = 8;
rows = 2;
cols = 32;

blocks_2d = reshape(a3, block_size*rows, []);
rearranged_blocks = permute(blocks_2d, [2, 1]);
rearranged_image = reshape(rearranged_blocks, cols, rows*block_size)';

%figure(15);
%imshow(rearranged_image, 'InitialMagnification','fit');