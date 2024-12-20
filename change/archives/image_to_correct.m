function [CorrectData, LabelData, TestData, count] = image_to_correct(image_name, TestCount)

    % 指定された画像を開く
    % 画像をモノクロに変更
    image = double(rgb2gray(imread(image_name)))/255.0; % 0~255を0~1にする
    SizeX = size(image,2) / 8;
    SizeY = size(image,1) / 8;
    
    % 8×8のブロック毎にCorrectDataに挿入
    count = 1;
    for i = 1:SizeY
        for j = 1:SizeX
            % 8x8のブロックを切り出す
            CorrectData(:, count) = reshape(image((i-1)*8+1:i*8, (j-1)*8+1:j*8)', 64, 1); 
            LabelData(:, count) = CorrectData(:,count);
            count = count+1;
        end
    end
    
    TestData = LabelData(:, 1:TestCount);
end



