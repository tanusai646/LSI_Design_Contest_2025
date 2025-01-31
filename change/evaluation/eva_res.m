clear; close all;
% PSNR比較検証用プログラム
block1_SW = readmatrix("block_PSNR1.csv");
block2_SW = readmatrix("block_PSNR2.csv");
block3_SW = readmatrix("block_PSNR3.csv");
block1_HW = readmatrix("PSNR1.CSV");
block2_HW = readmatrix("PSNR2.CSV");
block3_HW = readmatrix("PSNR3.CSV");


figure(10);
colormap(jet);
imagesc(block1_SW);
colorbar;
clim([0, 40]);

figure(11);
colormap(jet);
imagesc(block1_HW);
colorbar;
clim([0, 40]);


figure(20);
colormap(jet);
imagesc(block2_SW);
colorbar;
clim([0, 40]);

figure(21);
colormap(jet);
imagesc(block2_HW);
colorbar;
clim([0, 40]);

figure(30);
colormap(jet);
imagesc(block3_SW);
colorbar;
clim([0, 40]);

figure(31);
colormap(jet);
imagesc(block3_HW);
colorbar;
clim([0, 40]);

count1 = 0;
for i=1:32
    for j=1:32
        if(block1_HW(i,j) > 25)
            count1 = count1 + 1;
        end
    end
end

size1 = count1*8*16 + (32*32 - count1) * 16 * 16 * 8;
per1 = (512*512*8-size1) / (512*512*8);
count2 = 0;
for i=1:32
    for j=1:32
        if(block2_HW(i,j) > 25)
            count2 = count2 + 1;
        end
    end
end

size2 = count2*8*16 + (32*32 - count2) * 16 * 16 * 8;
per2 = (512*512*8-size2) / (512*512*8);
count3 = 0;
for i=1:32
    for j=1:32
        if(block3_HW(i,j) > 25)
            count3 = count3 + 1;
        end
    end
end

size3 = count3*8*16 + (32*32 - count3) * 16 * 16 * 8;
per3 = (512*512*8-size3) / (512*512*8);