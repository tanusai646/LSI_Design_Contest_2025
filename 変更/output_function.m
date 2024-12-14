function output_function(w3, b3)
fprintf("output_function\n");
for k = -1.5:0.02:1.5
    %disp(k);
    a2_1 = [k;k];
    z3_1 = w3 * a2_1 + b3;
    a3_1 = 1./(1+exp(-z3_1));
    %disp(a3_1);

    figure(20);
    imshow(reshape(a3_1(:,1),5,5), 'InitialMagnification','fit');

    drawnow;

    pause(0.1);
end

