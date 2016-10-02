function [output,fil_bank,ibu] = pretrain( input,fil_bank,ibu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    %d = 7;
    %n = 5;
    [d,~,n] = size(fil_bank);
    alpha = 1;
    %fil_bank = rand(d,d,n);
    [h,w] = size(input);
    feature = zeros(h-d+1,w-d+1,n);
    upsample = zeros(h-d+1,w-d+1,n);
    presence = zeros(1,n);
    feature_recon = zeros(h,w,n);
    new_recon = zeros(h,w,n);
    shift = zeros(1,n);
    for i = 1:n
        filter = squeeze(fil_bank(:,:,i));
        f_map = conv2(squeeze(input),rot90(filter,2),'valid');
        shift(i) = max(max(f_map));
        feature(:,:,i) = f_map;
        upsample(:,:,i) = (f_map == shift(i));
        if shift(i) == 0
            upsample(:,:,i) = 0;
        end
        %subplot(5,n,i+2*n),imshow(squeeze(upsample(:,:,i)));
        test = sum(sum(squeeze(upsample(:,:,i))));
        if test > 1
            fprintf('upsample has too many max points with value %f!\n',shift(i));
        end
        feature_recon(:,:,i) = conv2(squeeze(upsample(:,:,i)),filter,'full');
    end
    %[Z0,ibu] = sparse_trans(shift,ibu);
    Z0 = sigmoid(shift/d^2);
    for i = 1:n
        feature_recon(:,:,i) = feature_recon(:,:,i) *Z0(i);
    end
    output = sum(feature_recon,3);
    new_out = output;
    display(Z0);
    z = double(Z0);
    y = reshape(input,1,[]);
    pred = reshape(output,1,[]);
    delta = pred - y;
    error = delta*delta';
    %fprintf('error : %f\n',error);
    %get optimal Z
    %display(sum(sum((new_out-input) .* squeeze(feature_recon(:,:,1))))/(h*w));
    converge = 1;
    new_error = 0;
    while converge > (0.001 * n)
        for i = 1:n
            dz = sum(sum((new_out-input) .* squeeze(feature_recon(:,:,i)))) + alpha *(z(i) - Z0(i));
            z(i) = z(i) - 0.01* dz;
            if z(i)<0
                z(i) = 0;
            end
            
            if z(i)>1
                z(i) = 1;
            end
            
        end
        
        for i = 1:n
            filter = squeeze(fil_bank(:,:,i));
            new_recon(:,:,i) = conv2(squeeze(upsample(:,:,i)),filter,'full') .* z(i);
        end
        new_out = sum(new_recon,3);     
        new_pred = reshape(new_out,1,[]);
        converge = abs((new_pred-y)*(new_pred-y)' +(z-Z0) *(z-Z0)' - new_error);
        new_error = (new_pred-y)*(new_pred-y)' +(z-Z0) *(z-Z0)' ;
        %fprintf('error : %f\n',new_error);
        %display(z);
    end
    display(z);
    %Update filter
    %Decoder
    delta = new_pred - y;
    delta = reshape(delta,h,w);
    %figure,subplot(3,1,1),
    %imshow(input);
    %subplot(3,1,2),imshow(new_out);
    %subplot(3,1,3),imshow(delta);
    %figure;
    
    for i = 1:n
        %Not sure??
        %display(conv2(delta*z(i),squeeze(upsample(:,:,i)),'valid'));
        gradw = conv2(delta*z(i),rot90(squeeze(upsample(:,:,i)),2),'valid');
        fil_bank(:,:,i) = fil_bank(:,:,i) - 0.1 * gradw;
        subplot(4,n,i+1*n),imshow(mat2gray(-gradw)); 
    end
    %display(gradw);
    %figure;
    %Encoder
    %display(Z0);
    %display(z);
    for i = 1:n
        feature = conv2(input,rot90(squeeze(upsample(:,:,i)),2),'valid');
        gradw = (Z0(i)-z(i)).* sigmoidgrad(Z0(i))*feature;
        fil_bank(:,:,i) = fil_bank(:,:,i) - 0.1 * gradw;
        subplot(4,n,i+2*n),imshow(mat2gray(-gradw)); 
    end
    %display(gradw);
end

