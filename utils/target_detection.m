function [All_Num,time_per_image] = target_detection(readPath, savePath, tuneopts)

if isfield(tuneopts, 'temporal_step');   temporal_step = tuneopts.temporal_step;    end
if isfield(tuneopts, 'patchSize');               patchSize = tuneopts.patchSize;    end
if isfield(tuneopts, 'lambdaL');                 lambdaL = tuneopts.lambdaL;        end
if isfield(tuneopts, 'mu');                      mu = tuneopts.mu;                  end

slideStep = patchSize;

%% Get all image file names, please make sure that the image file order is correct by this reading way.
filesdir = dir([readPath '/*.jpg']);
if isempty( filesdir )
    filesdir = dir( [readPath '/*.bmp'] );
end
if isempty( filesdir )
    filesdir = dir([readPath '/*.png']);
end
if isempty( filesdir )
    fprintf('\n There is no any image in the folder of %s', readPath);
    return;
end
% get all image file names into a cell array;
files = { filesdir.name };
files = sort_nat(files);

iteration = 0;
All_Num = length(files);

%% begin to process images using mog based detection method.
sliding_step = temporal_step-1;
if mod(length(files),sliding_step)==0
    t_list =  1 : sliding_step : length(files)-temporal_step + 1;
else
    t_list =  [1 : sliding_step : length(files)-sliding_step + 1, length(files) - temporal_step + 1];
end

t1 = clock;
for t = t_list
    iteration = iteration + 1;
    spat_temp_ten = []; 
    priorWeight_ten = [];

    %% read images and construct the patch image
    for tt = 1 : temporal_step
        img = imread([readPath '/' files{tt+t-1}]);
        if size(img, 3) > 1 
            img = rgb2gray( img );
        end
        [imgHei, imgWid] = size(img);
        img = double(img);
        imwrite(mat2gray(img), [savePath '/' files{tt+t-1}]);


        %% construct patch tensor
        [tenF, patchNumber, patchPosition] = construct_patch_ten(img, patchSize, slideStep);
        spat_temp_ten(:,:,:,tt) = tenF;
    end

    [~,~,~,n4]=size(spat_temp_ten);

    %% DataSphering
    spat_temp_ten = DataSphering(spat_temp_ten);
    for i = 1:n4
        spat_temp_ten(:,:,:,i) = 255*mat2gray(spat_temp_ten(:,:,:,i));
    end

    %% Prior
    for kk = 1:temporal_step
        tarImg = zeros([imgHei, imgWid]);
        for ss = 1:patchNumber
            tar_tmp = spat_temp_ten(:,:,:,kk);
            position = patchPosition(:,:,ss);
            row = position(1);
            col = position(2);
            tarImg(row: row + patchSize-1, col:col + patchSize-1) = tar_tmp(:,:,ss);
        end
%         imwrite(mat2gray(tarImg), [savePath  '/' strtok([files{kk+t-1}],'.') '_DataSphering.jpg']);
        gray_img = mat2gray(tarImg);
    
        
        fft_img = fft2(gray_img);
        
        
        fft_img = fftshift(fft_img);
        [x, y] = meshgrid(-128:127, -128:127);        
        sigma = 45;  
        filter = exp(-(x.^2 + y.^2)/(2*sigma^2));
        filter = 1 - filter/max(filter(:));
        
        
        filtered_fft_img = fft_img .* filter;
        
        
        filtered_fft_img = ifftshift(filtered_fft_img);
        
        
        filtered_img = real(ifft2(filtered_fft_img));
%         imwrite(mat2gray(filtered_img),[savePath   '/' strtok([files{kk+t-1}],'.') '_filtered.jpg']);    
        [grad_x, grad_y] = gradient(filtered_img);
        

        grad_0 =grad_x;

        grad_45 = (grad_x + grad_y) / sqrt(2);

        grad_90 = grad_y;

        grad_135 = (-grad_x + grad_y) / sqrt(2);

        grad_180 = -grad_x;

        grad_225 =  (-grad_x - grad_y) / sqrt(2);

        grad_270 = -grad_y;

        grad_315 = (grad_x - grad_y) / sqrt(2);
        
        % 计算不同方向的梯度矩阵的平方
        grad_0_sq = grad_0.^2;
        grad_45_sq = grad_45.^2;
        grad_90_sq = grad_90.^2;
        grad_135_sq = grad_135.^2;
        grad_180_sq = grad_180.^2;
        grad_225_sq = grad_225.^2;
        grad_270_sq = grad_270.^2;
        grad_315_sq = grad_315.^2;
    
        Q1=mat2gray(grad_180_sq+grad_270_sq+grad_225_sq);
        Q2=mat2gray(grad_0_sq+grad_270_sq+grad_315_sq);
        Q3=mat2gray(grad_0_sq+grad_90_sq+grad_45_sq);
        Q4=mat2gray(grad_90_sq+grad_180_sq+grad_135_sq);
    
        k1 = mean(Q1(:))+2*var(Q1(:));
        k2 = mean(Q2(:))+2*var(Q2(:));
        k3 = mean(Q3(:))+2*var(Q3(:));
        k4 = mean(Q4(:))+2*var(Q4(:));
    
        prior = img .* ((double(Q1>k1).*Q1) + (double(Q2>k2).*Q2) + (double(Q3>k3).*Q3) + (double(Q4>k4).*Q4));
%         imwrite(mat2gray(prior), [savePath '/' strtok([files{kk+t-1}],'.') '_prior.jpg']);
        [tenP, ~, ~] = construct_patch_ten(prior, patchSize, slideStep);
        priorWeight_ten(:,:,:,tt) = tenP;
    end

    %% The proposed model
    Nway = size(spat_temp_ten);
    rs = 3;
    rt = 4;
    opts=[];
    opts.tol =0.001;%1e-4; 
    opts.max_iter = 400;
    opts.max_mu = 1e5;
    opts.gamma = 1.5;
    opts.lambda1 = lambdaL/((max(Nway(1),Nway(2))) * Nway(3));
    opts.lambda2 =  100*opts.lambda1; 
    opts.r = [rs,rs,rs,rt];
    opts.mu = mu;
    opts.r_max = [1,1,1,1]*7;
    opts.alpha = [1,1,1,3]/6;
    opts.beta = [1,1,1,3]/6;

    tenT=[];
    tenB=[];
    tenN=[];
    [tenB, tenT, tenN] = LRSD(spat_temp_ten,priorWeight_ten,opts);
    
    for kk = 1:temporal_step 
        tarImg = zeros([imgHei, imgWid]);
        BKGImg = zeros([imgHei, imgWid]);
        NImg = zeros([imgHei, imgWid]);
        for ss = 1:patchNumber
            tar_tmp = tenT(:,:,:,kk);
            BKG_tmp = tenB(:,:,:,kk);
            noise_tmp = tenN(:,:,:,kk);
            position = patchPosition(:,:,ss);
            row = position(1);
            col = position(2);
            tarImg(row: row + patchSize-1, col:col + patchSize-1) = tar_tmp(:,:,ss);
            BKGImg(row: row + patchSize-1, col:col + patchSize-1) = BKG_tmp(:,:,ss);
            NImg(row: row + patchSize-1, col:col + patchSize-1) = noise_tmp(:,:,ss);
        end
        if (kk+t-1) ~= length(files)
            imwrite(mat2gray(tarImg), [savePath '/' strtok([files{kk+t-1}],'.') '_tar.jpg']);
            E = tarImg;
            save([savePath '/' strtok([files{kk+t-1}],'.') '_tar.mat'],'E');
%             imwrite(mat2gray(BKGImg), [savePath  '/' strtok([files{kk+t-1}],'.') '_BKG.jpg']);
%             imwrite(mat2gray(NImg), [savePath '/' strtok([files{kk+t-1}],'.') '_noise.jpg']);
        end
    end
end
t2=clock;
time_all = etime(t2,t1);
time_per_image = time_all/(iteration*temporal_step);
disp(['Each image consumes time: ', num2str(time_per_image)]);
end