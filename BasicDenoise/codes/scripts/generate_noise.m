function generate_noise()

%% set parameters
input_path = '/home/heyp/data/DnCNN/Train400_p40_s8';

sigma = 50
save_noise_path = '/home/heyp/data/DnCNN/Train400_p40_s8_sigma50';

if exist('save_noise_path', 'var')
    if exist(save_noise_path, 'dir')
        disp(['It will cover ', save_noise_path]);
    else
        disp(['Make directory ', save_noise_path]);
        mkdir(save_noise_path);
    end
end


idx = 0;
filepaths = dir(fullfile(input_path,'*.*'));
total = length(filepaths)
fprintf('file number is %d\n', total);
for i = 1 : total
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('add noise to img[%d/%d]\t%s.\n', idx, total, imname);
        fprintf(str_rlt);

        % read image
        img = imread(fullfile(input_path, [imname, ext]));

        % additive Gaussian noise
        img = AddGaussianNoise(img, sigma);
        
        % imshow(img);
        if exist('save_noise_path', 'var')
            imwrite(img, fullfile(save_noise_path, [imname, '.png']));
        else
            fprintf('no exit save path: %s\n', save_noise_path);
        end
    end
end
end


% additive Gaussian noise
function input = AddGaussianNoise(label, noiseSigma)
randn('seed', 0);
noise = noiseSigma/255.*randn(size(label));
input = im2single(label) + single(noise);
input = im2uint8(input);
end