function generate_noise()

%% set parameters
input_path = '/media/hyp/Data/DataModel/d/ClassicalSR/Set5';
save_noise_path = '/media/hyp/Data/DataModel/d/ClassicalSR/Set5_noise';

if exist('save_noise_path', 'var')
    if exist(save_noise_path, 'dir')
        disp(['It will cover ', save_noise_path]);
    else
        mkdir(save_noise_path);
    end
end


idx = 0;
filepaths = dir(fullfile(input_path,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);

        % read image
        img = imread(fullfile(input_path, [imname, ext]));

        % additive Gaussian noise
        img = AddGaussianNoise(img, 20);
        
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