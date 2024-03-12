function [multichannel_noise,noise_type]= noise_generator(datapath,setind,channel_number,micro_pos,sdur,contain_CHIME_noise)


    addpath  D:\Audio\TIMIT_Dataset\tools\RIR-Generator-master
    addpath  D:\Audio\TIMIT_Dataset\tools\ANF-Generator-master
    addpath  D:\Audio\TIMIT_Dataset\tools\Wind-Generator-master
    
    % Multichannel Wind Noise Generation
    
    c = 340;                    % Sound velocity (m/s)
    fs = 16000;
    fft_length = 512;         % FFT length     
    ww = 2*pi*fs*(0:fft_length/2)/fft_length;
    if setind ==3
        noise_list = {'babble','white','wind'};
    else
        if contain_CHIME_noise == 0
            noise_list = {'babble','white','wind'};
%             noise_list = {'babble','white','wind','buccaneer2','f16','factory2','hfchannel','leopard','m109','volvo'};
        else
            noise_list = {'babble','white','wind','buccaneer2','destroyerengine','factory1','leopard','m109','machinegun','PED1','PED2','CAF1','CAF2','STR1','STR2','BUS1'};
        end
    end
        %     noise_list = {'babble','white'};
    Npoint = [0 0.6; 0 0.6;0.6 1];

    noiseind = randperm(length(noise_list),1);
    noise_type = noise_list{noiseind};
%     fprintf('%s\n',noise_type)
    if not(strcmp(noise_type,'wind'))
        [noise1,~] = audioread([datapath 'noises/' noise_list{noiseind} '_16khz.wav']);
        noise1 = noise1 - mean(noise1);

        noise_piece = zeros(sdur,channel_number);
        setbeg = randperm(round(Npoint(setind,2)*length(noise1)-sdur-1),1);
        for m=1:channel_number
            noise_piece(:,m) = noise1(setbeg:setbeg+sdur-1);  %截取噪声
        end

        DC = zeros(channel_number,channel_number,fft_length/2+1);
        for p = 1:channel_number
            for q = 1:channel_number
                if p == q
                    DC(p,q,:) = ones(1,1,fft_length/2+1);
                end
                 DC(p,q,:) = sinc(ww*norm(micro_pos(p,:)-micro_pos(q,:))/(c*pi)); % 改进为每个麦克风间距
            end
        end
        multichannel_noise = mix_signals(noise_piece,DC,'eigen'); % Using eigen instead of cholesky

    else
        
        type = 'const';
        uncorr_wind = generate_uncorr_wind(fs, sdur, type, channel_number); 
        wind = uncorr_wind.';           % Uncorrelated wind noise signals as input (not yet filtered)
        gamma = randperm(360,1);        % direction of arrival of the wind stream [rad]
        U = 1.8;                        % free-field wind speed [m/s]
        alpha1 = -0.125;                % experimental longitudinal decay
        alpha2 = -0.7;                  % experimental lateral decay
        
        DC = zeros(channel_number,channel_number,fft_length/2+1); % init of the coherence matrix

        % Build of the coherence matrix
        for p = 1:channel_number 
            for q = 1:channel_number
                if p == q
                    DC(p,q,:) = ones(1,1,fft_length/2+1);
                else
                    u = [cosd(gamma) sind(gamma) 0];
                    v = micro_pos(p,:)-micro_pos(q,:);
                    CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
                    d = norm(micro_pos(p,:)-micro_pos(q,:));
                    d_m = d*cos(CosTheta);             % DOA-dependent phase difference term
                    alpha = alpha1*abs(CosTheta) + ...
                    alpha2*abs(sqrt(1-CosTheta^2));     % DOA-dependent decay value
                    if p>q
                        DC(p,q,:) = exp(alpha*ww*d/(0.8*U)).*exp(1i*ww*abs(p-q)*d_m/(0.8*U));
                    else
                        DC(p,q,:) = exp(alpha*ww*d/(0.8*U)).*exp(-1i*ww*abs(p-q)*d_m/(0.8*U));
                    end
                end
            end
        end
        
        x = mix_signals(wind,DC,'eigen'); % Correlated wind noise signals (not yet filtered)
        
        multichannel_noise = zeros(size(x)); % init of the filtered wind noise samples

        lpc_coeff = [2.4804   -2.0032    0.5610   -0.0794    0.0392]; % AR parameters from real wind recordings
        lpc_order = 5; % AR order

        for i=1:channel_number
            % Filtering with the AR filter
            multichannel_noise(:,i) = filter(1,[1 -lpc_coeff],x(:,i)); 

            % Scaling to avoid clippings
            multichannel_noise(:,i) = multichannel_noise(:,i)/max(abs(multichannel_noise(:,i)))*0.95; 
        end

        
    end
end
