clc
clear


shuffle_enable = 1;
fs = 16000;


addpath  D:\Audio\TIMIT_Dataset\tools\Wind-Generator-master
TIMIT_speaker_path = '../../../TIMIT_Dataset/Data/';
cough_path = 'coughs/';

speech_data = 1;

if speech_data ==1
    clean_path = {[TIMIT_speaker_path 'train_val_clean/'],[TIMIT_speaker_path 'train_val_clean/'],[TIMIT_speaker_path 'test_cln/']};
else
    clean_path = {[cough_path 'train_val_clean/'],[cough_path 'train_val_clean/'],[cough_path 'test_cln/']};
end


sets = {'tr','dt'};

for setindx = 1:length(sets)
    set = sets{setindx};
    clean = [clean_path{setindx} '*.wav'];
    eval([set '_bth= dir(clean);']);
    eval([ 'isolated_cln = ' set '_bth;'])
    eval([set '_utt={};']);
    for uind = 1:length(isolated_cln)
        eval([set '_utt(end+1)= cellstr(' set '_bth(uind).name);'])
    end
end

TRAINNSR = [-10,5]; % Train/validation snr range

simu_rir_path = 'simulated_RIR/';

outnames = {'train_mixed_wav','validation_mixed_wav'};

train_total_data_len = 10;  % train data 10 hours
val_total_data_len = 3;     % validation data 3 hours

total_durations = [train_total_data_len*60*60*16000 val_total_data_len*60*60*16000];
sample_len = 90*fs;    % length of each long concatenated sequence
point_source_flag = 0;

% whether to cancel the echo and reverberation
direct_dur = 0.002*fs;   % dp duration reverberation and echo cancellation
echoic_dur = 0.05*fs;   % d50 duration reverberation cancellation

for i=1:2

    set = sets{i};
    accum_len = 1;
    
    total_duration = total_durations(i);
    
    outname = [sprintf('speech_dp_d50_reverb_simu_30h/') outnames{i}];

    if ~exist(outname,'dir')    
        mkdir(outname);
    end
    
    eval(['clean_utt = ' set '_utt;']);
    
    count = 0;
    
    while accum_len < total_duration
        
        
        rir_list = dir([simu_rir_path '*.mat']);
        rir_name = rir_list(randperm(length(rir_list),1)).name;
        
        load([simu_rir_path rir_name])
        nchan = str2num(rir_name(1));
        
        %   initialization
        utt_ind = randperm(length(clean_utt),1);
        s =  audioread([clean_path{i} clean_utt{utt_ind}]);
        while length(s) < sample_len
            utt_ind = randperm(length(clean_utt),1);
            su =  audioread([clean_path{i} clean_utt{utt_ind}]);
            s = [s; su];
        end
        sdur = length(s);
        
       
        

        recorded_speech = ones(sdur,nchan);
        for ch = 1:nchan
            recorded_speech(:,ch) = fftfilt(h(ch,:), s);
        end
        
        n_white = randn(sdur,nchan);
        WHITENSR_dB = -20-5*rand;
        whitensr = 10^(WHITENSR_dB/10);
        n_white=sqrt(whitensr/sum(sum(n_white.^2))*sum(sum(recorded_speech.^2)))*n_white;
        
        accum_len = accum_len + sdur;
        
        [n, envir] = noise_generator(i,nchan, micro_pos, sdur);
        if length(n)<sdur
            padding = zeros(sdur-length(n),nchan);
            n(length(n)+1:sdur,:) = padding;
        end
        
        if i <= 2                % training and validation data
                
            nsrdb = rand(1)*(TRAINNSR(2)-TRAINNSR(1))+TRAINNSR(1);   
%             nsrdb = 0;
            nsr = 10^(nsrdb/10);                
            n = sqrt(nsr/sum(sum(n.^2))*sum(sum(recorded_speech.^2)))*n;
            
            if point_source_flag == 1
                 x = recorded_speech + n + n_point_ori + n_white; 
                 real_snr = 10*log10(sum(sum(recorded_speech.^2))/sum(sum((n+n_point_ori).^2)));
            else
                x = recorded_speech + n + n_white; 
                real_snr = 10*log10(sum(sum(recorded_speech.^2))/sum(sum((n+n_white).^2)));
            end
            
            if shuffle_enable == 1
                
                refind = randperm(size(x,2),1); % Refenrence channel randomly chosen

                shuffled_4ch_mix_speech = zeros(size(x));
                shuffled_4ch_mix_speech(:,1) = x(:,refind);
                x(:,refind) = x(:,1);
                C = x(:,2:end);
                colrank = randperm(size(C,2));
                shuffled_4ch_mix_speech(:,2:end) = C(:,colrank);
            
                x(:,nchan+1) = recorded_speech(:,refind);
                
                ref_h = h(refind,:);
                
                [~,max_ind] = max(ref_h);
                echoinic_speech =  fftfilt(ref_h(1:max_ind+echoic_dur), s); 
                direct_speech = fftfilt(ref_h(1:max_ind+direct_dur), s);
                
                x(:,nchan+2) = echoinic_speech;
                x(:,nchan+3) = direct_speech;
 
                if point_source_flag == 1
                    audiowrite([outname '/'  rir_name(1:end-16) '_AC' num2str(ac_ind) '_' envir '_' num2str(real_snr) 'dB.wav'],x/max(max(abs(x))),fs);
                else
                    audiowrite([outname '/'  rir_name(1:end-16) '_' envir '_SNR_' num2str(real_snr) 'dB.wav'],x/max(max(abs(x))),fs);
                end
            else
                x(:,end+1) = recorded_speech(:,refIndx);
                audiowrite([outname '/' envir '_'  num2str(real_snr) 'dB.wav'],x/max(max(abs(x))),fs);
%                 audiowrite([outname '/' int2str(count) '.wav'],x/max(max(abs(x))),fs);
            end
            
            
        else
            nsrdb = 0;
            nsr = 10^(nsrdb/10);                
            n=sqrt(nsr/sum(sum(n.^2))*sum(sum(recorded_speech.^2)))*n;
            x=recorded_speech+n;    
            

            audiowrite([outname int2str(count) '_ms.wav'],x/max(max(abs(x))),fs);   
            audiowrite([outname int2str(count) '_noise.wav'],n/max(max(abs(n))),fs); 
%                 audiowrite([udir oname(1:end-4) '_cln.wav'],s/max(max(abs(s))),fs); 
%             audiowrite([outname int2str(count) '_refms.wav'],x(:,refIndx)/max(abs(x(:,refIndx))),fs);   
            audiowrite([outname int2str(count) '_cln.wav'],recorded_speech(:,refIndx)/max(abs(recorded_speech(:,refIndx))),fs); 
        end
        count = count+1;
    end
end
