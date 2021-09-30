% Generating enough multichannel RIR to convolve with single channel speech
% 


clc
clear

addpath  ../tools/RIR-Generator-master
addpath  ../tools/ANF-Generator-master
addpath  ../tools/Wind-Generator-master

c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)  
height = 1;                 % Microphone array height(m)  

refind = 1; % Refenrence channel index = 1

outpath = 'simulated_RIR/';

if ~exist(outpath,'dir')
    mkdir(outpath);
end  

% the occurance of each unique array with various channel number and
% various geometry
ch_num = [2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8];
type ={{'line'},{'line','cir'},{'line','cir','cir_point','irregular_line','irregular_random'},{'line','cir','cir_point','irregular_line','irregular_random'},{'line','cir','cir_point','irregular_line','irregular_random'},{'line','cir','cir_point','irregular_line','irregular_random'},{'line','cir','cir_point','irregular_line','irregular_random'}};
diameters = {[0.04, 0.06, 0.1, 0.15, 0.20, 0.25],[0.1,0.15,0.2,0.3],[0.8,0.15,0.25,0.35],[0.15,0.2,0.25,0.3],[0.1,0.2,0.25,0.3,0.4],[0.25,0.35,0.3,0.4],[0.2,0.25,0.4,0.5]};
theta = 0:10:360;
propos = 0:0.1:1;

% room parameters for RIR_Generator
rooms = {[6 6 3],[10,6,3],[20,10,6],[9,6,4],[10,10,3]};     % Roomsize
beta_list =  [0.125 0.137 0.258  0.154 0.153];              % RT60 coefficient

beta_max = 1;                                               % max RT60 time

count = 0;

while count < 1500
    
    % selecting room
    roomind = randperm(length(rooms),1);
    beta_min = beta_list(roomind);
    r = rooms{roomind};
    
    % selecting channel number
    channel_number = ch_num(randperm(length(ch_num),1));
    
    % Randomly selecting channel number
    beta = rand(1)*(beta_max-beta_min)+beta_min;
    % Microphone position 
    micro_pos = zeros(channel_number,3);
    
    % Microphone geometry 
    subtype = type{channel_number-1};
    shape = subtype{randperm(length(subtype),1)};
    subdiams = diameters{channel_number-1};
    diam = subdiams(randperm(length(subdiams),1));
    
    % Source-array relative position 
    src_ang = theta(randperm(length(theta),1));
    propo = propos(randperm(length(propos),1));
    center = [0.25*r(1) + (0.5*r(1))*rand 0.25*r(2) + (0.5*r(2))*rand height];
    rw = [(r(1)-center(1))/cosd(src_ang) (0-center(1))/cosd(src_ang) (r(2)-center(2))/sind(src_ang) (0-center(2))/sind(src_ang)];
    expected_ray = min(rw(rw>0));
    expected_line = min(5,expected_ray);
    rho=0:0.1:expected_ray;
    length_pro = 0.5 + propo*expected_line; 
    expected_src = [max(0.5,min(center(1)+ cosd(src_ang)*length_pro,r(1)-1)) max(0.5,min(center(2)+sind(src_ang)*length_pro,r(2)-1)) height];
    Arr_Src_dist = sqrt(sum((expected_src-center).^2,'all'));
    array_angle = randperm(360,1);
    
    % Shaping the array
    switch shape
        case 'line'
            if mod(channel_number,2) == 0
                for mic = 1:2:channel_number-1 
                    sgn = 1;
                    micro_pos(mic,:) = [center(1) + sgn*diam*(1/(2*(channel_number-1))+floor(mic/2)/(channel_number-1))*cosd(array_angle)  center(2)+ sgn*diam*(1/(2*(channel_number-1))+floor(mic/2)/(channel_number-1))*sind(array_angle) height];
                end
                for mic = 2:2:channel_number 
                    sgn = -1;
                    micro_pos(mic,:) = [center(1) + sgn*diam*(1/(2*(channel_number-1))+(floor(mic/2)-1)/(channel_number-1))*cosd(array_angle)  center(2)+ sgn*diam*(1/(2*(channel_number-1))+(floor(mic/2)-1)/(channel_number-1))*sind(array_angle) height];
                end  

            else 
                micro_pos(1,:) = [center(1) center(2) height];
                for mic = 2:2:channel_number 
                    sgn = 1;
                    micro_pos(mic,:) = [center(1) + sgn*diam/(2*(channel_number-1))*cosd(array_angle)  center(2)+ sgn*diam/(2*(channel_number-1))*sind(array_angle) height];
                end
                for mic = 3:2:channel_number 
                    sgn = -1;
                    micro_pos(mic,:) = [center(1) + sgn*diam/(2*(channel_number-1))*cosd(array_angle)  center(2)+ sgn*diam/(2*(channel_number-1))*sind(array_angle) height];
                end                    
            end

       case 'cir'

            for mic = 1:channel_number
                micro_pos(mic,:) = [center(1) + cosd(array_angle + (mic-1)*360/channel_number)*diam/2, center(2)+ sind(array_angle + (mic-1)*360/channel_number)*diam/2, height];
            end
       case 'cir_point'

            micro_pos(1,:) = [center(1) center(2) center(3)];
            for mic = 2:channel_number 
                micro_pos(mic,:) = [center(1) + cosd(array_angle + (mic-2)*360/(channel_number-1))*diam/2, center(2)+ sind(array_angle + (mic-2)*360/(channel_number-1))*diam/2, height];
            end
       case 'irregular_line'

            if mod(channel_number,2) == 0
                    init_length = 0.1*rand*diam + 0.1*diam;
                    micro_pos(1,:) = [center(1) + init_length*cosd(array_angle)  center(2) + init_length*sind(array_angle) height];
                    micro_pos(2,:) = [center(1) - init_length*cosd(array_angle)  center(2) - init_length*sind(array_angle) height];
                for mic_pair = 2:channel_number/2
                    random_length = 0.4*rand*diam + 0.1*diam;
                    micro_pos(2*mic_pair -1,:) = [micro_pos(2*mic_pair-3,1)+ random_length*cosd(array_angle) micro_pos(2*mic_pair-3,2)+ random_length*sind(array_angle)  height];
                    micro_pos(2*mic_pair,:) = [micro_pos(2*mic_pair-2,1)- random_length*cosd(array_angle) micro_pos(2*mic_pair-2,2)-random_length*sind(array_angle) height];
                end
            else 
                micro_pos(1,:) = [center(1) center(2) height];
                for mic_pair = 1:(channel_number-1)/2
                    random_length = 0.4*rand*diam + 0.1*diam;
                    micro_pos(2*mic_pair,:) = [micro_pos(max(1,2*mic_pair-2),1)+random_length*cosd(array_angle) micro_pos(max(1,2*mic_pair-2),2)+ random_length*sind(array_angle) height];
                    micro_pos(2*mic_pair+1,:) = [micro_pos(2*mic_pair-1,1)-random_length*cosd(array_angle) micro_pos(2*mic_pair-1,2)-random_length*sind(array_angle) height];
                end                  
            end

       case 'irregular_random'
           for mic = 1:channel_number 
               random_angle = randperm(360,1);
               rand_len = (rand*0.9+0.1)*(-1)^(randperm(2,1))/2;
               x1  = center(1) + diam*rand_len*cosd(random_angle);  
               y1 = center(2) + diam*rand_len*sind(random_angle);
               micro_pos(mic,:) = [center(1) + diam*rand_len*cosd(random_angle)  center(2)+ diam*rand_len*sind(random_angle) height];
           end
    end 
    % Generated multichannel RIR 
    h = rir_generator(c, fs, micro_pos, expected_src, r, beta);
    
    save([outpath int2str(channel_number) 'ch_' shape '_RT60_' sprintf('%.4f',beta) 's_D' num2str(diam*100) 'cm_ASdist' sprintf('%.2f',Arr_Src_dist) 'm.mat'],'h','center','r','beta','micro_pos','diam','shape')
    
    count = count + 1;
end


