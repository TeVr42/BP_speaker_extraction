function generate_dataset(nr_mix_samples, datafolder_name, wsj_folder) 

    % Delka nahravky (s)
    record_len = 5;
    
    % Max zdalenost mluvciho
    max_dist = 1.25; 
    
    % Maximalni uhel mluvciho
    max_angle = pi/6; % 60 (30 na kazdou)
    
    % Min vzdalenost hluku z pozadi
    min_dist_bg = 1.5;
    
    % Max pocet mluvcich v pozadi
    max_bg_speakers = 8;
    
    % Rozmer mistnosti
    room_size = [5 6 2.5];
    
    % Pozice mikrofonu
    ref_point_mic = [2.5 0.75 1.45];
    mics_pos = [2.47 0.75 1.48; 2.5 0.75 1.48; 2.53 0.75 1.48; 2.47 0.75 1.45; 2.5 0.75 1.45; 2.53 0.75 1.45; 2.47 0.75 1.42; 2.5 0.75 1.42; 2.53 0.75 1.42];
    
    % Nataveni fs
    fs = 16000;
    
    % wsj_folder = './wsj_cleaned/';
    wsj_files = dir(fullfile(wsj_folder, '*')); 
    wsj_files = wsj_files(~[wsj_files.isdir]); 
    wsj_filenames = {wsj_files.name}; 
    
    % Generovani jednotlivych mixu
    for j=1:nr_mix_samples
       
        % -------------------------------------------------------
        % Hlavni mluvci
        main_speaker_pos = generate_speaker_pos(max_dist, max_angle, ref_point_mic);
      
        random_index = randi(length(wsj_filenames));
        random_speaker = wsj_filenames{random_index};
        filename_speaker = fullfile(wsj_folder, random_speaker);
        [main_speaker_full, fs1] = audioread(filename_speaker);
    
        speakers_names = random_speaker;
    
        % -------------------------------------------------------
        % Priprava matic pro mluvci
        num_of_speakers_bg = randi([1, max_bg_speakers]);
        bg_speakers = zeros(num_of_speakers_bg, fs*record_len);
    
        main_speaker = zeros(1, fs*record_len);
        main_speaker = add_sound(main_speaker, main_speaker_full, 1);
        % -------------------------------------------------------
        % Mluvci v pozadi
        bg_speakers_pos = zeros(num_of_speakers_bg, 3);
        for i=1:num_of_speakers_bg
            bg_speakers_pos(i, :) = generate_bg_pos(min_dist_bg, ref_point_mic, room_size);
    
            found_uniquel_speaker = false;
            while ~found_uniquel_speaker
                random_index = randi(length(wsj_filenames));
                random_speaker = wsj_filenames{random_index};
                
                found_uniquel_speaker = true;
    
                % Zajisteni rozdilnych mluvcich - kontrola jmena
                for i_speaker_name = 1:i
                    if i == 1
                        speaker_name = speakers_names;
                    else
                        speaker_name = speakers_names(i_speaker_name, :);
                    end
                    if strcmp(random_speaker(1:3), speaker_name(1:3))
                        found_uniquel_speaker = false;
                        break;
                    end
                end
            end
    
            speakers_names = [speakers_names; random_speaker];
    
            filename_bg_speaker = fullfile(wsj_folder, random_speaker);
            [bg_speaker, fs2] = audioread(filename_bg_speaker);
            bg_speakers = add_sound(bg_speakers, bg_speaker, i);
    
            if fs1 ~= fs || fs2 ~= fs
                error('Nesedí vzorkovací frekvence nahrávek a RIR.');
            end
    
        end
            
        % Generovani RIR
        h_bg = zeros(9,4096,num_of_speakers_bg);
        h_main = zeros(9, 4096);
        for m=1:9
            for n=1:num_of_speakers_bg
                h_bg(m, :, n) = generate_rir(mics_pos(m, :),bg_speakers_pos(n, :),room_size);
            end
            h_main(m, :) = generate_rir(mics_pos(m, :),main_speaker_pos,room_size);
        end
        
        % Konvoluce
        bg_mixed_conv = zeros(9,length(main_speaker));
        main_conv = zeros(9,length(main_speaker));
        for m = 1:9
            for n = 1:num_of_speakers_bg
                conv_speaker = filter(h_bg(m,:,n), 1, bg_speakers(n,:));
                bg_mixed_conv(m, :) = bg_mixed_conv(m, :) + conv_speaker;
            end
            main_conv(m, :) = filter(h_main(m,:), 1, main_speaker);
        end
    
        % Pridání šumu: (Nastaveni SNR šumu)
        desired_SNR = rand()*4 + 18; % 18-22 dB
        
        signal_power = mean(main_conv.^2, "all");
        snr_linear = 10^(desired_SNR / 10);
        noise_power = signal_power / snr_linear;
        noise = sqrt(noise_power) * randn(9, size(bg_mixed_conv, 2));
        
        % Uprava SIR (nastaveni SIR speakeru v pozadi vs hlavniho rečníka):
        desired_SIR = rand()*5; % 0-5 dB
        
        sir_linear = 10^(desired_SIR / 10);
        int_power_old = mean(bg_mixed_conv.^2, "all");
        scale_factor = sqrt(signal_power / (int_power_old * sir_linear));
        bg_mixed_conv = scale_factor * bg_mixed_conv;
    
        bg_mixed_conv = bg_mixed_conv + noise;
    
        % Normalizace
        mix_conv = bg_mixed_conv + main_conv;
        max_abs = max(abs([mix_conv; bg_mixed_conv; main_conv]),[],"all");
    
        bg_mixed_conv = bg_mixed_conv / max_abs;
        main_conv = main_conv / max_abs;
        mix_conv = mix_conv / max_abs;
       
        % Ulozeni
        audiowrite(append('./',datafolder_name,'/s', num2str(j-1) , '.wav'), main_conv', fs);
        audiowrite(append('./',datafolder_name,'/y', num2str(j-1) , '.wav'), bg_mixed_conv', fs);
    
        % Ulozeni mixu - jen pro kontrolu
        audiowrite(append('./',datafolder_name,'/mix', num2str(j-1) , '.wav'), mix_conv', fs);
    end

end

function sounds = add_sound(sounds, new_sound, index)
    % Pridani zvuku do matice zvuku - uprava delky a nahodne umisteni
    len_sound = length(sounds);
    
    if length(new_sound) >= len_sound
        start_pos = randi([1, length(new_sound) - len_sound + 1]);
        sounds(index, :) = new_sound(start_pos:start_pos+len_sound-1);
        
    else
        len_new = length(new_sound);
        offset = randi([0, len_sound - len_new]);
        
        new_segment = zeros(1, len_sound);
        new_segment(offset+1:offset+len_new) = new_sound;
        
        sounds(index, :) = new_segment;
    end
end


function h = generate_rir(r,s,room_size)
    c = 340;                    % Sound velocity (m/s)
    fs = 16000;                 % Sample frequency (samples/s)
    L = room_size;              % Room dimensions [x y z] (m)
    nsample = 4096;             % Number of samples
    beta = 0.4;                 % Reverberation time (s)
    mtype = 'hypercardioid';    % Type of microphone
    order = -1;                 % -1 equals maximum reflection order!
    dim = 3;                    % Room dimension
    orientation = [pi/2 0];     % Microphone orientation (rad)
    hp_filter = 1;              % Disable high-pass filter

    h = rir_generator(c, fs, r, s, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
end

function pos = generate_speaker_pos(max_dist, max_angle, ref_point_mic)
    valueY = rand() * max_dist;
    limit1 = sqrt((max_dist^2) - (valueY^2)); % osetreni maximalni vzdalenosti
    limit2 = tan(max_angle) * valueY; % osetreni uhlu
    maxX = min(limit1, limit2);
    valueX = rand() * (2*maxX) - maxX; % moznost na obe strany
    valueZ = 1.3 + (1.85-1.3)*rand(); % vyska mluvciho

    pos = round([valueX + ref_point_mic(1), valueY + ref_point_mic(2), valueZ], 2);
end

function pos = generate_bg_pos(min_dist, ref_point_mic, room_size)
    found_pos = false;
    
    while ~found_pos
        valueX = rand() * room_size(1);
        valueY = rand() * room_size(2);

        dist = sqrt((valueX - ref_point_mic(1))^2 + (valueY - ref_point_mic(2))^2);
        if dist > min_dist
            found_pos = true;
        end
    end
    valueZ = 1.3 + (1.85-1.3)*rand();

    pos = round([valueX, valueY, valueZ],2);
end