clear all; close all; clc;

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
ref_bod_mic = [2.5 0.75 1.45];
mics_pos = [2.47 0.75 1.48; 2.5 0.75 1.48; 2.53 0.75 1.48; 2.47 0.75 1.45; 2.5 0.75 1.45; 2.53 0.75 1.45; 2.47 0.75 1.42; 2.5 0.75 1.42; 2.53 0.75 1.42];

main_speaker_posx = zeros(1000,2);
bg_speakers_posx = zeros(1000,2);

figure;
hold on;
grid on;
axis equal;

for i = 1:1000
main_speaker_posx(i, :) = generate_speaker_pos(max_dist, max_angle, ref_bod_mic);
end
for i = 1:20000
bg_speakers_posx(i, :) = generate_bg_pos(min_dist_bg, ref_bod_mic, room_size);
end

plot(main_speaker_posx(:,1), main_speaker_posx(:,2),".", 'DisplayName', 'Pozice hlavního mluvčího')
plot(bg_speakers_posx(:,1), bg_speakers_posx(:,2),".", 'DisplayName', 'Pozice interferujících mluvčích')
plot(mics_pos(:,1), mics_pos(:,2), '.', 'DisplayName', 'Pozice mikrofonu')

xlabel('X [m]', 'FontSize', 14);
ylabel('Y [m]', 'FontSize', 14);
title('Rozmístění mluvčích a mikrofonů v místnosti', 'FontSize', 18);
legend('FontSize', 15);
xlim([0 5]);
set(gca, 'FontSize', 14);

function pos = generate_speaker_pos(max_dist, max_angle, ref_bod_mic)
    valueY = rand() * max_dist;
    limit1 = sqrt((max_dist^2) - (valueY^2)); % osetreni maximalni vzdalenosti
    limit2 = tan(max_angle) * valueY; % osetreni uhlu
    maxX = min(limit1, limit2);
    valueX = rand() * (2*maxX) - maxX; % moznost na obe strany

    pos = round([valueX + ref_bod_mic(1), valueY + ref_bod_mic(2)], 2);
end

function pos = generate_bg_pos(min_dist, ref_bod_mic, room_size)
    found_pos = false;
    
    while ~found_pos
        valueX = rand() * room_size(1);
        valueY = rand() * room_size(2);

        dist = sqrt((valueX - ref_bod_mic(1))^2 + (valueY - ref_bod_mic(2))^2);
        if dist > min_dist
            found_pos = true;
        end
    end

    pos = round([valueX, valueY],2);
end