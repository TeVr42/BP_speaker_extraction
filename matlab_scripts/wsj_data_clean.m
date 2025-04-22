function wsj_data_clean(baseFolder, destinationFolder)
    % baseFolder = 'wsj0-2mix/cv';
    % destinationFolder = 'wsj_cleaned';

    s1Folder = fullfile(baseFolder, 's1');
    s2Folder = fullfile(baseFolder, 's2');

    s1Files = dir(fullfile(s1Folder, '*.wav'));
    s2Files = dir(fullfile(s2Folder, '*.wav'));

    for i = 1:length(s1Files)
        fileName = s1Files(i).name;
        first8 = fileName(1:8);
        destFilePath = fullfile(destinationFolder, [first8, '.wav']);
        if ~isfile(destFilePath)
            copyfile(fullfile(s1Folder, fileName), destFilePath);
        end

    end

    for i = 1:length(s2Files)
        fileName = s2Files(i).name;
        underscores = strfind(fileName, '_');
        second8 = fileName(underscores(2) + 1 : underscores(2) + 8);
        destFilePath = fullfile(destinationFolder, [second8, '.wav']);
        if ~isfile(destFilePath)
            copyfile(fullfile(s2Folder, fileName), destFilePath);
        end

    end

    disp('Processing completed.');
end