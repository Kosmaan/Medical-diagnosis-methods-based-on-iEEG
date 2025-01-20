clear all
close all

% Define base directory where dataset files are located
baseDirectory = 'F:\LICENTA\Dataset';

% Get user input for patient ID, number of electrodes, and frame overlap
patientID = input("Introduce ID-ul pacientului: ");
num_electrodes = input("Introduceti numarul de electrozi de selectat: ");
overlap_seconds = input("Introduceti suprapunerea cadrelor in secunde: ");

% Constants
frameTime = 6; % Frame duration in seconds
samplesPerSecond = 512; % Sampling rate
frameLength = frameTime * samplesPerSecond; % Length of one frame in samples
stepSize = frameLength - overlap_seconds * samplesPerSecond; % Step size in samples

patientFolder = sprintf('ID%d', patientID);
folderPath = fullfile(baseDirectory, patientFolder); % Build path to patient folder

% Check if the patient folder exists
if exist(folderPath, 'dir')
    % Find all .mat files in the patient folder
    filePattern = fullfile(folderPath, 'Sz*.mat');
    files = dir(filePattern);

    % Initialize variables for preallocation
    total_frames = 0;

    % First pass: Calculate the total number of frames for preallocation
    for fileIndex = 1:length(files)
        if ~strcmp(files(fileIndex).name, '.') && ~strcmp(files(fileIndex).name, '..')
            currentFile = fullfile(folderPath, files(fileIndex).name);

            % Load EEG data to calculate frames
            loadedData = load(currentFile, 'EEG');
            eegData = loadedData.EEG;
            T = size(eegData, 1); % Total samples in the file

            % Compute the number of frames for this file
            num_frames = floor((T - frameLength) / stepSize) + 1;
            total_frames = total_frames + num_frames * num_electrodes;
        end
    end

    % Preallocate feature_matrix and class_array
    feature_matrix = zeros(total_frames, frameLength); % Each row corresponds to one frame
    YTrain = zeros(total_frames, 1); % Class labels for each frame

    % Initialize index for inserting data
    current_index = 1;

    % Initialize cell arrays to store data for each selected electrode
    seizure_data_per_electrode = cell(1, num_electrodes);       % For class 1 (seizure)
    non_seizure_data_per_electrode = cell(1, num_electrodes);   % For class 0 + 2 (non-seizure)

    % Second pass: Populate feature_matrix and class_array
    for fileIndex = 1:length(files)
        if ~strcmp(files(fileIndex).name, '.') && ~strcmp(files(fileIndex).name, '..')
            currentFile = fullfile(folderPath, files(fileIndex).name);

            % Load EEG data from file
            loadedData = load(currentFile, 'EEG');
            eegData = loadedData.EEG;

            % Get dimensions of the EEG data
            [T, M] = size(eegData);

            % Compute standard deviations for electrode selection
            std_devs = std(eegData);
            [~, sorted_indices] = sort(std_devs, 'descend');
            selected_electrodes = sorted_indices(1:num_electrodes);

            % Define seizure start and end points
            seizure_start = 512 * 3 * 60; % Example start time
            seizure_end = T - seizure_start;

            for i = 1:num_electrodes
                electrode = selected_electrodes(i); % Get the electrode index

                % Append seizure data for this electrode
                seizure_data_per_electrode{i} = [seizure_data_per_electrode{i}; ...
                    eegData(seizure_start:seizure_end, electrode)];

                % Append non-seizure data for this electrode
                non_seizure_data_per_electrode{i} = [non_seizure_data_per_electrode{i}; ...
                    eegData(1:seizure_start-1, electrode); ... % Before seizure
                    eegData(seizure_end+1:end, electrode)];    % After seizure
            end
            % Loop through the EEG data with the specified step size
            for start_idx = 1:stepSize:(T - frameLength)
                % Determine the class based on the frame's position
                if start_idx + frameLength - 1 < seizure_start
                    class_label = 0; % Pre-seizure
                elseif start_idx >= seizure_start && start_idx + frameLength - 1 <= seizure_end
                    class_label = 1; % Seizure
                else
                    class_label = 2; % Post-seizure
                end

                % Extract data for each selected electrode and add to the feature matrix
                for electrode = selected_electrodes
                    frameData = eegData(start_idx:start_idx + frameLength - 1, electrode);
                    feature_matrix(current_index, :) = frameData'; % Assign frame to row
                    YTrain(current_index) = class_label; % Assign class label
                    current_index = current_index + 1;
                end
            end
        end
    end

    % Trim unused rows from the feature matrix and class array
    feature_matrix = feature_matrix(1:current_index-1, :);
    YTrain = YTrain(1:current_index-1);

    % Apply Butterworth filter to the feature matrix
    fL = 0.5;
    fH = 150;
    n = 5; % Filter order
    [bd, ad] = butter(n, [fL, fH] / (samplesPerSecond / 2));

    % Filter the data row-wise
    XTrain = filtfilt(bd, ad, feature_matrix')'; % Transpose for filtering
else
    fprintf('Folder %s does not exist.\n', patientFolder);
end

%%
% Find indices for each class
seizure_indices = find(YTrain == 1); % Indices for class 1 (seizure)
non_seizure_indices = find(YTrain == 0 | YTrain == 2); % Indices for classes 0 and 2 (non-seizure)

% Number of seizure frames
num_seizure_frames = length(seizure_indices);

% Randomly sample the same number of frames for non-seizure classes
rng(1); % For reproducibility
non_seizure_indices_sampled = randsample(non_seizure_indices, num_seizure_frames);

% Combine the balanced indices
balanced_indices = [seizure_indices; non_seizure_indices_sampled];

% Shuffle the combined indices
balanced_indices = balanced_indices(randperm(length(balanced_indices)));

% Create balanced XTrain and YTrain
XTrain = XTrain(balanced_indices, :);
YTrain = YTrain(balanced_indices, :);

% Total number of balanced samples
num_samples = size(XTrain, 1);

% Create a randomized index for splitting
indices = randperm(num_samples);

% Compute split point (75% train, 25% test)
split_index = round(0.75 * num_samples);

% Split the data into training and testing sets
XTest = XTrain(indices(split_index + 1:end), :);
YTest = YTrain(indices(split_index + 1:end), :);

XTrain = XTrain(indices(1:split_index), :);
YTrain = YTrain(indices(1:split_index), :);

%%
numTrees = 200;
% Mdl = fitcensemble(XTrain, YTrain, 'Method', 'Bag', 'Learner', 'tree', "NumLearningCycles", numTrees);
[YMTest, scoruriTest] = predict(Mdl, XTest);
[YMTrain, scoruriTrain] = predict(Mdl, XTrain);
AccTest = mean(YMTest == YTest);
AccTrain = mean(YMTrain == YTrain);
ccTrain = confusionchart(YTrain, YMTrain);
ccTest = confusionchart(YTest, YMTest);

% matriceRezultate(32,:) = [patientID, num_electrodes, numTrees, AccTrain, AccTest];