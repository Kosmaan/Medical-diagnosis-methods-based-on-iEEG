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
% Parameters for the figure
num_bins = 50; % Number of bins for the histograms
num_rows = num_electrodes;
num_cols = 2;

figure;

% Loop through each electrode to create histograms
for i = 1:num_electrodes
    % Seizure data histogram (class 1)
    subplot(num_rows, num_cols, (i-1)*2 + 1);
    histogram(seizure_data_per_electrode{i}, num_bins, 'FaceColor', 'r');
    title(sprintf('Electrode %d: Seizure', selected_electrodes(i)));
    xlabel('Amplitude');
    ylabel('Frequency');

    % Non-seizure data histogram (class 0 + 2)
    subplot(num_rows, num_cols, (i-1)*2 + 2);
    histogram(non_seizure_data_per_electrode{i}, num_bins, 'FaceColor', 'b');
    title(sprintf('Electrode %d: Non-Seizure', selected_electrodes(i)));
    xlabel('Amplitude');
    ylabel('Frequency');
end

sgtitle('Histograms for Seizure and Non-Seizure Data');
%%
% Initialize arrays to store mean and dispersion for each electrode and class
mean_seizure = zeros(num_electrodes, 1);
dispersion_seizure = zeros(num_electrodes, 1);

mean_non_seizure = zeros(num_electrodes, 1);
dispersion_non_seizure = zeros(num_electrodes, 1);

% Compute mean and dispersion for each electrode
for i = 1:num_electrodes
    % Seizure data (class 1)
    seizure_data = seizure_data_per_electrode{i};
    mean_seizure(i) = mean(seizure_data);
    dispersion_seizure(i) = var(seizure_data); % Variance as dispersion

    % Non-seizure data (class 0 + 2)
    non_seizure_data = non_seizure_data_per_electrode{i};
    mean_non_seizure(i) = mean(non_seizure_data);
    dispersion_non_seizure(i) = var(non_seizure_data);
end

disp('Mean and Dispersion for Seizure Data (Class 1):');
for i = 1:num_electrodes
    fprintf('Electrode %d: Mean = %.2f, Dispersion = %.2f\n', selected_electrodes(i), mean_seizure(i), dispersion_seizure(i));
end

disp('Mean and Dispersion for Non-Seizure Data (Classes 0 + 2):');
for i = 1:num_electrodes
    fprintf('Electrode %d: Mean = %.2f, Dispersion = %.2f\n', selected_electrodes(i), mean_non_seizure(i), dispersion_non_seizure(i));
end

%%
% Initialize variables to store the mean frame for each class and electrode
mean_frame_seizure = zeros(num_electrodes, frameLength);  % Mean frame for class 1
mean_frame_non_seizure = zeros(num_electrodes, frameLength);  % Mean frame for class 0+2

% Loop through each selected electrode
for i = 1:num_electrodes
    % Seizure data (class 1)
    seizure_data = seizure_data_per_electrode{i};
    num_seizure_frames = floor(length(seizure_data) / frameLength); % Number of full frames
    seizure_frames = reshape(seizure_data(1:num_seizure_frames * frameLength), frameLength, [])';
    mean_frame_seizure(i, :) = mean(seizure_frames, 1); % Compute mean frame for seizure data

    % Non-seizure data (class 0+2)
    non_seizure_data = non_seizure_data_per_electrode{i};
    num_non_seizure_frames = floor(length(non_seizure_data) / frameLength); % Number of full frames
    non_seizure_frames = reshape(non_seizure_data(1:num_non_seizure_frames * frameLength), frameLength, [])';
    mean_frame_non_seizure(i, :) = mean(non_seizure_frames, 1); % Compute mean frame for non-seizure data
end
%%
% Number of patients
num_patients = length(mean_frame_seizure_all);
num_electrodes = 5;
% Initialize distance matrices for each class
distance_matrix_seizure = zeros(num_patients);       % For seizure class
distance_matrix_non_seizure = zeros(num_patients);   % For non-seizure class

% Calculate distances for seizure class (class 1)
for p1 = 1:num_patients
    for p2 = 1:num_patients
        % Compute the distance for all electrodes and take the average
        total_distance = 0;
        for electrode = 1:num_electrodes
            frame1 = mean_frame_seizure_all{p1}(electrode, :); % Mean frame for patient p1
            frame2 = mean_frame_seizure_all{p2}(electrode, :); % Mean frame for patient p2
            total_distance = total_distance + sqrt(sum((frame1 - frame2).^2)); % Euclidean distance
        end
        distance_matrix_seizure(p1, p2) = total_distance / num_electrodes; % Average over electrodes
    end
end

% Calculate distances for non-seizure class (class 0+2)
for p1 = 1:num_patients
    for p2 = 1:num_patients
        % Compute the distance for all electrodes and take the average
        total_distance = 0;
        for electrode = 1:num_electrodes
            frame1 = mean_frame_non_seizure_all{p1}(electrode, :); % Mean frame for patient p1
            frame2 = mean_frame_non_seizure_all{p2}(electrode, :); % Mean frame for patient p2
            total_distance = total_distance + sqrt(sum((frame1 - frame2).^2)); % Euclidean distance
        end
        distance_matrix_non_seizure(p1, p2) = total_distance / num_electrodes; % Average over electrodes
    end
end

% Display the results
disp('Distance matrix for seizure class (Class 1):');
disp(distance_matrix_seizure);

disp('Distance matrix for non-seizure class (Class 0+2):');
disp(distance_matrix_non_seizure);
