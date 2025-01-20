clear all;
close all;

withTrain=0;

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

    % Initialize cell arrays to store sequences and labels
    sequences = {};
    labels = [];

    % Process each file
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

            % Define seizure start and end points (adjust as needed)
            seizure_start = 512 * 3 * 60; % Example start time
            seizure_end = T - seizure_start;

            % Loop through the EEG data with the specified step size
            for start_idx = 1:stepSize:(T - frameLength)
                % Determine the class based on the frame's position
                if start_idx + frameLength - 1 < seizure_start
                    class_label = 0; % Pre-seizure
                elseif start_idx >= seizure_start && start_idx + frameLength - 1 <= seizure_end
                    class_label = 1; % Seizure
                else
                    class_label = 0; % Post-seizure
                end

                % Extract data for each selected electrode
                frameData = eegData(start_idx:start_idx + frameLength - 1, selected_electrodes);

                % Append sequence and label
                sequences{end+1} = frameData; % Store as a cell array of matrices
                labels(end+1, 1) = class_label; % Store labels
            end
        end
    end

    % Convert labels to categorical
    labels = categorical(labels);

    % Split data into training and testing sets
    splitRatio = 0.8;
    numTrain = floor(splitRatio * length(sequences));
    idx = randperm(length(sequences));

    XTrain = sequences(idx(1:numTrain));
    YTrain = labels(idx(1:numTrain));

    XTest = sequences(idx(numTrain+1:end));
    YTest = labels(idx(numTrain+1:end));

    % Define LSTM network
    numChannels = num_electrodes;
    numHiddenUnits = 64;
    numClasses = numel(categories(labels));
    if withTrain
        layers = [ ...
            sequenceInputLayer(numChannels)
            bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
            bilstmLayer(numHiddenUnits / 2, 'OutputMode', 'last')
            fullyConnectedLayer(numClasses)
            softmaxLayer];
    
        % Specify training options
        options = trainingOptions("adam", ...
            MaxEpochs=200, ...
            InitialLearnRate=0.002,...
            GradientThreshold=1, ...
            Shuffle="never", ...
            Plots="training-progress", ...
            Metrics="accuracy", ...
            Verbose=false);


    % Train LSTM network
    
        net = trainnet(XTrain, YTrain, layers, 'crossentropy', options);
        feval(@save,['F:\LICENTA\Cod sursa\LSTM Models\LSTM_ID', num2str(patientID), '1.mat'],'net');
    else
        feval(@load,['F:\LICENTA\Cod sursa\LSTM Models\LSTM_ID', num2str(patientID), '.mat']);
    end

    acc = testnet(net,XTest,YTest,"accuracy");
    [YMTest, scoruriTest] = predict(net, XTest);

    % Save trained model
   % save('trainedLSTM.mat', 'net');
else
    fprintf('Folder %s does not exist.\n', patientFolder);
end