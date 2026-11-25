function exportReferenceData(dataPath, outPath)
% Exports TackleMMe step-1 reference data in a Python-readable format.
%
% MATLAB tables, string arrays, categoricals and datetimes are stored as
% opaque objects that scipy.io.loadmat cannot decode. This function
% converts them to plain cells/structs and saves with -v7.
%
% Tables become structs with fields:
%   tmmeType = 'table', columns (cellstr), rowNames (cellstr), data (1xN cell)
%
% Arguments:
%   dataPath (string): folder containing dico.mat, contextSpecificModelsBRCA.mat
%       and (optionally) BRCAProjectNo1.mat from Zenodo.
%   outPath (string): output folder (created if missing).

arguments
    dataPath (1,1) string
    outPath (1,1) string
end

if ~isfolder(outPath)
    mkdir(outPath);
end

% --- 1. dico (v7.3 table) ---
fprintf("Exporting dico...\n");
dico = load(fullfile(dataPath, "dico.mat")).dico;
writetable(dico, fullfile(outPath, "dico.csv"));

% --- 2. rFastcormics outputs (input to createProject) ---
fprintf("Exporting contextSpecificModelsBRCA...\n");
csm = load(fullfile(dataPath, "contextSpecificModelsBRCA.mat"));
fprintf("  class(paramsForPipelineControl.discretizedData) = %s\n", ...
    class(csm.paramsForPipelineControl.discretizedData));
csm = toPlain(csm);
save(fullfile(outPath, "contextSpecificModelsBRCA_plain.mat"), "-struct", "csm", "-v7");

% --- 3. Reference project after step 1 (output of createProject/addModelsToProject) ---
projectFile = fullfile(dataPath, "BRCAProjectNo1.mat");
if isfile(projectFile)
    fprintf("Exporting BRCAProjectNo1...\n");
    project = load(projectFile).BRCAProject;
    modelNames = fieldnames(project.models);
    % One variable per model to stay below the 2 GB per-variable limit of -v7
    out = struct();
    out.modelNames = modelNames;
    for i = 1:numel(modelNames)
        out.(modelNames{i}) = toPlain(project.models.(modelNames{i}));
    end
    save(fullfile(outPath, "BRCAProjectNo1_plain.mat"), "-struct", "out", "-v7");
else
    warning("BRCAProjectNo1.mat not found in %s, reference project not exported.", dataPath);
end

% --- 4. Tutorial workspace (exact inputs to createProject/addModelsToProject) ---
workspaceFile = fullfile(dataPath, "workspaceTutoNo1.mat");
if isfile(workspaceFile)
    fprintf("Exporting workspaceTutoNo1 (this can take a while)...\n");
    ws = load(workspaceFile);
    fprintf("  Variables: %s\n", strjoin(fieldnames(ws), ", "));
    if isfield(ws, "paramsForPipelineControl")
        fprintf("  class(workspace paramsForPipelineControl.discretizedData) = %s\n", ...
            class(ws.paramsForPipelineControl.discretizedData));
    end
    ws = toPlain(ws);
    save(fullfile(outPath, "workspaceTutoNo1_plain.mat"), "-struct", "ws", "-v7");
else
    warning("workspaceTutoNo1.mat not found in %s, tutorial workspace not exported.", dataPath);
end

fprintf("Done. Files written to %s\n", outPath);
end


function out = toPlain(x)
% Recursively converts opaque MATLAB types to scipy-readable ones.
    if istable(x)
        out = struct();
        out.tmmeType = 'table';
        out.columns = x.Properties.VariableNames;
        out.rowNames = x.Properties.RowNames;
        out.data = cell(1, width(x));
        for j = 1:width(x)
            out.data{j} = toPlain(x.(j));
        end
    elseif isstring(x)
        if isscalar(x)
            out = char(x);
        else
            out = cellstr(x);
        end
    elseif iscategorical(x) || isdatetime(x) || isduration(x)
        out = cellstr(string(x));
    elseif isstruct(x)
        out = x;
        fn = fieldnames(x);
        for i = 1:numel(x)
            for k = 1:numel(fn)
                out(i).(fn{k}) = toPlain(x(i).(fn{k}));
            end
        end
    elseif iscell(x)
        out = cellfun(@toPlain, x, 'UniformOutput', false);
    elseif isobject(x)
        warning("Unhandled class '%s' converted to string.", class(x));
        out = cellstr(string(x));
    else
        out = x; % numeric (incl. int8/sparse), logical, char
    end
end
