function make_observed_bin_FPR_figure_all12(projectRoot, outputDir)
% MAKE_OBSERVED_BIN_FPR_FIGURE_ALL12
% Reproduce Figure 5 from FINISHED observed-bin null-validation CSVs.
% No calibration or Monte Carlo simulations are rerun.
%
% Usage in p-value2.0 folder:
%   make_observed_bin_FPR_figure_all12(pwd)
%
% Requires in the independent and paired results directories:
%   FPR_BY_OBSERVED_BIN.csv
%
% Output: Figure_5_observed_effect_FPR_all12_80_95.{png,svg,fig}
% Panels A-C independent and D-F paired, four thresholds per panel,
% 80% (solid) and 95% (dashed) persistence.
%
% The y-axis is P_H0(at least one eligible crossing IN THE BIN) x 100.
% Isolated-bin probabilities are NONADDITIVE: an H0 trajectory may
% qualify in several bins. For the overall Type I error of the SINGLE
% adaptive observed-effect stopping rule, see FPR_BY_THRESHOLD_SUMMARY.csv
% or Table 2. Ineligible minimum-N bins are not plotted.

if nargin < 1 || isempty(projectRoot), projectRoot = pwd; end
if nargin < 2 || isempty(outputDir)
    outputDir=fullfile(projectRoot,'FPR_by_effect_figures','observed_effect_all12');
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end
folders={'ttest_persistence_80_95_results', ...
         'paired_ttest_persistence_80_95_results'};
D=cell(2,1);
for k=1:2
    filename=fullfile(projectRoot,folders{k},'FPR_BY_OBSERVED_BIN.csv');
    assert(exist(filename,'file')==2,'Cannot find file: %s',filename);
    D{k}=readtable(filename,'TextType','string');
    required={'persistence_target','bin_lower','criterion','threshold', ...
        'minimum_N','isolated_bin_FPR','null_trajectories'};
    assert(all(ismember(required,D{k}.Properties.VariableNames)), ...
        'Input table missing expected columns: %s',filename);
    assert(height(D{k})==21*12*2,'Expected 504 observed-bin rows: %s',filename);
    assert(all(D{k}.null_trajectories==800000),'Expected 800,000 null trajectories');
end

f=figure('Color','w','Position',[70 65 1760 1040]);
colors=[0.03 0.50 0.55;0.72 0.17 0.29;0.62 0.37 0.03;0.41 0.27 0.65];
familyNames={'p thresholds','Reference BF thresholds','p-matched BF thresholds'};
for design=1:2
    T=D{design};
    for family=1:3
        ax=subplot(2,3,(design-1)*3+family);hold(ax,'on');
        familyRows=select_family(T.criterion,family);
        cutoff=sort(unique(T.threshold(familyRows)));
        assert(numel(cutoff)==4,'Expected 4 thresholds in each family');
        for j=1:4
            for ti=1:2
                targets=[0.80 0.95];tar=targets(ti);
                use=familyRows & abs(T.threshold-cutoff(j)) < 1e-7*max(1,abs(cutoff(j))) & ...
                    abs(T.persistence_target-tar)<1e-8 & isfinite(T.minimum_N);
                x=T.bin_lower(use);
                y=100*T.isolated_bin_FPR(use);
                [x,ord]=sort(x);y=y(ord);
                y(x<=0|y<=0)=NaN;  % log axes: leave ineligible/zero bins blank
                if ti==1,ls='-';mk='o';else,ls='--';mk='s';end
                plot(ax,x,y,'Color',colors(j,:),'LineStyle',ls,'Marker',mk, ...
                    'LineWidth',1.9,'MarkerSize',3.3,'MarkerFaceColor','w', ...
                    'DisplayName',sprintf('%s: %d%%',threshold_label(family,cutoff(j)),round(tar*100)));
            end
        end
        set(ax,'XScale','log','YScale','log','XTick',[.1 .2 .3 .5 1 2 4 6], ...
            'YTick',[.0001 .001 .01 .1 1 5 10]);
        xlim(ax,[.09 6.35]);ylim(ax,[.0001 12]);
        xlabel(ax,'Observed |d| bin (lower boundary)','FontSize',12);
        ylabel(ax,'Isolated-bin false-positive rate (%)','FontSize',12);
        title(ax,sprintf('(%c) %s — %s',char(64+(design-1)*3+family), ...
            design_label(design),familyNames{family}),'FontSize',13);
        if design==1
            legend(ax,'Location','northeast','FontSize',8.1,'Box','off','NumColumns',2);
        end
        set(ax,'FontName','Arial','FontSize',11,'Box','off', ...
           'LineWidth',1.2,'TickDir','out','XMinorTick','off','YMinorTick','off');
        grid(ax,'off');
    end
end
try
    sgtitle('Null false-positive qualification probabilities across observed-effect bins', ...
        'FontSize',18,'FontWeight','bold');
catch
end
set(f,'PaperPositionMode','auto');
stem=fullfile(outputDir,'Figure_5_observed_effect_FPR_all12_80_95');
print(f,[stem '.png'],'-dpng','-r300');
try, print(f,[stem '.svg'],'-dsvg'); catch ME, warning('%s',ME.message); end
try, savefig(f,[stem '.fig']); catch ME, warning('%s',ME.message); end
close(f);
fprintf('Saved %s.{png,svg,fig}\n',stem);
end

function mask=select_family(labels,family)
labels=string(labels);
switch family
    case 1,mask=startsWith(labels,'p <');
    case 2,mask=contains(labels,'reference');
    case 3,mask=contains(labels,'n50 matching');
    otherwise,error('Unknown family');
end
end
function s=threshold_label(family,cutoff)
if family==1,s=sprintf('p < %g',cutoff);
else,s=sprintf('BF10 >= %g',cutoff);end
end
function s=design_label(k)
if k==1,s='Independent';else,s='Paired';end
end
