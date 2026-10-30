function make_ttest_FPR_effect_bin_figures(projectRoot, outputDir)
% MAKE_TTEST_FPR_EFFECT_BIN_FIGURES
% Publication figures from the FINISHED, independently validated null runs.
% Does not generate random numbers or rerun simulation/calibration.
%
% Usage (while in the p-value2.0 project folder):
%   make_ttest_FPR_effect_bin_figures(pwd)
% Or:
%   make_ttest_FPR_effect_bin_figures('C:\path\to\p-value2.0')
%
% Reads from each design's results folder:
%   FPR_BY_ASSUMED_EFFECT.csv
%   FPR_BY_OBSERVED_BIN.csv
%   FPR_BY_THRESHOLD_SUMMARY.csv
%
% Produces Figure4_MAIN_p05 (2 x 2, reader-oriented) and three
% 2 x 3 all-12-threshold figures (A-F: independent top, paired bottom):
%   Figure4_ASSUMED_FPR_all_thresholds
%   Figure4_OBSERVED_isolated_bin_FPR_all_thresholds
%   Figure4_OBSERVED_firststop_bin_contributions_all_thresholds
% Every figure is saved as .png (300 dpi), .svg, and .fig.
%
% IMPORTANT DEFINITIONS
% ASSUMED: one COMPLETE null stopping rule per assumed effect d and target.
% OBSERVED ISOLATED BIN: probability a null trajectory ever becomes eligible
% in that specific observed-effect bin. A trajectory may count in >1 bin:
% DO NOT sum these probabilities.
% OBSERVED FIRST-STOP: probability first eligible stopping occurs in that bin
% under the single adaptive observed-effect procedure. Contributions SUM
% to the total null Type I error rate for that procedure.
% NaN minimum-N entries are INELIGIBLE and never plotted, even when their
% null false-positive rate happens to be zero.
% One-sided 95% Wilson upper bound < 5% is the screening criterion for each
% COMPLETE procedure, not for the individual first-stop contributions.

if nargin < 1 || isempty(projectRoot), projectRoot = pwd; end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'FPR_by_effect_figures','publication');
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end
folders = {'ttest_persistence_80_95_results', ...
           'paired_ttest_persistence_80_95_results'};
D = cell(2,1);
for design = 1:2
    folder = fullfile(projectRoot,folders{design});
    files = {'FPR_BY_ASSUMED_EFFECT.csv','FPR_BY_OBSERVED_BIN.csv', ...
             'FPR_BY_THRESHOLD_SUMMARY.csv'};
    for j=1:numel(files)
        assert(exist(fullfile(folder,files{j}),'file')==2, ...
            'Missing %s. Run revalidate_ttests_FPR_by_effect_or_bin first.', ...
            fullfile(folder,files{j}));
    end
    D{design}.assumed = readtable(fullfile(folder,files{1}),'TextType','string');
    D{design}.observed = readtable(fullfile(folder,files{2}),'TextType','string');
    D{design}.summary = readtable(fullfile(folder,files{3}),'TextType','string');
    validate_tables(D{design});
end

% Main-text figure: matching the manuscript's p < .05 narrative.
make_main_p05(D,outputDir);

% Full-threshold plots retain the THREE threshold families from the
% persistence figures, and the two design rows from observed-effect figure.
make_six_panel(D,'assumed',outputDir, ...
    'Figure4_ASSUMED_FPR_all_thresholds');
make_six_panel(D,'isolated',outputDir, ...
    'Figure4_OBSERVED_isolated_bin_FPR_all_thresholds');
make_six_panel(D,'firststop',outputDir, ...
    'Figure4_OBSERVED_firststop_bin_contributions_all_thresholds');

fprintf('\nFour publication figures saved in:\n  %s\n',outputDir);
fprintf('No simulations were rerun.\n');
end

function validate_tables(Q)
A=Q.assumed; B=Q.observed; S=Q.summary;
requiredA={'persistence_target','assumed_d','criterion','threshold', ...
           'minimum_N','false_positive_rate','one_sided95_upper'};
requiredB={'persistence_target','bin_lower','criterion','threshold', ...
           'minimum_N','isolated_bin_FPR','first_stop_FPR_contribution', ...
           'first_stop_count','isolated_bin_count','overall_FPR'};
requiredS={'persistence_target','criterion','threshold','null_trajectories', ...
           'overall_observed_FPR','sum_firststop_bin_FPR','overall_upper95'};
assert(all(ismember(requiredA,A.Properties.VariableNames)), ...
    'Assumed-effect output lacks required columns.');
assert(all(ismember(requiredB,B.Properties.VariableNames)), ...
    'Observed-bin output lacks required columns.');
assert(all(ismember(requiredS,S.Properties.VariableNames)), ...
    'Summary output lacks required columns.');
assert(height(A)==26*12*2 && height(B)==21*12*2 && height(S)==12*2, ...
    'Unexpected number of assumed/observed/summary rows.');
assert(all(B.first_stop_count <= B.isolated_bin_count), ...
    'First-stop counts exceed isolated-bin counts.');
assert(all(B.first_stop_FPR_contribution <= B.isolated_bin_FPR+1e-12), ...
    'First-stop contributions exceed isolated-bin probabilities.');
assert(all(S.null_trajectories>=100000), ...
    'Input appears to be a smoke test. Full null validation required.');
assert(all(S.overall_upper95>=S.overall_observed_FPR), ...
    'Confidence-bound / overall-FPR inconsistency.');
for t=[.8 .95]
    for k=1:height(S)
        if abs(S.persistence_target(k)-t)>1e-8, continue; end
        sel=(abs(B.persistence_target-t)<1e-8) & ...
            (abs(B.threshold-S.threshold(k))<1e-7*max(1,abs(S.threshold(k))));
        assert(sum(sel)==21,'Expected 21 observed bins per criterion.');
        contributed=sum(B.first_stop_FPR_contribution(sel));
        overall=S.overall_observed_FPR(k);
        assert(abs(contributed-overall)<1e-9, ...
           'First-stop-bin contributions do not sum to total FPR.');
        assert(abs(S.sum_firststop_bin_FPR(k)-overall)<1e-9, ...
           'Reported sum and overall FPR differ.');
        assert(max(abs(B.overall_FPR(sel)-overall))<1e-9, ...
           'Inconsistent overall FPR within observed bins.');
    end
end
end

function make_main_p05(D,outdir)
f=figure('Color','w','Units','pixels','Position',[80 80 1320 910]);
colors=[0.13 0.39 0.66; 0.76 0.27 0.19];
for design=1:2
    A=D{design}.assumed;
    B=D{design}.observed;
    S=D{design}.summary;
    % Left: independent assumed-d rules; each x corresponds to a distinct rule.
    ax=subplot(2,2,(design-1)*2+1);hold(ax,'on');
    for ti=1:2
        target=[.8 .95];target=target(ti);
        x=A.assumed_d; y=100*A.false_positive_rate;
        ix=is_p05(A.criterion) & abs(A.persistence_target-target)<1e-8 ...
           & isfinite(A.minimum_N) & x>0 & y>0;
        [xs,ind]=sort(x(ix));ys=y(ix);ys=ys(ind);
        if ti==1,ls='-';mk='o';else,ls='--';mk='s';end
        plot(ax,xs,ys,'Color',colors(ti,:),'LineStyle',ls, ...
             'Marker',mk,'MarkerFaceColor','w','LineWidth',2.2, ...
             'MarkerSize',4.1,'DisplayName',sprintf('%d%% persistence',round(100*target)));
    end
    yline(ax,5,':','5%','LineWidth',1.2,'Color',[.38 .38 .38], ...
          'LabelVerticalAlignment','bottom');
    set(ax,'XScale','log');xlim(ax,[0.08 2.15]);ylim(ax,[0 51]);
    set(ax,'XTick',[.1 .2 .3 .5 1 2]);
    xlabel(ax,'Assumed population effect, d');
    ylabel(ax,'Overall null false-positive rate (%)');
    title(ax,sprintf('(%c) %s: assumed effect',char(64+(design-1)*2+1),design_label(design)));
    style_ax(ax);
    if design==1,legend(ax,'Location','northwest','Box','off','FontSize',10);end

    % Right: first eligible stop's observed bin; not a separate 5% test.
    ax=subplot(2,2,(design-1)*2+2);hold(ax,'on');
    for ti=1:2
        target=[.8 .95];target=target(ti);
        ix=is_p05(B.criterion) & abs(B.persistence_target-target)<1e-8 ...
           & isfinite(B.minimum_N) & B.bin_lower>0;
        x=B.bin_lower(ix);y=100*B.first_stop_FPR_contribution(ix);
        [x,order]=sort(x);y=y(order);
        if ti==1,ls='-';mk='o';else,ls='--';mk='s';end
        plot(ax,x,y,'Color',colors(ti,:),'LineStyle',ls, ...
             'Marker',mk,'MarkerFaceColor','w','LineWidth',2.2, ...
             'MarkerSize',4.1,'DisplayName',sprintf('%d%% persistence',round(100*target)));
    end
    overall80=100*S.overall_observed_FPR(is_p05(S.criterion) & abs(S.persistence_target-.8)<1e-8);
    overall95=100*S.overall_observed_FPR(is_p05(S.criterion) & abs(S.persistence_target-.95)<1e-8);
    txt=sprintf('Total FPR: 80%% = %.2f%%; 95%% = %.2f%%',overall80,overall95);
    text(ax,.97,.95,txt,'Units','normalized','HorizontalAlignment','right', ...
        'VerticalAlignment','top','FontSize',10,'BackgroundColor','w');
    xlim(ax,[.08 6.2]);set(ax,'XScale','log','XTick',[.1 .2 .3 .5 1 2 4 6]);
    if design==1,ylim(ax,[0 3.3]);else,ylim(ax,[0 3.3]);end
    xlabel(ax,'Observed |d| bin (lower boundary)');
    ylabel(ax,'Contribution to null false-positive rate (percentage points)');
    title(ax,sprintf('(%c) %s: observed effect',char(64+(design-1)*2+2),design_label(design)));
    style_ax(ax);
end
try
 sgtitle('p < .05: assumed-effect error versus observed-bin first stopping', ...
    'FontSize',17,'FontWeight','bold');
catch
end
export_figure(f,outdir,'Figure4_MAIN_p05');
end

function make_six_panel(D,metric,outdir,basename)
f=figure('Color','w','Units','pixels','Position',[60 60 1720 990]);
C=lines(4);
styles={'-','--'};
marks={'o','s'};
familyNames={'p-value thresholds','Reference Bayes factors', ...
             'p-matched Bayes factors (N = 50)'};
for design=1:2
   if strcmp(metric,'assumed'),T=D{design}.assumed;
   else,T=D{design}.observed;end
   for family=1:3
       ax=subplot(2,3,(design-1)*3+family);hold(ax,'on');
       selFamily=family_mask(T.criterion,family);
       cutoffs=unique(T.threshold(selFamily),'stable');
       % Sort within a family to preserve order of increasing stringency.
       cutoffs=sort(cutoffs(:));
       assert(numel(cutoffs)==4,'Expected four distinct thresholds per family.');
       for cutoffIdx=1:4
          cutoff=cutoffs(cutoffIdx);
          for ti=1:2
              tar=[.8 .95];target=tar(ti);
              use=selFamily & abs(T.threshold-cutoff)<1e-7*max(1,abs(cutoff)) ...
                    & abs(T.persistence_target-target)<1e-8 ...
                    & isfinite(T.minimum_N);
              if strcmp(metric,'assumed')
                 x=T.assumed_d(use);
                 y=100*T.false_positive_rate(use);
              else
                 x=T.bin_lower(use);
                 if strcmp(metric,'isolated')
                     y=100*T.isolated_bin_FPR(use);
                 else
                     y=100*T.first_stop_FPR_contribution(use);
                 end
              end
              good=isfinite(x)&x>0&isfinite(y)&y>0;
              x=x(good);y=y(good);
              [x,order]=sort(x);y=y(order);
              if isempty(x),continue;end
              plot(ax,x,y,'Color',C(cutoffIdx,:),'LineStyle',styles{ti}, ...
                  'Marker',marks{ti},'MarkerSize',3.2,'LineWidth',1.6, ...
                  'MarkerFaceColor','w', ...
                  'DisplayName',sprintf('%s | %d%%',short_threshold(family,cutoff),round(target*100)));
          end
       end
       if strcmp(metric,'assumed')
           yline(ax,5,':','5%','LineWidth',1.2, ...
                 'Color',[.45 .45 .45],'LabelHorizontalAlignment','left');
           xlim(ax,[.07 2.15]);ylim(ax,[.0001 60]);
           set(ax,'XTick',[.1 .2 .3 .5 1 2]);
           xlabel(ax,'Assumed population d');
           ylabel(ax,'Overall false-positive rate (%)');
       else
           xlim(ax,[.08 6.2]);ylim(ax,[.0001 10]);
           set(ax,'XTick',[.1 .2 .5 1 2 4 6]);
           xlabel(ax,'Observed |d| bin (lower boundary)');
           if strcmp(metric,'isolated')
              ylabel(ax,'Isolated-bin false-positive probability (%)');
           else
              ylabel(ax,'First-stop contribution (percentage points)');
           end
       end
       set(ax,'XScale','log','YScale','log');
       title(ax,sprintf('(%c) %s: %s', ...
           char(64+(design-1)*3+family),design_label(design),familyNames{family}));
       style_ax(ax);
       if design==1
           legend(ax,'Location','southwest','FontSize',8,'Box','off', ...
               'NumColumns',2);
       end
   end
end
switch metric
    case 'assumed'
        heading='Null false-positive rate of each assumed-effect stopping rule';
    case 'isolated'
        heading='Null probability of qualifying in each observed-effect bin (nonadditive)';
    otherwise
        heading='Null first-stop contributions by observed-effect bin (additive)';
end
try,sgtitle(heading,'FontSize',17,'FontWeight','bold');catch,end
export_figure(f,outdir,basename);
end

function tf=family_mask(labels,family)
labels=string(labels);
switch family
  case 1,tf=startsWith(labels,'p <');
  case 2,tf=contains(labels,'reference');
  case 3,tf=contains(labels,'n50 matching');
  otherwise,error('Unknown threshold family');
end
end

function tf=is_p05(labels)
tf=strcmp(strtrim(string(labels)),'p < 0.05');
end

function s=short_threshold(family,cutoff)
if family==1
    s=sprintf('p < %.4g',cutoff);
else
    s=sprintf('BF >= %.4g',cutoff);
end
end

function s=design_label(d)
if d==1,s='Independent';else,s='Paired';end
end

function style_ax(ax)
set(ax,'FontName','Arial','FontSize',11,'LineWidth',1.1, ...
    'Box','off','TickDir','out','TickLength',[.010 .012], ...
    'XMinorTick','off','YMinorTick','off');
grid(ax,'off');
end

function export_figure(fig,outputDir,stem)
set(fig,'PaperPositionMode','auto');
png=fullfile(outputDir,[stem '.png']);
print(fig,png,'-dpng','-r300');
try
    print(fig,fullfile(outputDir,[stem '.svg']),'-dsvg');
catch ME
    warning('SVG export failed: %s',ME.message);
end
try
    savefig(fig,fullfile(outputDir,[stem '.fig']));
catch ME
    warning('FIG export failed: %s',ME.message);
end
fprintf('Saved %s\n',png);
close(fig);
end
