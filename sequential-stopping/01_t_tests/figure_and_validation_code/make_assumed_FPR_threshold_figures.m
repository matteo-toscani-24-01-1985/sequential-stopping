function make_assumed_FPR_threshold_figures(projectRoot, outputDir)
% MAKE_ASSUMED_FPR_THRESHOLD_FIGURES
% Standalone MATLAB figure generator for completed t-test simulations.
% Produces false-positive-rate versus ASSUMED Cohen's d, including the
% stringent p and BF thresholds that make small-sample sequential stopping
% possible while controlling Type I error.
%
% Usage from the p-value2.0 directory:
%    make_assumed_FPR_threshold_figures(pwd)
%
% Input files (already produced; no simulation rerun needed):
%   ttest_persistence_80_95_results/FPR_BY_ASSUMED_EFFECT.csv
%   paired_ttest_persistence_80_95_results/FPR_BY_ASSUMED_EFFECT.csv
%
% Output 1 (main text): Figure_4_assumed_effect_FPR_95pct_focus
%   Two panels (independent and paired), 95% persistence, p < .05,
%   BF10 >= 10, p < .001, p < .0001; legends show min N at d = 2.
% Output 2 (all thresholds): Figure_4_assumed_effect_FPR_all12_80_95
%   Six panels: first row independent; second row paired.
%   Columns = four p thresholds, four reference BF thresholds,
%             four BF thresholds p-matched at N = 50.
%   Each panel shows 80% (solid) and 95% (dashed) persistence.
%
% All y-axes are overall null false-positive RATE (%); not minimum N.
% Each assumed d defines a distinct complete sequential stopping rule.
% The horizontal 5% line is a visual reference. FORMAL acceptance requires
% one-sided 95% Wilson UPPER confidence limit < 5%, as in the CSV.
% Simulations were independently run under H0 with 800,000 trajectories.
%
% Rates of exactly zero are omitted on logarithmic y-axes. Curves are
% undefined where no calibrated minimum N was available (NaN).

if nargin < 1 || isempty(projectRoot), projectRoot = pwd; end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'FPR_by_effect_figures','assumed_effect_corrected');
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end

sources = {'ttest_persistence_80_95_results', ...
           'paired_ttest_persistence_80_95_results'};
T = cell(2,1);
for design = 1:2
    path = fullfile(projectRoot,sources{design},'FPR_BY_ASSUMED_EFFECT.csv');
    assert(exist(path,'file')==2,'Missing required CSV: %s',path);
    T{design}=readtable(path,'TextType','string');
    required={'assumed_d','persistence_target','criterion','threshold', ...
              'minimum_N','false_positive_rate','one_sided95_upper'};
    assert(all(ismember(required,T{design}.Properties.VariableNames)), ...
           'Missing columns in %s',path);
    assert(height(T{design})==26*12*2, ...
           'Unexpected number of assumed-effect rows: %s',path);
end

make_focused(T,outputDir);
make_all12(T,outputDir);
fprintf('\nFigures saved to %s\n',outputDir);
fprintf('No Monte Carlo experiments were rerun.\n');
end

function make_focused(T,out)
f=figure('Color','w','Position',[50 50 1620 800]);
criteria={'p < 0.05','BF10 >= 10 (reference)', ...
          'p < 0.001','p < 0.0001'};
pretty={'p < .05','BF10 >= 10','p < .001','p < .0001'};
cols=lines(4);
for design=1:2
    ax=subplot(1,2,design);hold(ax,'on');
    A=T{design};
    for j=1:4
        idx=strcmp(strtrim(string(A.criterion)),criteria{j}) & ...
            abs(A.persistence_target-.95)<1e-9 & ...
            isfinite(A.minimum_N) & A.assumed_d>0 & A.false_positive_rate>0;
        x=A.assumed_d(idx);y=100*A.false_positive_rate(idx);
        [x,ii]=sort(x);y=y(ii);
        assert(~isempty(x),'Missing curve %s',criteria{j});
        idx2=strcmp(strtrim(string(A.criterion)),criteria{j}) & ...
            abs(A.persistence_target-.95)<1e-9 & abs(A.assumed_d-2)<1e-8;
        n2=A.minimum_N(idx2);
        assert(numel(n2)==1 && isfinite(n2),'Missing d=2 minimum N');
        lab=sprintf('%s  |  N at d=2: %g',pretty{j},n2);
        plot(ax,x,y,'-o','Color',cols(j,:),'LineWidth',2.6, ...
            'MarkerSize',4.7,'MarkerFaceColor','w','DisplayName',lab);
    end
    plot(ax,[.08 2.2],[5 5],'--','Color',[.35 .35 .35], ...
         'LineWidth',1.2,'HandleVisibility','off');
    text(ax,.081,5.65,'5% Type I error threshold', ...
         'Color',[.30 .30 .30],'FontSize',10);
    set(ax,'XScale','log','YScale','log','XTick',[.1 .2 .3 .5 1 2], ...
        'XTickLabel',{'0.1','0.2','0.3','0.5','1','2'}, ...
        'YTick',[.01 .05 .1 .5 1 5 10 50], ...
        'YTickLabel',{'0.01','0.05','0.1','0.5','1','5','10','50'});
    xlim(ax,[.08 2.2]);ylim(ax,[.01 70]);
    xlabel(ax,'Assumed population effect size (Cohen''s d)');
    ylabel(ax,'Overall sequential false-positive rate (%)');
    if design==1
        title(ax,'(A) Independent-samples t test');
    else
        title(ax,'(B) Paired-samples t test');
    end
    legend(ax,'Location','southoutside','NumColumns',2, ...
        'FontSize',9,'Box','off');
    style_ax(ax);
end
try
    sgtitle('95% persistence: stringent thresholds permit small N with FPR < 5%', ...
        'FontWeight','bold','FontSize',18);
catch
end
save_figure(f,out,'Figure_4_assumed_effect_FPR_95pct_focus');
end

function make_all12(T,out)
f=figure('Color','w','Position',[50 50 1760 1070]);
cols=lines(4);
familyNames={'Conventional p thresholds', ...
             'Reference Bayes factors', ...
             'Bayes factors matched at N = 50'};
for design=1:2
    A=T{design};
    for family=1:3
        ax=subplot(2,3,(design-1)*3+family);hold(ax,'on');
        fam=family_mask(A.criterion,family);
        cutoffs=sort(unique(A.threshold(fam)));
        assert(numel(cutoffs)==4,'Expected 4 thresholds per family');
        for j=1:4
            for ti=1:2
                target=[.8,.95];target=target(ti);
                idx=fam & abs(A.threshold-cutoffs(j))<1e-7*max(1,abs(cutoffs(j))) & ...
                    abs(A.persistence_target-target)<1e-8 & ...
                    isfinite(A.minimum_N) & A.assumed_d>0 & A.false_positive_rate>0;
                x=A.assumed_d(idx);y=100*A.false_positive_rate(idx);
                [x,ii]=sort(x);y=y(ii);
                if isempty(x),continue;end
                if ti==1,ln='-';mk='o';else,ln='--';mk='s';end
                label=sprintf('%s, %d%%',short_label(family,cutoffs(j)),round(target*100));
                plot(ax,x,y,'Color',cols(j,:),'LineStyle',ln,'Marker',mk, ...
                    'LineWidth',1.8,'MarkerSize',3.5,'MarkerFaceColor','w', ...
                    'DisplayName',label);
            end
        end
        plot(ax,[.08 2.2],[5 5],':','Color',[.35 .35 .35], ...
            'LineWidth',1.3,'HandleVisibility','off');
        set(ax,'XScale','log','YScale','log', ...
            'XTick',[.1 .2 .3 .5 1 2], ...
            'XTickLabel',{'0.1','0.2','0.3','0.5','1','2'}, ...
            'YTick',[.0001 .001 .01 .1 1 5 10 50], ...
            'YTickLabel',{'0.0001','0.001','0.01','0.1','1','5','10','50'});
        xlim(ax,[.08 2.2]);ylim(ax,[.0001 70]);
        xlabel(ax,'Assumed population Cohen''s d');
        ylabel(ax,'Overall false-positive rate (%)');
        title(ax,sprintf('(%c) %s: %s', ...
            char(64+(design-1)*3+family),design_label(design),familyNames{family}));
        if design==1
            legend(ax,'Location','southoutside','NumColumns',2, ...
                'FontSize',8,'Box','off');
        end
        style_ax(ax);
    end
end
try
    sgtitle('False-positive rates under assumed-effect sequential rules: all 12 thresholds', ...
        'FontWeight','bold','FontSize',18);
catch
end
save_figure(f,out,'Figure_4_assumed_effect_FPR_all12_80_95');
end

function yes=family_mask(labels,family)
labels=string(labels);
switch family
    case 1,yes=startsWith(labels,'p <');
    case 2,yes=contains(labels,'reference');
    case 3,yes=contains(labels,'n50 matching');
    otherwise,error('Unknown threshold family');
end
end

function lab=short_label(family,value)
if family==1
    lab=sprintf('p < %g',value);
else
    lab=sprintf('BF10 >= %g',value);
end
end

function lab=design_label(design)
if design==1,lab='Independent';else,lab='Paired';end
end

function style_ax(ax)
set(ax,'FontName','Arial','FontSize',11,'Box','off', ...
    'LineWidth',1.2,'TickDir','out','TickLength',[.012 .012], ...
    'XMinorTick','off','YMinorTick','off');
grid(ax,'off');
end

function save_figure(fig,out,stem)
set(fig,'PaperPositionMode','auto');
print(fig,fullfile(out,[stem '.png']),'-dpng','-r300');
try
    print(fig,fullfile(out,[stem '.svg']),'-dsvg');
catch ME
    warning('SVG unavailable: %s',ME.message);
end
try
    savefig(fig,fullfile(out,[stem '.fig']));
catch ME
    warning('MATLAB FIG unavailable: %s',ME.message);
end
fprintf('Saved %s\n',fullfile(out,stem));
close(fig);
end
