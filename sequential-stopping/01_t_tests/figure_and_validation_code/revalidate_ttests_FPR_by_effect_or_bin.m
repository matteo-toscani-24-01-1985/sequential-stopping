function revalidate_ttests_FPR_by_effect_or_bin(projectRoot, nNull)
% REVALIDATE_TTESTS_FPR_BY_EFFECT_OR_BIN
%
% Re-run HELD-OUT NULL validation, using the UNCHANGED checkpoint-calibrated
% stopping rules from the two original MATLAB persistence simulations:
%  1. independent-samples t test (participants PER GROUP)
%  2. paired-samples t test (participant PAIRS)
%
% ASSUMED-EFFECT RULE: each of the 26 assumed d values defines a DIFFERENT
% prespecified stopping rule. Output one null false-positive rate per
% (assumed d, evidence threshold, persistence target), with Wilson upper CI.
%
% OBSERVED-EFFECT RULE: one ADAPTIVE rule recomputes |d| and its bin at EACH
% checkpoint. For each observed |d| bin, report TWO distinct probabilities:
%  (a) Isolated-bin FPR: probability that a null trajectory EVER satisfies
%      the calibrated threshold/minimum-N condition IN THAT BIN at any look
%      (as though stopping were permitted in this one bin alone).
%  (b) First-stop FPR contribution: probability that a null trajectory first
%      stops in that bin under the full ADAPTIVE rule. Contributions across
%      bins must sum EXACTLY to the full rule's overall null FPR.
% Do NOT sum isolated-bin FPRs: one trajectory may qualify in several bins.
%
% REQUIREMENTS: The two existing FULL_results_80_95.mat files produced by
% the original standalone scripts. No Stats Toolbox is required. Existing
% calibration files are READ ONLY; output CSVs/PNGs have distinct names.
% The same null sampling generator, random seed and checkpoint boundaries
% as the original scripts are used. Full default validation is 800,000 null
% trajectories PER TEST DESIGN. For a quick smoke test specify nNull=1000;
% use the default for publication results.
%
% USAGE (run from the p-value2.0 project directory):
%   revalidate_ttests_FPR_by_effect_or_bin(pwd);
% OR
%   revalidate_ttests_FPR_by_effect_or_bin('path/to/p-value2.0', 1000);
%
% OUTPUT FOR EACH DESIGN in <projectRoot>/<design>_..._results:
%  FPR_BY_ASSUMED_EFFECT.csv
%  FPR_BY_OBSERVED_BIN.csv
%  FPR_BY_THRESHOLD_SUMMARY.csv
%  FPR_BY_EFFECT_VALIDATION_COUNTS.mat
% OUTPUT SIX-PANEL FIGURES in <projectRoot>/FPR_by_effect_figures:
%  FPR_assumed_effects.png
%  FPR_observed_isolated_bins.png
%  FPR_observed_firststop_contributions.png
%
% CONVENTION: binwise false-positive probabilities have the SAME denominator
% (number of null trajectories). This makes binwise contributions additive.
% The formal 5% Type I decision belongs to the COMPLETE adaptive rule,
% not to each contribution in isolation.

if nargin < 1 || isempty(projectRoot), projectRoot = pwd; end
if nargin < 2 || isempty(nNull), nNull = 0; end
validateattributes(nNull,{'numeric'},{'scalar','integer','nonnegative'});

names = {'independent','paired'};
folders = {'ttest_persistence_80_95_results', ...
           'paired_ttest_persistence_80_95_results'};
allAssumed = cell(2,1); allObserved = cell(2,1);

for design = 1:2
    sourceDir = fullfile(projectRoot,folders{design});
    sourceFile = fullfile(sourceDir,'FULL_results_80_95.mat');
    if ~exist(sourceFile,'file')
        error('Missing calibration file: %s. Run the original script first.',sourceFile);
    end
    R = load(sourceFile,'C','allowAssumed','allowObserved', ...
        'minAssumed','minObserved');
    fields = {'C','allowAssumed','allowObserved','minAssumed','minObserved'};
    for j=1:numel(fields)
        assert(isfield(R,fields{j}),'Missing %s in %s.',fields{j},sourceFile);
    end
    C = R.C;
    assert(isfield(C,'critical') && isfield(C,'observedBinEdges'), ...
        'Original simulation settings and critical boundaries are required.');
    assert(any(C.validationEffects==0), 'Null validation is not configured.');
    assert(size(R.allowAssumed,1)==numel(C.nGrid), 'Checkpoint mismatch.');
    assert(size(R.allowObserved,2)==C.B, 'Bin mismatch.');
    assert(all(size(C.critical)==[C.S,numel(C.nFull)]), ...
        'Saved critical boundaries are not aligned to the full time grid.');

    if nNull==0, reps=C.nValidateNull; else, reps=nNull; end
    fprintf('\n%s: validating %d null trajectories (both targets, all 12 thresholds).\n', ...
        names{design},reps);
    S=C.S; T=C.T; A=C.A; B=C.B; K=C.K;
    assumedCount=zeros(A,S,T);
    observedCount=zeros(S,T);
    naiveCount=zeros(S,1);
    firstBinCount=zeros(B,S,T);
    anyBinCount=zeros(B,S,T);

    % Same RNG initialization and null trajectory order as STAGE 3 of the
    % original codes: d=0 is first in C.validationEffects.
    rng(C.seedValidation,'twister');
    for offset=1:C.batchSize:reps
        batch=min(C.batchSize,reps-offset+1);
        [absT,obsD]=simulate_null_trajectory(batch,C,design);
        bins=discretize(obsD(:,C.initialIndex),C.observedBinEdges);
        assert(all(isfinite(bins(:))), 'Unclassified observed effect in null data.');
        checkpointCols=repmat(1:K,batch,1);
        lookup=sub2ind([K,B],checkpointCols,bins);

        for s=1:S
            passes=bsxfun(@ge,absT(:,C.initialIndex), ...
                C.critical(s,C.initialIndex));
            naiveCount(s)=naiveCount(s)+sum(any(passes,2));
            for t=1:T
                % FULL observed-effect adaptive stopping rule.
                allowed=R.allowObserved(:,:,s,t); % K x B
                qualifies=passes & allowed(lookup);
                [firstHit,firstCol]=max(qualifies,[],2);
                firstHit=logical(firstHit);
                observedCount(s,t)=observedCount(s,t)+sum(firstHit);

                % FIRST STOP: each trajectory is attributed to ONE bin.
                if any(firstHit)
                    stopRows=find(firstHit);
                    stopCols=firstCol(firstHit);
                    stopBins=bins(sub2ind([batch,K],stopRows,stopCols));
                    firstBinCount(:,s,t)=firstBinCount(:,s,t) + ...
                        accumarray(stopBins(:),1,[B,1],@sum,0);
                end

                % ISOLATED BIN RULE: count a trajectory once in EACH bin
                % in which it has at least one eligible crossing. It can
                % qualify in multiple bins along its trajectory.
                [qRows,qCols]=find(qualifies);
                if ~isempty(qRows)
                    qBins=bins(sub2ind([batch,K],qRows,qCols));
                    hitByTrajectoryAndBin=sparse(qRows,qBins,1,batch,B);
                    anyBinCount(:,s,t)=anyBinCount(:,s,t) + ...
                        full(sum(spones(hitByTrajectoryAndBin),1))';
                end

                % Assumed-effect FPR: each a is a distinct fixed rule;
                % data remain generated under TRUE d=0 in ALL cases.
                assumedAllowed=reshape(R.allowAssumed(:,s,:,t),[1,K,A]);
                assumedQualifies=bsxfun(@and,reshape(passes,[batch,K,1]), ...
                    assumedAllowed);
                hitsByAssumed=reshape(any(assumedQualifies,2),[batch,A]);
                assumedCount(:,s,t)=assumedCount(:,s,t) + ...
                    sum(hitsByAssumed,1)';
            end
        end
        if mod(ceil(offset/C.batchSize),250)==0 || offset+batch-1==reps
            fprintf('  %s: %d / %d null trajectories\n',names{design}, ...
                offset+batch-1,reps);
        end
    end

    for s=1:S
        for t=1:T
            total=observedCount(s,t);
            assert(sum(firstBinCount(:,s,t))==total, ...
                'First-stop bin contributions must sum to overall FPR count.');
            assert(all(firstBinCount(:,s,t)<=anyBinCount(:,s,t)), ...
                'First-stop count cannot exceed isolated-bin count.');
            assert(all(anyBinCount(:,s,t)<=total), ...
                'No bin can exceed overall null stopping count.');
        end
    end

    [assumedTable, observedTable, summaryTable] = ...
        make_validation_tables(C,R,reps,assumedCount,observedCount, ...
        firstBinCount,anyBinCount,naiveCount,names{design});
    if reps==C.nValidateNull
        outputDir=sourceDir;
    else
        outputDir=fullfile(sourceDir,sprintf('FPR_SMOKE_TEST_N%d',reps));
        if ~exist(outputDir,'dir'),mkdir(outputDir);end
    end
    writetable(assumedTable,fullfile(outputDir,'FPR_BY_ASSUMED_EFFECT.csv'));
    writetable(observedTable,fullfile(outputDir,'FPR_BY_OBSERVED_BIN.csv'));
    writetable(summaryTable,fullfile(outputDir,'FPR_BY_THRESHOLD_SUMMARY.csv'));
    save(fullfile(outputDir,'FPR_BY_EFFECT_VALIDATION_COUNTS.mat'), ...
        'assumedCount','observedCount','firstBinCount','anyBinCount', ...
        'naiveCount','reps');

    if reps==C.nValidateNull
        compare_to_existing_outputs(sourceDir,assumedTable,summaryTable,C);
    else
        fprintf('  Smoke-test mode: reference validation rates NOT asserted.\n');
    end
    allAssumed{design}=assumedTable;
    allObserved{design}=observedTable;
end

if nNull==0
    figureDir=fullfile(projectRoot,'FPR_by_effect_figures');
else
    figureDir=fullfile(projectRoot,sprintf('FPR_by_effect_figures_SMOKE_N%d',nNull));
end
if ~exist(figureDir,'dir'),mkdir(figureDir);end
plot_six_panel(allAssumed,'assumed',figureDir,'FPR_assumed_effects.png');
plot_six_panel(allObserved,'isolated',figureDir,'FPR_observed_isolated_bins.png');
plot_six_panel(allObserved,'firststop',figureDir, ...
    'FPR_observed_firststop_contributions.png');
fprintf('\nAll tables and figures saved. These are new validation outputs, not\n');
fprintf('the pre-existing observed-effect FPR table copied across bins.\n');
end

function [absT,obsD]=simulate_null_trajectory(batch,C,design)
% Same generator and arithmetic as simulate_trajectory(..., trueD=0) in
% each original MATLAB script, to match its held-out validation.
nEnd=C.nEnd;
nValues=C.nFull;
if design==1
    G1=randn(batch,nEnd);
    G2=randn(batch,nEnd);
    S1=cumsum(G1,2);S2=cumsum(G2,2);
    Q1=cumsum(G1.^2,2);Q2=cumsum(G2.^2,2);
    clear G1 G2
    x=S1(:,nValues);y=S2(:,nValues);
    v=2*nValues-2;
    sumsq=Q1(:,nValues)+Q2(:,nValues);
    pooled=bsxfun(@rdivide,sumsq-bsxfun(@rdivide,x.^2+y.^2,nValues),v);
    pooled=max(pooled,realmin('double'));
    obsD=abs(bsxfun(@rdivide,y-x,nValues))./sqrt(pooled);
    absT=bsxfun(@times,obsD,sqrt(nValues/2));
else
    D=randn(batch,nEnd);
    S=cumsum(D,2);
    Q=cumsum(D.^2,2);
    clear D
    x=S(:,nValues);
    v=nValues-1;
    sumsq=Q(:,nValues);
    varD=bsxfun(@rdivide,sumsq-bsxfun(@rdivide,x.^2,nValues),v);
    varD=max(varD,realmin('double'));
    obsD=abs(bsxfun(@rdivide,x,nValues))./sqrt(varD);
    absT=bsxfun(@times,obsD,sqrt(nValues));
end
end

function [ta,to,ts]=make_validation_tables(C,R,nNull,ca,co,cb,cm,cn,design)
A=C.A;B=C.B;S=C.S;T=C.T;z=C.z;
% Assumed: 26 d values x 12 evidence criteria x 2 persistence targets.
V=A*S*T;
designName=repmat({design},V,1);
assumed_d=zeros(V,1);persistence_target=zeros(V,1);
criterion=cell(V,1);threshold=zeros(V,1);minimum_N=zeros(V,1);
null_trajectories=nNull*ones(V,1);false_positive_count=zeros(V,1);
false_positive_rate=zeros(V,1);one_sided95_upper=zeros(V,1);
passes_5pct=false(V,1);unrestricted_FPR=zeros(V,1);r=0;
for t=1:T
    for a=1:A
        for s=1:S
            r=r+1;
            assumed_d(r)=C.effectGrid(a);persistence_target(r)=C.targets(t);
            criterion{r}=C.names{s};threshold(r)=C.cut(s);
            minimum_N(r)=R.minAssumed(a,s,t);
            false_positive_count(r)=ca(a,s,t);
            false_positive_rate(r)=ca(a,s,t)/nNull;
            one_sided95_upper(r)=wilson_upper(ca(a,s,t),nNull,z);
            passes_5pct(r)=one_sided95_upper(r)<C.fprLimit;
            unrestricted_FPR(r)=cn(s)/nNull;
        end
    end
end
ta=table(designName,persistence_target,assumed_d,criterion,threshold, ...
    minimum_N,null_trajectories,false_positive_count,false_positive_rate, ...
    one_sided95_upper,passes_5pct,unrestricted_FPR);

% Observed: 21 bins x 12 criteria x 2 persistence targets.
V=B*S*T;
designName=repmat({design},V,1);
persistence_target=zeros(V,1);observed_bin=cell(V,1);
bin_lower=zeros(V,1);bin_upper=zeros(V,1);
criterion=cell(V,1);threshold=zeros(V,1);minimum_N=zeros(V,1);
null_trajectories=nNull*ones(V,1);
first_stop_count=zeros(V,1);first_stop_FPR_contribution=zeros(V,1);
isolated_bin_count=zeros(V,1);isolated_bin_FPR=zeros(V,1);
isolated_bin_one_sided95_upper=zeros(V,1);
overall_FPR=zeros(V,1);overall_one_sided95_upper=zeros(V,1);
share_of_false_positive_stops=zeros(V,1);passes_overall_5pct=false(V,1);
r=0;
for t=1:T
    for b=1:B
        lo=C.observedBinEdges(b);hi=C.observedBinEdges(b+1);
        if isinf(hi), label=sprintf('[%.2f, Inf)',lo);
        else, label=sprintf('[%.2f, %.2f)',lo,hi);end
        for s=1:S
            r=r+1;
            persistence_target(r)=C.targets(t);
            observed_bin{r}=label;bin_lower(r)=lo;bin_upper(r)=hi;
            criterion{r}=C.names{s};threshold(r)=C.cut(s);
            minimum_N(r)=R.minObserved(b,s,t);
            first_stop_count(r)=cb(b,s,t);
            first_stop_FPR_contribution(r)=cb(b,s,t)/nNull;
            isolated_bin_count(r)=cm(b,s,t);
            isolated_bin_FPR(r)=cm(b,s,t)/nNull;
            isolated_bin_one_sided95_upper(r)=wilson_upper(cm(b,s,t),nNull,z);
            overall_FPR(r)=co(s,t)/nNull;
            overall_one_sided95_upper(r)=wilson_upper(co(s,t),nNull,z);
            if co(s,t)>0
                share_of_false_positive_stops(r)=cb(b,s,t)/co(s,t);
            else
                share_of_false_positive_stops(r)=NaN;
            end
            passes_overall_5pct(r)=overall_one_sided95_upper(r)<C.fprLimit;
        end
    end
end
to=table(designName,persistence_target,observed_bin,bin_lower,bin_upper, ...
    criterion,threshold,minimum_N,null_trajectories,first_stop_count, ...
    first_stop_FPR_contribution,isolated_bin_count,isolated_bin_FPR, ...
    isolated_bin_one_sided95_upper,overall_FPR,overall_one_sided95_upper, ...
    share_of_false_positive_stops,passes_overall_5pct);

% Compact one-row-per-threshold-and-target audit table.
V=S*T;
designName=repmat({design},V,1);
persistence_target=zeros(V,1);criterion=cell(V,1);threshold=zeros(V,1);
null_trajectories=nNull*ones(V,1);unrestricted_FPR=zeros(V,1);
overall_observed_FPR=zeros(V,1);overall_upper95=zeros(V,1);
sum_firststop_bin_FPR=zeros(V,1);pass_overall=false(V,1);r=0;
for t=1:T
    for s=1:S
        r=r+1;
        persistence_target(r)=C.targets(t);criterion{r}=C.names{s};
        threshold(r)=C.cut(s);unrestricted_FPR(r)=cn(s)/nNull;
        overall_observed_FPR(r)=co(s,t)/nNull;
        overall_upper95(r)=wilson_upper(co(s,t),nNull,z);
        sum_firststop_bin_FPR(r)=sum(cb(:,s,t))/nNull;
        pass_overall(r)=overall_upper95(r)<C.fprLimit;
    end
end
ts=table(designName,persistence_target,criterion,threshold, ...
    null_trajectories,unrestricted_FPR,overall_observed_FPR,overall_upper95, ...
    sum_firststop_bin_FPR,pass_overall);
end

function hi=wilson_upper(k,n,z)
p=k./n;
hi=(p+z^2./(2*n)+z*sqrt(p.*(1-p)./n+z^2./(4*n.^2)))./(1+z^2./n);
end

function compare_to_existing_outputs(folder,ta,ts,C)
% Independent regression test: exact results should match original H0
% validation because the same seed, batches and null generator are used.
refObs=fullfile(folder,'VALIDATION_observed_H0.csv');
if exist(refObs,'file')
    old=readtable(refObs);
    for i=1:height(ts)
        ids=(abs(old.persistence_target-ts.persistence_target(i))<1e-10) & ...
            (abs(old.threshold-ts.threshold(i))<1e-7);
        assert(nnz(ids)==1, 'Observed null reference row not unique.');
        assert(abs(old.overall_H0_FPR(ids)-ts.overall_observed_FPR(i))<1e-12, ...
            'Observed overall null FPR differs from original.');
    end
    fprintf('  PASS: all observed-effect overall FPRs reproduce original H0 CSV.\n');
end
refAss=fullfile(folder,'VALIDATION_assumed_H0.csv');
if exist(refAss,'file')
    old=readtable(refAss);
    for i=1:height(ta)
        ids=(abs(old.persistence_target-ta.persistence_target(i))<1e-10) & ...
            (abs(old.assumed_true_d-ta.assumed_d(i))<1e-10) & ...
            (abs(old.threshold-ta.threshold(i))<1e-7);
        assert(nnz(ids)==1, 'Assumed null reference row not unique.');
        assert(abs(old.overall_H0_FPR(ids)-ta.false_positive_rate(i))<1e-12, ...
            'Assumed null FPR differs from original.');
    end
    fprintf('  PASS: all assumed-effect FPRs reproduce original H0 CSV.\n');
end
fprintf('  Tested %d thresholds, %d targets, %d assumed effects, %d observed bins.\n', ...
    C.S,C.T,C.A,C.B);
end

function plot_six_panel(tables,kind,folder,outfile)
fig=figure('Color','w','Position',[70 80 1630 960],'Visible','off');
for design=1:2
    data=tables{design};
    for group=1:3
        ax=subplot(2,3,(design-1)*3+group);
        hold(ax,'on');
        if group==1, ss=1:4; fam='p thresholds';
        elseif group==2, ss=9:12;fam='Reference Bayes factors';
        else, ss=5:8;fam='BF matched at N=50';end
        % Thresholds are identified by color and persistence by line style.
        colors=lines(4);
        target=[0.8 0.95];
        for q=1:4
            for t=1:2
                if t==1,linestyle='-';else,linestyle='--';end
                if strcmp(kind,'assumed')
                    ids=find(abs(data.persistence_target-target(t))<1e-10 & ...
                             abs(data.threshold-data.threshold(ss(q)))<1e-8);
                    [x,order]=sort(data.assumed_d(ids));
                    y=data.false_positive_rate(ids(order))*100;
                    xl='Assumed population Cohen''s d';
                    yl='Null false-positive rate (%)';
                else
                    % Use criterion strings to select the same BF matching
                    % group in independent and paired tables.
                    thresholdId=data.threshold(ss(q));
                    ids=find(abs(data.persistence_target-target(t))<1e-10 & ...
                             abs(data.threshold-thresholdId)<1e-8);
                    [x,order]=sort(data.bin_lower(ids));
                    if strcmp(kind,'isolated')
                        y=data.isolated_bin_FPR(ids(order))*100;
                        yl='Isolated-bin null FPR (%)';
                    else
                        y=data.first_stop_FPR_contribution(ids(order))*100;
                        yl='First-stop FPR contribution (%)';
                    end
                    x(x>=6)=6.5;
                    xl='Observed absolute Cohen''s d (bin lower boundary)';
                end
                plot(ax,x,y,'Color',colors(q,:),'LineStyle',linestyle, ...
                    'LineWidth',1.6,'Marker',ternary_marker(t), ...
                    'MarkerSize',3,'DisplayName',sprintf('%s / %d%%', ...
                    threshold_label(group,q),round(target(t)*100)));
            end
        end
        if ~strcmp(kind,'firststop')
            plot(ax,xlim(ax),[5 5],'k:','LineWidth',1.15, ...
                'DisplayName','5% reference');
        end
        title(ax,sprintf('%s: %s', ...
            capitalize_design(design),fam),'FontWeight','bold');
        xlabel(ax,xl);ylabel(ax,yl);
        set(ax,'Box','off','TickDir','out','FontSize',10);
        grid(ax,'off');
        legend(ax,'Location','northeast','FontSize',6.5,'Box','off');
    end
end
if strcmp(kind,'assumed')
    name='False positives by ASSUMED effect (one rule per assumed d)';
elseif strcmp(kind,'isolated')
    name='Null false positives from EACH observed-effect bin in isolation';
else
    name='Contributions of FIRST stopping bin to overall null false positives';
end
try,sgtitle(name,'FontSize',17,'FontWeight','bold');catch,end
print(fig,fullfile(folder,outfile),'-dpng','-r220');
close(fig);
end

function result=ternary_marker(t)
if t==1, result='o';else,result='none';end
end

function s=threshold_label(group,q)
if group==1
    v=[.05 .01 .001 .0001];s=sprintf('p<%g',v(q));
elseif group==2
    v=[3 10 30 100];s=sprintf('BF>=%g',v(q));
else
    v=[.05 .01 .001 .0001];s=sprintf('BF p-matched %g',v(q));
end
end

function str=capitalize_design(design)
if design==1,str='Independent';else,str='Paired';end
end
