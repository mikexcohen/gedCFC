%% multivariate cross-frequency coupling, based on generalized eigendecomposition (gedCFC)
% Method 4: using delay-embedded matrices to define an empirical
%           spatiotemporal filter

% You will need the following files in the current directory or Matlab path:
%   - emptyEEG.mat
%   - filterFGx.m
%   - topoplotIndie.m
%   - eegfilt.m

% mikexcohen@gmail.com

clear

%% preliminaries

% mat file containing EEG, leadfield and channel locations
load emptyEEG
origEEG = EEG;

% indices of dipole locations
thetaDipole =  94;
gam1Dipole  = 109;
gam2Dipole  = 111;

whichOri = 1; % 1 for "EEG" and 2 for "MEG"

figure(1), clf
clim = [-45 45];

subplot(231), topoplotIndie(-lf.Gain(:,whichOri,thetaDipole),EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','off','shading','interp');
subplot(232), topoplotIndie(-lf.Gain(:,whichOri,gam1Dipole), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','off','shading','interp');
subplot(233), topoplotIndie(-lf.Gain(:,whichOri,gam2Dipole), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','off','shading','interp');

%% create a brain of correlated random data

% correlation matrix
cormat = rand(size(lf.Gain,3));
cormat = cormat*cormat';
% cormat = cormat-min(cormat(:));
cormat = .8*( cormat./max(cormat(:)) );
cormat(1:size(lf.Gain,3)+1:end) = 1;

% eigdecomp and create correlated random data
[evecs,evals] = eig(cormat);

% 1/f random data
ps   = bsxfun(@times, exp(1i*2*pi*rand(size(lf.Gain,3),floor(EEG.pnts/2))) , .1+exp(-(1:floor(EEG.pnts/2))/200) );
ps   = [ps zeros(size(lf.Gain,3),1) ps(:,end:-1:1)];
data = 100* real(ifft(ps,[],2))'*(evecs*sqrt(evals))';

%% simulate theta oscillation

thetafreq = 8;

% create data time series
ampl1     = 8+10*filterFGx(randn(1,EEG.pnts),EEG.srate,1,30);
freqmod1  = detrend(10*filterFGx(randn(1,EEG.pnts),EEG.srate,1,30));
thetawave = ampl1 .* sin(2*pi.*thetafreq.*EEG.times + 2*pi/EEG.srate*cumsum(freqmod1));

data(:,thetaDipole) = thetawave*10;

%% insert gamma burst around theta troughs

gamfreq = 75;

nwin = round(EEG.srate/thetafreq/5); % time window for gamma burst

% find troughs and peaks
troughs = find(diff(sign(diff( thetawave )))>0)+1;
troughs(troughs<nwin+1) = [];
troughs(troughs>EEG.pnts-nwin-1) = [];

% gamma burst insertion...
ttv = (-nwin:nwin)/EEG.srate;
for ti=1:length(troughs)
    data(troughs(ti)-nwin:troughs(ti)+nwin,gam1Dipole) = 10*sin(2*pi*gamfreq*ttv + 2*pi*rand-pi).*exp(-(ttv.^2)/.001) + randn(size(ttv));
end

% distractor gamma
data(:,gam2Dipole) = sin(2*pi*50*EEG.times);

% project data to scalp
EEG.data = detrend( data*squeeze(lf.Gain(:,whichOri,:))' )';

%% GED to identify theta peaks & troughs

% get theta and broadband covariances
thetafilt = filterFGx(EEG.data,EEG.srate,thetafreq,2);
thetacov  = (thetafilt*thetafilt')/EEG.pnts;
bbcov     = (EEG.data*EEG.data')/EEG.pnts;

% GED and get maps and component time series
[evecs,evals] = eig(thetacov,bbcov);
[~,maxcomp]   = sort(diag(evals));
thetamap      = inv(evecs');
thetamap      = thetamap(:,maxcomp(end));
thetacomp     = thetafilt' * evecs(:,maxcomp(end));

% fix sign of component map
[~,maxe] = max(abs(thetamap));
thetamap = thetamap * sign(thetamap(maxe));

% fix time series sign based on correlation with EEG
thetacomp = thetacomp * -sign(corr(thetacomp,filterFGx(EEG.data(maxe,:),EEG.srate,thetafreq,9)'));

% plot component map for comparison with simulation
figure(1), subplot(234)
topoplotIndie(thetamap,EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');

%% main analysis happens here

% create time-delay matrix
ndel = 60; % number of delay embeddings (even only, please!)
padorder = [ EEG.pnts-floor(ndel/2):EEG.pnts 1:floor(ndel/2)-1 ];

% now get troughs of component time series and remove edges
troughs = find(diff(sign(diff( thetacomp )))>0)+1;
troughs(troughs<ndel+1) = [];
troughs(troughs>EEG.pnts-ndel-1) = [];


% create the delay-embedded matrix
delEmb = zeros(EEG.nbchan*ndel,EEG.pnts);
for deli = 1:ndel
    delEmb( (1:EEG.nbchan)+(deli-1)*EEG.nbchan,:) = detrend( EEG.data(:,[padorder(deli):end 1:padorder(deli)-1])' )';
end

% covariance matrix
delEmb = bsxfun(@minus,delEmb,mean(delEmb,2));
delcov = (delEmb*delEmb')/size(delEmb,2);

% high-pass filter at 20 Hz
delEmb = eegfilt(delEmb,EEG.srate,20,0);


% covariances per trough
tcov = zeros(size(delEmb,1));
tla  = zeros(EEG.nbchan,ndel);

for ti=1:length(troughs)
    tmpsd = delEmb(:,troughs(ti)-ndel/2:troughs(ti)+ndel/2);
    tmpsd = bsxfun(@minus,tmpsd,mean(tmpsd,2));
    tcov  = tcov + (tmpsd*tmpsd')/ndel;
    
    % also get the trough-locked activity from the non-delayed rows
    tla = tla + zscore(tmpsd(1:EEG.nbchan,1:end-1));
end
% divide by N for mean
tla  = tla./ti;
tcov = tcov./ti;

% eigendecomposition and sort matrices
[evecs,evals] = eig( tcov,delcov );
[~,sidx] = sort(diag(evals));
evecs   = evecs(:,sidx);
maps    = tcov * evecs / (evecs' * tcov * evecs);

%% and plot

% get map of largest component and normalize
map = reshape(maps(:,end),EEG.nbchan,ndel);
map = (map-mean(map(:)))/std(map(:));

% channel sorting for plotting
[~,chansort] = sort([EEG.chanlocs.X]);


figure(2), clf
tv = 1000*(-ndel/2:ndel/2-1)/EEG.srate;

% show example time course
subplot(231), plot(EEG.times,data(:,[thetaDipole gam1Dipole]),'linew',2)
set(gca,'xlim',[2 2.4]+1), axis square
xlabel('Time (s)'), ylabel('Amplitude (a.u.)')
title('Sample time series')

% show trough-triggered average
subplot(232)
contourf(tv,1:64,tla(chansort,:),40,'linecolor','none')
axis square, xlabel('Time (ms)'), ylabel('Channel'), title('Trough-triggered average')

% show forward model of filter
subplot(233)
contourf(ndel/2+tv,1:64,map(chansort,:),40,'linecolor','none')
axis square, title('Component')
xlabel('Time (ms)'), ylabel('Channel')

% forward model of filter energy
figure(1), subplot(235)
topoplotIndie(mean(map.^2,2),EEG.chanlocs,'electrodes','off','numcontour',0,'shading','interp');
figure(2)


% time-domain filter and ERP
subplot(234)
plot(tv,zscore(tla(31,:)),'m^-','linew',1), hold on
plot(tv,zscore(map(31,:)),'ks-','linew',2)
axis square, xlabel('Peri-trough time (ms)'), ylabel('Amplitude (a.u.)')

% freq-domain filter and ERP
subplot(235)
plot(linspace(0,EEG.srate,500),abs(fft(zscore(tla(30,:)),500)).^2,'m^-','linew',1), hold on
plot(linspace(0,EEG.srate,500),abs(fft(zscore(map(30,:)),500)).^2,'ks-','linew',2)
set(gca,'xlim',[0 200])
axis square, xlabel('Frequency (Hz)'), ylabel('Power (a.u.)')
legend({'electrode';'STfilter'})


% power spectrum of EEG (hard-coded to electrode POz)
subplot(236)
plot(linspace(0,EEG.srate,EEG.pnts),smooth(abs(fft(EEG.data(30,:))/EEG.pnts).^2),'k','linew',2)
set(gca,'xlim',[0 200])
axis square, xlabel('Frequency (Hz)'), ylabel('Power')

%% 

