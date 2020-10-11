%% multivariate cross-frequency coupling, based on generalized eigendecomposition (gedCFC)
% Method 3: using the low-frequency rhythm as a convolution kernel on sphered data

% You will need the following files in the current directory or Matlab path:
%   - emptyEEG.mat 
%   - filterFGx.m
%   - topoplotIndie.m

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

subplot(231), topoplotIndie(-lf.Gain(:,whichOri,thetaDipole),EEG.chanlocs,'maplimits',clim,'electrodes','off','numcontour',0,'shading','interp');
subplot(232), topoplotIndie(-lf.Gain(:,whichOri,gam1Dipole), EEG.chanlocs,'maplimits',clim,'electrodes','off','numcontour',0,'shading','interp');
subplot(233), topoplotIndie(-lf.Gain(:,whichOri,gam2Dipole), EEG.chanlocs,'maplimits',clim,'electrodes','off','numcontour',0,'shading','interp');

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
data = 50 * real(ifft(ps,[],2))'*(evecs*sqrt(evals))';

%% replace one dipole with nonstationary theta oscillation

thetafreq = 6;

% create data time series
ampl1     = 8+10*filterFGx(randn(1,EEG.pnts),EEG.srate,1,30);
freqmod1  = detrend(10*filterFGx(randn(1,EEG.pnts),EEG.srate,1,30));
thetawave = ampl1 .* sin(2*pi.*thetafreq.*EEG.times + 2*pi/EEG.srate*cumsum(freqmod1));

%% add gamma bursts around theta troughs

gamfreq = 45;

mod1 =  1 * (.0+.9*(1+real(exp(1i*angle(hilbert(thetawave)))))/2).^4;
gamma1wave = mod1 .* sin(2*pi*gamfreq*EEG.times);

data(:,thetaDipole) = thetawave*10;
data(:,gam1Dipole)  = gamma1wave*10;
data(:,gam2Dipole)  = sin(2*pi*55*EEG.times)*10;

% project data to scalp
EEG.data = detrend( data*squeeze(lf.Gain(:,whichOri,:))' )';

%% GED to identify theta peaks & troughs

% GED to get theta network
thetafilt = filterFGx(EEG.data,EEG.srate,thetafreq,2);
thetacov  = (thetafilt*thetafilt')/EEG.pnts;
bbcov     = (EEG.data*EEG.data')/EEG.pnts;

% GED 
[evecs,evals] = eig(thetacov,bbcov);
[~,maxcomp]   = sort(diag(evals));

% get maps and component time series
thetamap      = inv(evecs');
thetamap      = thetamap(:,maxcomp(end));
thetacomp     = thetafilt' * evecs(:,maxcomp(end));

%% loop through upper frequencies

% higher frequencies
frex = linspace(10,90,50);

nwinTF = 60; % time window in indices for time-frequency window

% get peaks and troughs of theta component
troughs = find(diff(sign(diff( thetacomp )))>0)+1;
peeks   = find(diff(sign(diff( thetacomp )))<0)+1;
peeks(peeks<nwinTF+1 | peeks>EEG.pnts-nwinTF+1) = [];
troughs(troughs<nwinTF+1 | troughs>EEG.pnts-nwinTF+1) = [];

% initialize, etc
tf   = zeros(2,length(frex),nwinTF*2+1);
cfcc = zeros(2,length(frex));
hz   = linspace(0,EEG.srate/2,floor(EEG.pnts/2)+1);


for fi=1:length(frex)
    
    % covariance of narrowband power time series
    filtdata = abs(hilbert(filterFGx(EEG.data,EEG.srate,frex(fi),15)')').^2;
    filtdata = bsxfun(@minus,filtdata,mean(filtdata,2));
    bbcov    = (filtdata*filtdata')/(EEG.pnts-1);
    
    % sphere the data
    [evecsO,evalsO] = eig( bbcov );
    spheredata = ( filtdata'*evecsO*sqrt(inv(evalsO)) )'; % matrix Y in the paper
    
    % bias filter and covariance with data
    biasfilt = toeplitz(thetacomp);
    filtdat  = biasfilt*spheredata'; % this is called BY in the paper
    [evecsF,evalsF] = eig( filtdat'*filtdat );
    
    % compute weights and map
    jdw    = evecsO * sqrt(pinv(evalsO)) * evecsF;
    jdmaps = pinv(jdw)';
    
    % check for sign-flip
    [~,maxcomp] = max(diag(evalsF));
    [~,idx]     = max(abs(jdmaps(:,maxcomp)));
    compsign    = sign(jdmaps(idx,maxcomp));
    jdmaps(:,maxcomp) = jdmaps(:,maxcomp) * compsign;
    
    jddata = filtdata'*jdw(:,maxcomp)*compsign;
    
    % store data at the right frequency for plotting
    if fi==dsearchn(frex',gamfreq)
        jddata2keep = jddata;
    end
    
    % various synchronization quantifications
    cfcc(1,fi) = corr(jddata,abs(hilbert(gamma1wave))')^2;
    cfcc(2,fi) = abs(mean(exp(1i*( angle(hilbert(zscore(jddata)))-angle(hilbert(zscore(abs(hilbert(gamma1wave))') )))))).^2;
    cfcc(3,fi) = corr(filtdata(30,:)',abs(hilbert(gamma1wave))')^2; % hardcoded to POz
    
    %% peri-peak power spectra
    
    for peki=1:length(peeks)
        tf(1,fi,:) = squeeze(tf(1,fi,:)) + jddata(peeks(peki)-nwinTF:peeks(peki)+nwinTF);
    end
    tf(1,fi,:) = tf(1,fi,:)./peki;
    
    for teki=1:length(troughs)
        tf(2,fi,:) = squeeze(tf(2,fi,:)) + jddata(troughs(peki)-nwinTF:troughs(peki)+nwinTF);
    end
    tf(2,fi,:) = tf(2,fi,:)./teki;

end

%% plotting

figure(2), clf

% power spectrum of electrode
subplot(321)
plot(linspace(0,EEG.srate,EEG.pnts),abs(fft(EEG.data(30,:))/EEG.pnts).^2,'k','linew',2)
set(gca,'xlim',[0 100]), axis square
title('Channel POz')
xlabel('Frequency (Hz)'), ylabel('Power (a.u.)')

% power spectrum of JD component
subplot(322)
plot(linspace(0,EEG.srate,EEG.pnts),abs(fft(jddata2keep)/EEG.pnts).^2,'r','linew',2)
set(gca,'xlim',[0 100]), axis square
title('gedCFC component')
xlabel('Frequency (Hz)'), ylabel('Power (a.u.)')

% CFC strengths
subplot(323)
plot(frex,cfcc,'s-','linew',2,'markersize',5,'markerfacecolor','w')
hold on
plot([gamfreq gamfreq],get(gca,'ylim'),'k--'), axis square
set(gca,'xlim',[frex(1)-5 frex(end)+5])
xlabel('Frequency (Hz)'), ylabel('R^2 with simulated data')
legend({'comp. corr';'comp. PS';'bestelec'})


% peri-peak power spectrum
tv = 1000*(-nwinTF:nwinTF)/EEG.srate;
subplot(324)
contourf(tv,frex,squeeze(tf(1,:,:)).^2,40,'linecolor','none'), hold on
plot(get(gca,'xlim'),[45 45],'m--'), axis square
set(gca,'xtick',-50:25:50,'clim',[0 .5])
xlabel('Peri-peak time (ms)'), ylabel('Frequency (Hz)')

subplot(313)
plot(EEG.times,thetawave, EEG.times,gamma1wave*8,'linew',2)
set(gca,'xlim',[2 3])
xlabel('Time (s)'), ylabel('Amplitude (a.u.)')

%%
