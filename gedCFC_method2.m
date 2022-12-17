%% multivariate cross-frequency coupling, based on generalized eigendecomposition (gedCFC)
% Method 2: covariance matrices at two low-frequency phase values.

% You will need the following files in the current directory or Matlab path:
%   - emptyEEG.mat 
%   - filterFGx.mt 
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
whichOri    = 1; % 1 for "EEG" and 2 for "MEG"

% plot dipole projections
figure(1), clf
clim = [-45 45];

subplot(231), topoplotIndie(-lf.Gain(:,whichOri,thetaDipole),EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','off','shading','interp');
subplot(232), topoplotIndie(-lf.Gain(:,whichOri,gam1Dipole), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','off','shading','interp');
subplot(233), topoplotIndie(-lf.Gain(:,whichOri,gam2Dipole), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','off','shading','interp');

%% create correlated 1/f noise time series

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
data = 100* real(ifft(ps,[],2))' * (evecs*sqrt(evals))';

%% create nonstationary theta oscillation as low-frequency rhythm

thetafreq = 6;

% create data time series
ampl1     = 5+10*filterFGx(randn(1,EEG.pnts),EEG.srate,1,30);
freqmod1  = detrend(10*filterFGx(randn(1,EEG.pnts),EEG.srate,1,30));
thetawave = ampl1 .* sin(2*pi.*thetafreq.*EEG.times + 2*pi/EEG.srate*cumsum(freqmod1));

%% create gamma bursts locked to theta peaks/troughs

mod1 =   (.1+.9*(1+real(exp(1i*angle(hilbert(thetawave)))))/2);
mod2 = 1-(.1+.9*(1+real(exp(1i*angle(hilbert(thetawave)))))/2);

% gamma frequencies are hard-coded here to 40 and 45 Hz
gamma1wave = mod1 .* sin(2*pi*40*EEG.times);
gamma2wave = mod2 .* sin(2*pi*45*EEG.times);

% replace noise with theta/gamma simulated signal
data(:,thetaDipole) = thetawave*10;
data(:,gam1Dipole)  = gamma1wave*10;
data(:,gam2Dipole)  = gamma2wave*10;

% project data to scalp
EEG.data = detrend( data*squeeze(lf.Gain(:,whichOri,:))' )';

%% GED to identify theta component

% covariance matrices
thetafilt = filterFGx(EEG.data,EEG.srate,thetafreq,2);
thetacov  = (thetafilt*thetafilt')/EEG.pnts;
bbcov     = (EEG.data*EEG.data')/EEG.pnts;

% GED
[evecs,evals] = eig(thetacov,bbcov);
% maximum component
[~,maxcomp]   = sort(diag(evals));
thetamap      = thetacov*evecs;
thetamap      = thetamap(:,maxcomp(end));

% fix sign of map (max is positive)
[~,maxe] = max(abs(thetamap));
thetamap = thetamap * sign(thetamap(maxe));

% get theta component and its FFT
thetacomp  = thetafilt' * evecs(:,maxcomp(end));
thetacompX = fft(thetacomp);

%% get covariance matrices around peaks and troughs

nwin = ceil(EEG.srate/thetafreq/8); % 1/4 cycle window (1/8 on either side of peak/trough)

% find troughs and peaks
troughs = find(diff(sign(diff( thetacomp )))>0)+1;
peeks   = find(diff(sign(diff( thetacomp )))<0)+1;

% remove points too close to edges
troughs(troughs<nwin+1)          = [];
troughs(troughs>EEG.pnts-nwin-1) = [];
peeks(peeks<nwin+1)              = [];
peeks(peeks>EEG.pnts-nwin-1)     = [];


% trough covar
[covT,covP] = deal( zeros(EEG.nbchan) );
for ti=1:length(troughs)
    tmpdat = EEG.data(:,troughs(ti)-nwin:troughs(ti)+nwin);
    tmpdat = bsxfun(@minus,tmpdat,mean(tmpdat,2));
    covT   = covT + (tmpdat*tmpdat')/(nwin*2);
end
covT = covT./ti;


% peak covar
for ti=1:length(peeks)
    tmpdat = EEG.data(:,peeks(ti)-nwin:peeks(ti)+nwin);
    tmpdat = bsxfun(@minus,tmpdat,mean(tmpdat,2));
    covP   = covP + (tmpdat*tmpdat')/(nwin*2);
end
covP = covP./ti;

%% GED to get gamma peak/trough networks

[evecs,evals] = eig(covT,covP);
[~,sidx]   = sort(diag(evals));
evecs = evecs(:,sidx);
maps  = covT*evecs;

% trough/peak network are components with biggest/smallest eigenvalues
gammap1 = maps(:,1);
gammap2 = maps(:,end);

% fix sign of map (max is positive)
[~,maxe] = max(abs(gammap1));
gammap1  = gammap1 * sign(gammap1(maxe));
[~,maxe] = max(abs(gammap2));
gammap2  = gammap2 * sign(gammap2(maxe));


% plot components
figure(1)
subplot(234), topoplotIndie(thetamap, EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');
subplot(235), topoplotIndie(gammap2,  EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');
subplot(236), topoplotIndie(gammap1,  EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');

%% get time course and reconstruct topography and sources

nwinTF = 60; % window length (in indices) for TF plot

% remove edge points
peeks(peeks<nwinTF+1     | peeks>EEG.pnts-nwinTF+1) = [];
troughs(troughs<nwinTF+1 | troughs>EEG.pnts-nwinTF+1) = [];

% initialize
frex = linspace(10,90,70);
[mvarpac,sync] = deal( zeros(2,length(frex)) );
[tfP,tfT] = deal( zeros(2,length(frex),nwinTF*2+1) );

% complex conjugate of analytic time series (for synchronization)
% thetaphase = angle(hilbert(detrend(thetacomp)));
thetaconj  = conj(hilbert(detrend(thetacomp)));

for fi=1:length(frex)
    
    %% get filtered component time series
    
    gam1comp = filterFGx(EEG.data,EEG.srate,frex(fi),10)' * evecs(:,sidx(1));
    gam2comp = filterFGx(EEG.data,EEG.srate,frex(fi),10)' * evecs(:,sidx(end));
    
    %% peri-peak power spectra
    
    gam1pow = abs(hilbert(detrend(gam1comp)));
    gam2pow = abs(hilbert(detrend(gam2comp)));
    
    for peki=1:length(peeks)
        tfP(1,fi,:) = squeeze(tfP(1,fi,:)) + gam1pow(peeks(peki)-nwinTF:peeks(peki)+nwinTF);
        tfP(2,fi,:) = squeeze(tfP(2,fi,:)) + gam2pow(peeks(peki)-nwinTF:peeks(peki)+nwinTF);
    end
    tfP(:,fi,:) = tfP(:,fi,:)./peki;
    
    for teki=1:length(troughs)
        tfT(1,fi,:) = squeeze(tfT(1,fi,:)) + gam1pow(troughs(teki)-nwinTF:troughs(teki)+nwinTF);
        tfT(2,fi,:) = squeeze(tfT(2,fi,:)) + gam2pow(troughs(teki)-nwinTF:troughs(teki)+nwinTF);
    end
    tfT(:,fi,:) = tfT(:,fi,:)./teki;
    
    %% peak-vs-trough measure
    
    ttr = find(diff(sign(diff( gam1comp )))>0)+1;
    tps = find(diff(sign(diff( gam1comp )))<0)+1;
    mvarpac(1,fi) = mean(gam1comp(tps)) - mean(gam1comp(ttr));
    
    ttr = find(diff(sign(diff( gam2comp )))>0)+1;
    tps = find(diff(sign(diff( gam2comp )))<0)+1;
    mvarpac(2,fi) = mean(gam2comp(tps)) - mean(gam2comp(ttr));
    
    %% phase synchronization
    
    % now evaluate as phase synchronization with low-freq phase
    for compi=1:2
        
        if compi==1
            asconj = hilbert(detrend(abs(hilbert(gam1comp)))) .* thetaconj;
        else
            asconj = hilbert(detrend(abs(hilbert(gam2comp)))) .* conj(hilbert(thetacomp));
        end
        
        outsum  = nansum(asconj);
        outssq  = nansum(asconj.*conj(asconj));
        outsumw = nansum(abs(asconj));
        sync(compi,fi) = (outsum.*conj(outsum) - outssq)./(outsumw.*conj(outsumw) - outssq);
        
    end
    
end % end frequency loop

%% plotting

figure(2), clf

% peak-trough distances
subplot(311), plot(frex,mvarpac,'s-')
xlabel('Frequency (Hz)'), ylabel('CFC modulation')
legend({'gam1';'gam2'})

% smoothed squared phase synchronization (a.k.a. wpli)
subplot(312), plot(frex,sync,'s-')

subplot(313)
plot(EEG.times,thetacomp, EEG.times,zscore(abs(hilbert(filterFGx(EEG.data,EEG.srate,40,15)' * evecs(:,sidx(1))))), EEG.times,zscore(abs(hilbert(filterFGx(EEG.data,EEG.srate,45,15)' * evecs(:,sidx(end))))),'linew',2);
set(gca,'xlim',[2 4])
legend({'theta';'gam1';'gam2'})

%% plot TF power plots locked to peaks and troughs

figure(3), clf, colormap hot
subplot(221)
contourf(1000*(-nwinTF:nwinTF)/EEG.srate,frex,squeeze(tfP(1,:,:)),40,'linecolor','none')
set(gca,'clim',[0 3]/2), axis square
xlabel('Peri-peak time (ms)'), ylabel('Frequency (Hz)'), title('Network 1')
hold on, plot(get(gca,'xlim'),[45 45],'m--','linew',2)

subplot(222)
contourf(1000*(-nwinTF:nwinTF)/EEG.srate,frex,squeeze(tfP(2,:,:)),40,'linecolor','none')
set(gca,'clim',[0 3]), axis square
hold on, plot(get(gca,'xlim'),[45 45],'m--','linew',2)

subplot(223)
contourf(1000*(-nwinTF:nwinTF)/EEG.srate,frex,squeeze(tfT(1,:,:)),40,'linecolor','none')
set(gca,'clim',[0 3]), axis square
xlabel('Peri-trough time (ms)'), ylabel('Frequency (Hz)'), title('Network 2')
hold on, plot(get(gca,'xlim'),[40 40],'m--','linew',2)

subplot(224)
contourf(1000*(-nwinTF:nwinTF)/EEG.srate,frex,squeeze(tfT(2,:,:)),40,'linecolor','none')
set(gca,'clim',[0 3]), axis square
hold on, plot(get(gca,'xlim'),[40 40],'m--','linew',2)

%% power spectra and sample time courses

figure(4), clf

hz = linspace(0,EEG.srate,EEG.pnts);

subplot(331)
plot(hz,abs(thetacompX/EEG.pnts).^2,'linew',2), hold on
plot(hz,.2*abs(fft(EEG.data'*evecs(:,sidx(1)))/EEG.pnts),'linew',2)
plot(hz,.1*abs(fft(EEG.data'*evecs(:,sidx(end)))/EEG.pnts),'linew',2)
set(gca,'xlim',[0 60],'xtick',10:20:90), axis square
xlabel('Frequency (Hz)'), ylabel('Power (a.u.)')
title('Theta component power spectrum')


subplot(332)
plot(hz,abs(fft(EEG.data(30,:))/EEG.pnts),'k')
set(gca,'xlim',[0 90],'xtick',10:20:90), axis square
xlabel('Frequency (Hz)'), ylabel('Power (a.u.)')
title('EEG power spectrum @ max')


subplot(333)
plot(frex,mvarpac,'linew',2)
set(gca,'xlim',[0 90],'xtick',10:20:90), axis square
xlabel('Frequency (Hz)'), ylabel('CFC strength')
title('Peak-trough spectrum')


subplot(312)
plot(EEG.times,thetacomp, EEG.times,zscore(abs(hilbert(filterFGx(EEG.data,EEG.srate,40,15)' * evecs(:,sidx(1))))), EEG.times,zscore(abs(hilbert(filterFGx(EEG.data,EEG.srate,45,15)' * evecs(:,sidx(end))))),'linew',2);
set(gca,'xlim',[2 4])
legend({'theta';'gam1';'gam2'})
xlabel('Time (s)'), ylabel('EEG activity (a.u.)')
title('Theta and gamma components')


subplot(313)
plot(EEG.times,zscore(filterFGx(EEG.data(30,:),EEG.srate,thetafreq,5)'), EEG.times,zscore(abs(hilbert(filterFGx(EEG.data(31,:),EEG.srate,40,15)'))), EEG.times,zscore(abs(hilbert(filterFGx(EEG.data(63,:),EEG.srate,45,15)'))),'linew',2);
set(gca,'xlim',[2 4],'ylim',[-2.5 3.6])
title('Theta/gamma at max electrode')

%% now for traditional Euler PAC (this time using a debiasing term from van Driel et al., 2015)

dpac = zeros(EEG.nbchan,length(frex));

nperm     = 100;
cutpoints = randsample(10:EEG.pnts-10,nperm);
permpac   = zeros(1,nperm);

% takes a while for all electrodes, or just run POz
for chani=30%1:EEG.nbchan
    
    % filter for 6 Hz
    phase = angle(hilbert(filterFGx(EEG.data(chani,:),EEG.srate,thetafreq,5)));
    
    % filter over lots of higher frequencies
    for fi=1:length(frex)
        pow = abs(hilbert(filterFGx(EEG.data(chani,:),EEG.srate,frex(fi),6)));
        
        % zPACd
        obspac = abs(mean( pow.*( exp(1i*phase)-mean(exp(1i*phase)) ) ));
        for permi=1:nperm
            permpac(permi) = abs(mean( pow([cutpoints(permi):end 1:cutpoints(permi)-1]).*( exp(1i*phase)-mean(exp(1i*phase)) ) ));
        end
        
        dpac(chani,fi) = (obspac-mean(permpac))/std(permpac);
    end
end

% plot
figure(5), clf
clim = [-2 2];

subplot(221), imagesc(1:64,frex,dpac')
set(gca,'clim',clim)

subplot(222), topoplotIndie(dpac(:,dsearchn(frex',40)),EEG.chanlocs,'maplimits',clim);
subplot(223), topoplotIndie(dpac(:,dsearchn(frex',45)),EEG.chanlocs,'maplimits',clim);

subplot(224)
plot(frex,dpac(30,:),'k','linew',2)
set(gca,'xlim',[0 90],'xtick',10:20:90), axis square
xlabel('Frequency (Hz)'), ylabel('CFC strength')

%%
