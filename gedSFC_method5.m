%% multivariate cross-frequency coupling, based on generalized eigendecomposition (gedCFC)
% Method 5: spike-field coherence adapted from Methods 3 and 4 (sphered delay-embedded matrices)

% You will need the following files in the current directory or Matlab path:
%   - filterFGx.m

% mikexcohen@gmail.com

clear

%% initial parameters and random spike times

npnts  = 10000;
nchans = 16;
srate  = 1000;
data   = zeros(nchans,npnts);

% time window for waveform around each spike
npad   = 40; % even only, please!
npad2  = npad/2;

spikelocs = sort(randsample(100:npnts-100,500));

%% simulate random spike times with sine waves around them

% add patterns to data around spikes
for si=1:length(spikelocs)
    simsine = cos(2*pi*75*(-npad2:npad2)/srate + rand*2*pi) .* exp(-(-npad2:npad2).^2/200);
    data(:,spikelocs(si)-npad2:spikelocs(si)+npad2) = data(:,spikelocs(si)-npad2:spikelocs(si)+npad2) + bsxfun(@times,repmat(simsine,nchans,1),sin(exp(-(linspace(-1.5,1,nchans))).^2)');
end

% one example of a pattern
realpattern = bsxfun(@times,repmat(simsine,nchans,1),sin(exp(-(linspace(-1.5,1,nchans))).^2)');

% add random noise to data, and 80 Hz gamma oscillations (unrelated to spiking)
data = data + randn(size(data))*2;
data = data + bsxfun(@times,repmat(sin(2*pi*80*(1:npnts)/srate),16,1),exp(-(linspace(-1,1,16)).^2)');

%% the important stuff...

% produce augmented data
padorder = [ npnts-floor(npad2):npnts 1:floor(npad2)-1 ];

delEmb = zeros(16*npad,npnts);
for deli = 1:npad
    delEmb( (1:16)+(deli-1)*nchans,:) = detrend(data(:,[padorder(deli):end 1:padorder(deli)-1])')';
end

% sphere data
[evecsO,evalsO] = eig( (delEmb*delEmb')/size(delEmb,2) );
spheredata = (delEmb' * evecsO * sqrt(inv(evalsO)) )';
spheredata = reshape(spheredata,[npad*nchans npnts ]);


% sum covariances around spikes, then divide by N
spcov = zeros(size(delEmb,1));
for si=1:length(spikelocs)
    tmpdat = spheredata(:,spikelocs(si)-npad:spikelocs(si)+npad);
    spcov  = spcov + tmpdat*tmpdat'/size(tmpdat,2);
end
spcov = spcov/si;


% eigendecomposition of sphered matrix
[evecsF,evalsF] = eig( spcov );

% compute weights and map
jdw    = evecsO * sqrt(pinv(evalsO)) * evecsF;
jdmaps = pinv(jdw)';


% simple spike-triggered average
sta = zeros(nchans,npad+1);
for si=length(spikelocs)
    sta = sta+data(:,spikelocs(si)-npad2:spikelocs(si)+npad2);
end

%% plotting

figure(1), clf
tv = 1000*(-npad2:npad2)/srate;

% example peri-spike pattern
subplot(231)
contourf(tv,1:16,realpattern,40,'linecolor','none')
hold on, plot([0 0],get(gca,'ylim'),'k--'), axis square
title('Simulated pattern'), xlabel('Peri-spike time (ms)'), ylabel('"Cortical depth"')

% forward model of spatiotemporal component
subplot(232)
rmap = reshape(jdmaps(:,end)',nchans,npad);
contourf(npad2+tv(1:end-1),1:nchans,rmap,40,'linecolor','none')
hold on, plot(npad2+tv(1:end-1),8+1*zscore(rmap(9,:)),'k','linew',2)
title('Spatiotemporal component'), axis square
xlabel('Filter time (ms)'), ylabel('"Cortical depth"')

% Spike-triggered average
subplot(233)
imagesc(sta)
contourf(tv,1:nchans,sta,40,'linecolor','none')
hold on, plot([0 0],get(gca,'ylim'),'k--')
title('Spike-triggered average'), axis square
xlabel('Peri-spike time (ms)'), ylabel('"Cortical depth"')

% power spectrum of component and STA
subplot(2,3,[4 5])
plot(linspace(0,srate,200),abs(fft(zscore(rmap(8,:))/npad,200)).^2,'ks-','linew',2,'markersize',8,'markerfacecolor','w'), hold on
plot(linspace(0,srate,200),abs(fft(zscore(sta(8,:)) /npad,200)).^2,'bo-','linew',2,'markersize',8,'markerfacecolor','w')
set(gca,'xlim',[0 200])
xlabel('Frequencies (Hz)'), ylabel('Power (a.u.)')
legend({'STF';'STA'})

% example single-trial LFP traces
subplot(236)
for si=1:10
    plot(tv, 9*si + data(8,spikelocs(si)-npad2:spikelocs(si)+npad2),'k'), hold on
end
plot([0 0],get(gca,'ylim'),'k--')
xlabel('Peri-spike time (ms)'), ylabel('''Trials'' (spikes)')

%%

