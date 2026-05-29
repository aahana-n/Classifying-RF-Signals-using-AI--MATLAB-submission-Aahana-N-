clc;
clear;
close all;

%% ============================================================
%  GLOBAL PARAMETERS
%  fs = 2e6 matches USRP B210 baseband rate exactly
%% ============================================================
fs             = 2e6;
numSymbols     = 10;
symbolDuration = 1;                       % 1 second per symbol — slow enough
                                          % for peaks to land at exact freqs
samplesPerSymbol = fs * symbolDuration;   % = 2,000,000 samples/symbol
t = (0:samplesPerSymbol-1)' / fs;

rng(42);
symbolIndices = randi([0 1], numSymbols, 1);

%% ============================================================
%  GAINS
%  ASK(0.20) + FSK(0.25) + PSK(0.20) + Sine(0.15) = 0.80 peak
%  Simulink sine block: Amplitude = 0.15, Frequency = 100 Hz
%% ============================================================
gain_ask = 0.20;
gain_fsk = 0.25;
gain_psk = 0.20;

%% ============================================================
%  FREQUENCY PLAN
%  Sine (Simulink) :  100 Hz
%  ASK             :  200 Hz
%  PSK             :  300 Hz
%  FSK f0/f1       :  360/440 Hz
%% ============================================================
fc_ask = 200;
fc_psk = 300;
fc_fsk = 400;
fDev   = 40;
f0_fsk = fc_fsk - fDev;   % 360 Hz
f1_fsk = fc_fsk + fDev;   % 440 Hz

%% ============================================================
%  ASK Modulation (BASK) — raised cosine pulse shaping
%% ============================================================
rolloff      = 0.35;
rcFilterSpan = 6;
rcFilter     = rcosdesign(rolloff, rcFilterSpan, samplesPerSymbol, 'sqrt');

totalSamples   = numSymbols * samplesPerSymbol;
tTotal         = (0:totalSamples-1)' / fs;
carrierASKFull = cos(2*pi*fc_ask*tTotal);

askAmplitude = [0, 1];
askBaseband  = zeros(totalSamples, 1);
for k = 1:numSymbols
    idx = (k-1)*samplesPerSymbol + 1 : k*samplesPerSymbol;
    askBaseband(idx) = askAmplitude(symbolIndices(k)+1);
end
askBasebandShaped = conv(askBaseband, rcFilter, 'same');
askWaveform       = gain_ask * askBasebandShaped .* carrierASKFull;
askWaveform       = askWaveform - mean(askWaveform);   % DC removal

askConstellation  = askAmplitude(symbolIndices + 1).' .* cos(2*pi*fc_ask*(samplesPerSymbol/fs));

%% ============================================================
%  FSK Modulation (BFSK)
%% ============================================================
fskWaveform         = zeros(totalSamples, 1);
fskConstellationRaw = zeros(numSymbols, 1);

for k = 1:numSymbols
    idx = (k-1)*samplesPerSymbol + 1 : k*samplesPerSymbol;
    % Pure real tones only — complex phasor was causing frequency doubling
    sig = (1 - symbolIndices(k)) * cos(2*pi*f0_fsk*t) + ...
           symbolIndices(k)      * cos(2*pi*f1_fsk*t);
    fskWaveform(idx)       = gain_fsk * sig;
    fskConstellationRaw(k) = sig(end);
end
fskWaveform = fskWaveform - mean(fskWaveform);   % DC removal

%% ============================================================
%  PSK Modulation (BPSK)
%% ============================================================
pskWaveform      = zeros(totalSamples, 1);
pskConstellation = exp(1j * pi * symbolIndices);

for k = 1:numSymbols
    idx   = (k-1)*samplesPerSymbol + 1 : k*samplesPerSymbol;
    phase = pi * symbolIndices(k);
    pskWaveform(idx) = gain_psk * cos(2*pi*fc_psk*t + phase);
end
pskWaveform = pskWaveform - mean(pskWaveform);   % DC removal

%% ============================================================
%  TIME VECTOR & SIMULINK MATRICES
%  From Workspace block settings:
%    Sample time  = 5e-7  (= 1/2e6)
%    Output dtype = double
%% ============================================================
timeVector   = tTotal;

ASK_simulink = [timeVector, askWaveform];
FSK_simulink = [timeVector, fskWaveform];
PSK_simulink = [timeVector, pskWaveform];

%% ============================================================
%  PEAK REPORT
%% ============================================================
fprintf('\n=== Signal Peaks ===\n');
fprintf('  ASK  : %.3f  @ %d Hz\n',      max(abs(askWaveform)), fc_ask);
fprintf('  FSK  : %.3f  @ %d/%d Hz\n',   max(abs(fskWaveform)), f0_fsk, f1_fsk);
fprintf('  PSK  : %.3f  @ %d Hz\n',      max(abs(pskWaveform)), fc_psk);
fprintf('  Sine : 0.150 @ 100 Hz  (set in Simulink)\n');
fprintf('  Worst-case sum : %.3f  %s\n', ...
    max(abs(askWaveform)) + max(abs(fskWaveform)) + max(abs(pskWaveform)) + 0.15, ...
    '(<= 1.0 = OK)');

fprintf('\n=== Simulink Settings ===\n');
fprintf('  From Workspace sample time : 5e-7\n');
fprintf('  From Workspace dtype       : double\n');
fprintf('  Buffer size                : 64\n');
fprintf('  Sine amplitude             : 0.15\n');
fprintf('  Sine frequency             : 100 Hz\n');
fprintf('  USRP sample rate           : 2e6\n');
fprintf('  USRP Tx gain               : 32 dB\n');
fprintf('  USRP centre freq           : 2.45 GHz\n');
fprintf('\n=== Spectrum Analyzer Settings ===\n');
fprintf('  Sample rate   : 2e6\n');
fprintf('  Freq range    : 0 to 500 Hz  (enter as 0 to 0.5 in kHz view)\n');
fprintf('  Freq scale    : Linear\n');
fprintf('  RBW           : 1 Hz\n');

%% ============================================================
%  PLOTS
%% ============================================================
% Show first 3 symbols only — 3 × 2M = 6M samples, manageable to plot
plotSamples = min(3 * samplesPerSymbol, totalSamples);
tPlot       = tTotal(1:plotSamples);

figure('Name','Modulated Waveforms','NumberTitle','off');
subplot(3,1,1);
plot(tPlot, askWaveform(1:plotSamples), 'b', 'LineWidth', 1.0);
title(sprintf('ASK | f_c=%dHz | gain=%.2f | RC rolloff=%.2f', fc_ask, gain_ask, rolloff));
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(3,1,2);
plot(tPlot, fskWaveform(1:plotSamples), 'r', 'LineWidth', 1.0);
title(sprintf('FSK | f_0=%dHz f_1=%dHz | gain=%.2f', f0_fsk, f1_fsk, gain_fsk));
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(3,1,3);
plot(tPlot, pskWaveform(1:plotSamples), 'g', 'LineWidth', 1.0);
title(sprintf('PSK | f_c=%dHz | gain=%.2f', fc_psk, gain_psk));
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

sgtitle('Modulation Waveforms — fs=2MHz');

%% ============================================================
%  SPECTRUM  (zoom 0–500 Hz, matches Spectrum Analyzer view)
%% ============================================================
% Use first symbol only for quick FFT (2M points is sufficient resolution)
N_fft  = samplesPerSymbol;
f_ax   = (0:N_fft-1) * (fs/N_fft);
half   = 1:floor(N_fft/2);
zoom   = f_ax(half) <= 500;    % only plot 0–500 Hz

figure('Name','Baseband Spectra 0-500Hz','NumberTitle','off');
subplot(3,1,1);
Y_ask = fft(askWaveform(1:N_fft));
plot(f_ax(half(zoom)), 2*abs(Y_ask(half(zoom)))/N_fft, 'b', 'LineWidth', 1.2);
xline(100,    'k--', 'Sine 100Hz',       'LabelVerticalAlignment','bottom');
xline(fc_ask, 'b--', sprintf('ASK %dHz', fc_ask), 'LabelVerticalAlignment','bottom');
title('ASK Spectrum'); xlabel('Frequency (Hz)'); grid on;

subplot(3,1,2);
Y_fsk = fft(fskWaveform(1:N_fft));
plot(f_ax(half(zoom)), 2*abs(Y_fsk(half(zoom)))/N_fft, 'r', 'LineWidth', 1.2);
xline(f0_fsk, 'r--', sprintf('f0 %dHz', f0_fsk), 'LabelVerticalAlignment','bottom');
xline(f1_fsk, 'r-',  sprintf('f1 %dHz', f1_fsk), 'LabelVerticalAlignment','bottom');
title('FSK Spectrum'); xlabel('Frequency (Hz)'); grid on;

subplot(3,1,3);
Y_psk = fft(pskWaveform(1:N_fft));
plot(f_ax(half(zoom)), 2*abs(Y_psk(half(zoom)))/N_fft, 'g', 'LineWidth', 1.2);
xline(fc_psk, 'g--', sprintf('PSK %dHz', fc_psk), 'LabelVerticalAlignment','bottom');
title('PSK Spectrum'); xlabel('Frequency (Hz)'); grid on;

sgtitle('Individual Spectra — 0 to 500 Hz');

fprintf('\nDone. Matrices: ASK_simulink, FSK_simulink, PSK_simulink\n');
fprintf('Total samples per signal: %d (%.1f seconds)\n', totalSamples, totalSamples/fs);