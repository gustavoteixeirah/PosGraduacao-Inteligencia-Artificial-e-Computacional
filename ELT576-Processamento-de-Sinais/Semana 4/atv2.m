% Limpeza de Microfonia em Sinais de Áudio

% Limpa o ambiente
clear all, close all, clc;

% Carrega a amostra de áudio contaminada por microfonia e reproduz
pkg load signal
[Y, FS] = audioread ('bird2fil.wav');
soundsc(Y, FS)

% Visualiza o espectro do sinal contaminado
nfft = FS/2;
overlap = .5;
[spectra, freq] = pwelch(Y, nfft, overlap, nfft, FS, 'half', 'db');
figure;
plot(freq, spectra)
title('Espectro do sinal + ruído');
xlabel('Frequência (Hz)')

% Projeta e visualiza um filtro passa-baixas
fbaixa = 4000;
n = 4;
[B, A] = butter(n, fbaixa/(FS/2), 'low');
figure;
freqz(B, A, nfft, FS);

% Filtra o sinal com o filtro passa-baixas
Ypb = filtfilt(B, A, Y);

% Visualiza o espectro do sinal filtrado
[spectra, freq] = pwelch(Ypb, nfft, overlap, nfft, FS, 'half', 'db');
figure;
plot(freq, spectra)
soundsc(Ypb, FS)

% Projeta e visualiza um filtro rejeita-faixa
f1 = 4930;
f2 = 5070;
[B, A] = butter(n, [f1/(FS/2) f2/(FS/2)], 'stop');
figure;
freqz(B, A, nfft, FS);

% Filtra o sinal com o filtro rejeita-faixa
Ynotch = filtfilt(B, A, Y);

% Visualiza o espectro do sinal filtrado
[spectra, freq] = pwelch(Ynotch, nfft, overlap, nfft, FS, 'half', 'db');
figure;
plot(freq, spectra)
soundsc(Ynotch, FS)
