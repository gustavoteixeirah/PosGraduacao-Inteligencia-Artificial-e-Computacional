close all, clc;

pkg load signal;

fs = 2048;
t = 0:1/fs:5;
s = sin(2*pi*262*t);

figure;
stem(t,s);
axis([0 0.04 -30 30]);
title('Sinal de interesse');
xlabel('Tempo (s)');


ruido = load('ruido.txt');
x = s + ruido;
figure;
stem(t,x);
axis([0 0.04 -30 30]);
title('Sinal + ruído');
xlabel('Tempo (s)');

nfft = 1024;
overlap = .5;
[spectra,freq] =  pwelch(x, nfft, overlap, nfft, fs, 'half', 'db');
figure;
plot(freq, spectra);
title('Espectro do sinal + ruído');
xlabel('Frequência (Hz)');

f2 = 264; 
n = 4;
[B, A] = butter(n, [f1/(fs/2) f2/(fs/2)]); 


figure;
freqz(B, A, nfft, fs);

y = filtfilt(B, A, x);

figure;
plot(t,y);
axis([0 0.04 -30 30]);
title('Sinal de interesse recuperado');
xlabel('Tempo (s)');

