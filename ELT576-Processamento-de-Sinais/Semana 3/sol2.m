close all, clc;

%% Parte 1 - Roteiro
%Transformada de Fourier de curta duração

fs = 1000;
t = 0:1/fs:3;
f0 = 150;
t1 = 3;
f1 = 450;
B = (f1-f0)/t1;
y = cos(2*pi*(f0*t+B/2*t.^2));
Y = abs(fft(y));
F = linspace(0,fs/2,round(length(y)/2));
plot(F,Y(1:round(length(y)/2)))
xlabel('Frequência (Hz)')
ylabel('Magnitude')
[S,F,T] = espectrograma(y,64,20,64,fs);
figure
surf(T,F,10*log10(abs(S)),'EdgeColor','none');
axis xy; axis tight; view(0,90);
xlabel('Tempo (s)');
ylabel('Frequência (Hz)');
%soundsc(y,fs)