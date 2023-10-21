% Fecha todas as janelas de figuras abertas e limpa a janela de comando
close all, clc;

% Configura��es do sinal de curta dura��o
fs = 1000; % Frequ�ncia de amostragem em Hz
t = 0:1/fs:3; % Vetor de tempo de 0 a 3 segundos
f0 = 150; % Frequ�ncia inicial
t1 = 3; % Dura��o do sinal
f1 = 450; % Frequ�ncia final
B = (f1-f0)/t1; % Largura de banda
y = cos(2*pi*(f0*t + B/2*t.^2)); % Gera��o do sinal

% An�lise da Transformada de Fourier
Y = abs(fft(y)); % Calcula a Transformada de Fourier
F = linspace(0, fs/2, round(length(y)/2)); % Vetor de frequ�ncias para plotagem
plot(F, Y(1:round(length(y)/2))) % Plota a magnitude da Transformada de Fourier
xlabel('Frequ�ncia (Hz)')
ylabel('Magnitude')

% An�lise do Espectrograma
[S, F, T] = espectrograma(y, 64, 20, 64, fs); % Calcula o espectrograma
figure
surf(T, F, 10*log10(abs(S)), 'EdgeColor', 'none'); % Plota o espectrograma em 3D
axis xy; axis tight; view(0, 90);
xlabel('Tempo (s)');
ylabel('Frequ�ncia (Hz)');

% Reprodu��o do som
soundsc(y, fs);
