close all, clc;


% frequ�ncia de amostragem em hertz
Fs = 8000; 

% numero de pontos
L = 8000; 

% representa valores de tempo correspondentes a cada ponto de amostra
t = 0:1/Fs:(L-1)/Fs; 

% sinal que combina uma forma de onda senoidal e ru�do aleat�rio
x = 0.7*sin(2*pi*500*t)+sin(2*pi*2000*t)+2*randn(1,L); 


% Plot
figure()
subplot(221),plot(t,x) 
title('x(t)');
xlabel('t');
ylabel('Amplitude');

% Transformada de Fourier do sinal
X = fft(x); 

% An�lise da Transformada de Fourier
freq = (-(L/2-1):L/2)*Fs/L; % Vetor de frequ�ncia para plotagem
pfreq = (0:L/2)*Fs/L;
% Retorno ao Dom�nio do Tempo
xnew = real(ifft(X));

% Plot
subplot(2,2,2), plot(freq,abs(X)) % Plota a fun��o no dom�nio da frequ�ncia
title('X(jw)');
xlabel('Frequ�ncia');
ylabel('Amplitude');
subplot(2,2,3), plot(freq,abs(fftshift(X))) % Plota a fun��o centralizada
title('X(jw)(fftshift)');

% Plot
ylabel('Amplitude');
subplot(2,2,4), plot(pfreq,abs(X(1:L/2+1)))
title('X(jw) (sem a parte repetida)');
xlabel('Frequ�ncia');
ylabel('Amplitude');


