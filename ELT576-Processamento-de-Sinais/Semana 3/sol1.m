close all, clc;


%% Parte 1

Fs = 8000; % Define a frequ�ncia
L = 8000; % numero de pontos
t = 0:1/Fs:(L-1)/Fs; % Vetor de tempo
x = 0.7*sin(2*pi*500*t)+sin(2*pi*2000*t)+2*randn(1,L); % Fun��o de exemplo

figure()
subplot(221),plot(t,x) % Plota a fun��o no tempo
title('x(t)');
xlabel('t');
ylabel('Amplitude');

X = fft(x); % Realiza a transformada r�pida

freq = (-(L/2-1):L/2)*Fs/L; % Vetor de frequ�ncia para plotagem
pfreq = (0:L/2)*Fs/L;
xnew = real(ifft(X)); % Realiza a transformada inversa

subplot(2,2,2), plot(freq,abs(X)) % Plota a fun��o no dom�nio da frequ�ncia
title('X(jw)');
xlabel('Frequ�ncia');
ylabel('Amplitude');
subplot(2,2,3), plot(freq,abs(fftshift(X))) % Plota a fun��o centralizada
title('X(jw)(fftshift)');

ylabel('Amplitude');
subplot(2,2,4), plot(pfreq,abs(X(1:L/2+1)))
title('X(jw) (sem a parte repetida)');
xlabel('Frequ�ncia');
ylabel('Amplitude');


