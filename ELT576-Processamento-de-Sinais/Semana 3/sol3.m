
close all, clc;

%%
%Espectrogramas ou "Ouvindo imagens ..."
fs = 44100;

[lena fs]=audioread('lena.wav');
%soundsc(lena,fs);

nwin = fs/100;
nolap = 0%fs/10;
nfft = fs/4
%[S,F,T] = espectrograma(lena,nwin,nolap,nfft,fs); %versao mais detalhada

[S,F,T] = espectrograma(lena,256,40,256,fs);
figure
surf(T,F,10*log10(abs(S)),'EdgeColor','none');
axis xy; axis tight; view(0,90);
xlabel('Tempo (s)');
ylabel('Frequência (Hz)');
colormap(gray)