%% 
%Canto das baleias

load('whalecalls.mat');
x1=X1(1,:);
x2=X1(2,:);
x3=X1(3,:);
x4=X1(4,:);
sound(x1,fs);
sound(x2,fs);
sound(x3,fs);
sound(x4,fs);
figure
subplot(2,2,1)
[S,F,T] = espectrograma(x1,256,20,256,fs);
surf(T,F,10*log10(abs(S)),'EdgeColor','none');
axis xy; axis tight; view(0,90);
xlabel('Tempo (s)');
ylabel('Frequência (Hz)');
subplot(2,2,2)
[S,F,T] = espectrograma(x2,256,20,256,fs);
surf(T,F,10*log10(abs(S)),'EdgeColor','none');
axis xy; axis tight; view(0,90);
xlabel('Tempo (s)');
ylabel('Frequência (Hz)');
subplot(2,2,3)
[S,F,T] = espectrograma(x3,256,20,256,fs);
surf(T,F,10*log10(abs(S)),'EdgeColor','none');
axis xy; axis tight; view(0,90);
xlabel('Tempo (s)');
ylabel('Frequencia (Hz)');
subplot(2,2,4)
[S,F,T] = espectrograma(x4,256,20,256,fs);
surf(T,F,10*log10(abs(S)),'EdgeColor','none');
axis xy; axis tight; view(0,90);
xlabel('Tempo (s)');
ylabel('Frequência (Hz)');
