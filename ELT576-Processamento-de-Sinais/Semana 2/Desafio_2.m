arq = 'trumpet.mat';
trumpet = load(arq); % Carregando o arquivo trumpet
figure;plot(trumpet.y)
soundsc(trumpet.y, trumpet.Fs, 16)

h4 = [ones(1,50)/50 zeros(1,20)];
% Convolução da função h4 e dos dados do arquivo trumpet
y4 = conv(h4, trumpet.y);  
audiowrite('trumpet_conv_ex3.wav', y4, trumpet.Fs);
figure;
subplot(3,1,1)
plot(y4,'marker','none','LineWidth',1);
title('Convolução trumpet com resposta a impulso quadrado')
soundsc(y4, trumpet.Fs, 16)

% O sinal fica mais suave, parece mais "abafado"


% O que acontece se alterar o número de uns e zeros de h2?
h4 = [ones(1,100)/50 zeros(1,20)];
y4 = conv(h4, trumpet.y);    % Convolução da função h4 e dos dados do arquivo trumpet
subplot(3,1,2)
plot(y4,'marker','none','LineWidth',1);
title('Convolução trumpet com resposta a impulso aumentando o numero de uns')
soundsc(y4, trumpet.Fs, 16)
% Nesse caso parece q abafou novamente, mas em um outro tom

h4 = [ones(1,50)/50 zeros(1,2000)];
y4 = conv(h4, trumpet.y);  % Convolução da função h4 e dos dados do arquivo trumpet
subplot(3,1,3)
plot(y4,'marker','none','LineWidth',1);
title('Convolução trumpet com resposta a impulso aumentando o numero de zeros')
soundsc(y4, trumpet.Fs, 16)
% Nesse novamente, parece q abafou mas em outro tom, parece q esse filtro
% e o anterior ambos são de passa-baixa, mas alterando levemente o tom
