
close all, clc;

%% Parte 1 - Roteiro

% Exercício I

% Letra a)

h = [1 zeros(1,20) 0.5 zeros(1,10)]; % Definindo a função h
figure;
plot(h,'marker','none','LineWidth',1);
title('Resposta ao impulso - roteiro')
xlabel('t');
ylabel('Amplitude');
grid on
##salvaGrafico("Resposta ao Impulso - 1a");

% Letra b)

x = [0 1:10 ones(1,5)*5 zeros(1,10)]; % Definindo o sinal de entrada
figure;
stem(x,'marker','none','LineWidth',1);
title('Entrada - roteiro')
xlabel('t');
ylabel('Amplitude');
grid on
##salvaGrafico("Entrada - 1b");


% Letra c)

y = conv(x,h);                   % Convolução de x e h
t = length(y);
aux = zeros(1, t-length(x));
xi = [x aux];                    % Completando com zeros para igualar ao tamanho de y
aux2 = zeros(1, t-length(h));
hi = [h aux2];                   % Completando com zeros para igualar ao tamanho de y

figure;

subplot(3,1,1)
stem(xi,'marker','none','LineWidth',1);
title('x(t)');
ylabel('x');
xlabel('t');

subplot(3,1,2);
stem(hi,'marker','none','LineWidth',1);
title('h(t)');
ylabel('h');
xlabel('t');

subplot(3,1,3);
stem(y,'marker','none','LineWidth',1);
title('y(t)');
ylabel('y');
xlabel('t');

##salvaGrafico("Resposta ao Impulso, Entrada e Saída - 1c");



