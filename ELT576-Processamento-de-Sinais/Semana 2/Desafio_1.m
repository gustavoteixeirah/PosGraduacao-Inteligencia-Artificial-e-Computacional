% Desafio 1

% a
h = [1 zeros(1,20) 0.5 zeros(1,10)];
figure;
plot(h,'marker','none','LineWidth',1);


% b
x = [0 1:10 ones(1,5)*5 zeros(1,10)]; % Definindo o sinal de entrada
figure;
stem(x,'marker','none','LineWidth',1);


% c
y = conv(x,h);                   
t = length(y);

% Completando com zeros para igualar ao tamanho de y
aux = zeros(1, t-length(x));
xi = [x aux];                   
aux2 = zeros(1, t-length(h));
hi = [h aux2];                  

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