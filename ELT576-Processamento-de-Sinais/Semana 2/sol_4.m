close all
clc

% Parte IV

G = imread('texto.jpg');            % Carrega a imagem
figure; 
imshow(G, [0,255]);                 % Mostra a imagem

F6 = (1/9)*[ 1 1 1; 1 1 1 ; 1 1 1]; % Cria��o do filtro
y7 = conv2(G,F6);                   % Convolu��o da imagem com o filtro

##salvaGrafico("Exerc�cio 3_4_a");

figure;
imshow(y7, [0,255]);                        % Mostra a imagem convoluida

##salvaGrafico("Exerc�cio 3_4_b");

If7 = uint8(imresize(conv2(F6,G),size(G))); % Soma a imagem convoluida com a original
figure; 
imshow(If7,[0,255]);



