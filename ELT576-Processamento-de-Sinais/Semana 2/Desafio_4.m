G = imread('texto.jpg');
figure;
imshow(G, [0,255]);
title('Imagem Original');

F6 = (1/9)*[ 1 1 1; 1 1 1 ; 1 1 1];
y7 = conv2(G,F6);
figure;
imshow(y7, [0,255]);
title('Imagem Convolu�da com Filtro de M�dia');

If7 = uint8(imresize(conv2(F6,G),size(G)));
figure;
imshow(If7,[0,255]);
title('Soma da Imagem Original com a Convolu�da');


% Esse filtro � bem legal, ele ajuda a visualizar melhor textos
% que possam estar com partes faltando, pixels cortados, etc.
% Ap�s pesquisar sobre, descobri que se trata de um filtro de m�dia
% Tamb�m conhecido como um filtro de m�dia de vizinhan�a 3x3.
% � frequentemente usado em processamento de imagens para realizar 
% suaviza��o ou borramento, pois calcula a m�dia dos valores de 
% intensidade dos pixels em uma vizinhan�a local.