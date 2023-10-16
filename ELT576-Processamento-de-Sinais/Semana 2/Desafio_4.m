G = imread('texto.jpg');
figure;
imshow(G, [0,255]);
title('Imagem Original');

F6 = (1/9)*[ 1 1 1; 1 1 1 ; 1 1 1];
y7 = conv2(G,F6);
figure;
imshow(y7, [0,255]);
title('Imagem Convoluída com Filtro de Média');

If7 = uint8(imresize(conv2(F6,G),size(G)));
figure;
imshow(If7,[0,255]);
title('Soma da Imagem Original com a Convoluída');


% Esse filtro é bem legal, ele ajuda a visualizar melhor textos
% que possam estar com partes faltando, pixels cortados, etc.
% Após pesquisar sobre, descobri que se trata de um filtro de média
% Também conhecido como um filtro de média de vizinhança 3x3.
% é frequentemente usado em processamento de imagens para realizar 
% suavização ou borramento, pois calcula a média dos valores de 
% intensidade dos pixels em uma vizinhança local.