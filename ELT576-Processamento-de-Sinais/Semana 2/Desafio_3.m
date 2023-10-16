pkg load image; 

imgFile = 'lua.jpg'; 
imshow(I, [0 255]);

% Filtro de laplace
F = [-1 -1 -1;-1 8 -1;-1 -1 -1];  

y1 = conv2(I,F);
figure();
subplot(1,2,1)
% Exibe a imagem resultante da convolu��o
imshow(y1, [0,255]);   
% Soma da imagem resultante com a imagem original
If1 = uint8( double(I)+imresize(conv2(F,I),size(I))); 
subplot(1,2,2)
imshow(If1,[0,255]); 

% O processamento da imagem lunar envolveu tr�s etapas.
% Primeiro, a imagem original foi visualizada.
% Em seguida, aplicou-se um filtro de Laplace por convolu��o,
% real�ando contornos e detalhes.
% Finalmente, a soma da imagem convolu�da com a original 
% aumentou o contraste e destacou as caracter�sticas lunares. 
% Esse processo real�ou detalhes �nicos, como crateras e montanhas,
% tornando a superf�cie da lua mais vis�vel e discern�vel.