
% Parte I
warning('off'); 
pkg load image; 

imgFile = 'lua.jpg'; 

I = imread(imgFile);              % Carrega a imagem baixada previamente
figure;
image(I);               % Exibe a imagem da lua
figure;B
imshow(I, [0,255]); 
xlabel([ 'Imagem: ' imgFile ]); 

F = [-1 -1 -1;-1 8 -1;-1 -1 -1];  % Filtro de laplace, com o objetivo de realçar a imagem

y1 = conv2(I,F);       % Realiza a convolução da imagem da lua com o filtro F
figure()
subplot(1,2,1)
imshow(y1, [0,255]);   % Exibe a imagem resultante da convolução
If1 = uint8( double(I)+imresize(conv2(F,I),size(I))); % Soma da imagem resultante com a imagem original
subplot(1,2,2)
imshow(If1,[0,255]);   % Exibe a soma da imagem resultante com a imagem original
