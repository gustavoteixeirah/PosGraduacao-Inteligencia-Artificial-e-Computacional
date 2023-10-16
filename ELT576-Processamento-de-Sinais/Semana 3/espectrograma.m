function varargout = espectrograma(x,nwin,novlap,nfft,fs)
% Fun��o que calcula o espectrograma de um sinal x.
%
% Entradas:
% x => sinal
% nwin => n�mero de pontos por janela
% nolap => n�mero de pontos de "overlap". Deve ser 0 <= nolap < nwin.
% nfft => n�mero de pontos da transformada de Fourier
% fs => frequ�ncia de amostragem
%
% Sa�das:
% S => espectrograma
% F => vetor de frequ�ncia
% T => vetor de tempo
%
% Casos:
% 1. Retorna imagem do espectrograma
% espectrograma(...)
%
% 2. Retorna matriz do espectrograma
% S = espectrograma(...)
%
% 3. Retorna matriz do espectrograma e vetores de frequ�ncia e tempo
% [S,F,T] = espectrograma(...)

% Verifica n�mero de entradas e sa�das
if nargin < 5 || nargin > 5
    error('N�mero incorreto de argumentos de entrada')
end

if nargout > 4 || nargout == 2
    error('N�mero incorreto de argumentos de sa�da')
end
% Verifica validade de novlap
if novlap < 0 || novlap >= nwin
    error('N�mero de overlap deve ser maior ou igual a zero e menor que o tamanho da janela')
end
% calcula o tamanho de x e n�mero de janelas dado o n�mero de pontos
nx = length(x);
nw = floor((nx-novlap)/(nwin-novlap));

% divide o sinal em janelas
xin = zeros(nwin,nw);
xin(:,1) = x(1:nwin);
for i = 1:(nw-1)
    xin(:,i) = x(1+(nwin-novlap)*i:(nwin-novlap)*i+nwin);
end

% calcula a fft, vetor de tempo e de frequ�ncia
S = fft(xin,nfft);
S = S(1:nfft/2+1,:);
F = linspace(0,fs/2,nfft/2+1);
T = linspace(0,((nwin-novlap)*(nw-1)+nwin)/fs,nw);

% define sa�das
if nargout == 0
    [t,f] = meshgrid(T,F);
    surf(t,f,10*log10(abs(S)),'edgecolor','none')
    xlabel('Tempo (s)')
    ylabel('Frequ�ncia (Hz)')
    view(0,90)
    axis tight
elseif nargout == 1
    varargout{1} = S;
elseif nargout == 3
    varargout{1} = S;
    varargout{2} = F;
    varargout{3} = T;
end