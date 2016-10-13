import numpy
import cPickle
import gzip


def carregar():
    f = gzip.open('../mnist_teste/mnist.pkl.gz', 'rb')
    dados_treinamento, dados_validacao, dados_teste = cPickle.load(f)
    f.close()
    return dados_treinamento, dados_validacao, dados_teste


def load():
    dados_treinamento_suporte, dados_validacao_suporte, dados_teste_suporte = carregar()

    treinamento_entrada = [numpy.reshape(x, (784, 1)) for x in dados_treinamento_suporte[0]]
    treinament_resultado = [vetor_digitos(y) for y in dados_treinamento_suporte[1]]

    validacao_entrada = [numpy.reshape(x, (784, 1)) for x in dados_validacao_suporte[0]]
    validacao_resultado = [vetor_digitos(y) for y in dados_validacao_suporte[1]]

    vetor_final_teste = treinamento_entrada + validacao_entrada
    vetor_final_resultado = treinament_resultado + validacao_resultado
    vetor_final_projeto = zip(vetor_final_teste, vetor_final_resultado)

    teste_entrada = [numpy.reshape(x, (784, 1)) for x in dados_teste_suporte[0]]
    teste_resultado = [vetor_digitos(y) for y in dados_teste_suporte[1]]
    teste_dados_final = zip(teste_entrada, teste_resultado)

    return vetor_final_projeto, teste_dados_final


def vetor_digitos(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    # e = numpy.zeros((10, 1))
    # e[j] = 1.0
    if j == 0:
        return [1.0]
    else:
        return [0.0]


if __name__ == '__main__':
    aavetor_final_projeto, taaeste_dados_final = load()

    print 'fim'
