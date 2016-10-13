# -*- coding: utf-8 -*-
import numpy
import random
import Mnist_Loader
import Meu_Teste
import time


class RedeNeural(object):
    def __init__(self, tamanho, taxa_aprendizado, erro_desejado):
        '''
        :param tamanho: Tamanho da rede [784, x1, x2, ... , xn, 10], n é quantidade de camadas escondidas.
        :param taxa_aprendizado: taxa de aprendizado da rede
        :param erro_desejado: erro desejado para rede
        '''
        self.num_camadas = len(tamanho)
        self.tamanho = tamanho
        self.vetor_bias = [numpy.random.randn(num, 1) for num in tamanho[1:]]
        self.vetor_pesos = [numpy.random.randn(y, x) / numpy.sqrt(x) for x, y in zip(tamanho[:-1], tamanho[1:])]
        self.taxa_aprendizado = taxa_aprendizado
        self.erro_desehado = erro_desejado

    def gradiente_descendente(self, dados_treinamento, epocas, tamanho_batch, dados_teste=None):
        '''
        :param dados_treinamento: Dados do treinamento. Utilizando MNIST
        :param epocas: Quantidades de épocas caso o erro_desejado demore muito a ser atingido
        :param tamanho_batch: quantidade de elementos dentro do batch de treinamento
        :param dados_teste: Dados teste. Utilizando MNIST
        :return:
        '''
        num_teste = None
        if dados_teste:
            num_teste = len(dados_teste)
        tam = len(dados_treinamento)
        for i in xrange(epocas):
            random.shuffle(dados_treinamento)
            batchs = [dados_treinamento[k:k + tamanho_batch] for k in xrange(0, tam, tamanho_batch)]
            for batch in batchs:
                vetor_erros = self.pesos_e_bias(batch)
                erro_batch = numpy.sum(vetor_erros)
                # if erro_desejado > erro_batch:
                #     print "Erro: ", erro_desejado, " desejado atingido. Erro final do batch: ", erro_batch
                #     break
            if dados_teste:
                print "Época {0}: {1} / {2}".format(i, self.estimativa(dados_teste), num_teste)
            else:
                print "Época ", i, " finalizada."

        teste_lista, nome_lista = Meu_Teste.imagem_foto()

        for x, y in zip(teste_lista, nome_lista):
            print '--- file: ', y
            print self.feedforward_dados_teste(x)

    def pesos_e_bias(self, batch):
        '''
            :param batch: Batch de treinamento, tupla (x,y), onde x = matrix da imagem, y = resultado esperado
            :param taxa_aprendizado: Auto descritivo
            :return: retorna void, porém é principal para a atualização do self.vetor_bias e self.vetor_pesos
            '''
        # Aqui faz update dos pesos e bias da rede aplicando o
        # algoritmo de backpropagation para cada batch definido
        # anteriormente.
        # Batch é um Tuple (x, y), onde x é a
        # matriz Figura e y é o resultado.
        vetor_bias = [numpy.zeros(b.shape) for b in self.vetor_bias]
        vetor_pesos = [numpy.zeros(w.shape) for w in self.vetor_pesos]
        vetor_erros = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]

        for x, y in batch:
            delta_vetor_bias, delta_vetor_pesos, delta_vetor_erros = self.forw_back_prop(x, y)
            vetor_bias = [nb + dnb for nb, dnb in zip(vetor_bias, delta_vetor_bias)]
            vetor_pesos = [nw + dnw for nw, dnw in zip(vetor_pesos, delta_vetor_pesos)]
            vetor_erros += delta_vetor_erros

        vetor_erros /= len(batch)
        self.vetor_pesos = [w - self.taxa_aprendizado * nw for w, nw in zip(self.vetor_pesos, vetor_pesos)]
        self.vetor_bias = [b - self.taxa_aprendizado * nb for b, nb in zip(self.vetor_bias, vetor_bias)]
        return vetor_erros

    def forw_back_prop(self, x, y):
        '''
            :param x: Valores da imagem 28x28
            :param y: Resultado esperado entre 0-9
            :return: vetor_bias, vetor_pesos
            '''

        '''
        Forward propagation
        vetor_resultado = resultado da saída de cada neurônio
        vetor_ativação = resultado da saída de cada neurônio após a função sigmoid
        erro_resultante = diferença entre o esperado y com o gerado no vetor_ativação após a função de erro_quadrático
        '''

        vetor_bias = [numpy.zeros(b.shape) for b in self.vetor_bias]
        vetor_pesos = [numpy.zeros(w.shape) for w in self.vetor_pesos]

        resultado_ativacao = x
        vetor_ativacao = [x]
        vetor_resultado = []
        for bias, pesos in zip(self.vetor_bias, self.vetor_pesos):
            result = numpy.dot(pesos, resultado_ativacao) + bias
            vetor_resultado.append(result)
            resultado_ativacao = sigmoid(result)
            vetor_ativacao.append(resultado_ativacao)

        # Erro
        vetor_erro = cost_derivative(vetor_ativacao[-1], y)
        vetor_erro_quadratico = erro_quadratico_medio(vetor_erro)

        '''
            Backward propagation
            Faço gradiente descendente, aplico regra da cadeia.
            uso índices negativos, pois dessa forma posso variar a quantidade de camadas
            Python define que sempre indice -1 será o ultimo elemento.
        '''
        delta = cost_derivative(vetor_ativacao[-1], y) * sigmoid_derivada(vetor_resultado[-1])
        vetor_bias[-1] = delta
        vetor_pesos[-1] = numpy.dot(delta, vetor_ativacao[-2].transpose())

        for l in xrange(2, self.num_camadas):
            result = vetor_resultado[-l]
            sp = sigmoid_derivada(result)
            delta = numpy.dot(self.vetor_pesos[-l + 1].transpose(), delta) * sp
            vetor_bias[-l] = delta
            vetor_pesos[-l] = numpy.dot(delta, vetor_ativacao[-l - 1].transpose())
        return vetor_bias, vetor_pesos, vetor_erro_quadratico

    def estimativa(self, test_data):

        test_results = [(numpy.argmax(self.feedforward_dados_teste(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward_dados_teste(self, a):

        for b, w in zip(self.vetor_bias, self.vetor_pesos):
            a = sigmoid(numpy.dot(w, a) + b)
        return a


def cost_derivative(output_activations, y):
    return output_activations - y  # Miscellaneous functions


def sigmoid(z):
    # return 1.7159 * numpy.tanh((2/3)*z)
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_derivada(z):
    # return 1.14393 * numpy.arccosh((2/3)*z) * numpy.arccosh((2/3)*z)
    return sigmoid(z) * (1 - sigmoid(z))


def erro_quadratico_medio(vetor_valores):
    return vetor_valores * vetor_valores


def main():
    t0 = time.clock()
    training_data, validation_data, test_data = Mnist_Loader.load()
    net = RedeNeural([784, 200, 10], 0.04, 0.0001)
    net.gradiente_descendente(training_data, 100, 10, test_data)
    t1 = time.clock()
    print t1 - t0


if __name__ == '__main__':
    main()
