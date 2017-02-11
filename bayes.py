# Classificação é provavelmente a tarefa de data mining mais conhecida e utilizada.
# Ela consiste em associar objetos a um conjunto pré-definido de classes de acordo com as suas características.
# Implementado por ferramentas como MALLET, Apache Mahout e NLTK1, Naïve Bayes computa a probabilidade de um documento pertencer a
# uma determinada classe a partir da probabilidade a priori de um documento ser desta classe e das probabilidades condicionais de
# cada termo ocorrer em um documento da mesma classe

# Uso do TextBlob, processamento de texto simplificado
# TextBlob é uma biblioteca Python (2 e 3) para processamento de dados textuais. Ele fornece uma API simples para mergulhar em tarefas comuns de processamento de linguagem natural (PNL), tais como tagging de parte de fala,
# extração de frase nominal, análise de sentimento, classificação, tradução e muito mais.

# Abaixo, uma simples implementação do NaiveBayes a partir da biblioteca textblob do Python:
# Criação de classificador
import import numpy as np
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

# o carregamento de dados pode ser a partir de arquivos CSV, JSON, TSV
# Parametro que recebe as informações de treinamento no padrão JSON

treinar = \
    [
    ('O Bairro Barroso é o bairro com o maior índice de homicidios de Fortaleza.', 'sim'),('O time do Ceará começou bem o campeonato', 'sim'),
    ('Eu gosto de ir a praia com chuva.', 'nao'),('Trabalhar na área de Saúde é muito gratificante.', 'não'),
    ('Comer peixe faz bem a saúde.', 'sim'),('O Flamengo é o melhor time do Brasil', 'não'),
    ('Todo político é honesto', 'nao'),('A justiça no brasil funciona para todos', 'não'),
    ('I love the northeast','yes')
    ]
# recebe as opções para testar
opcoesteste = \
    [
    ('O Time do Fortaleza é o melhor do campeonato.', 'sim'),('Os processos de justiça aqui são rápidos', 'não'),
    ("Se chover não vá a praia, pode ser perigoso.", 'não'),("A política é uma vergonhosa em todos lugares", 'sim'),
    ("Ser honesto é sinônimo de responsável", 'sim'),("Eu não estou bem de saúde!", 'sim'),
    ]

# passagem de dados para o construtor
classifica01 = NaiveBayesClassifier(treinar)

# Classificação de Texto por amostragem
print(classifica01.classify("Carne vermelha faz bem"))
print(classifica01.classify("Eu não gosto de futebol"))
print(classifica01.classify("O time do Flamengo ganhou todas as partidas"))
print(classifica01.classify("I love the Brazil"))

# Classificação atraves do textblob
conteudo = TextBlob("O Jogo estava muito bom. O melhor time estava em campo, Flamengo"
                "Adoro Futebol", classifier=classifica01)
print(conteudo)

print(conteudo.classify())

# estrutura que percorre toda sentença
for frase in conteudo.sentences:
    print(frase)
    print(frase.classify())

# exibe a exatidão, precisão
print("Exatidão: {0}".format(classifica01.accuracy(opcoesteste)))

# Exibição das informações
classifica01.show_features(3)
