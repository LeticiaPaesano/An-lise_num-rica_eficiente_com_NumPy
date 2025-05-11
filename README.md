![Fotobibli](https://github.com/user-attachments/assets/a6bc37c9-997e-40f0-b649-fa056760d793)
# Estudo de NumPy com Análise de Preços de Maçãs

## Sobre 
Os estudos aqui apresentados baseiam-se no curso "NumPy: análise numérica eficiente com Python", realizado pela Alura e conduzido pelo instrutor [Allan Segovia](https://github.com/allanspadini).

O foco deste projeto é:

- Explorar a estrutura de arrays do NumPy.
- Realizar operações entre arrays.
- Trabalhar com dados reais (preços de maçãs em diferentes cidades).
- Aplicar técnicas de manipulação, análise e visualização de dados.
- Realizar regressão linear simples e ajustes de modelos, etc.

## Tecnologias Utilizadas
- Python 
- NumPy
- Matplotlib

## Conteúdo Estudado

### 1. Importação de Bibliotecas

```python
import numpy as np
import matplotlib.pyplot as plt
````
### 2. Carregamento dos Dados
Os dados foram carregados a partir de um arquivo CSV disponibilizado pela Alura.

```python
url = 'https://raw.githubusercontent.com/alura-cursos/numpy/dados/apples_ts.csv'
dado = np.loadtxt(url, delimiter=',', usecols=np.arange(1,88,1))
````
Carrega o arquivo ignorando a primeira coluna, pois representa apenas nomes de cidades.

### 3. Preparação dos Dados
Transpusemos o array para facilitar a manipulação (linhas viram colunas):

```python
dado_transposto = dado.T
```
Separação dos dados em meses e preços:
```python
datas = dado_transposto[:, 0]
precos = dado_transposto[:, 1:6]
```
Corrigimos o vetor de datas para representar corretamente os meses pois temos 87 meses:

```python
datas = np.arange(1, 88, 1)
```
### 4. Análise Inicial dos Preços
Preço de maçãs em Moscow ao longo dos meses:

```python
plt.plot(datas, precos[:, 0])
plt.title('Preço de maçãs em Moscow')
plt.xlabel('Mês')
plt.ylabel('Preço')
plt.show()
```
![Preco macas em Moscow](https://github.com/user-attachments/assets/2c00e4ba-f364-4fad-9c4c-66ceae125b06)

### 5. Separação dos Preços por Cidade

```python
Moscow = precos[:, 0]
Kaliningrad = precos[:, 1]
Petersburg = precos[:, 2]
Krasnodar = precos[:, 3]
Ekaterinburg = precos[:, 4]
```
### 6. Análise Comparativa dos Preços em Moscow
Analisamos o comportamento dos preços de Moscow ano a ano, separando blocos de 12 meses:

```python
Moscow_ano1 = Moscow[0:12]
Moscow_ano2 = Moscow[12:24]
Moscow_ano3 = Moscow[24:36]
Moscow_ano4 = Moscow[36:48]
```
Gráfico comparativo:

```python
plt.plot(np.arange(1,13), Moscow_ano1)
plt.plot(np.arange(1,13), Moscow_ano2)
plt.plot(np.arange(1,13), Moscow_ano3)
plt.plot(np.arange(1,13), Moscow_ano4)
plt.legend(['Ano 1', 'Ano 2', 'Ano 3', 'Ano 4'])
plt.title('Comparação Anual dos Preços - Moscow')
plt.xlabel('Meses')
plt.ylabel('Preço')
plt.grid()
plt.show()
```
Para complementar a análise gráfica, realizamos uma comparação numérica direta entre os dados do ano3 e ano4, utilizando a função ```np.allclose()```:

```python
np.allclose(Moscow_ano3, Moscow_ano4, atol=10)
```

![Comparacao anual dos precos](https://github.com/user-attachments/assets/970f59b4-e568-4dfe-894b-88b9db2e97d4)

### 7. Tratamento de Valores Nulos
Kaliningrad apresentava valores nulos (NaN)

```python
# Analisando gráfico de Kaliningrad (com NaN)
plt.plot(datas, Kaliningrad)
plt.title('Preços em Kaliningrad (com NaN)')
plt.xlabel('Meses')
plt.ylabel('Preço')
plt.grid()
plt.show()
```
![Precos em Kaliningrad](https://github.com/user-attachments/assets/8ea0a785-db71-4a0f-9138-ba196594ee87)

Corrigidos pela média dos vizinhos:

```python
Kaliningrad[4] = np.mean([Kaliningrad[3], Kaliningrad[5]])
```
![Precos em Kaliningrad corrigido](https://github.com/user-attachments/assets/56e7f8cc-dc59-4401-8d07-c181a7ac3ba4)

### 8. Cálculo de Estatísticas Simples
Cálculo de médias:

```python
np.mean(Moscow)
np.mean(Kaliningrad)
```
Cálculo de desvio padrão:

```python
np.std(Moscow)
```
### 9. Cálculo de Médias e Primeiras Tentativas de Regressão Linear
Antes de iniciar os ajustes de modelos, foi realizado o cálculo das médias de preços:

```python
print('Média Moscow:', np.mean(Moscow))
print('Média Kaliningrad:', np.mean(Kaliningrad))
```
Em seguida, iniciamos a regressão manual utilizando aproximações simples.
Primeira tentativa de ajuste utilizando uma função linear arbitrária:

```python
x = datas
y = 2*x + 80

plt.plot(datas, Moscow)
plt.plot(x, y)
plt.title('Primeira tentativa de ajuste (y=2x+80)')
plt.xlabel('Meses')
plt.ylabel('Preço')
plt.grid()
plt.show()
```
Gráfico da primeira tentativa:

![Primeira tentativa de ajuste  y=2x+80](https://github.com/user-attachments/assets/df3ef323-c4e3-4d68-b882-169e524b4937)

Cálculo do erro quadrático da primeira aproximação:

```python
print('Erro inicial:', np.sqrt(np.sum(np.power(Moscow - y, 2))))
```
Segunda tentativa de ajuste, ajustando o coeficiente angular:

```python
y = 0.52*x + 80

plt.plot(datas, Moscow)
plt.plot(x, y)
plt.title('Segunda tentativa de ajuste (y=0.52x+80)')
plt.xlabel('Meses')
plt.ylabel('Preço')
plt.grid()
plt.show()
```
Gráfico da segunda tentativa:

![Segunda tentativa de ajuste y=0 52x+80](https://github.com/user-attachments/assets/3187ac49-946b-4b46-aad9-8be8180bf494)

Cálculo do novo erro quadrático:

```python
print('Erro novo ajuste:', np.sqrt(np.sum(np.power(Moscow - y, 2))))
```

### 10. Cálculo Correto da Regressão Linear
Após as tentativas iniciais, realizamos o cálculo formal dos coeficientes da regressão linear utilizando as fórmulas matemáticas:
- Coeficiente angular (a)
- Coeficiente linear (b)

```python
  Y = Moscow
X = datas
n = np.size(Moscow)

# Cálculo dos coeficientes
a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y)) / (n*np.sum(X**2) - (np.sum(X))**2)
b = np.mean(Y) - a*np.mean(X)

# Construção da reta de regressão
y = a*X + b
```
Avaliação do erro da regressão correta (Norma 2):

```python
print('Norma 2 com ajuste correto:', np.linalg.norm(Moscow - y))
```
Visualização do ajuste final com previsões:

```python
plt.plot(datas, Moscow)
plt.plot(x, y)
plt.plot(41.5, 41.5*a + b, '*r') # Previsão para 41.5 meses
plt.plot(100, 100*a + b, '*r')   # Previsão para 100 meses
plt.title('Ajuste Final com Previsões')
plt.xlabel('Meses')
plt.ylabel('Preço')
plt.grid()
plt.show()
```
![Ajuste Final com Previsoes](https://github.com/user-attachments/assets/14ac85a8-d047-4d63-8b4b-1a19852a3dfd)

### 11. Manipulações Extras com NumPy
Criação de arrays:
```vetor_zeros = np.zeros(10)```: Criação de um vetor contendo 10 elementos, todos inicializados com o valor zero.

```vetor_uns = np.ones(10)```: Criação de um vetor de 10 elementos, todos com valor igual a um.

```vetor_identidade = np.eye(5)```: Geração de uma matriz identidade de 5x5, onde os elementos da diagonal principal são iguais a 1 e os demais são zero.

```vetor_linspace = np.linspace(0, 100, 5)```: Geração de um vetor com 5 números igualmente espaçados entre 0 e 100.

```python
vetor_zeros = np.zeros(10)
vetor_uns = np.ones(10)
vetor_identidade = np.eye(5)
vetor_linspace = np.linspace(0, 100, 5)
```
Fatiar:
```meses_pares = Moscow[1::2]```: Criação de um novo vetor contendo os elementos da variável Moscow nos índices pares (considerando que a indexação começa do zero).

```python
meses_pares = Moscow[1::2]
```
Indexação booleana:
Aplicação de uma máscara booleana para identificar quais elementos do vetor Moscow possuem valores superiores a 100.

```python
mascara_precos_altos = Moscow > 100
```
Remodelar:
Remodelagem do vetor Moscow para um array tridimensional com a forma (3, 4, 3), considerando os primeiros 36 elementos.

```python
Moscow_reshape = Moscow[:36].reshape(3,4,3)
```
Operações vetorizadas:
Aplicação de uma operação vetorizada que aumenta todos os elementos do vetor Moscow em 5%, simulando uma inflação de preços.

```python
Moscow_inflacao = Moscow * 1.05
```
Tipo de dados e cópia de arrays:
```manual_array = np.array([10, 20, 30, 40, 50])```: Criação de um array manual com os valores especificados.

```manual_array_float = manual_array.astype(float)```: Conversão do tipo de dados do array manual_array para o tipo float.

```array_copiado = manual_array.copy()```: Criação de uma cópia independente do array manual_array, evitando alterações no array original.

```python
manual_array = np.array([10, 20, 30, 40, 50])
manual_array_float = manual_array.astype(float)
array_copiado = manual_array.copy()
```

### 12. Exportação dos Dados
Salvando resultados em CSV:

```python
np.savetxt('dados.csv', dados, delimiter=',')
```

---
## Conclusão

foram realizadas tarefas essenciais no tratamento de dados com NumPy e Matplotlib, incluindo a geração de sequências de números aleatórios, a garantia de reprodutibilidade de resultados por meio da definição de sementes, o agrupamento e a manipulação de arrays, além da exportação de dados para arquivos. Essas atividades consolidam fundamentos importantes para o desenvolvimento de análises de dados consistentes e reprodutíveis.

### Referências
- [Curso NumPy: análise numérica eficiente com Python - Alura](https://www.alura.com.br/curso-online-numpy-analise-numerica-eficiente-pythons?srsltid=AfmBOopLEEvwP5m0KPhBnaxewkSazAmklgqtas7W36qMB_jKNocoVzek)

- [Documentação oficial do NumPy](https://numpy.org/doc/stable/)


