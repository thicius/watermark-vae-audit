# Auditoria de Modelos Generativos com Marca d'Água em Dados

Python
PyTorch
MNIST

## Sobre o Projeto

Este projeto investiga como **marcar um conjunto de dados de forma imperceptível** de modo que, se ele for usado no treinamento de um modelo generativo, o uso possa ser **auditado estatisticamente** a posteriori. O modelo generativo escolhido é um **Variational Autoencoder (VAE)** treinado no MNIST.

A pergunta central é: dado que o dono de uma base de imagens quer preservar a capacidade de verificar se terceiros usaram seus dados para treinar um modelo generativo, como inserir uma marca que seja ao mesmo tempo *imperceptível ao olho humano* e *detectável via teste estatístico*?

O projeto percorre três estratégias, em ordem crescente de sofisticação:

1. **Marca visível** — patch branco fixo no canto da imagem. Limite superior de detectabilidade, trivialmente removível.
2. **Marca spread-spectrum** — textura pseudo-aleatória de baixa amplitude adicionada aos pixels, imperceptível ao olho, detectável via correlação com a chave secreta do auditor.
3. **Estudo de ablação** — varredura sistemática sobre amplitude (ε) e fração de dados marcados, com heatmaps de sinal médio e t-estatístico.

Como extensão bônus, foi investigado o **efeito da capacidade do espaço latente** (LATENT_DIM ∈ {2, 16, 32}) sobre a detectabilidade da marca, mostrando que espaços latentes maiores preservam melhor os detalhes de alta frequência injetados pela marca.

---

## Estrutura do Repositório

```
.
├── watermark_vae.ipynb          # Notebook principal com todo o código e respostas
├── mnist_data/                  # Dataset MNIST (baixado automaticamente)
└── trabalho3_outputs/
    └── TVAO_2026-05-26/
        ├── vae_clean_latent16.pt        # Pesos do VAE limpo
        ├── vae_visible.pt               # Pesos do VAE treinado com marca visível
        ├── vae_ss.pt                    # Pesos do VAE treinado com marca spread-spectrum
        ├── ss_pattern.png               # Visualização do padrão secreto w
        ├── 1.5a_curvas_treinamento.png
        ├── 1.5b_reconstrucoes.png
        ├── 1.5c_espaco_latente.png
        ├── 1.5d_amostras_prior.png
        ├── 2.1_visuais_marcas.png
        ├── 2.3_auditoria_visual.png
        ├── 2.4_auditoria_estatistica.png
        ├── 3.2_ss_amostras.png
        ├── 3.4_detector_ingenuo.png
        ├── 3.6_detector_centralizado.png
        ├── 3.7_teste_hipotese.png
        └── 4.3_heatmaps_ablacao.png
```

---

## Metodologia

### Parte 1 — VAE Base

Implementação do zero de um VAE com arquitetura MLP:

- **Encoder**: 784 → 512 → 256 → (μ, log σ²) com dimensão latente configurável
- **Decoder**: latente → 256 → 512 → 784 com sigmoid na saída
- **Reparametrização**: z = μ + σ ⊙ ε, com ε ~ N(0, I)
- **Loss (ELBO)**: BCE de reconstrução + KL fechada para prior N(0, I)

A dimensão latente foi escolhida via busca em grade (LATENT_DIM ∈ {2, 4, 8, 12, 16, 20, 24, 28, 32}), com LATENT_DIM = 16 apresentando a menor loss de teste (94,31), com ganho marginal acima desse valor.

### Parte 2 — Marca Visível

Um patch branco 4×4 é estampado na posição [22:26, 22:26] de 20% das imagens de treino. A auditoria é feita calculando a intensidade média de pixel nessa região nas amostras geradas pelo VAE marcado versus o VAE limpo, com lift de **136×** entre os dois modelos.

### Parte 3 — Marca Spread-Spectrum

O padrão secreto w ∈ ℝ^(28×28) é gerado com semente do dono dos dados (OWNER_SEED), com média zero e normalizado para max|w| = 1. Cada imagem marcada é:

$$x' = \text{clamp}(x + \varepsilon \cdot w,\; 0, 1)$$

O detector de correlação calcula o score:

$$\text{score}(x) = \langle x - \mu_{\text{MNIST}},\; w \rangle$$

A centralização pela imagem média do MNIST é necessária porque E[x_clean] = μ_MNIST, o que introduz um offset sistemático no detector ingênuo. O t-estatístico de duas amostras independentes (N = 2000) atingiu **t = 19,14** com ε = 0,10 e fração = 0,20.

### Parte 4 — Estudo de Ablação

Varredura em grade sobre:
- ε ∈ {0,02; 0,05; 0,10; 0,15; 0,20}
- fração ∈ {0,02; 0,05; 0,10; 0,15; 0,20}

Para cada configuração, um VAE marcado é treinado por 10 épocas e auditado com N = 2000 amostras. Os resultados são visualizados como heatmaps de sinal médio e t-estatístico.

A configuração mínima recomendada para auditoria real é **ε = 0,05 e fração = 0,10**, que entrega t ≈ 9,1 com impacto mínimo na qualidade dos dados.

### Extensão Bônus — Efeito da Dimensão Latente

| LATENT_DIM | T-estatístico |
|:---:|:---:|
| 2 | 17,31 |
| 16 (original) | 22,30 |
| 32 | 23,27 |

Espaços latentes maiores preservam mais detalhes de alta frequência, tornando a marca mais detectável. Espaços muito pequenos atuam como filtro de baixa frequência, atenuando o sinal da marca.

---

## Como Executar

### Direto no Navegador

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Abra o notebook no Google Colab e execute as células em ordem. O MNIST é baixado automaticamente.

### Localmente

Requer Python 3.9+ e GPU recomendada para a varredura da Parte 4.

```bash
# Clone o repositório
git clone https://github.com/thicius/watermark-vae-audit.git
cd watermark-vae-audit

# Instale as dependências
pip install torch torchvision numpy matplotlib scipy

# Execute o notebook
jupyter notebook watermark_vae.ipynb
```

> **Atenção:** a varredura da Parte 4 treina 25 VAEs (5 valores de ε × 5 valores de fração). Em CPU, isso pode levar várias horas. Reduzir `EPOCHS_ABLATION` de 10 para 5 é uma alternativa viável para exploração rápida.

---

## Dependências

| Biblioteca | Uso |
|---|---|
| `torch` | VAE, treinamento, geração de amostras |
| `torchvision` | Carregamento do MNIST |
| `numpy` | PCA manual (SVD), RNG para seleção de índices |
| `matplotlib` | Todas as visualizações |
| `scipy` | Referência para t-estatístico (bônus) |

---

## Resultados Principais

- VAE base bem ajustado: loss de treino e teste convergem juntas sem overfitting.
- Marca visível detectável com lift de 136× na intensidade média do patch.
- Marca spread-spectrum detectável com t = 19,14 usando apenas 2000 amostras.
- Configuração mínima prática: ε = 0,05, fração = 10% dos dados, t ≈ 9,1.
- Dimensão latente maior preserva melhor o sinal da marca (t sobe de 17,31 para 23,27).
