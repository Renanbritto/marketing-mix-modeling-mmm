# 📊 Marketing Mix Modeling (MMM) & Budget Optimizer

[![CI Pipeline](https://github.com/Renanbritto/marketing-mix-modeling-mmm/actions/workflows/ci.yml/badge.svg)](https://github.com/Renanbritto/marketing-mix-modeling-mmm/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)

Motor analítico e preditivo de **Marketing Mix Modeling (MMM)** com modelagem econométrica de carryover temporal (**Adstock geométrico**), curvas de saturação com rendimentos decrescentes (**Curvas de Hill sigmoides**), regressão regularizada **Ridge** e otimizador de alocação de orçamento multicanal com restrições via **SciPy SLSQP**.

---

## 🎯 Problema de Negócio & Objetivos

Em cenários com múltiplos canais de aquisição (Meta Ads, Google Ads, TikTok Ads, Influenciadores, CRM) e restrições de privacidade (LGPD, iOS 14.5+ ATT, deprecation de cookies 3rd-party), modelos determinísticos de atribuição baseados em cliques (Last-Click, First-Click) tornam-se enviesados e subestimam o efeito residual e de topo de funil da mídia.

Este projeto resolve esse desafio implementando um framework econométrico estatisticamente rigoroso que:
1. **Mede o Efeito Residual (Adstock):** Modela o decaimento temporal e retenção de memória da mídia ao longo das semanas.
2. **Identifica Saturação de Canal (Hill Saturation):** Detecta o ponto exato em que investir mais em um canal gera retorno marginal decrescente (mROAS < 1.0).
3. **Decompõe a Receita Incremental:** Isola o baseline orgânico (vendas que ocorreriam sem marketing), efeito da sazonalidade e a contribuição real de cada canal de mídia.
4. **Otimiza Orçamentos com Restrições Matemáticas:** Encontra a distribuição ótima de budget que maximiza a receita esperada respeitando limites mínimos e máximos por canal.

---

## 📐 Fundamentação Econométrica & Matemática

### 1. Adstock Geométrico (Efeito Carryover)
Modela a retenção residual do impacto publicitário através de uma média móvel ponderada decrescente:

$$\text{Adstock}_t = x_t + \lambda \cdot \text{Adstock}_{t-1}$$

Onde:
- $x_t$ é o investimento da semana $t$.
- $\lambda \in [0, 1]$ é a taxa de retenção/decaimento (*decay rate*).
- Meia-vida da mídia: $T_{1/2} = \frac{\ln(0.5)}{\ln(\lambda)}$.

### 2. Curva de Saturação de Hill (Rendimentos Decrescentes)
Mapeia o investimento transformado pelo Adstock em uma escala sigmoidal $[0, 1]$:

$$\text{Hill}(x) = \frac{x^S}{K^S + x^S}$$

Onde:
- $K > 0$ é o ponto de meia-saturação (*half-saturation point*), nível de gasto que atinge 50% do efeito máximo.
- $S > 0$ é o parâmetro de inclinação/forma (*shape/steepness*).

### 3. ROAS Marginal (mROAS)
A derivada analítica da função de Hill multiplicada pelo coeficiente de regressão $\beta$:

$$\text{mROAS}(x) = \frac{d}{dx} \left[ \beta \cdot \frac{x^S}{K^S + x^S} \right] = \frac{\beta \cdot S \cdot K^S \cdot x^{S-1}}{(K^S + x^S)^2}$$

Classificação de Saúde do Canal:
- **`SCALING`** ($mROAS \ge 3.0$): Alto retorno marginal. Canal pronto para forte aceleração.
- **`EFFICIENT`** ($1.5 \le mROAS < 3.0$): Operação saudável e balanceada.
- **`SATURATED`** ($0.8 \le mROAS < 1.5$): Ponto de inflexão. Retornos decrescentes evidentes.
- **`OVER_INVESTED`** ($mROAS < 0.8$): Destruição de valor marginal. Realocação recomendada.

### 4. Otimização de Orçamento (SciPy SLSQP)
Maximiza a receita total sob restrições lineares e de fronteira:

$$\max_{\mathbf{w}} \quad \text{Receita}(\mathbf{w}) = \beta_0 + \sum_{i=1}^M \beta_i \cdot \text{Hill}_i(\text{Adstock}_i(w_i))$$

$$\text{sujeito a:} \quad \sum_{i=1}^M w_i = B_{\text{total}}, \quad L_i \le w_i \le U_i, \quad \forall i$$

---

## 🏛️ Arquitetura do Repositório

```
marketing-mix-modeling-mmm/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Pipeline CI (Lint + Pytest 3.11/3.12)
├── src/
│   ├── __init__.py
│   ├── api/                           # Camada HTTP e Controladores
│   │   ├── __init__.py
│   │   ├── routes.py                  # Endpoints REST (Simulacao, Otimizacao, Metricas)
│   │   └── server.py                  # Factory FastAPI, CORS, Jinja2
│   ├── config/                        # Configuracoes Centralizadas
│   │   ├── __init__.py
│   │   └── settings.py               # Pydantic Settings & Defaults de Canais
│   ├── data/                          # Geracao e Ingestao de Dados
│   │   ├── __init__.py
│   │   └── sample_generator.py        # Gerador estocastico com sazonalidade e ruido
│   ├── domain/                        # Entidades e Contratos
│   │   ├── __init__.py
│   │   ├── models.py                  # Enums, Dataclasses e Health Status
│   │   └── schemas.py                 # Pydantic Request/Response Schemas
│   ├── engine/                        # Motor Econometrico Core
│   │   ├── __init__.py
│   │   ├── transformations.py         # Funcoes puras NumPy (Adstock, Hill, mROAS)
│   │   ├── mmm_model.py              # Pipeline completo (Fit, Predict, Decompose)
│   │   └── optimizer.py               # Otimizador SciPy SLSQP Constrained
│   ├── templates/                     # Interface Web
│   │   └── dashboard.html             # Dashboard interativo com Plotly.js
│   └── static/                        # Assets estaticos
├── tests/                             # Suite de Testes Automatizados (26+ testes)
│   ├── __init__.py
│   ├── test_transformations.py        # Testes de Adstock e Hill
│   ├── test_mmm_model.py             # Testes do ajuste, R2 e decomposicao
│   ├── test_optimizer.py              # Testes do algoritmo de otimizacao
│   ├── test_sample_generator.py       # Testes do gerador estocastico
│   └── test_api.py                    # Testes de integracao dos endpoints
├── output/                            # Relatorios analiticos exportados
├── main.py                            # CLI unificada (--run-pipeline, --serve-api)
├── Dockerfile                         # Build multi-stage minimalista
├── docker-compose.yml                 # Orquestracao do servico local
├── requirements.txt                   # Dependencias diretas
├── pyproject.toml                     # Metadados e configuracao de ferramentas
├── .gitignore
├── LICENSE                            # MIT License
└── README.md                          # Documentacao tecnica
```

---

## 🚀 Instalação e Execução

### Opção 1: Ambiente Local (Python 3.11+)

```bash
# 1. Clonar o repositorio
git clone https://github.com/Renanbritto/marketing-mix-modeling-mmm.git
cd marketing-mix-modeling-mmm

# 2. Criar e ativar ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Opção 2: Docker & Docker Compose

```bash
# Construir e rodar o container
docker-compose up --build
```
Acesse o dashboard em: **`http://localhost:8000`**

---

## 💻 Uso via Linha de Comando (CLI)

O arquivo `main.py` disponibiliza comandos completos:

```bash
# Executar pipeline analitico completo (treino + decomposicao + otimizacao + exportacao)
python main.py --run-pipeline

# Iniciar o servidor HTTP FastAPI e o Dashboard interativo
python main.py --serve-api --host 0.0.0.0 --port 8000

# Executar suite completa de testes automatizados com Pytest
python main.py --run-tests
```

---

## 🌐 Endpoints da API REST

Documentação interativa Swagger disponível em: **`http://localhost:8000/docs`**

| Método | Rota | Descrição |
|:---|:---|:---|
| `GET` | `/` | Dashboard analítico interativo em HTML/Plotly |
| `GET` | `/health` | Health check operacional do container |
| `GET` | `/api/v1/model/metrics` | Métricas de ajuste econométrico ($R^2$, MAPE, MAE, NRMSE) |
| `GET` | `/api/v1/model/decomposition` | Decomposição temporal de receita (semana a semana) |
| `GET` | `/api/v1/model/response-curves` | Curvas de saturação Hill e pontos de corte por canal |
| `POST` | `/api/v1/simulate` | Simulação what-if de cenários customizados de investimento |
| `POST` | `/api/v1/optimize` | Otimização matemática de orçamento com restrições mín/máx |
| `GET` | `/api/v1/channels/summary` | Resumo executivo de ROAS, mROAS, saúde e recomendações |

### Exemplo de Chamada de Otimização (`cURL`)

```bash
curl -X POST "http://localhost:8000/api/v1/optimize" \
     -H "Content-Type: application/json" \
     -d '{
       "total_budget": 100000.0,
       "min_per_channel": { "crm_email": 2000.0 },
       "max_per_channel": { "meta_ads": 50000.0 }
     }'
```

---

## 🧪 Testes Automatizados

A suíte conta com mais de 26 testes cobrindo testes unitários, estabilidade matemática e integração de API:

```bash
pytest -v tests/
```

---

## 📄 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para obter detalhes.
